"""Comprehensive unit and integration test suite for the TruthLayer Fact-Check Agent pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.cache import AuditReportCache
from src.claim_extractor import ClaimExtractor
from src.config import AppSettings
from src.demo_data import build_demo_trap_report
from src.models import (
    ClaimCategory,
    ExtractedClaim,
    PageContent,
    TrapType,
    Verdict,
    WebSource,
)
from src.pdf_parser import (
    compute_pdf_sha256,
    extract_pdf_pages,
    format_pages_for_llm,
    parse_pdf,
    table_to_markdown,
)
from src.report_exporter import export_csv, export_html, export_json, export_markdown
from src.search_client import LiveSearchClient, normalize_url, score_source_authority
from src.verifier import FactCheckEngine, enforce_citation_grounding


def _build_minimal_pdf(content_stream: str) -> bytes:
    """Build a valid 1-page PDF byte payload in pure Python (zero external PDF writer dependency)."""
    stream_bytes = content_stream.encode("latin-1")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\n"
            b"endobj\n"
        ),
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            f"5 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        ),
    ]

    buf = bytearray(b"%PDF-1.4\n")
    offsets: List[int] = [0]
    for obj in objects:
        offsets.append(len(buf))
        buf.extend(obj)

    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    buf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    buf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("latin-1")
    )
    return bytes(buf)


def _build_table_pdf_bytes() -> bytes:
    """Build a real 1-page PDF containing body text AND a ruled 2x2 table parsable by pdfplumber."""
    # Draw outer box and grid lines for a 2-column x 3-row table + place text inside cells
    stream = "\n".join(
        [
            "BT /F1 12 Tf 72 740 Td (Quarterly Financial Summary Report) Tj ET",
            "0.8 w",
            # Horizontal grid lines (y = 700, 675, 650, 625)
            "72 700 m 472 700 l S",
            "72 675 m 472 675 l S",
            "72 650 m 472 650 l S",
            "72 625 m 472 625 l S",
            # Vertical grid lines (x = 72, 272, 472)
            "72 700 m 72 625 l S",
            "272 700 m 272 625 l S",
            "472 700 m 472 625 l S",
            # Row 1 (Headers)
            "BT /F1 10 Tf 82 684 Td (Company) Tj ET",
            "BT /F1 10 Tf 282 684 Td (FY2024 Revenue) Tj ET",
            # Row 2
            "BT /F1 10 Tf 82 659 Td (Apple Inc) Tj ET",
            "BT /F1 10 Tf 282 659 Td ($180B claimed) Tj ET",
            # Row 3
            "BT /F1 10 Tf 82 634 Td (NVIDIA Corp) Tj ET",
            "BT /F1 10 Tf 282 634 Td ($27B claimed) Tj ET",
        ]
    )
    return _build_minimal_pdf(stream)


class FakeLLMClient:
    """Deterministic mock LLM client with call counting for offline CI testing."""

    def __init__(self, extraction_payload: Dict[str, Any], verdict_map: Dict[str, Dict[str, Any]]):
        self.extraction_payload = extraction_payload
        self.verdict_map = verdict_map
        self.call_count = 0

    def complete_json(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        self.call_count += 1
        if "Extract up to" in prompt:
            return self.extraction_payload
        for key, response in self.verdict_map.items():
            if key in prompt:
                return response
        return {
            "verdict": "False",
            "trap_type": "Fabricated Claim",
            "confidence": 0.91,
            "explanation": "No evidence found.",
            "corrected_fact": "Claim could not be substantiated.",
            "cited_urls": [],
        }


class FakeSearchClient:
    """Deterministic mock dual-query web search client with call counting."""

    def __init__(self, results_by_query: Dict[str, List[WebSource]]):
        self.results_by_query = results_by_query
        self.call_count = 0
        self.queries_seen: List[tuple[str, str]] = []

    def search(self, query: str) -> List[WebSource]:
        self.call_count += 1
        return self.results_by_query.get(query, [])

    def search_dual(self, primary_query: str, adversarial_query: str | None = None) -> List[WebSource]:
        self.call_count += 2 if adversarial_query else 1
        self.queries_seen.append((primary_query, adversarial_query or ""))
        combined: List[WebSource] = []
        combined.extend(self.results_by_query.get(primary_query, []))
        if adversarial_query:
            combined.extend(self.results_by_query.get(adversarial_query, []))
        return combined


# ---------------------------------------------------------------------------
# 1. PDF Ingestion, Table Extraction, SHA-256 Hashing & Scanned PDF Detection
# ---------------------------------------------------------------------------

def test_pdf_parser_reads_sample_trap_document_and_sha256() -> None:
    sample_pdf = Path(__file__).resolve().parent.parent / "sample_docs" / "sample_trap_document.pdf"
    assert sample_pdf.exists(), "sample_trap_document.pdf must exist"

    raw_bytes = sample_pdf.read_bytes()
    parsed = parse_pdf(raw_bytes)
    assert parsed.total_pages == 1
    assert len(parsed.pages) == 1
    assert not parsed.is_scanned_or_image_only
    assert parsed.warning_message is None
    assert parsed.sha256_hash == compute_pdf_sha256(raw_bytes)
    assert len(parsed.sha256_hash) == 64
    assert "November 30, 2022" in parsed.pages[0].text
    assert "1.21 billion" in parsed.pages[0].text

    pages_legacy, total_legacy = extract_pdf_pages(str(sample_pdf))
    assert total_legacy == 1 and len(pages_legacy) == 1

    formatted = format_pages_for_llm(parsed.pages)
    assert "=== [Page 1] ===" in formatted


def test_pdf_parser_extracts_markdown_tables_with_pdfplumber() -> None:
    pdf_bytes = _build_table_pdf_bytes()
    parsed = parse_pdf(pdf_bytes)

    assert parsed.total_pages == 1
    assert parsed.table_count >= 1
    assert len(parsed.pages) == 1
    assert len(parsed.pages[0].tables_markdown) >= 1

    md_table = parsed.pages[0].tables_markdown[0]
    assert "| Company | FY2024 Revenue |" in md_table
    assert "| --- | --- |" in md_table
    assert "Apple Inc" in md_table
    assert "$180B claimed" in md_table
    assert "[Table 1]" in parsed.pages[0].text


def test_table_to_markdown_escapes_pipes_and_drops_empty_rows() -> None:
    raw = [
        ["Metric | Name", "2024", None],
        [None, None, None],
        ["Revenue", "$391B", ""],
    ]
    md = table_to_markdown(raw)
    assert "| Metric \\| Name | 2024 |" in md
    assert "| Revenue | $391B |" in md


def test_pdf_parser_detects_scanned_or_image_only_pdf() -> None:
    # A valid PDF with only a vector rectangle and zero selectable text
    scanned_like_pdf = _build_minimal_pdf("0.5 g 50 50 500 700 re f")
    parsed = parse_pdf(scanned_like_pdf)

    assert parsed.total_pages == 1
    assert parsed.is_scanned_or_image_only is True
    assert parsed.warning_message is not None
    assert "OCR" in parsed.warning_message


# ---------------------------------------------------------------------------
# 2. Dual-Query Extraction, Search Deduplication, Authority Ranking & Fallback
# ---------------------------------------------------------------------------

def test_claim_extractor_generates_dual_queries_and_deduplicates() -> None:
    payload = {
        "claims": [
            {
                "claim_id": 1,
                "claim_text": "Apple FY2024 revenue was $180 billion.",
                "original_quote": "Apple's total annual revenue in fiscal year 2024 was $180 billion",
                "page_number": 1,
                "category": "Financial",
                "primary_query": "Apple FY2024 annual revenue Form 10-K SEC",
                "adversarial_query": "Apple FY2024 revenue 180 billion vs 391 billion fact check",
            },
            {
                # Duplicate claim should be skipped
                "claim_id": 2,
                "claim_text": "Apple FY2024 revenue was $180 billion!",
                "original_quote": "Apple FY2024 revenue was $180 billion",
                "page_number": 1,
                "category": "Financial",
                "primary_query": "Apple FY2024 revenue",
                "adversarial_query": "Apple FY2024 revenue fact check",
            },
        ]
    }
    llm = FakeLLMClient(payload, {})
    extractor = ClaimExtractor(llm, max_claims=10)  # type: ignore[arg-type]
    claims = extractor.extract([PageContent(page_number=1, text="Sample doc")])

    assert len(claims) == 1
    assert claims[0].primary_query == "Apple FY2024 annual revenue Form 10-K SEC"
    assert "fact check" in claims[0].adversarial_query
    assert claims[0].search_query == claims[0].primary_query


def test_dual_query_search_deduplication_and_authority_ranking() -> None:
    assert normalize_url("https://www.sec.gov/filing/?utm_source=x&id=10") == normalize_url(
        "https://sec.gov/filing?id=10"
    )
    gov_score, gov_tier = score_source_authority("https://www.sec.gov/Archives/edgar/data/aapl-10k.htm")
    reuters_score, _ = score_source_authority("https://www.reuters.com/technology/apple-revenue/")
    blog_score, blog_tier = score_source_authority("https://random-seo.blogspot.com/apple-guess")

    assert gov_score > reuters_score > blog_score
    assert gov_tier == "Primary / Official"
    assert blog_tier == "Low-Signal"

    client = LiveSearchClient(tavily_api_key="dummy-key", max_results=5)

    primary_results = [
        WebSource(
            title="Random SEO Blog Post",
            url="https://random.medium.com/apple-rev",
            snippet="Someone guessed Apple revenue.",
            provider="tavily",
            query_origin="primary",
        ),
        WebSource(
            title="Apple Form 10-K Filing",
            url="https://www.sec.gov/edgar/apple-2024/?utm_source=tavily",
            snippet="Net sales $391.04B.",
            published_date="2024-11-01",
            provider="tavily",
            query_origin="primary",
        ),
    ]
    adversarial_results = [
        WebSource(
            title="Apple Form 10-K Filing",
            url="https://sec.gov/edgar/apple-2024",
            snippet="Apple Inc. reported total net sales of $391.035 billion for fiscal year 2024.",
            published_date="2024-11-01",
            provider="tavily",
            query_origin="adversarial",
        ),
        WebSource(
            title="Reuters Apple Earnings",
            url="https://www.reuters.com/technology/apple-fy24/",
            snippet="Apple annual revenue reached $391 billion.",
            published_date="2024-10-31",
            provider="tavily",
            query_origin="adversarial",
        ),
    ]

    with patch.object(
        client,
        "_execute_single_query",
        side_effect=[primary_results, adversarial_results],
    ):
        fused = client.search_dual("Apple FY2024 revenue", "Apple FY2024 revenue 2025 2026 fact check")

    # SEC duplicate collapsed into single corroborated result and ranked #1 ahead of Reuters and Medium
    assert len(fused) == 3
    assert fused[0].domain == "sec.gov"
    assert fused[0].query_origin == "both"
    assert "$391.035 billion" in fused[0].snippet
    assert fused[1].domain == "reuters.com"
    assert fused[2].domain == "random.medium.com"


def test_tavily_failure_falls_back_to_duckduckgo() -> None:
    client = LiveSearchClient(tavily_api_key="bad-key", max_results=4)
    ddg_source = WebSource(
        title="UN World Population Prospects",
        url="https://population.un.org/wpp/",
        snippet="India is the most populous country with over 1.44 billion people.",
        provider="duckduckgo",
    )
    with patch.object(client, "_search_tavily", return_value=[]), patch.object(
        client, "_search_duckduckgo", return_value=[ddg_source]
    ) as mock_ddg:
        res = client.search("India population UN")

    mock_ddg.assert_called_once()
    assert len(res) == 1
    assert res[0].provider == "duckduckgo"
    assert res[0].domain == "population.un.org"


# ---------------------------------------------------------------------------
# 3. Citation Grounding Guardrail
# ---------------------------------------------------------------------------

def test_citation_grounding_guardrail_strips_hallucinated_urls() -> None:
    retrieved = [
        WebSource(
            title="Apple FY2024 Results",
            url="https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/",
            snippet="Apple posted FY2024 revenue of $391.0 billion.",
            domain="apple.com",
            authority_score=11,
            authority_tier="Primary / Official",
        )
    ]

    ordered, grounded, stripped, clean_expl, clean_corr = enforce_citation_grounding(
        sources=retrieved,
        raw_cited_urls=[
            "https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/",
            "https://hallucinated-example.com/fake-sec-filing",
        ],
        explanation=(
            "Per [Fake Filing](https://hallucinated-example.com/inline-lie) and Apple Newsroom "
            "(https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/), revenue was $391B."
        ),
        corrected_fact="Apple FY2024 revenue was $391.04B (see https://invented.org/made-up).",
    )

    assert grounded == ["https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/"]
    assert "https://hallucinated-example.com/fake-sec-filing" in stripped
    assert "https://hallucinated-example.com/inline-lie" in stripped
    assert "https://invented.org/made-up" in stripped
    assert "hallucinated-example.com" not in clean_expl
    assert "invented.org" not in (clean_corr or "")
    assert "Fake Filing" in clean_expl
    assert len(ordered) == 1


# ---------------------------------------------------------------------------
# 4. End-to-End Pipeline: All 3 Verdicts, Trap Types, SHA-256 Cache & 4 Exports
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline_verdicts_traps_cache_and_all_four_exports() -> None:
    sample_pdf = Path(__file__).resolve().parent.parent / "sample_docs" / "sample_trap_document.pdf"
    parsed = parse_pdf(sample_pdf.read_bytes())

    extraction_payload = {
        "claims": [
            {
                "claim_id": 1,
                "claim_text": "OpenAI launched ChatGPT on November 30, 2022.",
                "original_quote": "OpenAI officially launched ChatGPT to the public on November 30, 2022",
                "page_number": 1,
                "category": "Date / Timeline",
                "primary_query": "OpenAI ChatGPT launch date",
                "adversarial_query": "OpenAI ChatGPT launch November 30 2022 official",
            },
            {
                "claim_id": 2,
                "claim_text": "India's current population is 1.21 billion people.",
                "original_quote": "India's population stands at 1.21 billion people",
                "page_number": 1,
                "category": "Statistic",
                "primary_query": "India current population UN",
                "adversarial_query": "India population 2025 2026 surpassed China UN",
            },
            {
                "claim_id": 3,
                "claim_text": "A 2024 Gartner report stated 99% of enterprises quit cloud computing.",
                "original_quote": "99% of enterprises quit cloud computing",
                "page_number": 1,
                "category": "Market Claim",
                "primary_query": "Gartner enterprise cloud adoption 2024",
                "adversarial_query": "Gartner 99 percent quit cloud fact check",
            },
            {
                "claim_id": 4,
                "claim_text": "NVIDIA acquired OpenAI in March 2025 for $420 billion.",
                "original_quote": "NVIDIA acquired OpenAI in March 2025 for $420 billion",
                "page_number": 1,
                "category": "Financial",
                "primary_query": "NVIDIA acquired OpenAI 420 billion",
                "adversarial_query": "did NVIDIA acquire OpenAI 2025 2026 fact check",
            },
        ]
    }

    verdict_map = {
        "OpenAI launched ChatGPT": {
            "verdict": "Verified",
            "trap_type": "None",
            "confidence": 0.99,
            "explanation": "Confirmed by OpenAI's official announcement.",
            "corrected_fact": None,
            "cited_urls": ["https://openai.com/index/chatgpt/"],
        },
        "India's current population": {
            "verdict": "Inaccurate",
            "trap_type": "Outdated Stat",
            "confidence": 0.97,
            "explanation": "1.21 billion is from India's 2011 census and is outdated.",
            "corrected_fact": "India's population exceeded 1.45 billion in 2025–2026 and is #1 globally (UN).",
            "cited_urls": [
                "https://population.un.org/",
                "https://fake-hallucinated-url.invalid/bogus",
            ],
        },
        "99% of enterprises quit cloud": {
            "verdict": "Inaccurate",
            "trap_type": "Misattributed Source",
            "confidence": 0.94,
            "explanation": "Gartner reported cloud spending growth, not an exodus.",
            "corrected_fact": "Gartner projected worldwide public cloud spending to grow >20% in 2024–2025.",
            "cited_urls": ["https://www.gartner.com/en/newsroom"],
        },
    }

    search_results = {
        "OpenAI ChatGPT launch date": [
            WebSource(
                title="Introducing ChatGPT - OpenAI",
                url="https://openai.com/index/chatgpt/",
                snippet="ChatGPT was launched on November 30, 2022.",
                provider="tavily",
            )
        ],
        "India current population UN": [
            WebSource(
                title="World Population Prospects - United Nations",
                url="https://population.un.org/",
                snippet="India surpassed China in 2023 with over 1.44 billion inhabitants.",
                provider="tavily",
            )
        ],
        "Gartner enterprise cloud adoption 2024": [
            WebSource(
                title="Gartner Forecasts Worldwide Public Cloud Spending",
                url="https://www.gartner.com/en/newsroom",
                snippet="Public cloud end-user spending continues rapid double-digit growth.",
                provider="tavily",
            )
        ],
        # Empty results for claim #4 triggers automatic False + Fabricated Claim
        "NVIDIA acquired OpenAI 420 billion": [],
    }

    settings = AppSettings(
        gemini_api_key="test-key",
        openai_api_key=None,
        tavily_api_key="test-tavily",
        max_workers=2,
        enable_cache=True,
    )
    fake_llm = FakeLLMClient(extraction_payload, verdict_map)
    fake_search = FakeSearchClient(search_results)
    shared_cache = AuditReportCache(max_entries=16)

    engine = FactCheckEngine(
        settings=settings,
        llm_client=fake_llm,  # type: ignore[arg-type]
        search_client=fake_search,  # type: ignore[arg-type]
        report_cache=shared_cache,
    )

    progress_events: List[int] = []
    report = engine.audit_document(
        document_name="sample_trap_document.pdf",
        pages=parsed.pages,
        total_pages=parsed.total_pages,
        document_sha256=parsed.sha256_hash,
        progress_cb=lambda done, total, msg: progress_events.append(done),
    )

    # Verdict, trap type, grounding, and telemetry checks
    assert report.total_claims == 4
    assert report.verified_count == 1
    assert report.inaccurate_count == 2
    assert report.false_count == 1
    assert report.trust_score == 25.0
    assert report.document_sha256 == parsed.sha256_hash
    assert report.telemetry.cache_hit is False
    assert report.telemetry.llm_calls == 4  # 1 extraction + 3 adjudicated (claim #4 had 0 search results)
    assert report.telemetry.search_calls == 8  # 4 claims x 2 dual queries
    assert report.telemetry.ungrounded_urls_stripped == 1

    assert report.results[0].verdict == Verdict.VERIFIED
    assert report.results[0].trap_type == TrapType.NONE
    assert report.results[0].corrected_fact is None

    assert report.results[1].verdict == Verdict.INACCURATE
    assert report.results[1].trap_type == TrapType.OUTDATED_STAT
    assert "1.45 billion" in (report.results[1].corrected_fact or "")
    assert report.results[1].cited_urls == ["https://population.un.org/"]
    assert "https://fake-hallucinated-url.invalid/bogus" in report.results[1].stripped_ungrounded_urls

    assert report.results[2].verdict == Verdict.INACCURATE
    assert report.results[2].trap_type == TrapType.MISATTRIBUTED_SOURCE

    assert report.results[3].verdict == Verdict.FALSE
    assert report.results[3].trap_type == TrapType.FABRICATED_CLAIM
    assert report.results[3].corrected_fact is not None
    assert len(progress_events) >= 4

    # SHA-256 Cache Hit Test — zero additional LLM or search calls
    calls_before = (fake_llm.call_count, fake_search.call_count)
    cached_report = engine.audit_document(
        document_name="renamed_copy.pdf",
        pages=parsed.pages,
        total_pages=parsed.total_pages,
        document_sha256=parsed.sha256_hash,
    )
    assert cached_report.telemetry.cache_hit is True
    assert cached_report.telemetry.llm_calls == 0
    assert cached_report.telemetry.search_calls == 0
    assert cached_report.document_name == "renamed_copy.pdf"
    assert (fake_llm.call_count, fake_search.call_count) == calls_before

    # Bypass cache forces a fresh audit
    fresh_report = engine.audit_document(
        document_name="sample_trap_document.pdf",
        pages=parsed.pages,
        total_pages=parsed.total_pages,
        document_sha256=parsed.sha256_hash,
        bypass_cache=True,
    )
    assert fresh_report.telemetry.cache_hit is False
    assert fake_llm.call_count > calls_before[0]

    # Verify all 4 export formats: JSON, CSV, Markdown, Standalone HTML
    json_out = json.loads(export_json(report))
    assert json_out["document_name"] == "sample_trap_document.pdf"
    assert json_out["document_sha256"] == parsed.sha256_hash
    assert json_out["telemetry"]["ungrounded_urls_stripped"] == 1
    assert len(json_out["results"]) == 4

    csv_out = export_csv(report)
    assert "Claim ID,Page,Category,Verdict,Trap Type,Confidence" in csv_out
    assert "Outdated Stat" in csv_out
    assert "Fabricated Claim" in csv_out

    md_out = export_markdown(report)
    assert "# Fact-Check Audit Report: sample_trap_document.pdf" in md_out
    assert "Outdated Stat" in md_out
    assert "Verified Real Fact" in md_out

    html_out = export_html(report)
    assert "<!DOCTYPE html>" in html_out
    assert "TruthLayer AI" in html_out
    assert "TRAP: Outdated Stat" in html_out
    assert "Verified Real Fact" in html_out
    assert parsed.sha256_hash[:16] in html_out


def test_demo_report_validates_and_exports_cleanly() -> None:
    demo = build_demo_trap_report()
    assert demo.total_claims == 6
    assert demo.verified_count == 2
    assert demo.inaccurate_count == 2
    assert demo.false_count == 2
    assert TrapType.OUTDATED_STAT.value in demo.trap_breakdown
    assert TrapType.NUMERIC_DISTORTION.value in demo.trap_breakdown
    assert TrapType.FABRICATED_CLAIM.value in demo.trap_breakdown

    html_str = export_html(demo)
    assert "<html" in html_str
    assert "TruthLayer" in html_str
    assert "Outdated Stat" in export_markdown(demo)
    assert "Numeric Distortion" in export_csv(demo)
    assert len(json.loads(export_json(demo))["results"]) == 6


def test_html_export_escapes_script_injection_and_mandatory_corrected_fact() -> None:
    from src.models import FactCheckReport, VerificationResult

    xss_claim = ExtractedClaim(
        claim_id=1,
        claim_text="Revenue was <script>alert(1)</script> $10B in 2024.",
        original_quote="<script>alert(1)</script>",
        page_number=1,
        category=ClaimCategory.FINANCIAL,
        primary_query="company 2024 revenue",
    )
    settings = AppSettings(gemini_api_key="k", openai_api_key=None, tavily_api_key="t")
    # LLM returns Inaccurate with empty corrected_fact and trap_type=None -> engine enforces both!
    fake_llm = FakeLLMClient(
        {"claims": []},
        {
            "Revenue was": {
                "verdict": "Inaccurate",
                "trap_type": "None",
                "confidence": 0.88,
                "explanation": "Actual revenue was $42B.",
                "corrected_fact": None,
                "cited_urls": ["https://sec.gov/10k"],
            }
        },
    )
    fake_search = FakeSearchClient(
        {
            "company 2024 revenue": [
                WebSource(
                    title="Form 10-K",
                    url="https://sec.gov/10k",
                    snippet="FY2024 revenue was $42B.",
                    provider="tavily",
                )
            ]
        }
    )
    engine = FactCheckEngine(
        settings=settings,
        llm_client=fake_llm,  # type: ignore[arg-type]
        search_client=fake_search,  # type: ignore[arg-type]
    )
    res: VerificationResult = engine.verify_claim(xss_claim)
    assert res.verdict == Verdict.INACCURATE
    assert res.trap_type == TrapType.NUMERIC_DISTORTION
    assert res.corrected_fact is not None and len(res.corrected_fact) > 5

    rep = FactCheckReport(
        document_name="xss_<img src=x>.pdf",
        page_count=1,
        llm_provider="gemini",
        search_provider="tavily",
        results=[res],
    )
    html_out = export_html(rep)
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out


def test_llm_client_retries_rate_limits_and_fails_over_to_openai() -> None:
    from src.llm_client import LLMClient

    settings = AppSettings(
        gemini_api_key="gem-key",
        openai_api_key="oai-key",
        tavily_api_key=None,
    )
    client = LLMClient(settings)

    # First test: Gemini recovers after a 429 rate-limit error
    with patch("src.llm_client.time.sleep", return_value=None), patch.object(
        client,
        "_call_gemini",
        side_effect=[RuntimeError("429 RESOURCE_EXHAUSTED"), '{"ok": true}'],
    ) as mock_gem:
        result = client.complete_json("test prompt", max_retries=2)
        assert result == {"ok": True}
        assert mock_gem.call_count == 2
        assert client.call_count == 2

    # Second test: Gemini fails permanently -> seamless failover to OpenAI
    client.reset_call_count()
    with patch("src.llm_client.time.sleep", return_value=None), patch.object(
        client, "_call_gemini", side_effect=RuntimeError("429 quota exceeded")
    ), patch.object(
        client, "_call_openai", return_value='{"fallback": "openai"}'
    ) as mock_oai:
        result2 = client.complete_json("test prompt", max_retries=1)
        assert result2 == {"fallback": "openai"}
        mock_oai.assert_called_once()


def test_llm_client_gemini_multi_model_fallback_on_404_or_503() -> None:
    from src.llm_client import LLMClient

    settings = AppSettings(
        gemini_api_key="gem-key",
        openai_api_key=None,
        tavily_api_key=None,
        gemini_model="gemini-deprecated-test",
    )
    client = LLMClient(settings)

    mock_sdk = MagicMock()
    good_resp = MagicMock()
    good_resp.text = '{"status": "recovered"}'
    mock_sdk.models.generate_content.side_effect = [
        RuntimeError("404 NOT_FOUND: model no longer available"),
        RuntimeError("503 UNAVAILABLE: high demand"),
        good_resp,
    ]

    with patch.object(client, "_get_gemini_client", return_value=mock_sdk):
        out = client.complete_json("verify claim", max_retries=0)

    assert out == {"status": "recovered"}
    assert mock_sdk.models.generate_content.call_count == 3
    assert client.model_name == "gemini-3.1-flash-lite"


def test_config_lookup_supports_aliases_and_nested_toml_sections() -> None:
    from src.config import GEMINI_KEY_ALIASES, _lookup_mapping_secret

    assert _lookup_mapping_secret({"GOOGLE_API_KEY": "abc123"}, GEMINI_KEY_ALIASES) == "abc123"
    assert (
        _lookup_mapping_secret({"secrets": {"gemini_api_key": "nested-key"}}, GEMINI_KEY_ALIASES)
        == "nested-key"
    )
    assert _lookup_mapping_secret({"OTHER_KEY": "nope"}, GEMINI_KEY_ALIASES) is None




