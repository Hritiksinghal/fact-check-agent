"""Concurrent trap-document adjudication engine with citation grounding guardrail, telemetry, and SHA-256 caching."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

from src.cache import AuditReportCache
from src.claim_extractor import ClaimExtractor
from src.config import AppSettings
from src.llm_client import LLMClient
from src.models import (
    AuditTelemetry,
    ExtractedClaim,
    FactCheckReport,
    PageContent,
    TrapType,
    Verdict,
    VerificationResult,
    WebSource,
)
from src.prompts import build_verification_prompt
from src.search_client import LiveSearchClient, normalize_url

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

_INLINE_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _format_evidence(sources: List[WebSource]) -> str:
    if not sources:
        return "NO LIVE WEB EVIDENCE FOUND FOR THIS QUERY."
    blocks = []
    for idx, src in enumerate(sources, start=1):
        date_str = f" | Published: {src.published_date}" if src.published_date else ""
        origin_str = f" | Origin: {src.query_origin}"
        blocks.append(
            f"[Source {idx}] {src.title} ({src.domain} · {src.authority_tier}{date_str}{origin_str})\n"
            f"URL: {src.url}\n"
            f"Excerpt: {src.snippet}"
        )
    return "\n\n".join(blocks)


def _parse_verdict(raw: str) -> Verdict:
    cleaned = (raw or "").strip().lower()
    if cleaned == "verified":
        return Verdict.VERIFIED
    if cleaned == "inaccurate":
        return Verdict.INACCURATE
    return Verdict.FALSE


def _parse_trap_type(raw: str, verdict: Verdict) -> TrapType:
    if verdict == Verdict.VERIFIED:
        return TrapType.NONE
    cleaned = (raw or "").strip().lower()
    mapping = {
        "outdated stat": TrapType.OUTDATED_STAT,
        "outdated": TrapType.OUTDATED_STAT,
        "stale stat": TrapType.OUTDATED_STAT,
        "numeric distortion": TrapType.NUMERIC_DISTORTION,
        "numeric": TrapType.NUMERIC_DISTORTION,
        "wrong number": TrapType.NUMERIC_DISTORTION,
        "misattributed source": TrapType.MISATTRIBUTED_SOURCE,
        "misattributed": TrapType.MISATTRIBUTED_SOURCE,
        "fabricated claim": TrapType.FABRICATED_CLAIM,
        "fabricated": TrapType.FABRICATED_CLAIM,
        "none": TrapType.NONE,
    }
    parsed = mapping.get(cleaned, TrapType.NONE)
    if parsed == TrapType.NONE:
        return TrapType.FABRICATED_CLAIM if verdict == Verdict.FALSE else TrapType.NUMERIC_DISTORTION
    return parsed


def _strip_ungrounded_inline_urls(
    text: str,
    allowed_norm_urls: set[str],
    stripped_sink: List[str],
) -> str:
    """Remove any URL from free-form text that was not actually retrieved by web search."""
    if not text:
        return text

    def _replace_md_link(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        clean_url = url.rstrip(".,;:")
        if normalize_url(clean_url) in allowed_norm_urls:
            return match.group(0)
        if clean_url not in stripped_sink:
            stripped_sink.append(clean_url)
        return label

    sanitized = _MD_LINK_RE.sub(_replace_md_link, text)

    def _replace_bare_url(match: re.Match[str]) -> str:
        raw_url = match.group(0)
        clean_url = raw_url.rstrip(".,;:")
        trailing = raw_url[len(clean_url) :]
        if normalize_url(clean_url) in allowed_norm_urls:
            return raw_url
        if clean_url not in stripped_sink:
            stripped_sink.append(clean_url)
        return trailing

    sanitized = _INLINE_URL_RE.sub(_replace_bare_url, sanitized)
    sanitized = re.sub(r"\(\s*\)", "", sanitized)
    sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()
    return sanitized


def enforce_citation_grounding(
    sources: List[WebSource],
    raw_cited_urls: object,
    explanation: str,
    corrected_fact: Optional[str],
) -> Tuple[List[WebSource], List[str], List[str], str, Optional[str]]:
    """Strict citation grounding guardrail: strip any cited URL not present in retrieved sources."""
    retrieved_by_norm: Dict[str, WebSource] = {}
    for src in sources:
        norm = normalize_url(src.url)
        if norm and norm not in retrieved_by_norm:
            retrieved_by_norm[norm] = src

    allowed_norms = set(retrieved_by_norm.keys())
    grounded_urls: List[str] = []
    stripped_urls: List[str] = []

    if isinstance(raw_cited_urls, list):
        for item in raw_cited_urls:
            candidate = str(item or "").strip()
            if not candidate:
                continue
            norm = normalize_url(candidate)
            if norm in retrieved_by_norm:
                canonical = retrieved_by_norm[norm].url
                if canonical not in grounded_urls:
                    grounded_urls.append(canonical)
            else:
                if candidate not in stripped_urls:
                    stripped_urls.append(candidate)

    clean_explanation = _strip_ungrounded_inline_urls(explanation, allowed_norms, stripped_urls)
    clean_corrected = (
        _strip_ungrounded_inline_urls(corrected_fact, allowed_norms, stripped_urls)
        if corrected_fact
        else None
    )

    # Order sources so explicitly grounded LLM citations come first, followed by authority-ranked extras
    grounded_norm_set = {normalize_url(u) for u in grounded_urls}
    ordered_sources = [s for s in sources if normalize_url(s.url) in grounded_norm_set] + [
        s for s in sources if normalize_url(s.url) not in grounded_norm_set
    ]

    if not grounded_urls and ordered_sources:
        grounded_urls = [s.url for s in ordered_sources[:3]]

    return ordered_sources, grounded_urls, stripped_urls, clean_explanation, clean_corrected


class FactCheckEngine:
    """Orchestrates PDF claim extraction, dual-query live search fusion, adjudication, and caching."""

    def __init__(
        self,
        settings: AppSettings,
        llm_client: Optional[LLMClient] = None,
        search_client: Optional[LiveSearchClient] = None,
        report_cache: Optional[AuditReportCache] = None,
    ) -> None:
        self.settings = settings
        self.llm = llm_client or LLMClient(settings)
        self.search = search_client or LiveSearchClient(
            tavily_api_key=settings.tavily_api_key,
            max_results=settings.search_results_per_claim,
            timeout_sec=settings.request_timeout_sec,
        )
        self.extractor = ClaimExtractor(self.llm, max_claims=settings.max_claims)
        self.cache = report_cache if report_cache is not None else AuditReportCache()

    def _run_search(self, claim: ExtractedClaim) -> List[WebSource]:
        if hasattr(self.search, "search_dual"):
            return self.search.search_dual(claim.primary_query, claim.adversarial_query)
        return self.search.search(claim.primary_query)

    def verify_claim(self, claim: ExtractedClaim) -> VerificationResult:
        """Verify a single extracted claim against dual-query authority-ranked live web evidence."""
        t0 = time.perf_counter()
        sources = self._run_search(claim)

        if not sources:
            elapsed = round(time.perf_counter() - t0, 2)
            return VerificationResult(
                claim=claim,
                verdict=Verdict.FALSE,
                trap_type=TrapType.FABRICATED_CLAIM,
                confidence=0.72,
                explanation=(
                    "Dual live web searches returned no credible supporting sources for this claim. "
                    "Treat as fabricated or unsubstantiated until primary documentation is provided."
                ),
                corrected_fact="No authoritative live web source confirms this claim or referenced event.",
                sources=[],
                cited_urls=[],
                stripped_ungrounded_urls=[],
                search_query_used=claim.primary_query,
                adversarial_query_used=claim.adversarial_query,
                latency_sec=elapsed,
            )

        evidence_block = _format_evidence(sources)
        prompt = build_verification_prompt(
            claim_text=claim.claim_text,
            original_quote=claim.original_quote,
            evidence_block=evidence_block,
        )

        try:
            verdict_data = self.llm.complete_json(prompt)
            verdict = _parse_verdict(str(verdict_data.get("verdict", "False")))
            trap_type = _parse_trap_type(str(verdict_data.get("trap_type", "None")), verdict)
            confidence = float(verdict_data.get("confidence", 0.78))
            confidence = max(0.0, min(1.0, round(confidence, 2)))

            explanation = str(
                verdict_data.get("explanation") or "Adjudicated against retrieved live web sources."
            ).strip()
            corrected = verdict_data.get("corrected_fact")
            if corrected is not None:
                corrected = str(corrected).strip() or None

            if verdict == Verdict.VERIFIED:
                corrected = None
                trap_type = TrapType.NONE
            elif not corrected:
                corrected = (
                    f"Claim flagged as {verdict.value} ({trap_type.value}): {explanation}"
                )

            (
                ordered_sources,
                grounded_urls,
                stripped_urls,
                clean_explanation,
                clean_corrected,
            ) = enforce_citation_grounding(
                sources=sources,
                raw_cited_urls=verdict_data.get("cited_urls", []),
                explanation=explanation,
                corrected_fact=corrected,
            )

            if verdict != Verdict.VERIFIED and not clean_corrected:
                clean_corrected = (
                    "Authoritative retrieved sources refute this claim; review cited evidence."
                )

            elapsed = round(time.perf_counter() - t0, 2)
            return VerificationResult(
                claim=claim,
                verdict=verdict,
                trap_type=trap_type,
                confidence=confidence,
                explanation=clean_explanation,
                corrected_fact=clean_corrected,
                sources=ordered_sources,
                cited_urls=grounded_urls,
                stripped_ungrounded_urls=stripped_urls,
                search_query_used=claim.primary_query,
                adversarial_query_used=claim.adversarial_query,
                latency_sec=elapsed,
            )
        except Exception as exc:
            logger.exception("Verification failed for claim #%d: %s", claim.claim_id, exc)
            elapsed = round(time.perf_counter() - t0, 2)
            return VerificationResult(
                claim=claim,
                verdict=Verdict.FALSE,
                trap_type=TrapType.FABRICATED_CLAIM,
                confidence=0.30,
                explanation=f"Automated verification encountered an error: {exc}",
                corrected_fact="Verification could not complete due to an upstream error; inspect sources manually.",
                sources=sources,
                cited_urls=[s.url for s in sources[:3]],
                stripped_ungrounded_urls=[],
                search_query_used=claim.primary_query,
                adversarial_query_used=claim.adversarial_query,
                latency_sec=elapsed,
            )

    def _get_call_counts(self) -> Tuple[int, int]:
        llm_calls = int(getattr(self.llm, "call_count", 0))
        search_calls = int(getattr(self.search, "call_count", 0))
        return llm_calls, search_calls

    def audit_document(
        self,
        document_name: str,
        pages: List[PageContent],
        total_pages: int,
        document_sha256: str = "",
        bypass_cache: bool = False,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> FactCheckReport:
        """Run the full Extract -> Dual-Search -> Adjudicate -> Ground -> Report pipeline."""
        audit_start = time.perf_counter()

        # Instant SHA-256 cache hit for repeat uploads
        if self.settings.enable_cache and document_sha256 and not bypass_cache:
            cached = self.cache.get(document_sha256)
            if cached is not None:
                if progress_cb:
                    progress_cb(1, 1, f"Instant SHA-256 cache hit ({document_sha256[:12]}…) — 0 API calls used.")
                cached.document_name = document_name
                cached.telemetry = cached.telemetry.model_copy(
                    update={
                        "cache_hit": True,
                        "total_latency_sec": round(time.perf_counter() - audit_start, 3),
                        "llm_calls": 0,
                        "search_calls": 0,
                    }
                )
                return cached

        llm_start, search_start = self._get_call_counts()

        if progress_cb:
            progress_cb(0, 1, "Decomposing PDF & tables into atomic verifiable claims...")

        extract_start = time.perf_counter()
        claims = self.extractor.extract(pages)
        extract_elapsed = round(time.perf_counter() - extract_start, 2)

        if not claims:
            llm_end, search_end = self._get_call_counts()
            total_elapsed = round(time.perf_counter() - audit_start, 2)
            return FactCheckReport(
                document_name=document_name,
                document_sha256=document_sha256,
                page_count=total_pages,
                llm_provider=self.settings.active_llm_provider,
                llm_model=self.settings.active_llm_model,
                search_provider=self.settings.active_search_provider,
                results=[],
                telemetry=AuditTelemetry(
                    total_latency_sec=total_elapsed,
                    extraction_latency_sec=extract_elapsed,
                    verification_latency_sec=0.0,
                    avg_claim_latency_sec=0.0,
                    llm_calls=max(0, llm_end - llm_start),
                    search_calls=max(0, search_end - search_start),
                    cache_hit=False,
                    ungrounded_urls_stripped=0,
                ),
            )

        total = len(claims)
        results_by_id: Dict[int, VerificationResult] = {}
        completed = 0
        verify_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
            future_map = {executor.submit(self.verify_claim, c): c for c in claims}
            for future in as_completed(future_map):
                claim = future_map[future]
                res = future.result()
                results_by_id[claim.claim_id] = res
                completed += 1
                if progress_cb:
                    progress_cb(
                        completed,
                        total,
                        f"Adjudicated {completed}/{total} [{res.verdict.value}]: {claim.claim_text[:60]}...",
                    )

        verify_elapsed = round(time.perf_counter() - verify_start, 2)
        total_elapsed = round(time.perf_counter() - audit_start, 2)
        ordered_results = [results_by_id[c.claim_id] for c in claims if c.claim_id in results_by_id]

        llm_end, search_end = self._get_call_counts()
        avg_claim_latency = (
            round(sum(r.latency_sec for r in ordered_results) / len(ordered_results), 2)
            if ordered_results
            else 0.0
        )
        stripped_total = sum(len(r.stripped_ungrounded_urls) for r in ordered_results)

        report = FactCheckReport(
            document_name=document_name,
            document_sha256=document_sha256,
            page_count=total_pages,
            llm_provider=self.settings.active_llm_provider,
            llm_model=str(getattr(self.llm, "model_name", self.settings.active_llm_model)),
            search_provider=self.settings.active_search_provider,
            results=ordered_results,
            telemetry=AuditTelemetry(
                total_latency_sec=total_elapsed,
                extraction_latency_sec=extract_elapsed,
                verification_latency_sec=verify_elapsed,
                avg_claim_latency_sec=avg_claim_latency,
                llm_calls=max(0, llm_end - llm_start),
                search_calls=max(0, search_end - search_start),
                cache_hit=False,
                ungrounded_urls_stripped=stripped_total,
            ),
        )

        if self.settings.enable_cache and document_sha256:
            self.cache.put(document_sha256, report)

        return report
