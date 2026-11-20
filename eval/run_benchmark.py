"""Automated 10-claim Adversarial Trap-Document Benchmark for TruthLayer AI.

Evaluates live end-to-end claim verification against a 10-claim ground-truth Trap Document:
  - 3 Verified claims (immutable dates / official effective dates)
  - 4 Inaccurate claims (2 Outdated Stats + 2 Numeric Distortions)
  - 3 False claims (Fabricated regulations, M&A deals, and scientific claims)

Computes:
  - Overall Verdict Accuracy
  - Trap Detection Recall
  - False Positive Rate (on Verified claims)
  - Trap-Type Classification Accuracy
  - Citation Grounding Rate
  - Average Latency per Claim

Usage:
    python eval/run_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.models import ClaimCategory, ExtractedClaim, TrapType, Verdict
from src.search_client import normalize_url
from src.verifier import FactCheckEngine


@dataclass(frozen=True)
class GroundTruthCase:
    claim: ExtractedClaim
    expected_verdict: Verdict
    expected_trap_type: TrapType
    notes: str


BENCHMARK_CASES: List[GroundTruthCase] = [
    # --- 3 VERIFIED CLAIMS ---
    GroundTruthCase(
        claim=ExtractedClaim(
            claim_id=1,
            claim_text="OpenAI officially launched ChatGPT to the public on November 30, 2022.",
            original_quote="OpenAI officially launched ChatGPT to the public on November 30, 2022.",
            page_number=1,
            category=ClaimCategory.DATE_TIMELINE,
            primary_query="OpenAI ChatGPT official public launch date",
            adversarial_query="ChatGPT launch date November 30 2022 official OpenAI announcement",
        ),
        expected_verdict=Verdict.VERIFIED,
        expected_trap_type=TrapType.NONE,
        notes="Immutable historical product launch date.",
    ),
    GroundTruthCase(
        claim=ExtractedClaim(
            claim_id=2,
            claim_text="Google was founded in September 1998 by Larry Page and Sergey Brin while Ph.D. students at Stanford University.",
            original_quote="Google was founded in September 1998 by Larry Page and Sergey Brin at Stanford University.",
            page_number=1,
            category=ClaimCategory.DATE_TIMELINE,
            primary_query="Google founding date Larry Page Sergey Brin Stanford University",
            adversarial_query="Google founded September 1998 official corporate history",
        ),
        expected_verdict=Verdict.VERIFIED,
        expected_trap_type=TrapType.NONE,
        notes="Immutable historical corporate founding fact.",
    ),
    GroundTruthCase(
        claim=ExtractedClaim(
            claim_id=3,
            claim_text="The European Union AI Act officially entered into force on August 1, 2024.",
            original_quote="The European Union AI Act officially entered into force on August 1, 2024.",
            page_number=1,
            category=ClaimCategory.REGULATORY,
            primary_query="European Union AI Act entered into force date European Commission",
            adversarial_query="EU AI Act August 1 2024 official entry into force europa.eu",
        ),
        expected_verdict=Verdict.VERIFIED,
        expected_trap_type=TrapType.NONE,
        notes="Official European Commission regulatory date.",
    ),
    # --- 4 INACCURATE CLAIMS (2 Outdated Stats + 2 Numeric Distortions) ---
    GroundTruthCase(
        claim=ExtractedClaim(
            claim_id=4,
            claim_text="As of today, India's population stands at 1.21 billion people, making it the second most populous country in the world behind China.",
            original_quote="As of today, India's population stands at 1.21 billion people, making it the second most populous country behind China.",
            page_number=1,
            category=ClaimCategory.STATISTIC,
            primary_query="India current population UN World Bank",
            adversarial_query="India population 2025 2026 surpassed China most populous country UN",
        ),
        expected_verdict=Verdict.INACCURATE,
        expected_trap_type=TrapType.OUTDATED_STAT,
        notes="2011 census stat (1.21B) presented as current; India >1.44B and #1 since 2023.",
    ),
    GroundTruthCase(
        claim=ExtractedClaim(
            claim_id=5,
            claim_text="As of today, the global human population is approximately 6.8 billion people.",
            original_quote="As of today, the global human population is approximately 6.8 billion people.",
            page_number=1,
            category=ClaimCategory.STATISTIC,
            primary_query="world human population United Nations World Bank",
            adversarial_query="current world population 2025 2026 over 8 billion UN",
        ),
        expected_verdict=Verdict.INACCURATE,
        expected_trap_type=TrapType.OUTDATED_STAT,
        notes="Stale ~2009 global population figure presented as current; actual >8.1B.",
    ),
    GroundTruthCase(
        claim=ExtractedClaim(
            claim_id=6,
            claim_text="Apple's total annual revenue in fiscal year 2024 was $180 billion, down sharply from prior years.",
            original_quote="Apple's total annual revenue in fiscal year 2024 was $180 billion, down sharply from prior years.",
            page_number=2,
            category=ClaimCategory.FINANCIAL,
            primary_query="Apple FY2024 total net sales revenue Form 10-K SEC",
            adversarial_query="Apple fiscal year 2024 annual revenue 391 billion official results",
        ),
        expected_verdict=Verdict.INACCURATE,
        expected_trap_type=TrapType.NUMERIC_DISTORTION,
        notes="Severely distorted figure; real FY2024 Apple revenue was $391.04 billion (+2% YoY).",
    ),
    GroundTruthCase(
        claim=ExtractedClaim(
            claim_id=7,
            claim_text="NVIDIA's total annual revenue in fiscal year 2025 (ended January 2025) was $27.0 billion.",
            original_quote="NVIDIA's total annual revenue in fiscal year 2025 (ended January 2025) was $27.0 billion.",
            page_number=2,
            category=ClaimCategory.FINANCIAL,
            primary_query="NVIDIA fiscal year 2025 annual revenue Form 10-K investor relations",
            adversarial_query="NVIDIA FY2025 full year revenue 130.5 billion official earnings",
        ),
        expected_verdict=Verdict.INACCURATE,
        expected_trap_type=TrapType.NUMERIC_DISTORTION,
        notes="Distorted figure (cites FY2023 $27B instead of FY2025 $130.5B).",
    ),
    # --- 3 FABRICATED FALSE CLAIMS ---
    GroundTruthCase(
        claim=ExtractedClaim(
            claim_id=8,
            claim_text="According to a 2025 United Nations Digital Commerce Mandate, all European Union websites are legally required to pay a 15% 'AI Citation Tax' on organic search traffic starting January 2026.",
            original_quote="all European Union websites are legally required to pay a 15% 'AI Citation Tax' on organic search traffic starting January 2026.",
            page_number=2,
            category=ClaimCategory.REGULATORY,
            primary_query="United Nations Digital Commerce Mandate EU AI Citation Tax",
            adversarial_query="EU 15 percent AI Citation Tax organic search 2026 fact check hoax",
        ),
        expected_verdict=Verdict.FALSE,
        expected_trap_type=TrapType.FABRICATED_CLAIM,
        notes="Completely fabricated UN regulation and EU tax.",
    ),
    GroundTruthCase(
        claim=ExtractedClaim(
            claim_id=9,
            claim_text="NVIDIA acquired OpenAI in March 2025 for $420 billion in an all-cash transaction.",
            original_quote="NVIDIA acquired OpenAI in March 2025 for $420 billion in an all-cash transaction.",
            page_number=2,
            category=ClaimCategory.FINANCIAL,
            primary_query="NVIDIA acquired OpenAI March 2025 420 billion acquisition",
            adversarial_query="did NVIDIA buy OpenAI 2025 2026 independent company fact check",
        ),
        expected_verdict=Verdict.FALSE,
        expected_trap_type=TrapType.FABRICATED_CLAIM,
        notes="Fabricated $420B acquisition.",
    ),
    GroundTruthCase(
        claim=ExtractedClaim(
            claim_id=10,
            claim_text="In October 2024, NASA officially confirmed the discovery of a 40-mile-wide subterranean ocean of liquid methane beneath the surface of Mars's moon Phobos.",
            original_quote="In October 2024, NASA officially confirmed the discovery of a 40-mile-wide subterranean ocean of liquid methane beneath Mars's moon Phobos.",
            page_number=2,
            category=ClaimCategory.TECHNICAL,
            primary_query="NASA Phobos subterranean ocean liquid methane October 2024",
            adversarial_query="Phobos liquid methane ocean NASA discovery hoax fact check",
        ),
        expected_verdict=Verdict.FALSE,
        expected_trap_type=TrapType.FABRICATED_CLAIM,
        notes="Fabricated NASA discovery (Phobos itself is only ~14 miles wide!).",
    ),
]


def run_benchmark() -> Dict[str, Any]:
    settings = load_settings(enable_cache=False)
    if settings.active_llm_provider == "none":
        raise SystemExit(
            "ERROR: No LLM API key found in .env or environment variables.\n"
            "Set GEMINI_API_KEY (or OPENAI_API_KEY) in .env and re-run `python eval/run_benchmark.py`."
        )

    engine = FactCheckEngine(settings)
    print(
        f"Running TruthLayer 10-Claim Trap-Document Benchmark\n"
        f"  LLM:    {settings.active_llm_provider} ({settings.active_llm_model})\n"
        f"  Search: {settings.active_search_provider}\n"
        f"{'-' * 72}"
    )

    wall_start = time.perf_counter()
    results_by_id: Dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_map = {executor.submit(engine.verify_claim, case.claim): case for case in BENCHMARK_CASES}
        for future in as_completed(future_map):
            case = future_map[future]
            res = future.result()
            results_by_id[case.claim.claim_id] = res
            print(f"  completed claim #{case.claim.claim_id:02d} ({res.verdict.value}, {res.latency_sec:.2f}s)", flush=True)

    case_outputs: List[Dict[str, Any]] = []
    exact_verdict_hits = 0
    trap_total = 0
    trap_caught = 0
    verified_total = 0
    false_positives = 0
    trap_type_hits = 0
    total_cited_urls = 0
    grounded_cited_urls = 0
    claim_latencies: List[float] = []

    for case in BENCHMARK_CASES:
        res = results_by_id[case.claim.claim_id]
        claim_latencies.append(res.latency_sec)

        verdict_ok = res.verdict == case.expected_verdict
        if verdict_ok:
            exact_verdict_hits += 1

        is_trap = case.expected_verdict != Verdict.VERIFIED
        if is_trap:
            trap_total += 1
            if res.verdict in (Verdict.INACCURATE, Verdict.FALSE):
                trap_caught += 1
        else:
            verified_total += 1
            if res.verdict != Verdict.VERIFIED:
                false_positives += 1

        trap_type_ok = res.trap_type == case.expected_trap_type
        if trap_type_ok:
            trap_type_hits += 1

        retrieved_norms = {normalize_url(s.url) for s in res.sources}
        for url in res.cited_urls:
            total_cited_urls += 1
            if normalize_url(url) in retrieved_norms:
                grounded_cited_urls += 1

        status_icon = "PASS" if verdict_ok else "MISS"
        print(
            f"[{status_icon}] #{case.claim.claim_id:02d} | Expected={case.expected_verdict.value:<10} "
            f"Got={res.verdict.value:<10} | Trap={res.trap_type.value:<20} | "
            f"Conf={int(res.confidence * 100):3d}% | {res.latency_sec:.2f}s"
        )
        case_outputs.append(
            {
                "claim_id": case.claim.claim_id,
                "claim_text": case.claim.claim_text,
                "expected_verdict": case.expected_verdict.value,
                "predicted_verdict": res.verdict.value,
                "verdict_correct": verdict_ok,
                "expected_trap_type": case.expected_trap_type.value,
                "predicted_trap_type": res.trap_type.value,
                "trap_type_correct": trap_type_ok,
                "confidence": res.confidence,
                "corrected_fact": res.corrected_fact,
                "cited_urls": res.cited_urls,
                "stripped_ungrounded_urls": res.stripped_ungrounded_urls,
                "latency_sec": res.latency_sec,
            }
        )

    wall_total = round(time.perf_counter() - wall_start, 2)
    n = len(BENCHMARK_CASES)
    overall_accuracy = round((exact_verdict_hits / n) * 100.0, 1)
    trap_recall = round((trap_caught / max(trap_total, 1)) * 100.0, 1)
    fp_rate = round((false_positives / max(verified_total, 1)) * 100.0, 1)
    trap_type_accuracy = round((trap_type_hits / n) * 100.0, 1)
    citation_grounding_rate = (
        round((grounded_cited_urls / total_cited_urls) * 100.0, 1) if total_cited_urls else 100.0
    )
    avg_latency = round(sum(claim_latencies) / n, 2)

    summary = {
        "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "llm_provider": settings.active_llm_provider,
        "llm_model": getattr(engine.llm, "model_name", settings.active_llm_model),
        "search_provider": settings.active_search_provider,
        "total_claims": n,
        "overall_verdict_accuracy_pct": overall_accuracy,
        "exact_verdict_hits": exact_verdict_hits,
        "trap_detection_recall_pct": trap_recall,
        "traps_caught": trap_caught,
        "traps_total": trap_total,
        "false_positive_rate_pct": fp_rate,
        "trap_type_accuracy_pct": trap_type_accuracy,
        "citation_grounding_rate_pct": citation_grounding_rate,
        "avg_latency_per_claim_sec": avg_latency,
        "total_wall_time_sec": wall_total,
        "cases": case_outputs,
    }

    out_path = Path(__file__).parent / "benchmark_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 72)
    print(f"Overall Verdict Accuracy:   {overall_accuracy:.1f}% ({exact_verdict_hits}/{n})")
    print(f"Trap Detection Recall:      {trap_recall:.1f}% ({trap_caught}/{trap_total})")
    print(f"False Positive Rate:        {fp_rate:.1f}% ({false_positives}/{verified_total})")
    print(f"Trap-Type Accuracy:         {trap_type_accuracy:.1f}% ({trap_type_hits}/{n})")
    print(f"Citation Grounding Rate:    {citation_grounding_rate:.1f}% ({grounded_cited_urls}/{total_cited_urls})")
    print(f"Avg Latency per Claim:      {avg_latency:.2f}s (total wall time: {wall_total:.2f}s)")
    print(f"Saved detailed results to:  {out_path}")
    return summary


if __name__ == "__main__":
    run_benchmark()
