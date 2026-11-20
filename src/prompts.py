"""Prompt templates engineered for atomic claim decomposition and adversarial trap-document adjudication."""

from __future__ import annotations

from datetime import datetime, timezone


def build_extraction_prompt(document_text: str, max_claims: int = 25) -> str:
    """Prompt to extract atomic, checkable claims with verbatim quotes and dual search queries."""
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    return f"""You are a Principal Fact-Checking Analyst auditing a business, policy, or marketing PDF document.
Today's date is {today}.

TASK:
Extract up to {max_claims} atomic, objectively verifiable factual claims from the document below (including facts inside Markdown tables).

EXTRACT ONLY OBJECTIVELY VERIFIABLE CLAIMS:
1. Statistics, percentages, population counts, market shares, user counts, and growth rates.
2. Financial figures (revenues, profits, valuations, M&A deals, funding rounds, prices, GDP, interest rates).
3. Dates, launch dates, founding years, regulatory effective dates, and historical milestones.
4. Technical specifications, model benchmarks, parameter counts, or scientific discoveries.
5. Named regulations, mandates, laws, studies, surveys, or official organizational reports.

STRICTLY SKIP:
- Subjective opinions, marketing fluff, or vague adjectives ("world-class", "industry-leading", "revolutionary").
- Purely internal aspirations or unquantified forward-looking statements.

DECOMPOSITION & PROVENANCE RULES:
- Split compound sentences with multiple independent facts into separate atomic claims.
- Resolve ALL pronouns ("it", "the company", "they", "the mandate") to full explicit entity names inside `claim_text`.
- Capture `original_quote` as a SHORT, EXACT VERBATIM substring from the document text (so it can be highlighted directly in the PDF viewer).
- Record the exact integer `page_number` matching the `=== [Page N] ===` anchor.
- Generate TWO complementary web search queries (5-12 words each) per claim:
  1. `primary_query`: Official/authoritative lookup targeting the primary source (entity + metric + official filing/org, e.g. "Apple FY2024 annual revenue Form 10-K SEC").
  2. `adversarial_query`: Temporal or fact-check lookup targeting the latest 2025-2026 official data or explicitly checking for hoaxes/rumors/outdated figures (e.g. "India current population 2025 2026 UN World Bank" or "NVIDIA acquired OpenAI 420 billion fact check rumor").

Return ONLY valid JSON matching this exact schema:
{{
  "claims": [
    {{
      "claim_id": 1,
      "claim_text": "Self-contained atomic factual claim with resolved entity names",
      "original_quote": "Exact verbatim quote copied from the document text",
      "page_number": 1,
      "category": "Statistic | Financial | Date / Timeline | Technical | Market Claim | Regulatory | Other",
      "primary_query": "official authoritative search query",
      "adversarial_query": "2025 2026 temporal or fact-check debunk query"
    }}
  ]
}}

DOCUMENT TEXT:
{document_text}
"""


def build_verification_prompt(
    claim_text: str,
    original_quote: str,
    evidence_block: str,
) -> str:
    """Prompt to adjudicate a single claim against fused live web evidence with strict trap tagging."""
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    return f"""You are an adversarial Fact-Checking Adjudicator operating as a high-precision Truth Layer.
Today's date is {today}.

Verify the CLAIM below strictly against the LIVE WEB EVIDENCE provided.
Assume the document may be an adversarial "Trap Document" containing intentional lies, outdated stats presented as current, subtly altered numbers/dates, or misattributed studies.

VERDICT TAXONOMY (choose EXACTLY ONE):
1. "Verified": Live web evidence confirms the claim's numbers, dates, entities, and core assertion (minor rounding within ~2-3% is acceptable unless exact precision matters). Historical claims accurately anchored to a past date are Verified if true for that date.
2. "Inaccurate": The underlying entity, metric, or event exists, BUT the claim uses an outdated figure presented as current, distorts/alters a number/unit/date, exaggerates the stat, or misattributes the source/study.
3. "False": The claim is completely fabricated, refers to a non-existent event/regulation/acquisition/discovery, or is directly refuted with no credible supporting evidence.

TRAP TYPE TAXONOMY (choose EXACTLY ONE):
- "None": MUST be used when verdict is "Verified".
- "Outdated Stat": Stale historical figure presented as current or today's value (e.g., citing a 2011 census figure as today's population, or presenting a retired status as current).
- "Numeric Distortion": Wrong magnitude, altered dollar/percentage/count, flipped trend direction, or altered date/year for a real entity or metric.
- "Misattributed Source": Real statistic or claim wrongly attributed to a different organization, report, law, or study.
- "Fabricated Claim": Invented law/tax, fake M&A deal, fake discovery, or non-existent event/statistic.

STRICT ADJUDICATION & GROUNDING RULES:
- Base your judgment strictly on the LIVE WEB EVIDENCE below. Prefer higher-authority primary sources (.gov, .edu, SEC, UN, World Bank, Reuters, official newsrooms) over secondary blogs.
- If verdict is "Inaccurate" or "False", you MUST populate `corrected_fact` with the exact verified real figure/date/fact AND authoritative source name supported by the evidence. Never leave `corrected_fact` null or vague for a flagged claim.
- If verdict is "Verified", you MUST set `corrected_fact` to null and `trap_type` to "None".
- Populate `cited_urls` ONLY with exact URLs copied verbatim from the `URL:` lines in LIVE WEB EVIDENCE below. NEVER invent, modify, or guess URLs.
- Do not include any external URL inside `explanation` or `corrected_fact` unless it appears verbatim in LIVE WEB EVIDENCE.
- Provide a calibrated `confidence` score between 0.0 and 1.0.

Return ONLY valid JSON matching this exact schema:
{{
  "verdict": "Verified" | "Inaccurate" | "False",
  "trap_type": "None" | "Outdated Stat" | "Numeric Distortion" | "Misattributed Source" | "Fabricated Claim",
  "confidence": 0.94,
  "explanation": "2-3 crisp sentences comparing the document claim directly against the highest-authority retrieved evidence.",
  "corrected_fact": "Exact real verified figure/date and source name, or null if Verified.",
  "cited_urls": ["https://exact-url-from-evidence-block"]
}}

CLAIM TO VERIFY:
{claim_text}

ORIGINAL PDF QUOTE:
"{original_quote}"

LIVE WEB EVIDENCE:
{evidence_block}
"""
