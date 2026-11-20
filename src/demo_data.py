"""Pre-computed realistic Trap Document audit report for instant zero-key preview mode."""

from __future__ import annotations

from src.models import (
    AuditTelemetry,
    ClaimCategory,
    ExtractedClaim,
    FactCheckReport,
    TrapType,
    Verdict,
    VerificationResult,
    WebSource,
)


def build_demo_trap_report(document_sha256: str = "demo-preview-sha256-trap-doc") -> FactCheckReport:
    """Return a realistic, citation-grounded audit of sample_trap_document.pdf."""
    results = [
        VerificationResult(
            claim=ExtractedClaim(
                claim_id=1,
                claim_text="OpenAI officially launched ChatGPT to the public on November 30, 2022.",
                original_quote="OpenAI officially launched ChatGPT to the public on November 30, 2022",
                page_number=1,
                category=ClaimCategory.DATE_TIMELINE,
                primary_query="OpenAI ChatGPT official public launch date",
                adversarial_query="ChatGPT release date November 30 2022 OpenAI official confirmation",
            ),
            verdict=Verdict.VERIFIED,
            trap_type=TrapType.NONE,
            confidence=0.99,
            explanation="Primary official documentation confirms OpenAI released ChatGPT to the public on November 30, 2022.",
            corrected_fact=None,
            sources=[
                WebSource(
                    title="Introducing ChatGPT — OpenAI",
                    url="https://openai.com/index/chatgpt/",
                    snippet="We've trained a model called ChatGPT which interacts in a conversational way. Published November 30, 2022.",
                    domain="openai.com",
                    authority_score=8,
                    authority_tier="Primary / Official",
                    published_date="2022-11-30",
                    provider="tavily",
                    query_origin="both",
                )
            ],
            cited_urls=["https://openai.com/index/chatgpt/"],
            search_query_used="OpenAI ChatGPT official public launch date",
            adversarial_query_used="ChatGPT release date November 30 2022 OpenAI official confirmation",
            latency_sec=1.42,
        ),
        VerificationResult(
            claim=ExtractedClaim(
                claim_id=2,
                claim_text="Google was founded in September 1998 by Larry Page and Sergey Brin at Stanford University.",
                original_quote="Google was originally founded in September 1998 by Larry Page and Sergey Brin",
                page_number=1,
                category=ClaimCategory.DATE_TIMELINE,
                primary_query="Google founding date Larry Page Sergey Brin Stanford",
                adversarial_query="Google founded September 1998 Stanford official history",
            ),
            verdict=Verdict.VERIFIED,
            trap_type=TrapType.NONE,
            confidence=0.99,
            explanation="Confirmed. Google LLC was incorporated on September 4, 1998 by Larry Page and Sergey Brin while Ph.D. candidates at Stanford University.",
            corrected_fact=None,
            sources=[
                WebSource(
                    title="About Google — Our History",
                    url="https://about.google/company-info/our-story/",
                    snippet="Larry Page and Sergey Brin founded Google in September 1998 while Ph.D. students at Stanford University.",
                    domain="about.google",
                    authority_score=8,
                    authority_tier="Primary / Official",
                    provider="tavily",
                    query_origin="both",
                )
            ],
            cited_urls=["https://about.google/company-info/our-story/"],
            search_query_used="Google founding date Larry Page Sergey Brin Stanford",
            adversarial_query_used="Google founded September 1998 Stanford official history",
            latency_sec=1.35,
        ),
        VerificationResult(
            claim=ExtractedClaim(
                claim_id=3,
                claim_text="India's population currently stands at 1.21 billion, making it the second most populous country behind China.",
                original_quote="India's population stands at 1.21 billion people, making it the second most populous",
                page_number=1,
                category=ClaimCategory.STATISTIC,
                primary_query="India current population UN World Bank",
                adversarial_query="India population 2025 2026 most populous country surpassed China UN",
            ),
            verdict=Verdict.INACCURATE,
            trap_type=TrapType.OUTDATED_STAT,
            confidence=0.98,
            explanation="Outdated statistic trap: 1.21 billion comes from India's 2011 Census. India overtook China in 2023 to become the world's most populous country.",
            corrected_fact="India is the world's most populous country with over 1.45 billion residents as of 2025–2026, ahead of China (United Nations DESA / World Bank).",
            sources=[
                WebSource(
                    title="India to Overtake China as World's Most Populous Country — UN DESA",
                    url="https://www.un.org/en/desa/india-overtake-china-world-most-populous-country-april-2023-united-nations-projects",
                    snippet="India surpassed China in 2023 with a population exceeding 1.425 billion and projected above 1.45 billion in 2025-2026.",
                    domain="un.org",
                    authority_score=8,
                    authority_tier="Primary / Official",
                    published_date="2023-04-24",
                    provider="tavily",
                    query_origin="both",
                )
            ],
            cited_urls=[
                "https://www.un.org/en/desa/india-overtake-china-world-most-populous-country-april-2023-united-nations-projects"
            ],
            search_query_used="India current population UN World Bank",
            adversarial_query_used="India population 2025 2026 most populous country surpassed China UN",
            latency_sec=1.68,
        ),
        VerificationResult(
            claim=ExtractedClaim(
                claim_id=4,
                claim_text="Apple's total annual revenue in fiscal year 2024 was $180 billion, down sharply from prior years.",
                original_quote="Apple's total annual revenue in fiscal year 2024 was $180 billion, down sharply from prior years.",
                page_number=1,
                category=ClaimCategory.FINANCIAL,
                primary_query="Apple FY2024 annual revenue Form 10-K SEC",
                adversarial_query="Apple fiscal year 2024 total net sales 391 billion official results",
            ),
            verdict=Verdict.INACCURATE,
            trap_type=TrapType.NUMERIC_DISTORTION,
            confidence=0.99,
            explanation="Severe numeric distortion. Apple's official FY2024 financial results reported $391.04 billion in total net sales — more than double the $180 billion claimed and up 2% YoY.",
            corrected_fact="Apple's total annual net sales for fiscal year 2024 were $391.04 billion, up 2.0% year-over-year from $383.29 billion in FY2023 (Apple FY2024 Form 10-K / Newsroom).",
            sources=[
                WebSource(
                    title="Apple Reports Fourth Quarter & FY2024 Results — Apple Newsroom",
                    url="https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/",
                    snippet="Apple posted full-year fiscal 2024 revenue of $391.0 billion, up from $383.3 billion in fiscal 2023.",
                    domain="apple.com",
                    authority_score=11,
                    authority_tier="Primary / Official",
                    published_date="2024-10-31",
                    provider="tavily",
                    query_origin="both",
                )
            ],
            cited_urls=["https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/"],
            search_query_used="Apple FY2024 annual revenue Form 10-K SEC",
            adversarial_query_used="Apple fiscal year 2024 total net sales 391 billion official results",
            latency_sec=1.54,
        ),
        VerificationResult(
            claim=ExtractedClaim(
                claim_id=5,
                claim_text="A 2025 United Nations Digital Commerce Mandate requires all EU websites to pay a 15% 'AI Citation Tax' starting January 2026.",
                original_quote="all European Union websites are legally required to pay a 15% 'AI Citation Tax' on organic search traffic starting January 2026.",
                page_number=1,
                category=ClaimCategory.REGULATORY,
                primary_query="United Nations Digital Commerce Mandate EU AI Citation Tax",
                adversarial_query="EU 15 percent AI Citation Tax organic search 2026 hoax fact check",
            ),
            verdict=Verdict.FALSE,
            trap_type=TrapType.FABRICATED_CLAIM,
            confidence=0.98,
            explanation="Completely fabricated regulation. Neither the United Nations nor the European Commission has enacted any 'Digital Commerce Mandate' or 15% 'AI Citation Tax' on EU websites.",
            corrected_fact="No UN Digital Commerce Mandate or 15% EU AI Citation Tax exists; EU AI regulation is governed by the EU AI Act (entered into force August 1, 2024), which imposes no tax on organic search traffic (European Commission).",
            sources=[
                WebSource(
                    title="Shaping Europe's Digital Future: AI Act — European Commission",
                    url="https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai",
                    snippet="The EU AI Act establishes a risk-based regulatory framework for AI systems in the European Union.",
                    domain="digital-strategy.ec.europa.eu",
                    authority_score=8,
                    authority_tier="Primary / Official",
                    provider="tavily",
                    query_origin="primary",
                )
            ],
            cited_urls=["https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai"],
            search_query_used="United Nations Digital Commerce Mandate EU AI Citation Tax",
            adversarial_query_used="EU 15 percent AI Citation Tax organic search 2026 hoax fact check",
            latency_sec=1.61,
        ),
        VerificationResult(
            claim=ExtractedClaim(
                claim_id=6,
                claim_text="NVIDIA acquired OpenAI in March 2025 for $420 billion in an all-cash transaction.",
                original_quote="NVIDIA acquired OpenAI in March 2025 for $420 billion in an all-cash transaction.",
                page_number=1,
                category=ClaimCategory.FINANCIAL,
                primary_query="NVIDIA acquired OpenAI March 2025 420 billion",
                adversarial_query="did NVIDIA buy OpenAI 2025 2026 fact check independent",
            ),
            verdict=Verdict.FALSE,
            trap_type=TrapType.FABRICATED_CLAIM,
            confidence=0.99,
            explanation="Fabricated M&A transaction. NVIDIA has not acquired OpenAI; OpenAI operates as an independent AI company while partnering with NVIDIA on GPU infrastructure.",
            corrected_fact="NVIDIA did not acquire OpenAI for $420 billion. OpenAI remains an independent company with Microsoft and other institutional investors (OpenAI Newsroom / Reuters).",
            sources=[
                WebSource(
                    title="OpenAI Official Newsroom",
                    url="https://openai.com/news/",
                    snippet="OpenAI operates as an independent AI research and deployment company building frontier models and ChatGPT.",
                    domain="openai.com",
                    authority_score=8,
                    authority_tier="Primary / Official",
                    provider="tavily",
                    query_origin="both",
                )
            ],
            cited_urls=["https://openai.com/news/"],
            search_query_used="NVIDIA acquired OpenAI March 2025 420 billion",
            adversarial_query_used="did NVIDIA buy OpenAI 2025 2026 fact check independent",
            latency_sec=1.49,
        ),
    ]

    return FactCheckReport(
        document_name="sample_trap_document.pdf",
        document_sha256=document_sha256,
        page_count=1,
        llm_provider="gemini",
        llm_model="gemini-3.5-flash-lite (demo preview)",
        search_provider="tavily (cached preview)",
        results=results,
        telemetry=AuditTelemetry(
            total_latency_sec=4.82,
            extraction_latency_sec=1.18,
            verification_latency_sec=3.64,
            avg_claim_latency_sec=1.52,
            llm_calls=7,
            search_calls=12,
            cache_hit=False,
            ungrounded_urls_stripped=0,
        ),
    )
