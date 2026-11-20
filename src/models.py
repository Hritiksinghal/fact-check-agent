"""Strict Pydantic v2 schemas for PDF parsing, claim extraction, web evidence, and verification reports."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator


class Verdict(str, Enum):
    """Assessment-mandated fact-check verdict taxonomy."""

    VERIFIED = "Verified"
    INACCURATE = "Inaccurate"
    FALSE = "False"


class TrapType(str, Enum):
    """Adversarial trap-document error classification."""

    NONE = "None"
    OUTDATED_STAT = "Outdated Stat"
    NUMERIC_DISTORTION = "Numeric Distortion"
    MISATTRIBUTED_SOURCE = "Misattributed Source"
    FABRICATED_CLAIM = "Fabricated Claim"


class ClaimCategory(str, Enum):
    """Categorization of factual claims extracted from business/marketing PDFs."""

    STATISTIC = "Statistic"
    FINANCIAL = "Financial"
    DATE_TIMELINE = "Date / Timeline"
    TECHNICAL = "Technical"
    MARKET_CLAIM = "Market Claim"
    REGULATORY = "Regulatory"
    OTHER = "Other"


class PageContent(BaseModel):
    """Text and Markdown-formatted tables extracted from a single PDF page with 1-based provenance."""

    page_number: int = Field(..., ge=1)
    text: str
    tables_markdown: List[str] = Field(default_factory=list)
    extraction_method: str = "pypdf"


class ParsedDocument(BaseModel):
    """Complete result of layout- and table-aware PDF ingestion."""

    pages: List[PageContent] = Field(default_factory=list)
    total_pages: int = Field(..., ge=0)
    sha256_hash: str = ""
    is_scanned_or_image_only: bool = False
    warning_message: Optional[str] = None
    table_count: int = 0


class ExtractedClaim(BaseModel):
    """An atomic, verifiable factual claim extracted from the uploaded PDF."""

    claim_id: int = Field(..., ge=1)
    claim_text: str = Field(..., min_length=5, description="Self-contained factual statement to verify.")
    original_quote: str = Field(..., description="Verbatim snippet from the PDF for highlight provenance.")
    page_number: int = Field(..., ge=1)
    category: ClaimCategory = Field(default=ClaimCategory.STATISTIC)
    primary_query: str = Field(
        default="",
        description="Official/authoritative web search query for entity + metric.",
    )
    adversarial_query: str = Field(
        default="",
        description="Temporal/fact-check query targeting 2025-2026 official figures or debunking rumors.",
    )
    search_query: str = Field(
        default="",
        description="Backward-compatible alias for primary_query.",
    )

    @model_validator(mode="after")
    def _sync_queries(self) -> "ExtractedClaim":
        primary = (self.primary_query or self.search_query or self.claim_text[:110]).strip()
        self.primary_query = primary
        self.search_query = primary
        if not self.adversarial_query.strip():
            self.adversarial_query = f"{primary} latest 2025 2026 fact check"
        else:
            self.adversarial_query = self.adversarial_query.strip()
        return self


class WebSource(BaseModel):
    """A single authority-ranked web search result used as evidence during verification."""

    title: str
    url: str
    snippet: str
    domain: str = ""
    authority_score: int = 0
    authority_tier: str = "Standard"
    published_date: Optional[str] = None
    provider: str = "tavily"
    query_origin: str = "primary"

    @model_validator(mode="after")
    def _populate_domain(self) -> "WebSource":
        if not self.domain and self.url:
            try:
                host = urlparse(self.url).netloc.lower().removeprefix("www.")
                self.domain = host or "source"
            except Exception:
                self.domain = "source"
        return self


class VerificationResult(BaseModel):
    """Final adjudicated verdict, trap tag, and grounded evidence trail for a single claim."""

    claim: ExtractedClaim
    verdict: Verdict
    trap_type: TrapType = Field(default=TrapType.NONE)
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., description="Concise, evidence-grounded reasoning for the verdict.")
    corrected_fact: Optional[str] = Field(
        default=None,
        description="Accurate, up-to-date real fact when verdict is Inaccurate or False.",
    )
    sources: List[WebSource] = Field(default_factory=list)
    cited_urls: List[str] = Field(default_factory=list)
    stripped_ungrounded_urls: List[str] = Field(default_factory=list)
    search_query_used: str = ""
    adversarial_query_used: str = ""
    latency_sec: float = 0.0


class AuditTelemetry(BaseModel):
    """Performance, cost, and safety guardrail telemetry for a completed PDF audit."""

    total_latency_sec: float = 0.0
    extraction_latency_sec: float = 0.0
    verification_latency_sec: float = 0.0
    avg_claim_latency_sec: float = 0.0
    llm_calls: int = 0
    search_calls: int = 0
    cache_hit: bool = False
    ungrounded_urls_stripped: int = 0


class FactCheckReport(BaseModel):
    """Complete audit report for an uploaded PDF document."""

    document_name: str
    document_sha256: str = ""
    page_count: int
    checked_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )
    llm_provider: str
    llm_model: str = ""
    search_provider: str
    results: List[VerificationResult] = Field(default_factory=list)
    telemetry: AuditTelemetry = Field(default_factory=AuditTelemetry)

    @property
    def total_claims(self) -> int:
        return len(self.results)

    @property
    def verified_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.VERIFIED)

    @property
    def inaccurate_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.INACCURATE)

    @property
    def false_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.FALSE)

    @property
    def flagged_count(self) -> int:
        return self.inaccurate_count + self.false_count

    @property
    def trust_score(self) -> float:
        """Percentage of claims verified as accurate (0–100)."""
        if not self.results:
            return 0.0
        return round((self.verified_count / len(self.results)) * 100.0, 1)

    @property
    def trap_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for res in self.results:
            if res.trap_type != TrapType.NONE:
                counts[res.trap_type.value] = counts.get(res.trap_type.value, 0) + 1
        return counts
