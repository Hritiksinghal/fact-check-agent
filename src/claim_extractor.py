"""Atomic claim decomposition transforming parsed PDF pages into verifiable claims with dual queries."""

from __future__ import annotations

import logging
import re
from typing import List

from src.llm_client import LLMClient
from src.models import ClaimCategory, ExtractedClaim, PageContent
from src.pdf_parser import format_pages_for_llm
from src.prompts import build_extraction_prompt

logger = logging.getLogger(__name__)


def _parse_category(raw: str) -> ClaimCategory:
    normalized = (raw or "").strip().lower()
    mapping = {
        "statistic": ClaimCategory.STATISTIC,
        "statistics": ClaimCategory.STATISTIC,
        "financial": ClaimCategory.FINANCIAL,
        "finance": ClaimCategory.FINANCIAL,
        "date / timeline": ClaimCategory.DATE_TIMELINE,
        "date/timeline": ClaimCategory.DATE_TIMELINE,
        "date": ClaimCategory.DATE_TIMELINE,
        "timeline": ClaimCategory.DATE_TIMELINE,
        "technical": ClaimCategory.TECHNICAL,
        "tech": ClaimCategory.TECHNICAL,
        "market claim": ClaimCategory.MARKET_CLAIM,
        "market": ClaimCategory.MARKET_CLAIM,
        "regulatory": ClaimCategory.REGULATORY,
        "regulation": ClaimCategory.REGULATORY,
        "policy": ClaimCategory.REGULATORY,
        "study": ClaimCategory.STATISTIC,
    }
    return mapping.get(normalized, ClaimCategory.OTHER)


def _normalize_claim_fingerprint(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class ClaimExtractor:
    """Extracts atomic factual claims with verbatim quotes and dual (primary + adversarial) queries."""

    def __init__(self, llm_client: LLMClient, max_claims: int = 25) -> None:
        self.llm_client = llm_client
        self.max_claims = max_claims

    def extract(self, pages: List[PageContent]) -> List[ExtractedClaim]:
        if not pages:
            return []

        formatted_doc = format_pages_for_llm(pages)
        prompt = build_extraction_prompt(formatted_doc, max_claims=self.max_claims)
        payload = self.llm_client.complete_json(prompt)

        raw_claims = payload.get("claims", [])
        extracted: List[ExtractedClaim] = []
        seen_fingerprints: set[str] = set()
        valid_page_nums = {p.page_number for p in pages}
        default_page = pages[0].page_number

        next_id = 1
        for raw_idx, item in enumerate(raw_claims, start=1):
            if len(extracted) >= self.max_claims:
                break
            try:
                claim_text = str(item.get("claim_text", "")).strip()
                if len(claim_text) < 5:
                    continue

                fingerprint = _normalize_claim_fingerprint(claim_text)
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)

                page_num = int(item.get("page_number", default_page))
                if page_num not in valid_page_nums:
                    page_num = default_page

                original_quote = str(item.get("original_quote") or claim_text).strip()
                primary_query = str(
                    item.get("primary_query") or item.get("search_query") or claim_text[:110]
                ).strip()
                adversarial_query = str(item.get("adversarial_query") or "").strip()

                extracted.append(
                    ExtractedClaim(
                        claim_id=next_id,
                        claim_text=claim_text,
                        original_quote=original_quote,
                        page_number=page_num,
                        category=_parse_category(str(item.get("category", "Statistic"))),
                        primary_query=primary_query,
                        adversarial_query=adversarial_query,
                        search_query=primary_query,
                    )
                )
                next_id += 1
            except Exception as exc:
                logger.warning("Skipping malformed extracted claim #%d: %s", raw_idx, exc)

        return extracted
