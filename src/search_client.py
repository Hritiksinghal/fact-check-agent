"""Authority-ranked dual-query live web search fusion (Tavily primary + DuckDuckGo zero-key fallback)."""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from src.models import WebSource

logger = logging.getLogger(__name__)

# Tracking parameters stripped during canonical URL deduplication
TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "si",
}

PRIMARY_AUTHORITY_TLDS = (".gov", ".edu", ".mil", ".gov.uk", ".gov.in", ".gov.au", ".gov.ca")
INSTITUTIONAL_TLDS = (".org", ".int", ".ac.uk")

# Tier 1: Official government, intergovernmental, regulatory, and corporate primary sources
PRIMARY_OFFICIAL_DOMAINS = (
    "sec.gov",
    "un.org",
    "worldbank.org",
    "imf.org",
    "who.int",
    "oecd.org",
    "wto.org",
    "europa.eu",
    "nih.gov",
    "cdc.gov",
    "bls.gov",
    "census.gov",
    "bea.gov",
    "federalreserve.gov",
    "nasa.gov",
    "nist.gov",
    "bis.org",
    "openai.com",
    "about.google",
    "blog.google",
    "apple.com",
    "microsoft.com",
    "nvidia.com",
    "anthropic.com",
    "meta.com",
    "python.org",
)

# Tier 2: Established news wires, peer-reviewed journals, and authoritative research/fact-checkers
TRUSTED_MEDIA_RESEARCH_DOMAINS = (
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "economist.com",
    "pewresearch.org",
    "ourworldindata.org",
    "statista.com",
    "gartner.com",
    "mckinsey.com",
    "similarweb.com",
    "gs.statcounter.com",
    "nature.com",
    "science.org",
    "ieee.org",
    "acm.org",
    "arxiv.org",
    "thelancet.com",
    "nejm.org",
    "snopes.com",
    "factcheck.org",
    "politifact.com",
    "fullfact.org",
    "bbc.com",
    "bbc.co.uk",
    "cnbc.com",
    "wikipedia.org",
    "britannica.com",
)

# Low-signal SEO blogs, Q&A forums, and unvetted UGC platforms
LOW_SIGNAL_DOMAINS = (
    "quora.com",
    "reddit.com",
    "pinterest.com",
    "medium.com",
    "blogspot.com",
    "wordpress.com",
    "wixsite.com",
    "tumblr.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
)


def extract_domain(url: str) -> str:
    """Return a clean domain label (without leading www.) for display and scoring."""
    try:
        host = urlparse(url).netloc.lower().strip()
        if ":" in host:
            host = host.split(":", 1)[0]
        return host.removeprefix("www.") or "source"
    except Exception:
        return "source"


def normalize_url(url: str) -> str:
    """Canonicalize URLs for deduplication and citation-grounding equality checks."""
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    try:
        parsed = urlparse(cleaned)
        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.rstrip("/") or ""
        query_items = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k.lower() not in TRACKING_QUERY_PARAMS and not k.lower().startswith("utm_")
        ]
        query_items.sort()
        query = urlencode(query_items)
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return cleaned.rstrip("/").lower()


def _host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def score_source_authority(url: str) -> Tuple[int, str]:
    """Score domain authority to rank primary official sources above SEO blogs."""
    try:
        parsed = urlparse(url)
        host = extract_domain(url)
        path = parsed.path.lower()
    except Exception:
        return 0, "Standard"

    score = 0
    tier = "Standard"

    is_primary_tld = any(host.endswith(tld) for tld in PRIMARY_AUTHORITY_TLDS)
    is_primary_domain = any(_host_matches(host, d) for d in PRIMARY_OFFICIAL_DOMAINS)
    is_ir_or_press = (
        host.startswith(("investor.", "investors.", "ir.", "newsroom.", "press."))
        or any(seg in path for seg in ("/newsroom", "/investor-relations", "/press-release", "/form-10-k"))
    )

    if is_primary_tld or is_primary_domain:
        score += 8
        tier = "Primary / Official"
    elif any(_host_matches(host, d) for d in TRUSTED_MEDIA_RESEARCH_DOMAINS):
        score += 5
        tier = "Trusted Research / Wire"
    elif any(host.endswith(tld) for tld in INSTITUTIONAL_TLDS):
        score += 3
        tier = "Institutional"

    if is_ir_or_press:
        score += 3
        if tier == "Standard":
            tier = "Official Press / IR"

    if any(_host_matches(host, d) for d in LOW_SIGNAL_DOMAINS):
        score -= 4
        tier = "Low-Signal"

    return score, tier


class LiveSearchClient:
    """Unified dual-query web search wrapper supporting Tavily API with DuckDuckGo fallback."""

    TAVILY_ENDPOINT = "https://api.tavily.com/search"

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        max_results: int = 6,
        timeout_sec: int = 20,
    ) -> None:
        self.tavily_api_key = tavily_api_key
        self.max_results = max_results
        self.timeout_sec = timeout_sec
        self._lock = threading.Lock()
        self._ddg_lock = threading.Lock()
        self._last_ddg_ts = 0.0
        self._call_count = 0

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def reset_call_count(self) -> None:
        with self._lock:
            self._call_count = 0

    def _increment_calls(self) -> None:
        with self._lock:
            self._call_count += 1

    def search(self, query: str) -> List[WebSource]:
        """Run a single live web search with Tavily -> DuckDuckGo fallback and authority ranking."""
        raw = self._execute_single_query(query, query_origin="primary")
        return self._deduplicate_and_rank(raw)

    def search_dual(
        self,
        primary_query: str,
        adversarial_query: Optional[str] = None,
    ) -> List[WebSource]:
        """Run both primary and adversarial queries, fuse results, deduplicate URLs, and rank by authority."""
        combined: List[WebSource] = []
        primary_clean = (primary_query or "").strip()
        adv_clean = (adversarial_query or "").strip()

        if primary_clean:
            combined.extend(self._execute_single_query(primary_clean, query_origin="primary"))

        if adv_clean and adv_clean.lower() != primary_clean.lower():
            combined.extend(self._execute_single_query(adv_clean, query_origin="adversarial"))

        return self._deduplicate_and_rank(combined)

    def _execute_single_query(self, query: str, query_origin: str) -> List[WebSource]:
        self._increment_calls()
        sources: List[WebSource] = []
        if self.tavily_api_key:
            sources = self._search_tavily(query, query_origin=query_origin)
        if not sources:
            sources = self._search_duckduckgo(query, query_origin=query_origin)
        return sources

    def _enrich_source(self, src: WebSource) -> WebSource:
        score, tier = score_source_authority(src.url)
        src.domain = extract_domain(src.url)
        src.authority_score = score
        src.authority_tier = tier
        return src

    def _search_tavily(self, query: str, query_origin: str = "primary", max_retries: int = 2) -> List[WebSource]:
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max(4, self.max_results),
            "include_answer": False,
        }
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    self.TAVILY_ENDPOINT,
                    json=payload,
                    timeout=self.timeout_sec,
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    time.sleep(1.2 * (2**attempt) + random.uniform(0.1, 0.4))
                    continue
                resp.raise_for_status()
                data = resp.json()
                results: List[WebSource] = []
                for item in data.get("results", []):
                    url = (item.get("url") or "").strip()
                    snippet = (item.get("content") or "").strip()
                    if url and snippet:
                        results.append(
                            self._enrich_source(
                                WebSource(
                                    title=(item.get("title") or url).strip(),
                                    url=url,
                                    snippet=snippet[:950],
                                    published_date=item.get("published_date"),
                                    provider="tavily",
                                    query_origin=query_origin,
                                )
                            )
                        )
                return results
            except Exception as exc:
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                logger.warning(
                    "Tavily search failed for '%s': %s; falling back to DuckDuckGo.",
                    query,
                    exc,
                )
                return []
        return []

    def _ddg_throttle(self) -> None:
        """Space DuckDuckGo requests slightly across threads to prevent burst throttling."""
        with self._ddg_lock:
            now = time.monotonic()
            wait = 0.4 - (now - self._last_ddg_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_ddg_ts = time.monotonic()

    def _search_duckduckgo(
        self,
        query: str,
        query_origin: str = "primary",
        max_retries: int = 1,
    ) -> List[WebSource]:
        from ddgs import DDGS

        words = query.split()
        short_query = " ".join(words[:5]) if len(words) > 5 else query
        attempt_queries = [query]
        if max_retries >= 1:
            attempt_queries.append(short_query)

        for attempt, active_query in enumerate(attempt_queries):
            try:
                self._ddg_throttle()
                results: List[WebSource] = []
                with DDGS(timeout=min(8, self.timeout_sec)) as ddgs:
                    raw_items = ddgs.text(active_query, max_results=max(4, self.max_results))
                    for item in raw_items or []:
                        url = (item.get("href") or item.get("link") or "").strip()
                        snippet = (item.get("body") or item.get("snippet") or "").strip()
                        if url and snippet:
                            results.append(
                                self._enrich_source(
                                    WebSource(
                                        title=(item.get("title") or url).strip(),
                                        url=url,
                                        snippet=snippet[:950],
                                        published_date=item.get("date") or item.get("published_date"),
                                        provider="duckduckgo",
                                        query_origin=query_origin,
                                    )
                                )
                            )
                if results:
                    return results
            except Exception as exc:
                if attempt < len(attempt_queries) - 1:
                    time.sleep(0.35 + random.uniform(0.05, 0.2))
                    continue
                logger.warning("DuckDuckGo search failed for '%s': %s", active_query, exc)
                return []
        return []

    def _deduplicate_and_rank(self, sources: List[WebSource]) -> List[WebSource]:
        by_url: Dict[str, WebSource] = {}
        for raw_src in sources:
            src = self._enrich_source(raw_src.model_copy())
            norm = normalize_url(src.url)
            if not norm:
                continue
            existing = by_url.get(norm)
            if existing is None:
                by_url[norm] = src
            else:
                # Merge corroborated duplicate across primary + adversarial queries
                if existing.query_origin != src.query_origin:
                    existing.query_origin = "both"
                if len(src.snippet) > len(existing.snippet):
                    existing.snippet = src.snippet
                if not existing.published_date and src.published_date:
                    existing.published_date = src.published_date
                if src.authority_score > existing.authority_score:
                    existing.authority_score = src.authority_score
                    existing.authority_tier = src.authority_tier

        ranked = list(by_url.values())
        ranked.sort(
            key=lambda s: (
                s.authority_score + (1 if s.query_origin == "both" else 0),
                1 if s.published_date else 0,
                len(s.snippet),
            ),
            reverse=True,
        )
        return ranked[: self.max_results]
