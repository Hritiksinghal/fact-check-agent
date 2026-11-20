"""Centralized configuration loader supporting environment variables, project .env, and Streamlit secrets."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()


def _lookup_mapping_secret(mapping: Mapping[str, Any], target_key: str) -> Optional[str]:
    """Search a Streamlit secrets mapping (top-level and nested sections, case-insensitive)."""
    target_lower = target_key.lower()
    # 1. Direct / case-insensitive top-level match
    for k, v in mapping.items():
        if str(k).lower() == target_lower and not isinstance(v, Mapping):
            val = str(v).strip()
            if val:
                return val
    # 2. Nested TOML section match (e.g. [general] or [secrets])
    for v in mapping.values():
        if isinstance(v, Mapping):
            found = _lookup_mapping_secret(v, target_key)
            if found:
                return found
    return None


def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a secret from environment variables first, then Streamlit secrets (flat or nested)."""
    for env_key in (key, key.lower()):
        val = os.getenv(env_key)
        if val and val.strip():
            return val.strip()
    try:
        import streamlit as st

        found = _lookup_mapping_secret(st.secrets, key)
        if found:
            return found
    except Exception:
        pass
    return default


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings for LLM inference, dual-query web search, and verification concurrency."""

    gemini_api_key: Optional[str]
    openai_api_key: Optional[str]
    tavily_api_key: Optional[str]
    gemini_model: str = "gemini-3.5-flash-lite"
    openai_model: str = "gpt-4o-mini"
    max_claims: int = 25
    max_workers: int = 4
    search_results_per_claim: int = 6
    request_timeout_sec: int = 25
    enable_cache: bool = True

    @property
    def active_llm_provider(self) -> str:
        if self.gemini_api_key:
            return "gemini"
        if self.openai_api_key:
            return "openai"
        return "none"

    @property
    def active_llm_model(self) -> str:
        if self.gemini_api_key:
            return self.gemini_model
        if self.openai_api_key:
            return self.openai_model
        return "none"

    @property
    def active_search_provider(self) -> str:
        if self.tavily_api_key:
            return "tavily"
        return "duckduckgo"


def load_settings(
    gemini_override: Optional[str] = None,
    openai_override: Optional[str] = None,
    tavily_override: Optional[str] = None,
    max_claims: int = 25,
    enable_cache: bool = True,
) -> AppSettings:
    """Build runtime settings with optional UI-provided key overrides."""
    return AppSettings(
        gemini_api_key=(gemini_override or "").strip() or _get_secret("GEMINI_API_KEY"),
        openai_api_key=(openai_override or "").strip() or _get_secret("OPENAI_API_KEY"),
        tavily_api_key=(tavily_override or "").strip() or _get_secret("TAVILY_API_KEY"),
        gemini_model=_get_secret("GEMINI_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite",
        openai_model=_get_secret("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        max_claims=max_claims,
        enable_cache=enable_cache,
    )
