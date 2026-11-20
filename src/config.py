"""Centralized configuration loader supporting environment variables, project .env, and Streamlit secrets."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

GEMINI_KEY_ALIASES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GEMINI_API_KEY", "GEMINI_KEY")
OPENAI_KEY_ALIASES = ("OPENAI_API_KEY", "OPENAI_KEY")
TAVILY_KEY_ALIASES = ("TAVILY_API_KEY", "TAVILY_KEY")


def _reload_local_dotenv() -> None:
    """Load project-root .env and cwd .env so updates take effect without server restarts."""
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    load_dotenv(override=False)


_reload_local_dotenv()


def _lookup_mapping_secret(mapping: Mapping[str, Any], target_keys: Sequence[str]) -> Optional[str]:
    """Search a secrets mapping (top-level and nested sections, case-insensitive)."""
    targets = {k.lower() for k in target_keys}
    for k, v in mapping.items():
        if str(k).lower() in targets and not isinstance(v, Mapping):
            val = str(v).strip()
            if val:
                return val
    for v in mapping.values():
        if isinstance(v, Mapping):
            found = _lookup_mapping_secret(v, target_keys)
            if found:
                return found
    return None


def _read_local_secrets_toml() -> Mapping[str, Any]:
    """Read project-local .streamlit/secrets.toml directly as a fallback if cwd differs."""
    toml_path = _PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not toml_path.exists():
        return {}
    try:
        return tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_detected_secret_names() -> List[str]:
    """Return ONLY the key names (never values) currently visible in Streamlit Secrets for safe debugging."""
    names: List[str] = []
    try:
        import streamlit as st

        raw = st.secrets.to_dict() if hasattr(st.secrets, "to_dict") else dict(st.secrets)
        for k, v in raw.items():
            if isinstance(v, Mapping):
                for sub_k in v.keys():
                    names.append(f"{k}.{sub_k}")
            else:
                names.append(str(k))
    except Exception:
        pass
    return sorted(set(names))


def _get_secret(keys: str | Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """Read a secret from environment variables, Streamlit st.secrets, or project secrets.toml."""
    key_list = [keys] if isinstance(keys, str) else list(keys)

    # 1. Environment variables (including .env)
    for k in key_list:
        for variant in (k, k.lower()):
            val = os.getenv(variant)
            if val and val.strip():
                return val.strip()

    # 2. Streamlit Cloud / runtime st.secrets
    try:
        import streamlit as st

        raw = st.secrets.to_dict() if hasattr(st.secrets, "to_dict") else st.secrets
        found = _lookup_mapping_secret(raw, key_list)
        if found:
            return found
    except Exception:
        pass

    # 3. Project-local .streamlit/secrets.toml direct fallback
    local_toml = _read_local_secrets_toml()
    if local_toml:
        found = _lookup_mapping_secret(local_toml, key_list)
        if found:
            return found

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
    _reload_local_dotenv()
    return AppSettings(
        gemini_api_key=(gemini_override or "").strip() or _get_secret(GEMINI_KEY_ALIASES),
        openai_api_key=(openai_override or "").strip() or _get_secret(OPENAI_KEY_ALIASES),
        tavily_api_key=(tavily_override or "").strip() or _get_secret(TAVILY_KEY_ALIASES),
        gemini_model=_get_secret("GEMINI_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite",
        openai_model=_get_secret("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        max_claims=max_claims,
        enable_cache=enable_cache,
    )
