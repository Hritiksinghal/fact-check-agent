"""Resilient, provider-agnostic structured JSON LLM client with rate-limit-aware retries and telemetry."""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from typing import Any, Dict, Optional

from src.config import AppSettings

logger = logging.getLogger(__name__)

GEMINI_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
)


def _is_model_unavailable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "404",
            "not_found",
            "no longer available",
            "503",
            "unavailable",
            "high demand",
            "overloaded",
            "timeout",
            "timed out",
        )
    )


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Parse JSON safely even if wrapped in markdown code fences or prose."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "429",
            "rate limit",
            "rate_limit",
            "resource_exhausted",
            "quota",
            "too many requests",
        )
    )


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return _is_rate_limit_error(exc) or any(
        token in msg
        for token in (
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "connection reset",
            "overloaded",
            "empty response",
            "jsondecodeerror",
        )
    )


class LLMClient:
    """Thin, thread-safe wrapper around Gemini (primary) and OpenAI (fallback) for strict JSON tasks."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.provider = settings.active_llm_provider
        self.model_name = settings.active_llm_model
        self._active_gemini_model = settings.gemini_model
        if self.provider == "none":
            raise ValueError(
                "No LLM API key configured. Provide GEMINI_API_KEY or OPENAI_API_KEY "
                "via environment variables, Streamlit secrets, or the sidebar."
            )
        self._lock = threading.Lock()
        self._call_count = 0
        self._gemini_client: Optional[Any] = None
        self._openai_client: Optional[Any] = None

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

    def complete_json(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Generate a structured JSON response with rate-limit-aware exponential backoff."""
        try:
            return self._complete_with_provider(self.provider, prompt, max_retries=max_retries)
        except Exception as primary_exc:
            # Seamless failover to OpenAI if Gemini fails permanently and OpenAI key is also configured
            if self.provider == "gemini" and self.settings.openai_api_key:
                logger.warning(
                    "Primary Gemini provider failed (%s); failing over to OpenAI (%s).",
                    primary_exc,
                    self.settings.openai_model,
                )
                return self._complete_with_provider("openai", prompt, max_retries=max_retries)
            raise

    def _complete_with_provider(
        self,
        provider: str,
        prompt: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                self._increment_calls()
                if provider == "gemini":
                    raw = self._call_gemini(prompt)
                else:
                    raw = self._call_openai(prompt)
                return _extract_json_object(raw)
            except Exception as exc:
                last_error = exc
                rate_limited = _is_rate_limit_error(exc)
                transient = _is_transient_error(exc)
                logger.warning(
                    "LLM call attempt %d/%d failed (%s, rate_limited=%s): %s",
                    attempt + 1,
                    max_retries + 1,
                    provider,
                    rate_limited,
                    exc,
                )
                if attempt < max_retries and transient:
                    base_delay = 2.5 if rate_limited else 1.2
                    sleep_sec = min(18.0, base_delay * (2**attempt) + random.uniform(0.15, 0.75))
                    time.sleep(sleep_sec)
                elif attempt < max_retries:
                    time.sleep(0.8)
                else:
                    break
        raise RuntimeError(
            f"LLM inference failed ({provider}) after {max_retries + 1} attempts: {last_error}"
        )

    def _get_gemini_client(self) -> Any:
        if self._gemini_client is None:
            from google import genai
            from google.genai import types

            try:
                http_opts = types.HttpOptions(
                    timeout=self.settings.request_timeout_sec * 1000,
                    retry_options=types.HttpRetryOptions(attempts=2),
                )
                self._gemini_client = genai.Client(
                    api_key=self.settings.gemini_api_key,
                    http_options=http_opts,
                )
            except Exception:
                self._gemini_client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._gemini_client

    def _get_openai_client(self) -> Any:
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.request_timeout_sec,
            )
        return self._openai_client

    def _call_gemini(self, prompt: str) -> str:
        from google.genai import types

        client = self._get_gemini_client()
        candidates = [self._active_gemini_model] + [
            m for m in GEMINI_FALLBACK_MODELS if m != self._active_gemini_model
        ]
        last_exc: Optional[Exception] = None
        for idx, model_id in enumerate(candidates):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                if not response.text:
                    raise ValueError(f"Empty response received from Gemini model '{model_id}'.")
                if model_id != self._active_gemini_model:
                    with self._lock:
                        self._active_gemini_model = model_id
                        self.model_name = model_id
                return response.text
            except Exception as exc:
                last_exc = exc
                if idx < len(candidates) - 1 and (
                    _is_model_unavailable_error(exc) or _is_rate_limit_error(exc)
                ):
                    logger.warning(
                        "Gemini model '%s' unavailable/throttled (%s); stepping to fallback '%s'.",
                        model_id,
                        exc,
                        candidates[idx + 1],
                    )
                    continue
                raise
        raise last_exc or RuntimeError("All Gemini fallback models failed.")

    def _call_openai(self, prompt: str) -> str:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise fact-checking engine. Always return strict JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response received from OpenAI API.")
        return content
