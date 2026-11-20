"""Thread-safe SHA-256 report cache for instantaneous repeat PDF audits."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

from src.models import FactCheckReport


class AuditReportCache:
    """Bounded LRU cache storing completed FactCheckReport objects keyed by PDF SHA-256 hash."""

    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max(1, max_entries)
        self._store: OrderedDict[str, FactCheckReport] = OrderedDict()
        self._lock = threading.Lock()

    def _normalize_key(self, sha256_hash: str) -> str:
        return (sha256_hash or "").strip().lower()

    def get(self, sha256_hash: str) -> Optional[FactCheckReport]:
        """Return a deep copy of a cached report if present, else None."""
        key = self._normalize_key(sha256_hash)
        if not key:
            return None
        with self._lock:
            report = self._store.get(key)
            if report is None:
                return None
            self._store.move_to_end(key)
            return report.model_copy(deep=True)

    def put(self, sha256_hash: str, report: FactCheckReport) -> None:
        """Store a deep copy of a completed report under its PDF SHA-256 hash."""
        key = self._normalize_key(sha256_hash)
        if not key:
            return
        with self._lock:
            self._store[key] = report.model_copy(deep=True)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def has(self, sha256_hash: str) -> bool:
        key = self._normalize_key(sha256_hash)
        if not key:
            return False
        with self._lock:
            return key in self._store

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
