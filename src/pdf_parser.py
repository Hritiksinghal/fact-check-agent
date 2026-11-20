"""Layout- and table-aware PDF ingestion with exact [Page N] anchors, SHA-256 hashing, and OCR detection."""

from __future__ import annotations

import hashlib
import io
import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from pypdf import PdfReader

from src.models import PageContent, ParsedDocument

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS_FOR_NON_SCANNED = 25


def compute_pdf_sha256(raw_bytes: bytes) -> str:
    """Compute a stable SHA-256 hex digest for PDF deduplication and audit caching."""
    return hashlib.sha256(raw_bytes).hexdigest()


def _load_raw_bytes(source: Union[bytes, io.BytesIO, str, Path]) -> bytes:
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            return f.read()
    if isinstance(source, io.BytesIO):
        return source.getvalue()
    return source


def _normalize_whitespace(text: str) -> str:
    """Clean extraction artifacts while preserving paragraph boundaries."""
    text = text.replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned.strip()


def _clean_table_cell(cell: Optional[str]) -> str:
    if cell is None:
        return ""
    cleaned = re.sub(r"\s+", " ", cell.replace("\x00", "")).strip()
    return cleaned.replace("|", "\\|")


def table_to_markdown(raw_table: Sequence[Sequence[Optional[str]]]) -> str:
    """Convert a 2D pdfplumber table into a clean GitHub-Flavored Markdown table."""
    cleaned_rows: List[List[str]] = []
    for row in raw_table:
        if not row:
            continue
        cells = [_clean_table_cell(c) for c in row]
        if any(cells):
            cleaned_rows.append(cells)

    if len(cleaned_rows) < 2:
        return ""

    col_count = max(len(r) for r in cleaned_rows)
    if col_count < 2:
        return ""

    padded = [r + [""] * (col_count - len(r)) for r in cleaned_rows]
    # Drop completely empty columns
    active_cols = [c for c in range(col_count) if any(r[c] for r in padded)]
    if len(active_cols) < 2:
        return ""

    normalized = [[r[c] for c in active_cols] for r in padded]
    header = normalized[0]
    body = normalized[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_tables_from_plumber_page(plumber_page: object) -> List[str]:
    """Extract all valid tables on a pdfplumber page and format them as Markdown."""
    markdown_tables: List[str] = []
    try:
        raw_tables = plumber_page.extract_tables() or []  # type: ignore[attr-defined]
    except Exception as exc:
        logger.debug("pdfplumber table extraction skipped on page: %s", exc)
        return markdown_tables

    for raw_table in raw_tables:
        md = table_to_markdown(raw_table)
        if md:
            markdown_tables.append(md)
    return markdown_tables


def parse_pdf(source: Union[bytes, io.BytesIO, str, Path]) -> ParsedDocument:
    """Parse a PDF using pypdf + pdfplumber for body text, Markdown tables, SHA-256, and scan detection."""
    raw_bytes = _load_raw_bytes(source)
    sha256_hash = compute_pdf_sha256(raw_bytes)

    reader = PdfReader(io.BytesIO(raw_bytes))
    total_pages = len(reader.pages)
    pages: List[PageContent] = []
    total_tables = 0
    total_non_ws_chars = 0
    has_embedded_images = False

    plumber_pdf = None
    try:
        try:
            import pdfplumber

            plumber_pdf = pdfplumber.open(io.BytesIO(raw_bytes))
        except Exception as exc:
            logger.warning("pdfplumber could not open PDF (%s); proceeding with pypdf only.", exc)
            plumber_pdf = None

        for idx, pypdf_page in enumerate(reader.pages, start=1):
            method = "pypdf"
            body_text = _normalize_whitespace(pypdf_page.extract_text() or "")

            if not has_embedded_images:
                try:
                    if len(pypdf_page.images) > 0:
                        has_embedded_images = True
                except Exception:
                    pass

            tables_md: List[str] = []
            if plumber_pdf is not None and idx - 1 < len(plumber_pdf.pages):
                p_page = plumber_pdf.pages[idx - 1]
                if not has_embedded_images:
                    try:
                        if getattr(p_page, "images", None):
                            has_embedded_images = True
                    except Exception:
                        pass

                # Fallback to pdfplumber body text when pypdf yields sparse output
                if len(body_text) < 50:
                    try:
                        plumber_text = _normalize_whitespace(p_page.extract_text() or "")
                        if len(plumber_text) > len(body_text):
                            body_text = plumber_text
                            method = "pdfplumber"
                    except Exception:
                        pass

                tables_md = _extract_tables_from_plumber_page(p_page)
                if tables_md:
                    total_tables += len(tables_md)
                    method = f"{method}+tables"

            combined_parts: List[str] = []
            if body_text:
                combined_parts.append(body_text)
            for t_idx, md_table in enumerate(tables_md, start=1):
                combined_parts.append(f"[Table {t_idx}]\n{md_table}")

            combined_text = "\n\n".join(combined_parts).strip()
            total_non_ws_chars += len(re.sub(r"\s+", "", combined_text))

            if combined_text:
                pages.append(
                    PageContent(
                        page_number=idx,
                        text=combined_text,
                        tables_markdown=tables_md,
                        extraction_method=method,
                    )
                )
    finally:
        if plumber_pdf is not None:
            try:
                plumber_pdf.close()
            except Exception:
                pass

    is_scanned = total_pages > 0 and (not pages or total_non_ws_chars < MIN_TEXT_CHARS_FOR_NON_SCANNED)
    warning_message: Optional[str] = None
    if is_scanned:
        scan_hint = "embedded image scans" if has_embedded_images else "no selectable text layer"
        warning_message = (
            f"Scanned or image-only PDF detected ({total_pages} page(s), {scan_hint}). "
            "TruthLayer requires selectable text to extract and highlight claims accurately. "
            "Run OCR on this PDF (e.g., Adobe Acrobat, macOS Preview, or `ocrmypdf input.pdf output.pdf`) "
            "and upload the OCR-processed PDF."
        )

    return ParsedDocument(
        pages=pages,
        total_pages=total_pages,
        sha256_hash=sha256_hash,
        is_scanned_or_image_only=is_scanned,
        warning_message=warning_message,
        table_count=total_tables,
    )


def extract_pdf_pages(source: Union[bytes, io.BytesIO, str, Path]) -> Tuple[List[PageContent], int]:
    """Backward-compatible helper returning (pages, total_pages) via parse_pdf."""
    parsed = parse_pdf(source)
    return parsed.pages, parsed.total_pages


def format_pages_for_llm(pages: List[PageContent], max_chars: int = 42000) -> str:
    """Format extracted pages and Markdown tables with explicit [Page N] anchors."""
    chunks: List[str] = []
    used = 0
    for p in pages:
        block = f"=== [Page {p.page_number}] ===\n{p.text}\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                chunks.append(block[:remaining] + "\n...[truncated]")
            break
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks)
