"""
PDF downloader and section-aware parser using PyMuPDF (fitz).

Heuristic section detection:
    - A line is a heading if its font size is ≥ median + 2pt AND it's
      short (< 120 chars). This is intentionally simple; "good enough" for
      most arXiv PDFs, with a known limitation on non-standard layouts.
    - Falls back gracefully: if section detection finds < 2 sections,
      we dump all text under a single "body" key (parse_status = "partial").
"""

from __future__ import annotations

import os
import re
import tempfile
from statistics import median
from typing import Optional

import pymupdf as fitz  # PyMuPDF (fitz alias deprecated)
import requests


def download_pdf(pdf_url: str, dest_dir: Optional[str] = None) -> str:
    """Download a PDF from `pdf_url` and return the local file path."""
    dest_dir = dest_dir or tempfile.mkdtemp(prefix="arxiv_")
    filename = pdf_url.rstrip("/").split("/")[-1]
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    path = os.path.join(dest_dir, filename)
    resp = requests.get(pdf_url, timeout=60)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def _extract_blocks(pdf_path: str) -> list[dict]:
    """Extract text blocks with font-size metadata from each page."""
    doc = fitz.open(pdf_path)
    blocks = []
    for page in doc:
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text_parts = []
                sizes = []
                for span in line.get("spans", []):
                    text_parts.append(span.get("text", ""))
                    sizes.append(span.get("size", 10))
                text = " ".join(text_parts).strip()
                if text:
                    blocks.append({
                        "text": text,
                        "size": max(sizes) if sizes else 10,
                    })
    doc.close()
    return blocks


def _detect_sections(blocks: list[dict]) -> dict[str, str]:
    """
    Group blocks into sections keyed by detected heading text.
    Returns a dict: { section_name: section_body_text, ... }
    """
    if not blocks:
        return {}

    all_sizes = [b["size"] for b in blocks]
    med_size = median(all_sizes)
    heading_threshold = med_size + 2.0

    sections: dict[str, str] = {}
    current_heading = "preamble"
    current_text: list[str] = []

    for block in blocks:
        text = block["text"]
        size = block["size"]

        is_heading = (
            size >= heading_threshold
            and len(text) < 120
            and not text.strip().startswith("(")
        )

        if is_heading:
            if current_text:
                body = "\n".join(current_text).strip()
                if body:
                    sections[current_heading] = body
            cleaned = re.sub(r"^[\d.]+\s*", "", text).strip()
            current_heading = cleaned if cleaned else text.strip()
            current_text = []
        else:
            current_text.append(text)

    if current_text:
        body = "\n".join(current_text).strip()
        if body:
            sections[current_heading] = body

    return sections


def parse_pdf(
    pdf_path: str,
) -> tuple[dict[str, str], str]:
    """
    Parse a local PDF into sections.

    Returns:
        (sections_dict, parse_status)
        parse_status is one of: "ok", "partial", "failed"
    """
    try:
        blocks = _extract_blocks(pdf_path)
    except Exception:
        return {}, "failed"

    if not blocks:
        return {}, "failed"

    sections = _detect_sections(blocks)

    if len(sections) < 2:
        all_text = "\n".join(b["text"] for b in blocks)
        return {"body": all_text}, "partial"

    return sections, "ok"
