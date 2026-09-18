"""
Section-aware text chunker.

Chunks within section boundaries — never across them.
Each chunk is tagged with metadata for downstream citation.
"""

from __future__ import annotations


def chunk_sections(
    sections: dict[str, str],
    arxiv_id: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> list[dict]:
    """
    Split section text into overlapping chunks.

    Returns a list of dicts:
        {
            "arxiv_id": str,
            "section": str,
            "chunk_id": str,
            "text": str,
        }
    """
    chunks: list[dict] = []

    for section_name, text in sections.items():
        text = " ".join(text.split())
        if not text:
            continue

        start = 0
        idx = 0
        while start < len(text):
            end = start + max_chars
            chunk_text = text[start:end]

            chunks.append({
                "arxiv_id": arxiv_id,
                "section": section_name,
                "chunk_id": f"{_slugify(section_name)}_{idx}",
                "text": chunk_text,
            })

            idx += 1
            start = end - overlap
            if start < 0:
                start = 0
            if end >= len(text):
                break

    return chunks


def _slugify(text: str) -> str:
    """Turn a section name into a safe slug for chunk IDs."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "section"
