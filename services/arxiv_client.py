"""
arXiv API wrapper — search by topic and fetch by ID/URL.

Uses the `arxiv` pip package which wraps the Atom API.
No hand-rolled XML parsing.
"""

from __future__ import annotations

import re
from typing import Optional

import arxiv

from models.schemas import PaperMetadata


_ARXIV_ID_PATTERN = re.compile(
    r"(?:^|\b)"
    r"(\d{4}\.\d{4,5})"
    r"(?:v\d+)?"
    r"(?:$|\b)",
)
_ARXIV_URL_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)


def detect_arxiv_id(query: str) -> Optional[str]:
    """Return an arXiv ID if the query contains one, else None."""
    m = _ARXIV_URL_PATTERN.search(query)
    if m:
        return m.group(1)
    m = _ARXIV_ID_PATTERN.search(query)
    if m:
        return m.group(1)
    return None


def _result_to_metadata(result: arxiv.Result) -> PaperMetadata:
    """Convert an arxiv.Result into our domain model."""
    entry_url = result.entry_id
    short_id_match = re.search(r"(\d{4}\.\d{4,5})", entry_url)
    short_id = short_id_match.group(1) if short_id_match else entry_url

    return PaperMetadata(
        arxiv_id=short_id,
        title=result.title.replace("\n", " ").strip(),
        authors=[a.name for a in result.authors],
        abstract=result.summary.replace("\n", " ").strip(),
        published=result.published.strftime("%Y-%m-%d") if result.published else "",
        pdf_url=result.pdf_url or "",
        categories=[c for c in (result.categories or [])],
    )


def fetch_by_id(arxiv_id: str) -> Optional[PaperMetadata]:
    """Fetch a single paper by its arXiv ID. Returns None if not found."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        return None
    return _result_to_metadata(results[0])


def search_by_topic(query: str, max_results: int = 10) -> list[PaperMetadata]:
    """Search arXiv by a topic query string. Returns up to max_results papers."""
    client = arxiv.Client()
    advanced_query = f'ti:"{query}" OR all:"{query}"'
    search = arxiv.Search(
        query=advanced_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [_result_to_metadata(r) for r in client.results(search)]
