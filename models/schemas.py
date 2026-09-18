"""
Pydantic data models and typed state dictionary for the arXiv agent.

PaperMetadata  — what we know about a paper before we parse it
Briefing       — the structured output the user receives
QAExchange     — one question-answer pair in the QA loop
AgentState     — the mutable state bag carried through every graph node
"""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Lightweight metadata returned by arXiv search / direct ID fetch."""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str                          # ISO-8601 date string
    pdf_url: str
    categories: list[str] = Field(default_factory=list)


class Briefing(BaseModel):
    """Structured executive briefing produced by the summarize node."""
    title: str
    authors: list[str]
    arxiv_id: str
    published: str
    link: str
    summary: str                            # 1 paragraph, plain English
    problem_statement: str
    method: list[str]                       # bullet points
    key_results: list[str]
    limitations: list[str]                  # never empty — see grounding strategy
    follow_up_questions: list[str]
    source_quality: Literal["full_text", "abstract_only"]


class QAExchange(BaseModel):
    """A single QA round-trip stored in conversation history."""
    question: str
    answer: str
    grounded: bool
    cited_chunks: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    raw_query: str                          # original user input, never mutated
    active_query: str                       # working copy — reformulation edits this

    mode: Literal["paper_id", "topic"]
    candidates: list[PaperMetadata]
    selected_paper: PaperMetadata | None
    reformulation_attempts: int             # track how many times we've broadened

    parsed_sections: dict[str, str]         # section_name -> text
    parse_status: Literal["ok", "partial", "failed"]

    vector_store_ref: str                   # collection path / name

    briefing: Briefing | None
    conversation_history: list[QAExchange]

    errors: list[str]
    _last_branch: str                       # routing key consumed by Graph.run()
