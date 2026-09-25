"""
Node functions — one per stage of the agent pipeline.

Each function has the signature:
    (state: AgentState) -> AgentState

It reads what it needs, does its work, sets state["_last_branch"],
and returns the mutated state.
"""

from __future__ import annotations

import json
import os
import tempfile

from models.schemas import (
    AgentState,
    Briefing,
    PaperMetadata,
    QAExchange,
)
from services import arxiv_client, llm_client
from services.chunker import chunk_sections
from services.embedder import embed_query, embed_texts
from services.pdf_parser import download_pdf, parse_pdf
from services.vector_store import (
    collection_exists,
    index_chunks,
    query_similar,
)
from prompts.templates import (
    BRIEFING_PROMPT,
    QA_PROMPT,
    QA_DECLINE_MSG,
    REFORMULATE_PROMPT,
    TARGETED_QUERIES,
)

# Confidence threshold for QA grounding (dense cosine similarity)
QA_CONFIDENCE_THRESHOLD = 0.35

MAX_REFORMULATION_ATTEMPTS = 2


# 1. ROUTE — deterministic regex, no LLM call

def route(state: AgentState) -> AgentState:
    """Classify the input as an arXiv ID/URL or a topic query."""
    raw = state["raw_query"]
    detected_id = arxiv_client.detect_arxiv_id(raw)

    if detected_id:
        state["mode"] = "paper_id"
        state["active_query"] = detected_id
        print(f"  Detected arXiv ID: {detected_id}")
        state["_last_branch"] = "id"
    else:
        state["mode"] = "topic"
        state["active_query"] = raw
        print(f"  Topic query: {raw}")
        state["_last_branch"] = "topic"

    return state


# 2. RETRIEVE - search arXiv by topic

def retrieve(state: AgentState) -> AgentState:
    """Search arXiv for candidate papers matching the active query."""
    query = state["active_query"]
    print(f"  Searching arXiv for: '{query}'")
    candidates = arxiv_client.search_by_topic(query, max_results=10)
    state["candidates"] = candidates
    count = len(candidates)
    print(f"  Found {count} candidate(s)")

    if count == 0:
        state["_last_branch"] = "zero"
    elif count == 1:
        state["selected_paper"] = candidates[0]
        print(f"  Auto-selected: {candidates[0].title}")
        state["_last_branch"] = "one"
    else:
        state["_last_branch"] = "many"

    return state


# 3. REFORMULATE — broaden a failing query

def reformulate(state: AgentState) -> AgentState:
    """Broaden the query if zero candidates were found. Gives up after MAX attempts."""
    attempts = state.get("reformulation_attempts", 0)

    if attempts >= MAX_REFORMULATION_ATTEMPTS:
        print(f"  Gave up after {attempts} reformulation attempts.")
        state["errors"] = state.get("errors", []) + [
            f"No papers found after {attempts} query reformulations."
        ]
        state["_last_branch"] = "give_up"
        return state

    raw = state["raw_query"]
    prompt = REFORMULATE_PROMPT.format(original_query=raw, attempt=attempts + 1)

    try:
        broader = llm_client.complete(prompt, temperature=0.5, max_tokens=100)
        broader = broader.strip().strip('"').strip("'")
        print(f"  Reformulated: '{raw}' → '{broader}'")
    except Exception as e:
        broader = raw
        state["errors"] = state.get("errors", []) + [f"Reformulation LLM error: {e}"]
        print(f"  Reformulation failed, retrying original query")

    state["active_query"] = broader
    state["reformulation_attempts"] = attempts + 1
    state["_last_branch"] = "retry"
    return state


# 4. RANK — pick the best candidate from many

def rank(state: AgentState) -> AgentState:
    """
    Rank candidates by cosine similarity of (query embedding, abstract embedding)
    with recency as tiebreak. Deterministic — no LLM call.
    """
    candidates = state.get("candidates", [])
    query = state["active_query"]

    if not candidates:
        state["_last_branch"] = "zero"
        return state

    texts = [query] + [c.abstract for c in candidates]
    embeddings = embed_texts(texts)
    q_emb = embeddings[0]

    import numpy as np
    q_vec = np.array(q_emb)
    scored = []
    for i, candidate in enumerate(candidates):
        c_vec = np.array(embeddings[i + 1])
        sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9))
        scored.append((sim, candidate))

    scored.sort(key=lambda x: (x[0], x[1].published), reverse=True)

    best = scored[0][1]
    print(f"  Ranked {len(candidates)} candidates")
    print(f"  Selected: {best.title} (sim={scored[0][0]:.3f})")
    state["selected_paper"] = best
    state["_last_branch"] = "ranked"
    return state


# 5. FETCH_PARSE — download PDF and extract sections

def fetch_parse(state: AgentState) -> AgentState:
    """Download and parse the selected paper's PDF."""
    paper: PaperMetadata | None = state.get("selected_paper")

    if paper is None:
        arxiv_id = state["active_query"]
        print(f"  Fetching paper by ID: {arxiv_id}")
        paper = arxiv_client.fetch_by_id(arxiv_id)
        if paper is None:
            state["errors"] = state.get("errors", []) + [
                f"Paper {arxiv_id} not found on arXiv."
            ]
            state["parse_status"] = "failed"
            state["_last_branch"] = "failed"
            return state
        state["selected_paper"] = paper

    print(f"  Paper: {paper.title}")
    print(f"  PDF:   {paper.pdf_url}")

    if paper.pdf_url:
        try:
            pdf_path = download_pdf(paper.pdf_url)
            sections, status = parse_pdf(pdf_path)
            state["parsed_sections"] = sections
            state["parse_status"] = status
            print(f"  Parse status: {status} ({len(sections)} sections)")
            if sections:
                print(f"  Sections: {', '.join(sections.keys())}")

            try:
                os.remove(pdf_path)
            except OSError:
                pass

            if status != "failed":
                state["_last_branch"] = status
                return state
        except Exception as e:
            print(f"  PDF download/parse failed: {e}")
            state["errors"] = state.get("errors", []) + [f"PDF error: {e}"]

    print("  Falling back to abstract-only mode")
    state["parsed_sections"] = {"abstract": paper.abstract}
    state["parse_status"] = "partial"
    state["_last_branch"] = "partial"
    return state


# 6. CHUNK_EMBED — section-aware chunking + vector indexing

def chunk_embed(state: AgentState) -> AgentState:
    """Chunk parsed sections and index them in the vector store."""
    sections = state.get("parsed_sections", {})
    paper = state["selected_paper"]
    arxiv_id = paper.arxiv_id

    if not sections:
        state["errors"] = state.get("errors", []) + ["No sections to chunk."]
        state["_last_branch"] = "default"
        return state

    chunks = chunk_sections(sections, arxiv_id)
    print(f"  Created {len(chunks)} chunks from {len(sections)} sections")

    col_name = index_chunks(chunks, arxiv_id)
    state["vector_store_ref"] = col_name
    print(f"  Indexed in collection: {col_name}")

    state["_last_branch"] = "default"
    return state


# 7. SUMMARIZE — structured briefing via targeted retrieval

def _gather_targeted_context(
    arxiv_id: str, paper_title: str, abstract: str
) -> str:
    """
    Run targeted retrieval for each briefing field, then merge into
    a single labelled context block for the LLM.
    """
    sections: list[str] = []

    general = query_similar(
        query=f"{paper_title} {abstract[:200]}",
        arxiv_id=arxiv_id,
        top_k=5,
    )
    if general:
        block = "\n".join(f"[{c['section']}] {c['text']}" for c in general)
        sections.append(f"### General overview\n{block}")

    for field, query_str in TARGETED_QUERIES.items():
        hits = query_similar(
            query=query_str,
            arxiv_id=arxiv_id,
            top_k=4,
        )
        if hits:
            block = "\n".join(f"[{c['section']}] {c['text']}" for c in hits)
            sections.append(f"### Context for {field}\n{block}")

    return "\n\n---\n\n".join(sections)


def summarize(state: AgentState) -> AgentState:
    """Generate a structured Briefing using field-by-field targeted retrieval."""
    paper = state["selected_paper"]
    arxiv_id = paper.arxiv_id
    parse_status = state.get("parse_status", "partial")

    print("  Running targeted retrieval for each briefing field...")
    context_text = _gather_targeted_context(arxiv_id, paper.title, paper.abstract)

    source_quality = "full_text" if parse_status == "ok" else "abstract_only"
    prompt = BRIEFING_PROMPT.format(
        title=paper.title,
        authors=", ".join(paper.authors),
        arxiv_id=arxiv_id,
        published=paper.published,
        link=f"https://arxiv.org/abs/{arxiv_id}",
        abstract=paper.abstract,
        context=context_text,
        source_quality=source_quality,
    )

    print("  Generating structured briefing...")

    try:
        result = llm_client.complete_json(prompt, max_tokens=3000)

        briefing = Briefing(
            title=result.get("title", paper.title),
            authors=result.get("authors", paper.authors),
            arxiv_id=result.get("arxiv_id", arxiv_id),
            published=result.get("published", paper.published),
            link=result.get("link", f"https://arxiv.org/abs/{arxiv_id}"),
            summary=result.get("summary", ""),
            problem_statement=result.get("problem_statement", ""),
            method=result.get("method", []),
            key_results=result.get("key_results", []),
            limitations=result.get("limitations", ["No limitations explicitly discussed by the authors."]),
            follow_up_questions=result.get("follow_up_questions", []),
            source_quality=source_quality,
        )

        if not briefing.limitations:
            briefing.limitations = ["No limitations explicitly discussed by the authors."]

        state["briefing"] = briefing
        print("  ✓ Briefing generated successfully")

    except Exception as e:
        state["errors"] = state.get("errors", []) + [f"Summarization error: {e}"]
        print(f"  ✗ Summarization failed: {e}")

    state["_last_branch"] = "default"
    return state


# 8. QA_LOOP — interactive question answering

def qa_loop(state: AgentState) -> AgentState:
    """
    Interactive QA loop. Asks the user for questions, retrieves relevant
    chunks, gates on confidence, and generates grounded answers.

    Sets _last_branch to "continue" to loop, "exit" to stop.
    """
    paper = state["selected_paper"]
    arxiv_id = paper.arxiv_id
    history = state.get("conversation_history", [])

    print("\n" + "═" * 60)
    print("  📝 QA Mode — ask questions about the paper")
    print("  Type 'quit' or 'exit' to stop")
    print("═" * 60)

    while True:
        try:
            question = input("\n  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        retrieval_query = f"{paper.title} {question}"
        hits = query_similar(query=retrieval_query, arxiv_id=arxiv_id, top_k=5)

        if not hits:
            print(f"\n  {QA_DECLINE_MSG}")
            history.append(QAExchange(
                question=question,
                answer=QA_DECLINE_MSG,
                grounded=False,
            ))
            continue

        top_sim = max(h["similarity"] for h in hits)

        if top_sim < QA_CONFIDENCE_THRESHOLD:
            print(f"\n  ⚠ Low confidence (similarity={top_sim:.3f})")
            print(f"  {QA_DECLINE_MSG}")
            history.append(QAExchange(
                question=question,
                answer=QA_DECLINE_MSG,
                grounded=False,
            ))
            continue

        chunk_context = "\n\n".join(
            f"[{h['section']} | chunk {h['chunk_id']}]\n{h['text']}"
            for h in hits
        )
        cited_chunks = [f"{h['section']}:{h['chunk_id']}" for h in hits[:3]]

        prompt = QA_PROMPT.format(
            question=question,
            context=chunk_context,
            paper_title=paper.title,
        )

        try:
            answer = llm_client.complete(prompt, max_tokens=1024).strip()

            if not answer:
                answer = llm_client.complete(prompt, max_tokens=1024).strip()

            if not answer:
                answer = "The model returned an empty response. Try rephrasing the question with more specific terms."
                print(f"\n  ⚠ Empty response from LLM, showing fallback message.")
                history.append(QAExchange(
                    question=question,
                    answer=answer,
                    grounded=False,
                ))
                continue

            print(f"\n  Answer: {answer}")
            print(f"  Sources: {', '.join(cited_chunks)}")

            history.append(QAExchange(
                question=question,
                answer=answer,
                grounded=True,
                cited_chunks=cited_chunks,
            ))
        except Exception as e:
            err_msg = f"LLM error during QA: {e}"
            print(f"\n  ✗ {err_msg}")
            state["errors"] = state.get("errors", []) + [err_msg]

    state["conversation_history"] = history
    state["_last_branch"] = "exit"
    return state
