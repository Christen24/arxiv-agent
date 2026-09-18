"""
arXiv Paper Digest & QA Agent — CLI entrypoint.

Usage:
    python main.py                     # interactive prompt
    python main.py "attention is all you need"
    python main.py 2301.12345
    python main.py https://arxiv.org/abs/2301.12345
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from graph.state import Graph
from graph.nodes import (
    route,
    retrieve,
    reformulate,
    rank,
    fetch_parse,
    chunk_embed,
    summarize,
    qa_loop,
)
from models.schemas import AgentState


def build_graph() -> Graph:
    """Wire up all nodes and edges into the agent graph."""
    g = Graph()

    g.add_node("route",         route,         {"id": "fetch_parse", "topic": "retrieve"})
    g.add_node("retrieve",      retrieve,      {"zero": "reformulate", "one": "fetch_parse", "many": "rank"})
    g.add_node("reformulate",   reformulate,   {"retry": "retrieve", "give_up": "END"})
    g.add_node("rank",          rank,          {"ranked": "fetch_parse", "zero": "END"})
    g.add_node("fetch_parse",   fetch_parse,   {"ok": "chunk_embed", "partial": "chunk_embed", "failed": "END"})
    g.add_node("chunk_embed",   chunk_embed,   {"default": "summarize"})
    g.add_node("summarize",     summarize,     {"default": "END"})

    g.set_entry("route")
    return g


def print_briefing(state: AgentState) -> None:
    """Pretty-print the structured briefing."""
    briefing = state.get("briefing")
    if not briefing:
        print("\n  ⚠ No briefing was generated.")
        errors = state.get("errors", [])
        if errors:
            print("  Errors:")
            for e in errors:
                print(f"    - {e}")
        return

    print("\n" + "═" * 60)
    print("  📋 STRUCTURED BRIEFING")
    print("═" * 60)
    print(f"\n  Title:     {briefing.title}")
    print(f"  Authors:   {', '.join(briefing.authors)}")
    print(f"  arXiv ID:  {briefing.arxiv_id}")
    print(f"  Published: {briefing.published}")
    print(f"  Link:      {briefing.link}")
    print(f"  Source:    {briefing.source_quality}")

    print(f"\n  Summary:\n    {briefing.summary}")

    print(f"\n  Problem Statement:\n    {briefing.problem_statement}")

    print("\n  Method:")
    for step in briefing.method:
        print(f"    • {step}")

    print("\n  Key Results:")
    for r in briefing.key_results:
        print(f"    • {r}")

    print("\n  Limitations:")
    for l in briefing.limitations:
        print(f"    • {l}")

    print("\n  Follow-up Questions:")
    for q in briefing.follow_up_questions:
        print(f"    ? {q}")
    print()


def print_qa_history(state: AgentState) -> None:
    """Print the QA conversation history."""
    history = state.get("conversation_history", [])
    if not history:
        return

    print("\n" + "═" * 60)
    print("  💬 QA CONVERSATION HISTORY")
    print("═" * 60)

    for i, qa in enumerate(history, 1):
        print(f"\n  Q{i}: {qa.question}")
        print(f"  A{i}: {qa.answer}")
        if qa.grounded and qa.cited_chunks:
            print(f"  Sources: {', '.join(qa.cited_chunks)}")
        elif not qa.grounded:
            print("  [Not grounded — declined or low confidence]")
    print()


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("\n  arXiv Paper Digest & QA Agent")
        print("  " + "─" * 40)
        print("  Enter a topic, arXiv ID, or arXiv URL:")
        query = input("  > ").strip()

    if not query:
        print("  No input provided. Exiting.")
        sys.exit(1)

    state: AgentState = {
        "raw_query": query,
        "active_query": query,
        "mode": "topic",
        "candidates": [],
        "selected_paper": None,
        "reformulation_attempts": 0,
        "parsed_sections": {},
        "parse_status": "ok",
        "vector_store_ref": "",
        "briefing": None,
        "conversation_history": [],
        "errors": [],
        "_last_branch": "",
    }

    graph = build_graph()
    state = graph.run(state)

    print_briefing(state)

    if state.get("briefing"):
        state = qa_loop(state)

    print_qa_history(state)

    errors = state.get("errors", [])
    if errors:
        print("\n  ⚠ Errors encountered during this run:")
        for e in errors:
            print(f"    - {e}")


if __name__ == "__main__":
    main()
