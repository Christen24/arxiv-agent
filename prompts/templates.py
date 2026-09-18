"""
Prompt templates for each LLM call in the agent.

One template per purpose. All use str.format() — no f-strings,
so they're safe to read at import time.
"""

# Briefing generation — structured JSON output

BRIEFING_PROMPT = """\
You are a research analyst. Given the following paper metadata and retrieved text chunks,
produce a structured briefing as a JSON object.

## Paper metadata
- Title: {title}
- Authors: {authors}
- arXiv ID: {arxiv_id}
- Published: {published}
- Link: {link}
- Source quality: {source_quality}

## Abstract
{abstract}

## Retrieved context (from the paper)
{context}

## Output format
Return ONLY a valid JSON object with these exact keys:
{{
    "title": "<paper title>",
    "authors": ["<author1>", "<author2>"],
    "arxiv_id": "<arXiv ID>",
    "published": "<date>",
    "link": "<URL>",
    "summary": "<1 paragraph plain-English summary of the entire paper>",
    "problem_statement": "<what problem does this paper address?>",
    "method": ["<step 1>", "<step 2>", "..."],
    "key_results": ["<result 1>", "<result 2>", "..."],
    "limitations": ["<limitation 1>", "<limitation 2>", "..."],
    "follow_up_questions": ["<question a reader might ask>", "..."]
}}

IMPORTANT RULES:
1. "limitations" must NEVER be empty. If the paper does not explicitly discuss limitations,
   infer reasonable ones conservatively and prefix each with "Not explicitly discussed: ".
2. "method" should be concrete steps, not vague descriptions.
3. "summary" should be understandable by someone outside the field.
4. Ground every claim in the retrieved context. Do not hallucinate details not present.
"""

# QA — grounded answer from retrieved chunks

QA_PROMPT = """\
You are a research assistant answering questions about the paper: "{paper_title}"

## Retrieved context (from the paper)
{context}

## Question
{question}

## Instructions
- Answer ONLY using the information in the retrieved context above.
- If the context does not contain enough information to answer the question,
  say so explicitly — do NOT guess or hallucinate.
- Be concise and precise.
- Reference which section the information came from when possible.
"""

# QA decline message (used when confidence is below threshold)

QA_DECLINE_MSG = (
    "I could not find information in the paper that is relevant enough "
    "to answer this question confidently. The paper may not cover this topic, "
    "or my retrieval may have missed it."
)

# Query reformulation — broaden a failing search

REFORMULATE_PROMPT = """\
The following search query returned zero results on arXiv:
  "{original_query}"

This is reformulation attempt {attempt}.

Produce a single, broader search query that is more likely to find relevant
papers on arXiv. The query should still target the same research area but
use more general or alternative terminology.

Return ONLY the new query string, nothing else.
"""

# Targeted retrieval queries — one per briefing field

TARGETED_QUERIES: dict[str, str] = {
    "problem_statement": "problem motivation objective goal introduction why",
    "method": "method approach methodology architecture model design proposed technique algorithm",
    "key_results": "results experiments evaluation performance accuracy metrics benchmark comparison",
    "limitations": "limitations drawbacks weaknesses future work discussion caveats shortcomings",
    "follow_up_questions": "conclusion future work open questions implications applications extensions",
}

