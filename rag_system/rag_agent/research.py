"""Multi-query deep research over indexed documents."""

import hashlib
import time

from rag_agent.graph import get_retriever
from config.settings import RESEARCH_MAX_SUB_QUERIES, RESEARCH_MAX_TOKENS
from retrieval.query_expansion import expand_query
from synthesis.generator import generate_answer
from synthesis.research_prompt import RESEARCH_SYSTEM_INSTRUCTIONS, build_research_prompt

RESEARCH_ANGLES = (
    "Definition and overview of {topic}",
    "How {topic} works — mechanism and process",
    "Key concepts, examples, and applications of {topic}",
    "Important details, advantages, and limitations of {topic}",
)


def parse_research_topic(text: str) -> str:
    """Extract topic from /research command or natural phrasing."""
    topic = text.strip()
    if topic.lower().startswith("/research"):
        topic = topic[len("/research") :].strip()
        if topic.startswith(":"):
            topic = topic[1:].strip()

    lowered = topic.lower()
    for prefix in ("research on ", "research about ", "do research on ", "do research about "):
        if lowered.startswith(prefix):
            topic = topic[len(prefix) :].strip()
            break

    if lowered.startswith("research ") and not lowered.startswith("research on"):
        rest = topic[len("research") :].strip()
        if rest and not rest.lower().startswith(("on ", "about ")):
            topic = rest

    return topic.strip(" ?.")


def generate_research_queries(topic: str, max_queries: int | None = None) -> list[str]:
    if max_queries is None:
        max_queries = RESEARCH_MAX_SUB_QUERIES
    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            queries.append(q.strip())

    add(topic)
    for q in expand_query(topic, num_expansions=4):
        add(q)
    for template in RESEARCH_ANGLES:
        add(template.format(topic=topic))

    return queries[:max_queries]


def _dedupe_docs(docs: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for doc in docs:
        key = hashlib.md5(doc[:300].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def gather_research_docs(topic: str, top_k_per_query: int = 5) -> tuple[list[str], list[str]]:
    retriever = get_retriever()
    sub_queries = generate_research_queries(topic)
    all_docs: list[str] = []

    for sub_q in sub_queries:
        hits = retriever.search(sub_q, top_k=top_k_per_query, show_timing=False)
        all_docs.extend(hits)

    return _dedupe_docs(all_docs), sub_queries


def run_research(
    topic_or_query: str,
    top_k_per_query: int = 5,
    max_docs: int = 15,
    show_timing: bool = False,
) -> dict:
    start = time.time()
    topic = parse_research_topic(topic_or_query)
    if not topic:
        return {
            "topic": "",
            "answer": "Please provide a research topic (e.g. `/research backpropagation`).",
            "docs": [],
            "sub_queries": [],
            "steps": ["parse_topic"],
            "timing": {"total_ms": 0},
        }

    retrieve_start = time.time()
    docs, sub_queries = gather_research_docs(topic, top_k_per_query=top_k_per_query)
    docs = docs[:max_docs]
    retrieve_ms = round((time.time() - retrieve_start) * 1000, 1)

    if not docs:
        return {
            "topic": topic,
            "answer": f"No relevant excerpts found in the index for: **{topic}**",
            "docs": [],
            "sub_queries": sub_queries,
            "steps": ["parse_topic", "multi_retrieve"],
            "timing": {"retrieval_ms": retrieve_ms, "total_ms": round((time.time() - start) * 1000, 1)},
        }

    prompt = build_research_prompt(topic, docs, sub_queries)

    gen_start = time.time()
    if show_timing:
        print(f"[Research] Topic: {topic}")
        print(f"[Research] Sub-queries: {len(sub_queries)}, unique chunks: {len(docs)}")

    from config.settings import USE_OPENAI, OPENAI_API_KEY

    if USE_OPENAI and OPENAI_API_KEY:
        from synthesis.openai_generator import generate_answer as openai_generate
        from synthesis.postprocessor import clean_answer

        raw = openai_generate(
            prompt,
            query=topic,
            system_instructions=RESEARCH_SYSTEM_INSTRUCTIONS,
            max_tokens=RESEARCH_MAX_TOKENS,
        )
        answer = clean_answer(raw)
    else:
        answer = generate_answer(prompt, query=topic)

    gen_ms = round((time.time() - gen_start) * 1000, 1)
    total_ms = round((time.time() - start) * 1000, 1)

    if show_timing:
        print(f"[Research] Retrieval: {retrieve_ms}ms, Generation: {gen_ms}ms, Total: {total_ms}ms")

    return {
        "topic": topic,
        "answer": answer,
        "docs": docs,
        "sub_queries": sub_queries,
        "steps": ["parse_topic", "multi_retrieve", "build_research_prompt", "generate_report"],
        "timing": {
            "retrieval_ms": retrieve_ms,
            "generation_ms": gen_ms,
            "total_ms": total_ms,
        },
    }
