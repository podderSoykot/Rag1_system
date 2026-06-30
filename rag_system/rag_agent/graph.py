import hashlib
import time
from typing import Literal

from langgraph.graph import END, START, StateGraph

from rag_agent.state import RAGState
from config.settings import CACHE_ENABLED, CACHE_SIZE, EMB_MODEL_NAME, VECTOR_DB_DIR
from retrieval.retriever import Retriever
from synthesis.generator import generate_answer
from synthesis.prompt_builder import build_prompt

_retriever_instance = None
_result_cache: dict[str, str] = {}
_compiled_graph = None


def get_retriever() -> Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever(str(VECTOR_DB_DIR), EMB_MODEL_NAME)
    return _retriever_instance


def reset_retriever() -> None:
    global _retriever_instance
    _retriever_instance = None


def _get_cache_key(query: str, top_k: int) -> str:
    key_string = f"{query.lower().strip()}:{top_k}"
    return hashlib.md5(key_string.encode()).hexdigest()


def _append_step(state: RAGState, step: str) -> list[str]:
    steps = list(state.get("steps") or [])
    steps.append(step)
    return steps


def check_cache(state: RAGState) -> RAGState:
    steps = _append_step(state, "check_cache")
    if not state.get("use_cache", CACHE_ENABLED):
        return {"cache_hit": False, "steps": steps}

    cache_key = _get_cache_key(state["query"], state["top_k"])
    if cache_key in _result_cache:
        return {
            "answer": _result_cache[cache_key],
            "cache_hit": True,
            "steps": steps,
        }
    return {"cache_hit": False, "steps": steps}


def retrieve(state: RAGState) -> RAGState:
    start = time.time()
    docs = get_retriever().search(
        state["query"],
        top_k=state["top_k"],
        show_timing=state.get("show_timing", True),
    )
    timing = dict(state.get("timing") or {})
    timing["retrieval_ms"] = round((time.time() - start) * 1000, 1)
    return {
        "docs": docs,
        "timing": timing,
        "steps": _append_step(state, "retrieve"),
    }


def build_prompt_node(state: RAGState) -> RAGState:
    start = time.time()
    prompt = build_prompt(state["query"], state.get("docs") or [])
    timing = dict(state.get("timing") or {})
    timing["prompt_ms"] = round((time.time() - start) * 1000, 1)
    return {
        "prompt": prompt,
        "timing": timing,
        "steps": _append_step(state, "build_prompt"),
    }


def generate_node(state: RAGState) -> RAGState:
    start = time.time()
    answer = generate_answer(state["prompt"], query=state["query"])
    timing = dict(state.get("timing") or {})
    timing["generation_ms"] = round((time.time() - start) * 1000, 1)
    timing["total_ms"] = round(
        timing.get("retrieval_ms", 0)
        + timing.get("prompt_ms", 0)
        + timing.get("generation_ms", 0),
        1,
    )

    if state.get("use_cache", CACHE_ENABLED):
        cache_key = _get_cache_key(state["query"], state["top_k"])
        if len(_result_cache) >= CACHE_SIZE:
            oldest_key = next(iter(_result_cache))
            del _result_cache[oldest_key]
        _result_cache[cache_key] = answer

    if state.get("show_timing", True):
        print("\n[Timing]")
        print(f"  Retrieval:  {timing.get('retrieval_ms', 0):.0f}ms")
        print(f"  Prompt:     {timing.get('prompt_ms', 0):.0f}ms")
        print(f"  Generation: {timing.get('generation_ms', 0):.0f}ms")
        print(f"  Total:      {timing.get('total_ms', 0):.0f}ms")

    return {
        "answer": answer,
        "timing": timing,
        "steps": _append_step(state, "generate"),
    }


def route_after_cache(state: RAGState) -> Literal["end", "retrieve"]:
    if state.get("cache_hit"):
        if state.get("show_timing", True):
            print("[Cache Hit] Returning cached result")
        return "end"
    return "retrieve"


def build_rag_graph():
    workflow = StateGraph(RAGState)
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "check_cache")
    workflow.add_conditional_edges(
        "check_cache",
        route_after_cache,
        {"end": END, "retrieve": "retrieve"},
    )
    workflow.add_edge("retrieve", "build_prompt")
    workflow.add_edge("build_prompt", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


def get_rag_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_rag_graph()
    return _compiled_graph


def run_rag_graph(
    query: str,
    top_k: int = 3,
    use_cache: bool | None = None,
    show_timing: bool = True,
) -> RAGState:
    graph = get_rag_graph()
    initial_state: RAGState = {
        "query": query,
        "top_k": top_k,
        "use_cache": CACHE_ENABLED if use_cache is None else use_cache,
        "show_timing": show_timing,
        "steps": [],
    }
    return graph.invoke(initial_state)
