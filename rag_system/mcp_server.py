"""MCP server exposing RAG system tools for Cursor and other MCP clients."""

import sys
import time

from mcp.server.fastmcp import FastMCP

TOOLS = [
    ("rag_query", "Answer a question (retrieve + generate)"),
    ("rag_retrieve", "Retrieve source chunks only"),
    ("rag_status", "Check index health"),
    ("rag_ingest", "Rebuild indexes from PDFs"),
]


def _quick_index_status():
    from pathlib import Path

    from config.settings import VECTOR_DB_DIR

    index_dir = Path(VECTOR_DB_DIR)
    has_vectors = (index_dir / "faiss_index.bin").exists() or (
        index_dir / "embeddings.npy"
    ).exists()
    has_metadata = (index_dir / "vector_metadata.pkl").exists()
    has_tfidf = (index_dir / "tfidf_index.pkl").exists()
    if not has_vectors or not has_metadata or not has_tfidf:
        return "needs_ingestion", "Indexes not found"
    return "ready", "Indexes present"


def log_mcp_startup(startup_s: float) -> None:
    """Log startup banner to stderr (stdout is reserved for MCP stdio)."""
    lines = [
        "",
        "=" * 60,
        "RAG MCP Server",
        "=" * 60,
        f"  Startup:   {startup_s:.1f}s",
        "  Transport: stdio",
        "",
        "  Tools:",
    ]
    for name, desc in TOOLS:
        lines.append(f"    - {name}: {desc}")

    try:
        from config.settings import (
            OLLAMA_MODEL,
            OPENAI_API_KEY,
            OPENAI_MODEL,
            USE_OLLAMA,
            USE_OPENAI,
        )

        if USE_OPENAI and OPENAI_API_KEY:
            lines.append(f"\n  Generation: OpenAI ({OPENAI_MODEL})")
        elif USE_OLLAMA:
            lines.append(f"\n  Generation: Ollama ({OLLAMA_MODEL})")
    except Exception:
        pass

    status, reason = _quick_index_status()
    lines.append(f"  Index status: {status} ({reason})")
    lines.extend(
        [
            "",
            "  Ready — waiting for MCP client (Ctrl+C to stop)",
            "=" * 60,
            "",
        ]
    )
    print("\n".join(lines), file=sys.stderr, flush=True)

mcp = FastMCP(
    "RAG System",
    instructions=(
        "Retrieval-Augmented Generation over indexed PDF documents. "
        "Use rag_query to answer questions, rag_retrieve to inspect source chunks, "
        "rag_status to check index health, and rag_ingest to rebuild indexes."
    ),
)


@mcp.tool()
def rag_query(query: str, top_k: int = 3) -> str:
    """Answer a question using indexed documents (retrieve + generate)."""
    from rag_agent.graph import run_rag_graph

    result = run_rag_graph(query, top_k=top_k, show_timing=False)
    answer = result.get("answer", "")
    docs = result.get("docs") or []
    if docs:
        sources = "\n".join(f"- {doc[:200]}..." for doc in docs[:top_k])
        return f"{answer}\n\n---\nSources ({len(docs)} chunks):\n{sources}"
    return answer


@mcp.tool()
def rag_retrieve(query: str, top_k: int = 5) -> str:
    """Retrieve relevant document chunks without generating an answer."""
    from rag_agent.graph import get_retriever

    docs = get_retriever().search(query, top_k=top_k, show_timing=False)
    if not docs:
        return "No relevant chunks found."
    return "\n\n---\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(docs))


@mcp.tool()
def rag_status() -> str:
    """Check whether document indexes exist and if ingestion is needed."""
    from main import needs_ingestion

    needs_it, reason = needs_ingestion()
    status = "needs_ingestion" if needs_it else "ready"
    return f"status={status}\nreason={reason}"


@mcp.tool()
def rag_ingest(force: bool = False) -> str:
    """Run the ingestion pipeline (PDF extract, chunk, embed, index)."""
    from rag_agent.graph import reset_retriever
    from main import run_ingestion

    run_ingestion(force=force)
    reset_retriever()
    return f"Ingestion complete (force={force})"


if __name__ == "__main__":
    print("[Info] Loading RAG MCP server...", file=sys.stderr, flush=True)
    t0 = time.time()
    log_mcp_startup(time.time() - t0)
    mcp.run()
