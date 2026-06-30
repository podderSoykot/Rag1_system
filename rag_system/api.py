"""FastAPI backend for the RAG frontend."""

import logging
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api_security import api_error, client_key, rate_limiter, save_upload_limited
from debug_log import debug_log
from config.settings import (
    API_DOCS_ENABLED,
    DATA_RAW,
    MAX_UPLOAD_FILES,
    MAX_UPLOAD_SIZE_MB,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    RATE_LIMIT_INGEST_PER_MIN,
    RATE_LIMIT_QUERY_PER_MIN,
    RATE_LIMIT_RESEARCH_PER_MIN,
    RATE_LIMIT_UPLOAD_PER_MIN,
    USE_OLLAMA,
    USE_OPENAI,
    VECTOR_DB_DIR,
)
from main import needs_ingestion, run_ingestion
from rag_agent.graph import get_retriever, reset_retriever, run_rag_graph

logging.basicConfig(level=logging.INFO)

_ingestion_lock = threading.Lock()
_ingestion_state = {
    "running": False,
    "message": "Idle",
    "error": None,
    "completed_at": None,
}
_MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _set_ingestion_state(running: bool, message: str, error: str | None = None):
    with _ingestion_lock:
        _ingestion_state["running"] = running
        _ingestion_state["message"] = message
        _ingestion_state["error"] = error
        if not running and error is None:
            _ingestion_state["completed_at"] = __import__("time").time()


def _run_ingestion_job(force: bool = False):
    _set_ingestion_state(True, "Ingestion in progress…")
    try:
        run_ingestion(force=force)
        reset_retriever()
        _set_ingestion_state(False, "Ingestion complete")
    except Exception as exc:
        logging.exception("Ingestion job failed")
        _set_ingestion_state(False, "Ingestion failed", "Ingestion failed")


def _index_exists() -> bool:
    """True if a searchable index is on disk (may be stale while re-indexing)."""
    index_dir = Path(VECTOR_DB_DIR)
    has_vectors = (index_dir / "faiss_index.bin").exists() or (
        index_dir / "embeddings.npy"
    ).exists()
    has_metadata = (index_dir / "vector_metadata.pkl").exists()
    has_tfidf = (index_dir / "tfidf_index.pkl").exists()
    return has_vectors and has_metadata and has_tfidf


def _ingestion_running() -> bool:
    with _ingestion_lock:
        return _ingestion_state["running"]


def _safe_filename(name: str) -> str:
    base = Path(name).name
    if not base.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"Only PDF files are allowed: {name}")
    if base.startswith(".") or ".." in base:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {name}")
    return base


def _rate_limit(request: Request, bucket: str, limit: int) -> None:
    rate_limiter.check(bucket, client_key(request.client.host if request.client else None), limit)


@asynccontextmanager
async def lifespan(app: FastAPI):
    needs_it, reason = needs_ingestion()
    app.state.index_ready = not needs_it
    app.state.index_reason = reason
    yield


app = FastAPI(
    title="RAG System API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if API_DOCS_ENABLED else None,
    redoc_url="/redoc" if API_DOCS_ENABLED else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    timing: dict
    steps: list[str]
    cache_hit: bool = False


class StatusResponse(BaseModel):
    status: str
    reason: str
    generation_backend: str
    chat_available: bool
    ingestion_running: bool


class IngestRequest(BaseModel):
    force: bool = False


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=2000)
    top_k_per_query: int = Field(default=4, ge=1, le=8)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/status", response_model=StatusResponse)
def status():
    needs_it, reason = needs_ingestion()
    has_index = _index_exists()
    running = _ingestion_running()

    if running:
        display_status = "indexing"
    elif not has_index:
        display_status = "needs_ingestion"
    elif needs_it:
        display_status = "ready"
    else:
        display_status = "ready"

    if USE_OPENAI and OPENAI_API_KEY:
        backend = f"OpenAI ({OPENAI_MODEL})"
    elif USE_OLLAMA:
        backend = "Ollama"
    else:
        backend = "Local LLM"
    return StatusResponse(
        status=display_status,
        reason=reason if needs_it or running else "All files up to date",
        generation_backend=backend,
        chat_available=has_index,
        ingestion_running=running,
    )


@app.get("/api/documents")
def list_documents():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    docs = []
    for path in sorted(DATA_RAW.glob("*.pdf")):
        stat = path.stat()
        docs.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
            }
        )
    return {"documents": docs, "count": len(docs)}


@app.get("/api/ingest/status")
def ingest_status():
    with _ingestion_lock:
        state = dict(_ingestion_state)
    if state.get("error") and state["error"] != "Ingestion failed":
        state["error"] = "Ingestion failed"
    return state


@app.post("/api/upload")
async def upload_pdfs(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    ingest: bool = Query(default=True),
):
    _rate_limit(request, "upload", RATE_LIMIT_UPLOAD_PER_MIN)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {MAX_UPLOAD_FILES} per upload.",
        )

    with _ingestion_lock:
        if _ingestion_state["running"]:
            raise HTTPException(status_code=409, detail="Ingestion already in progress")

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for upload in files:
        filename = _safe_filename(upload.filename or "document.pdf")
        dest = DATA_RAW / filename
        await save_upload_limited(upload, dest, _MAX_UPLOAD_BYTES)
        saved.append(filename)

    response = {
        "uploaded": saved,
        "count": len(saved),
        "ingestion_started": False,
    }

    if ingest:
        background_tasks.add_task(_run_ingestion_job, False)
        response["ingestion_started"] = True
        response["message"] = f"Uploaded {len(saved)} file(s). Indexing started."

    return response


@app.post("/api/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    _rate_limit(request, "query", RATE_LIMIT_QUERY_PER_MIN)
    debug_log("api.py:query", "start", {"qlen": len(body.query)}, "H3")

    if not _index_exists():
        raise HTTPException(
            status_code=503,
            detail="No searchable index yet. Upload PDFs and run ingestion first.",
        )

    try:
        result = run_rag_graph(
            body.query,
            top_k=body.top_k,
            show_timing=False,
        )
    except Exception as exc:
        raise api_error(exc, context="query") from exc

    docs = result.get("docs") or []
    debug_log("api.py:query", "ok", {"sources": len(docs)}, "H3")
    return QueryResponse(
        answer=result.get("answer", ""),
        sources=docs,
        timing=result.get("timing") or {},
        steps=result.get("steps") or [],
        cache_hit=bool(result.get("cache_hit")),
    )


@app.post("/api/retrieve")
def retrieve(body: QueryRequest, request: Request):
    _rate_limit(request, "query", RATE_LIMIT_QUERY_PER_MIN)

    if not _index_exists():
        raise HTTPException(status_code=503, detail="No searchable index yet.")

    try:
        docs = get_retriever().search(body.query, top_k=body.top_k, show_timing=False)
    except Exception as exc:
        raise api_error(exc, context="retrieve") from exc

    return {"sources": docs, "count": len(docs)}


@app.post("/api/research")
def research(body: ResearchRequest, request: Request):
    _rate_limit(request, "research", RATE_LIMIT_RESEARCH_PER_MIN)

    if not _index_exists():
        raise HTTPException(status_code=503, detail="No searchable index yet.")

    try:
        from rag_agent.research import run_research

        result = run_research(
            body.topic,
            top_k_per_query=body.top_k_per_query,
            show_timing=False,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise api_error(exc, context="research") from exc

    return {
        "topic": result.get("topic", body.topic),
        "answer": result.get("answer", ""),
        "sources": result.get("docs") or [],
        "sub_queries": result.get("sub_queries") or [],
        "timing": result.get("timing") or {},
        "steps": result.get("steps") or [],
    }


@app.post("/api/ingest")
def ingest(body: IngestRequest, request: Request, background_tasks: BackgroundTasks):
    _rate_limit(request, "ingest", RATE_LIMIT_INGEST_PER_MIN)

    with _ingestion_lock:
        if _ingestion_state["running"]:
            raise HTTPException(status_code=409, detail="Ingestion already in progress")

    background_tasks.add_task(_run_ingestion_job, body.force)
    return {"message": "Ingestion started", "force": body.force}
