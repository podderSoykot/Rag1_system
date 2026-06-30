"""FastAPI backend for the RAG frontend."""

import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import DATA_RAW, OPENAI_API_KEY, OPENAI_MODEL, USE_OLLAMA, USE_OPENAI, VECTOR_DB_DIR
from main import needs_ingestion, run_ingestion
from rag_agent.graph import get_retriever, reset_retriever, run_rag_graph

_ingestion_lock = threading.Lock()
_ingestion_state = {
    "running": False,
    "message": "Idle",
    "error": None,
    "completed_at": None,
}


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
        _set_ingestion_state(False, "Ingestion failed", str(exc))


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    needs_it, reason = needs_ingestion()
    app.state.index_ready = not needs_it
    app.state.index_reason = reason
    yield


app = FastAPI(title="RAG System API", version="1.0.0", lifespan=lifespan)

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
        return dict(_ingestion_state)


@app.post("/api/upload")
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    ingest: bool = Query(default=True),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    with _ingestion_lock:
        if _ingestion_state["running"]:
            raise HTTPException(status_code=409, detail="Ingestion already in progress")

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for upload in files:
        filename = _safe_filename(upload.filename or "document.pdf")
        dest = DATA_RAW / filename
        with dest.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
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
def query(body: QueryRequest):
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
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    docs = result.get("docs") or []
    return QueryResponse(
        answer=result.get("answer", ""),
        sources=docs,
        timing=result.get("timing") or {},
        steps=result.get("steps") or [],
        cache_hit=bool(result.get("cache_hit")),
    )


@app.post("/api/retrieve")
def retrieve(body: QueryRequest):
    if not _index_exists():
        raise HTTPException(status_code=503, detail="No searchable index yet.")

    try:
        docs = get_retriever().search(body.query, top_k=body.top_k, show_timing=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"sources": docs, "count": len(docs)}


@app.post("/api/ingest")
def ingest(body: IngestRequest, background_tasks: BackgroundTasks):
    with _ingestion_lock:
        if _ingestion_state["running"]:
            raise HTTPException(status_code=409, detail="Ingestion already in progress")

    background_tasks.add_task(_run_ingestion_job, body.force)
    return {"message": "Ingestion started", "force": body.force}
