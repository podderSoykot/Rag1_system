import os
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from config.settings import EMB_MODEL_NAME, HF_HUB_OFFLINE
from debug_log import debug_log

# Cache for embedding model (singleton pattern)
_embedding_model_cache = {}


def _hf_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_cached_on_disk(model_name: str) -> bool:
    slug = "models--" + model_name.replace("/", "--")
    snapshots = _hf_cache_dir() / slug / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def get_embedding_model(model_name: str = EMB_MODEL_NAME):
    """Get or create cached embedding model instance."""
    if model_name not in _embedding_model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cached = _model_cached_on_disk(model_name)
        local_only = HF_HUB_OFFLINE or cached

        # #region agent log
        debug_log(
            "embedder.py:get_embedding_model",
            "load start",
            {"model": model_name, "cached_on_disk": cached, "local_only": local_only, "device": device},
            "H6",
        )
        # #endregion

        print(f"[Loading] Embedding model: {model_name}")
        if device == "cuda":
            print("  Using GPU for faster embeddings")
        else:
            print("  Using CPU (slower, but will work)")
        if local_only:
            print("  Using local cache (skipping HuggingFace network checks)")

        load_start = time.time()
        _embedding_model_cache[model_name] = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=local_only,
        )
        load_s = round(time.time() - load_start, 2)

        # #region agent log
        debug_log(
            "embedder.py:get_embedding_model",
            "load done",
            {"model": model_name, "load_s": load_s, "local_only": local_only},
            "H6",
        )
        # #endregion

        print(f"[OK] Embedding model loaded and cached on {device} ({load_s}s)")
    return _embedding_model_cache[model_name]
