"""Document embeddings.

``tfidf``: TF-IDF + truncated SVD (LSA). No model download, deterministic; a no-neural reference.

``sbert``: any sentence-transformers model, chosen with ``sbert_model``. This is how both the
2022-style arm (``sentence-transformers/allenai-specter``) and the modern arms
(``BAAI/bge-small-en-v1.5``, ``BAAI/bge-m3``, ...) are run. Needs ``pip install 'alignmap[sbert]'``.

All embedders return L2-normalised rows. Embeddings can be cached on disk (``cache_dir``), keyed by
the exact documents and parameters, so comparing several clustering choices re-uses one embedding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def embed_tfidf_lsa(
    docs: list[str],
    *,
    dims: int = 128,
    seed: int = 0,
    min_df: int = 2,
    max_df: float = 0.5,
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 200_000,
) -> np.ndarray:
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        max_features=max_features,
    )
    X = vec.fit_transform(docs)
    if X.shape[1] < 3:
        raise ValueError(
            f"TF-IDF vocabulary has only {X.shape[1]} terms; the corpus is too small or too uniform."
        )
    n_components = max(2, min(dims, X.shape[0] - 1, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    return normalize(svd.fit_transform(X))


def embed_sbert(
    docs: list[str],
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
    max_seq_length: int | None = None,
    doc_prefix: str = "",
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'sbert' embedder needs sentence-transformers: pip install 'alignmap[sbert]'"
        ) from exc
    model = SentenceTransformer(model_name)
    if max_seq_length:
        model.max_seq_length = int(max_seq_length)
    texts = [doc_prefix + d for d in docs] if doc_prefix else docs
    return np.asarray(
        model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    )


def _cache_key(docs: list[str], params: dict) -> str:
    h = hashlib.sha1(json.dumps(params, sort_keys=True).encode())
    for d in docs:
        h.update(d.encode("utf-8", "ignore"))
        h.update(b"\0")
    return h.hexdigest()[:20]


def embed(
    docs: list[str],
    method: str = "tfidf",
    *,
    dims: int = 128,
    seed: int = 0,
    sbert_model: str = "BAAI/bge-small-en-v1.5",
    max_seq_length: int | None = None,
    doc_prefix: str = "",
    cache_dir=None,
) -> np.ndarray:
    if method not in ("tfidf", "sbert"):
        raise ValueError(f"Unknown embedder {method!r} (choose 'tfidf' or 'sbert')")
    params = (
        {"method": "tfidf", "dims": dims, "seed": seed}
        if method == "tfidf"
        else {"method": "sbert", "model": sbert_model, "max_seq_length": max_seq_length, "prefix": doc_prefix}
    )
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"emb_{_cache_key(docs, params)}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    if method == "tfidf":
        Z = embed_tfidf_lsa(docs, dims=dims, seed=seed)
    else:
        Z = embed_sbert(docs, model_name=sbert_model, max_seq_length=max_seq_length, doc_prefix=doc_prefix)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, Z)
    return Z
