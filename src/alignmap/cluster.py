"""Dimensionality reduction, clustering, quality diagnostics and keyword labels.

Two clustering families are supported so they can be compared on the same corpus:

* ``kmeans``: a fixed number of clusters k (what the 2022 analysis used, with k = 5);
* ``hdbscan``: density-based, finds the number of clusters itself and marks documents that
  belong to no dense region as noise (label -1) instead of forcing them into a subfield.

Either can run on the raw embeddings or after a UMAP reduction (``reducer="umap"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score


@dataclass(frozen=True)
class ClusterSpec:
    reducer: str = "none"  # "none" | "umap"
    umap_components: int = 5
    umap_neighbors: int = 30
    clusterer: str = "kmeans"  # "kmeans" | "hdbscan"
    k: int | None = 20
    min_cluster_size: float = 0.01  # HDBSCAN; values < 1 are a fraction of the documents

    def describe(self) -> str:
        red = (
            f"UMAP({self.umap_components}d, n_neighbors={self.umap_neighbors})"
            if self.reducer == "umap"
            else "no reduction"
        )
        if self.clusterer == "kmeans":
            return f"{red} → k-means (k={self.k if self.k is not None else 'auto'})"
        size = f"{self.min_cluster_size:.1%} of documents" if self.min_cluster_size < 1 else int(self.min_cluster_size)
        return f"{red} → HDBSCAN (min_cluster_size={size})"


def reduce_umap(
    Z: np.ndarray, *, n_components: int = 5, n_neighbors: int = 30, min_dist: float = 0.0, seed: int = 0
) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("UMAP is not installed: pip install 'alignmap[umap]'") from exc
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=max(2, min(n_neighbors, len(Z) - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    ).fit_transform(Z)


def reduce(Z: np.ndarray, spec: ClusterSpec, seed: int = 0) -> np.ndarray:
    if spec.reducer == "none":
        return Z
    if spec.reducer == "umap":
        return reduce_umap(Z, n_components=spec.umap_components, n_neighbors=spec.umap_neighbors, seed=seed)
    raise ValueError(f"Unknown reducer {spec.reducer!r} (choose 'none' or 'umap')")


def fit_kmeans(Z: np.ndarray, k: int, seed: int = 0, n_init: int = 10) -> KMeans:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z)


def resolve_min_cluster_size(value: float, n: int) -> int:
    return max(5, int(round(value * n))) if value < 1 else int(value)


def fit_hdbscan(X: np.ndarray, min_cluster_size: int) -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    return HDBSCAN(min_cluster_size=int(min_cluster_size)).fit(X).labels_


def cluster_reduced(X: np.ndarray, spec: ClusterSpec, seed: int = 0, *, quick: bool = False) -> np.ndarray:
    """Cluster already-reduced vectors. Labels are 0..K-1, with -1 for HDBSCAN noise."""
    if spec.clusterer == "kmeans":
        if spec.k is None:
            raise ValueError("k must be resolved before clustering (use select_k)")
        return fit_kmeans(X, spec.k, seed=seed, n_init=3 if quick else 10).labels_
    if spec.clusterer == "hdbscan":
        return fit_hdbscan(X, resolve_min_cluster_size(spec.min_cluster_size, len(X)))
    raise ValueError(f"Unknown clusterer {spec.clusterer!r} (choose 'kmeans' or 'hdbscan')")


def run_clustering(Z: np.ndarray, spec: ClusterSpec, seed: int = 0, *, quick: bool = False) -> np.ndarray:
    return cluster_reduced(reduce(Z, spec, seed), spec, seed, quick=quick)


def sampled_silhouette(Z: np.ndarray, labels: np.ndarray, seed: int = 0, sample: int = 4000) -> float:
    """Silhouette over clustered (non-noise) documents, on a random sample for speed."""
    labels = np.asarray(labels)
    mask = labels >= 0
    if mask.sum() < 3 or len(np.unique(labels[mask])) < 2:
        return float("nan")
    Zm, lm = Z[mask], labels[mask]
    return float(
        silhouette_score(Zm, lm, sample_size=None if len(Zm) <= sample else sample, random_state=seed)
    )


def select_k(
    X: np.ndarray, k_values: Iterable[int], seed: int = 0
) -> tuple[int, dict[int, float]]:
    """Pick the k with the best (sampled) silhouette score; also returns all scores."""
    scores: dict[int, float] = {}
    for k in k_values:
        if not 2 <= k < len(X):
            continue
        km = fit_kmeans(X, k, seed=seed, n_init=3)
        scores[int(k)] = sampled_silhouette(X, km.labels_, seed=seed)
    if not scores:
        raise ValueError("No valid k in the requested range for this corpus size")
    best = max(scores, key=scores.get)
    return best, scores


def stability(
    Z: np.ndarray,
    cluster_fn: Callable[[np.ndarray, int], np.ndarray],
    *,
    n_rounds: int = 5,
    frac: float = 0.8,
    seed: int = 0,
) -> float:
    """Mean adjusted Rand index between clusterings fitted on different random 80% subsamples.

    ``cluster_fn(Z_subsample, seed)`` runs the *whole* reduce-and-cluster procedure, so the score
    reflects the full pipeline. Two fits are compared on the documents both saw; noise points
    (label -1) in either fit are left out. Close to 1 = the structure does not depend on which
    documents were sampled.
    """
    rng = np.random.default_rng(seed)
    n = len(Z)
    fits = []
    for r in range(n_rounds):
        idx = np.sort(rng.choice(n, size=max(10, int(frac * n)), replace=False))
        fits.append((idx, np.asarray(cluster_fn(Z[idx], seed + r))))
    scores = []
    for (ia, la), (ib, lb) in combinations(fits, 2):
        _, pa, pb = np.intersect1d(ia, ib, return_indices=True)
        a, b = la[pa], lb[pb]
        keep = (a >= 0) & (b >= 0)
        if keep.sum() >= 2:
            scores.append(adjusted_rand_score(a[keep], b[keep]))
    return float(np.mean(scores)) if scores else float("nan")


def ctfidf_keywords(
    docs: list[str],
    labels: np.ndarray,
    *,
    top_n: int = 8,
    min_df: int = 3,
    max_df: float = 0.5,
    ngram_range: tuple[int, int] = (1, 2),
) -> dict[int, list[str]]:
    """Class-based TF-IDF keywords (the weighting used by BERTopic). Noise (-1) is ignored.

    weight(t, c) = tf(t, c) * log(1 + A / f(t)), where tf is the term's share of cluster
    c's words, f(t) its total count over all clusters and A the mean words per cluster.
    Redundant terms (one contained in another, e.g. "reward" / "reward model") are skipped.
    """
    labels = np.asarray(labels)
    mask = labels >= 0
    docs = [d for d, m in zip(docs, mask) if m]
    labels = labels[mask]
    clusters = np.unique(labels)
    cv = CountVectorizer(
        stop_words="english", ngram_range=ngram_range, min_df=min_df, max_df=max_df
    )
    X = cv.fit_transform(docs)
    inv = np.searchsorted(clusters, labels)
    G = sparse.csr_matrix(
        (np.ones(len(docs)), (inv, np.arange(len(docs)))), shape=(len(clusters), len(docs))
    )
    Xc = np.asarray((G @ X).todense())
    tf = Xc / np.maximum(Xc.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + Xc.sum(axis=1).mean() / np.maximum(Xc.sum(axis=0), 1))
    W = tf * idf
    vocab = cv.get_feature_names_out()

    out: dict[int, list[str]] = {}
    for row, c in enumerate(clusters):
        chosen: list[str] = []
        chosen_tokens: list[set[str]] = []
        for j in np.argsort(-W[row]):
            tokens = set(vocab[j].split())
            if any(tokens <= s or s <= tokens for s in chosen_tokens):
                continue
            chosen.append(str(vocab[j]))
            chosen_tokens.append(tokens)
            if len(chosen) == top_n:
                break
        out[int(c)] = chosen
    return out


def centroids(Z: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Unit-length mean embedding of each cluster 0..k-1 (in the original embedding space)."""
    out = np.zeros((k, Z.shape[1]))
    for c in range(k):
        members = labels == c
        if members.any():
            out[c] = Z[members].mean(axis=0)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def representative_docs(
    Z: np.ndarray, labels: np.ndarray, centers: np.ndarray, n: int = 5
) -> dict[int, list[int]]:
    """Row indices of the n documents closest (cosine) to each cluster centre."""
    unit = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    out: dict[int, list[int]] = {}
    for k in range(len(centers)):
        idx = np.flatnonzero(labels == k)
        if idx.size == 0:
            out[k] = []
            continue
        sims = Z[idx] @ unit[k]
        out[k] = idx[np.argsort(-sims)[:n]].tolist()
    return out
