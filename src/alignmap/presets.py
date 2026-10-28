"""Named configurations, used to compare methods on identical data.

``2022-style`` is an approximate reconstruction of the pipeline described by Kirchner et al. (2022):
SPECTER embeddings of title + abstract, UMAP with n_neighbors = 250, k-means with k = 5. The paper does
not state the UMAP output dimension (we use 5) and the analysis code is not public, so this arm is a
reference point for comparison, not an exact replication.

The ``modern-*`` presets swap in general-purpose long-context embedders and a density-based
clusterer that finds the number of subfields itself. Whether they actually agree better with
human tags is what ``alignmap compare`` measures; nothing here assumes they do.
"""

from __future__ import annotations

from dataclasses import replace

PRESETS: dict[str, dict] = {
    "lsa-baseline": dict(embedder="tfidf", reducer="none", clusterer="kmeans", k=20),
    "2022-style": dict(
        embedder="sbert", sbert_model="sentence-transformers/allenai-specter", doc_format="specter",
        max_seq_length=512, reducer="umap", umap_neighbors=250, umap_components=5,
        clusterer="kmeans", k=5,
    ),
    "modern-fast": dict(
        embedder="sbert", sbert_model="BAAI/bge-small-en-v1.5", max_seq_length=512,
        reducer="umap", umap_neighbors=30, umap_components=10, clusterer="hdbscan", min_cluster_size=0.01,
    ),
    "modern-strong": dict(
        embedder="sbert", sbert_model="BAAI/bge-m3", max_seq_length=1024,
        reducer="umap", umap_neighbors=30, umap_components=10, clusterer="hdbscan", min_cluster_size=0.01,
    ),
}


def apply_preset(cfg, name: str):
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    return replace(cfg, **PRESETS[name])
