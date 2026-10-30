"""End-to-end run: records -> clean -> embed -> cluster -> label -> trends -> report."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from . import __version__, report
from .cluster import (
    ClusterSpec,
    centroids,
    cluster_reduced,
    ctfidf_keywords,
    reduce,
    representative_docs,
    run_clustering,
    sampled_silhouette,
    select_k,
    stability,
)
from .corpus import build_corpus, coverage_table, coverage_warnings, docs_for_embedding
from .embed import embed
from .evaluate import GENERIC_TAGS, agreement, tag_labels
from .labeling import llm_label_clusters
from .trends import share_by_year, shift_table, shift_table_adjusted, usable_groups


@dataclass
class RunConfig:
    # corpus
    sources: tuple[str, ...] | None = None  # None keeps every source in the input
    drop_tags: tuple[str, ...] = ()  # posts carrying any of these community tags are left out
    min_chars: int = 200
    max_chars: int = 3000
    # representation
    embedder: str = "tfidf"  # "tfidf" | "sbert"
    dims: int = 128  # LSA dimensions (tfidf only)
    sbert_model: str = "BAAI/bge-small-en-v1.5"
    max_seq_length: int | None = None
    doc_format: str = "plain"  # "plain" | "specter"
    doc_prefix: str = ""
    # clustering
    reducer: str = "none"  # "none" | "umap"
    umap_components: int = 5
    umap_neighbors: int = 30
    clusterer: str = "kmeans"  # "kmeans" | "hdbscan"
    k: int | None = 20  # None -> pick by silhouette over k_range (k-means only)
    k_range: tuple[int, int, int] = (8, 30, 2)  # start, stop (inclusive), step
    min_cluster_size: float = 0.01  # HDBSCAN; < 1 means a fraction of the documents
    # analysis
    baseline: tuple[int, int] = (2021, 2022)
    recent: tuple[int, int] = (2025, 2026)
    min_group_docs: int = 30
    seed: int = 0
    n_boot: int = 5000
    top_keywords: int = 8
    n_representatives: int = 5
    tag_top: int = 30
    tag_exclude: tuple[str, ...] = tuple(sorted(GENERIC_TAGS))
    llm_labels: bool = False
    llm_model: str | None = None

    def spec(self) -> ClusterSpec:
        return ClusterSpec(
            reducer=self.reducer, umap_components=self.umap_components, umap_neighbors=self.umap_neighbors,
            clusterer=self.clusterer, k=self.k, min_cluster_size=self.min_cluster_size,
        )


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot serialise {type(obj)}")


def cluster_corpus(df: pd.DataFrame, cfg: RunConfig, *, cache_dir=None) -> dict:
    """Embed the corpus and cluster it. Returns embeddings, raw labels (-1 = noise) and the final spec."""
    Z = embed(
        docs_for_embedding(df, cfg.doc_format), cfg.embedder, dims=cfg.dims, seed=cfg.seed,
        sbert_model=cfg.sbert_model, max_seq_length=cfg.max_seq_length, doc_prefix=cfg.doc_prefix,
        cache_dir=cache_dir,
    )
    spec = cfg.spec()
    X = reduce(Z, spec, cfg.seed)
    k_scan: dict[int, float] = {}
    if spec.clusterer == "kmeans" and spec.k is None:
        start, stop, step = cfg.k_range
        k, k_scan = select_k(X, range(start, stop + 1, step), seed=cfg.seed)
        spec = replace(spec, k=k)
    labels = cluster_reduced(X, spec, cfg.seed)
    return {"Z": Z, "labels": labels, "spec": spec, "k_scan": k_scan}


def _renumber(raw: np.ndarray) -> np.ndarray:
    """Number clusters by size (0 = largest); noise stays -1."""
    ids, counts = np.unique(raw[raw >= 0], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    lookup = np.full(int(raw.max()) + 1, -1, dtype=int)
    lookup[order] = np.arange(len(order))
    return np.where(raw >= 0, lookup[np.maximum(raw, 0)], -1)


def tag_truth(df: pd.DataFrame, cfg: RunConfig):
    """Single topic label per post from community tags, or (None, []) if the corpus has too few."""
    if not df["tags"].map(len).any():
        return None, []
    truth, used = tag_labels(df, top=cfg.tag_top, exclude=cfg.tag_exclude)
    return (truth, used) if truth.notna().sum() >= 20 and len(used) >= 2 else (None, [])


def run_pipeline(
    records, out_dir, cfg: RunConfig | None = None, *, provenance: dict | None = None,
    llm_client=None, cache_dir=None,
) -> dict:
    cfg = cfg or RunConfig()
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    df, stats = build_corpus(
        records, sources=cfg.sources, min_chars=cfg.min_chars, max_chars=cfg.max_chars, drop_tags=cfg.drop_tags
    )
    for name, (lo, hi) in (("baseline", cfg.baseline), ("recent", cfg.recent)):
        if not df["year"].between(lo, hi).any():
            raise ValueError(
                f"No papers in the {name} period {lo}-{hi}. Papers per year: {stats['by_year']}. "
                "Adjust --baseline / --recent or add more years."
            )

    t0 = time.time()
    res = cluster_corpus(df, cfg, cache_dir=cache_dir)
    Z, spec, k_scan = res["Z"], res["spec"], res["k_scan"]
    raw = res["labels"]
    if not (raw >= 0).any():
        raise ValueError("Clustering found no clusters (everything is noise); lower --min-cluster-size")
    labels = _renumber(raw)
    n_clusters = int(labels.max()) + 1
    df["cluster"] = labels
    seconds = time.time() - t0

    centers = centroids(Z, labels, n_clusters)
    silhouette = sampled_silhouette(Z, labels, seed=cfg.seed)
    stab = stability(Z, lambda Zs, s: run_clustering(Zs, spec, s, quick=True), seed=cfg.seed)

    keywords = ctfidf_keywords(df["doc"].tolist(), labels, top_n=cfg.top_keywords)
    reps = representative_docs(Z, labels, centers, n=cfg.n_representatives)

    names = {c: ", ".join(keywords[c][:3]) or "unlabelled" for c in range(n_clusters)}
    summaries = {c: "" for c in range(n_clusters)}
    llm_used = {c: False for c in range(n_clusters)}
    if cfg.llm_labels:
        batch = [
            {"cluster": c, "keywords": keywords[c], "titles": df.loc[reps[c], "title"].tolist()}
            for c in range(n_clusters)
        ]
        for c, lab in llm_label_clusters(batch, model=cfg.llm_model, client=llm_client).items():
            names[c], summaries[c], llm_used[c] = lab["name"], lab["summary"], lab["llm"]

    clusters = []
    for c in range(n_clusters):
        n_c = int((labels == c).sum())
        clusters.append(
            {
                "id": c, "label": f"C{c:02d} · {names[c]}", "name": names[c], "summary": summaries[c],
                "llm": llm_used[c], "keywords": keywords[c], "n": n_c, "share": n_c / len(df),
                "representatives": [
                    {"title": df.at[i, "title"], "url": df.at[i, "url"], "year": int(df.at[i, "year"])}
                    for i in reps[c]
                ],
            }
        )
    n_noise = int((labels < 0).sum())
    label_of = {c["id"]: c["label"] for c in clusters}
    if n_noise:
        label_of[-1] = "Unassigned (noise)"

    share = share_by_year(df)
    pooled = shift_table(df, cfg.baseline, cfg.recent, n_boot=cfg.n_boot, seed=cfg.seed)
    groups = usable_groups(df, cfg.baseline, cfg.recent, min_docs=cfg.min_group_docs)
    adjusted = (
        shift_table_adjusted(
            df, cfg.baseline, cfg.recent, min_docs=cfg.min_group_docs, n_boot=cfg.n_boot, seed=cfg.seed
        )
        if len(groups) >= 2
        else None
    )
    excluded_groups = sorted(set(df["group"]) - set(groups))
    coverage = coverage_table(df)
    warnings = coverage_warnings(df)

    truth, used_tags = tag_truth(df, cfg)
    agree = agreement(labels, truth) if truth is not None else None

    banner = None
    if all(str(s).startswith("synthetic") for s in df["source"].unique()):
        banner = (
            "SYNTHETIC DEMO DATA: generated by `alignmap demo` to exercise the pipeline. "
            "It says nothing about the real alignment literature."
        )
    suffix = " (synthetic)" if banner else ""
    main = adjusted if adjusted is not None else pooled

    report.plot_share_heatmap(
        share, label_of, stats["by_year"], out_dir / "figures" / "share_heatmap.png",
        "Share of papers per cluster and year" + suffix,
    )
    report.plot_shift(
        main, label_of, out_dir / "figures" / "shift.png", cfg.baseline, cfg.recent,
        ("Source-adjusted change in share" if adjusted is not None else "Change in share") + suffix,
        pooled=pooled if adjusted is not None else None,
    )

    df[["id", "source", "group", "year", "cluster", "title", "url"]].to_csv(out_dir / "assignments.csv", index=False)
    share.to_csv(out_dir / "share_by_year.csv")
    coverage.to_csv(out_dir / "coverage.csv")
    pooled.assign(label=pooled["cluster"].map(label_of)).to_csv(out_dir / "shift_pooled.csv", index=False)
    if adjusted is not None:
        adjusted.assign(label=adjusted["cluster"].map(label_of)).to_csv(out_dir / "shift_adjusted.csv", index=False)
    (out_dir / "clusters.json").write_text(json.dumps(clusters, indent=2, ensure_ascii=False), encoding="utf-8")

    cfg_dict = asdict(cfg)
    report.write_report(
        out_dir / "report.md",
        {
            "banner": banner, "stats": stats, "cfg": cfg_dict, "n_clusters": n_clusters,
            "clustering": spec.describe(), "k_scan": k_scan, "n_noise": n_noise,
            "silhouette": silhouette, "stability": stab, "clusters": clusters,
            "shift": main, "pooled": pooled, "adjusted": adjusted is not None,
            "groups": groups, "excluded_groups": excluded_groups,
            "coverage": coverage, "coverage_warnings": warnings,
            "agreement": agree, "used_tags": used_tags,
            "baseline": cfg.baseline, "recent": cfg.recent,
            "llm_model": cfg.llm_model or "default model", "label_of": label_of,
        },
    )

    summary = {
        "alignmap": __version__, "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "config": cfg_dict, "provenance": provenance or {}, "corpus": stats,
        "clustering": spec.describe(), "n_clusters": n_clusters, "noise_fraction": n_noise / len(df),
        "k_scan": k_scan, "silhouette": silhouette, "stability_ari": stab,
        "tag_agreement": agree, "tags_used": used_tags, "seconds_embed_and_cluster": round(seconds, 1),
        "source_adjusted": adjusted is not None, "groups_used": groups, "groups_excluded": excluded_groups,
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return {"df": df, "clusters": clusters, "shift": main, "pooled": pooled, "adjusted": adjusted,
            "share": share, "summary": summary}
