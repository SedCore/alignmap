"""Compare pipelines (embedder x reducer x clusterer) on the SAME corpus.

For each arm we report how many clusters it finds, how much it leaves unassigned, how stable the
partition is across subsamples, how well it agrees with the community's own topic tags, and how long
embedding + clustering took. Use a fresh ``--cache-dir`` for honest timings (cached embeddings take
almost no time).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .cluster import run_clustering, stability
from .evaluate import agreement
from .pipeline import RunConfig, _renumber, cluster_corpus, tag_truth


def compare_arms(
    df: pd.DataFrame,
    arms: dict[str, RunConfig],
    *,
    cache_dir=None,
    n_rounds: int = 3,
    log=print,
) -> tuple[pd.DataFrame, list[str]]:
    """Returns (table with one row per arm, tags used for the agreement scores)."""
    rows = []
    used_tags: list[str] = []
    for name, cfg in arms.items():
        log(f"[{name}] {cfg.embedder}:{cfg.sbert_model if cfg.embedder == 'sbert' else 'lsa'} / {cfg.spec().describe()}")
        t0 = time.time()
        res = cluster_corpus(df, cfg, cache_dir=cache_dir)
        seconds = time.time() - t0
        raw = res["labels"]
        row = {
            "arm": name,
            "embedder": cfg.sbert_model if cfg.embedder == "sbert" else "tfidf+lsa",
            "clustering": res["spec"].describe(),
            "n_clusters": int(raw.max()) + 1 if (raw >= 0).any() else 0,
            "noise_fraction": float((raw < 0).mean()),
            "seconds": round(seconds, 1),
        }
        spec = res["spec"]
        row["stability_ari"] = (
            stability(res["Z"], lambda Zs, s, spec=spec: run_clustering(Zs, spec, s, quick=True),
                      n_rounds=n_rounds, seed=cfg.seed)
            if row["n_clusters"] > 1 else float("nan")
        )
        truth, used = tag_truth(df, cfg)
        if truth is not None:
            used_tags = used
            labels = _renumber(raw) if (raw >= 0).any() else raw
            row.update({f"tags_{k}": v for k, v in agreement(labels, truth).items()})
        rows.append(row)
    return pd.DataFrame(rows), used_tags


def write_compare(table: pd.DataFrame, used_tags: list[str], path, *, n_docs: int, sources: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path.with_suffix(".csv"), index=False)

    cols = [
        ("arm", "Arm"), ("embedder", "Embedder"), ("clustering", "Clustering"), ("n_clusters", "Clusters"),
        ("noise_fraction", "Noise"), ("stability_ari", "Stability (ARI)"), ("tags_ami", "Tag agreement (AMI)"),
        ("tags_v_measure", "V-measure"), ("tags_coverage", "Tagged posts clustered"), ("seconds", "Seconds"),
    ]
    cols = [(c, h) for c, h in cols if c in table.columns]

    def fmt(c, v):
        if isinstance(v, float):
            if c in ("noise_fraction", "tags_coverage"):
                return f"{v:.1%}"
            return "—" if pd.isna(v) else f"{v:.3f}" if c != "seconds" else f"{v:.0f}"
        return str(v).replace("|", "\\|")

    lines = [
        "# Method comparison", "",
        f"{n_docs:,} papers from " + ", ".join(f"{k} ({v:,})" for k, v in sources.items()) + ".", "",
        "| " + " | ".join(h for _, h in cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(fmt(c, row[c]) for c, _ in cols) + " |")
    stability_note = (
        "*Stability* is the mean adjusted Rand index between runs on random 80% subsamples; *Noise* is the "
        "share of papers a density-based method leaves unassigned."
    )
    if "tags_ami" in table.columns:
        note = (
            "**How to read this.** *Tag agreement* is the adjusted mutual information between a method's clusters "
            "and the topic tags that LessWrong / Alignment Forum / EA Forum authors attached to their posts. The "
            "method never sees the tags, so a higher value means its clusters match human topic judgements "
            "better. AMI corrects for chance and for different numbers of clusters; it is computed on tagged "
            "posts the method placed in a cluster, so read it together with *Tagged posts clustered* and "
            f"*Noise*. {stability_note} Tags are weak supervision: unevenly applied, multi-label (reduced to "
            "one), and coarser than some clusters."
        )
    else:
        note = (
            "**How to read this.** This corpus has no community tags, so methods cannot be scored against human "
            f"topic labels here; only structure is compared. {stability_note}"
        )
    lines += ["", note, ""]
    if used_tags:
        lines += ["Tags used as reference labels: " + ", ".join(f"`{t}`" for t in used_tags) + ".", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
