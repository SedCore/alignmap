"""Figures and the Markdown report.

Chart choices: a sequential one-hue heatmap for "share of papers by cluster and year" (many
clusters are too many for categorical colour), and a diverging bar chart for "larger vs smaller
share" with grey for changes whose interval includes zero, so only evidence gets colour. Direction
is also carried by position and text, never by colour alone. When a source-adjusted estimate is
plotted, hollow diamonds mark the unadjusted (pooled) estimate so composition effects are visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GROW, SHRINK, NO_CHANGE = "#2a78d6", "#e34948", "#c3c2b7"
SEQUENTIAL = [  # one-hue ramp, steps 100 -> 700
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def _trunc(text: str, n: int = 54) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _style(ax, grid_axis: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3, color=AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_share_heatmap(
    share: pd.DataFrame,
    names: dict[int, str],
    year_counts: dict[int, int],
    path: Path,
    title: str,
) -> None:
    order = share.mean(axis=1).sort_values(ascending=False).index
    M = share.loc[order].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(10, max(3.6, 0.34 * len(order) + 1.9)), facecolor=SURFACE)
    im = ax.imshow(
        M,
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL),
        vmin=0,
        vmax=float(M.max()),
        interpolation="nearest",
    )
    _style(ax, "x")
    ax.grid(False)
    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels([f"{y}\nn={year_counts.get(int(y), 0):,}" for y in share.columns])
    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels([_trunc(names[int(c)]) for c in order], color=INK)
    ax.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True)  # 2px surface gaps between cells
    ax.set_yticks(np.arange(-0.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v >= 0.5:
                ax.text(
                    j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center",
                    fontsize=7.5, color="white" if im.norm(v) > 0.55 else INK,
                )
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("% of the year's papers", color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, fontweight="bold", pad=10)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_shift(
    shift: pd.DataFrame,
    names: dict[int, str],
    path: Path,
    baseline: tuple[int, int],
    recent: tuple[int, int],
    title: str,
    pooled: pd.DataFrame | None = None,
) -> None:
    d = shift.sort_values("log2_ratio").reset_index(drop=True)  # biggest growth ends up on top
    y = np.arange(len(d))
    colors = np.where(
        d["significant"], np.where(d["log2_ratio"] > 0, GROW, SHRINK), NO_CHANGE
    )
    fig, ax = plt.subplots(figsize=(10, max(3.6, 0.36 * len(d) + 2.4)), facecolor=SURFACE)
    _style(ax, "x")
    ax.barh(y, d["log2_ratio"], height=0.55, color=colors, zorder=2)
    ax.errorbar(
        d["log2_ratio"], y,
        xerr=[d["log2_ratio"] - d["ci_low"], d["ci_high"] - d["log2_ratio"]],
        fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3,
    )
    extent = [d["ci_high"].max(), -d["ci_low"].min()]
    if pooled is not None:
        pooled_vals = pooled.set_index("cluster")["log2_ratio"].reindex(d["cluster"]).to_numpy()
        ax.scatter(pooled_vals, y, marker="D", s=26, facecolors=SURFACE, edgecolors=INK, linewidths=1.1, zorder=4)
        extent += [np.nanmax(pooled_vals), -np.nanmin(pooled_vals)]
    ax.axvline(0, color=INK2, linewidth=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([_trunc(names[int(c)]) for c in d["cluster"]], color=INK)
    span = float(max(extent)) * 1.18
    ax.set_xlim(-span, span)
    for yi, (_, row) in zip(y, d.iterrows()):  # label only the flagged bars
        if row["significant"]:
            right = row["log2_ratio"] > 0
            ax.text(
                (row["ci_high"] if right else row["ci_low"]) + (0.03 if right else -0.03) * span,
                yi, f"{row['log2_ratio']:+.1f}", va="center", ha="left" if right else "right",
                fontsize=8, color=INK,
            )
    ax.set_xlabel(
        f"log₂ ratio of share of papers, {recent[0]}–{recent[1]} vs {baseline[0]}–{baseline[1]}"
        "   (← smaller share · larger share →)",
        color=INK2, fontsize=9,
    )
    handles = [
        Patch(color=GROW, label="Larger share (interval excludes 0)"),
        Patch(color=SHRINK, label="Smaller share (interval excludes 0)"),
        Patch(color=NO_CHANGE, label="No clear change"),
    ]
    if pooled is not None:
        handles.append(
            Line2D([], [], marker="D", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK,
                   markersize=6, label="Unadjusted (pooled) estimate")
        )
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=len(handles),
        frameon=False, fontsize=8.5, labelcolor=INK2,
    )
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, fontweight="bold", pad=10)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _esc(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(path: Path, ctx: dict) -> None:
    stats, cfg, shift, pooled = ctx["stats"], ctx["cfg"], ctx["shift"], ctx["pooled"]
    names = ctx["label_of"]
    b, r = ctx["baseline"], ctx["recent"]
    lines: list[str] = [f"# Alignment-research map: {r[0]}–{r[1]} vs {b[0]}–{b[1]}", ""]
    if ctx.get("banner"):
        lines += [f"> **{ctx['banner']}**", ""]

    dropped = (
        f"{stats['dropped_source']} from other sources, {stats['dropped_no_date']} without a usable date, "
        f"{stats['dropped_short']} too short, {stats['dropped_duplicate']} duplicates"
    )
    lines += [
        f"**Corpus.** {stats['n_kept']:,} papers kept of {stats['n_input']:,} read ({dropped}). "
        "Sources: " + ", ".join(f"{k} ({v:,})" for k, v in stats["by_source"].items()) + ".",
        "",
        f"**Structure.** {ctx['clustering']} → {ctx['n_clusters']} clusters"
        + (f", {_pct(ctx['n_noise'] / stats['n_kept'])} unassigned" if ctx["n_noise"] else "")
        + f" · silhouette {ctx['silhouette']:.3f} · stability (ARI across 80% subsamples) {ctx['stability']:.3f}.",
        "",
    ]
    ag = ctx.get("agreement")
    if ag and not pd.isna(ag["ami"]):
        lines += [
            f"**Agreement with community tags.** AMI {ag['ami']:.3f}, V-measure {ag['v_measure']:.3f} "
            f"on {ag['n_evaluated']:,} tagged posts ({ag['coverage']:.0%} of tagged posts were clustered). "
            "Tags used: " + ", ".join(f"`{t}`" for t in ctx["used_tags"]) + ".",
            "",
        ]

    lines += ["## Sources and coverage", "", "Papers kept per source and year:", ""]
    cov = ctx["coverage"]
    lines.append("| Source | " + " | ".join(str(c) for c in cov.columns) + " |")
    lines.append("|---|" + "|".join("---:" for _ in cov.columns) + "|")
    for src, row in cov.iterrows():
        lines.append(f"| {_esc(src)} | " + " | ".join(f"{int(v):,}" for v in row) + " |")
    lines.append("")
    for w in ctx["coverage_warnings"]:
        lines.append(f"- ⚠ {w}")
    if ctx["coverage_warnings"]:
        lines.append("")

    lines += ["## Which subfields changed share?", ""]
    if ctx["adjusted"]:
        lines += [
            "The main result is **source-adjusted**: cluster shares are computed within each source group "
            f"({', '.join(ctx['groups'])}) and combined with fixed weights, so a change in the mix of venues "
            "cannot masquerade as a change in topics. Hollow diamonds in the figure show the unadjusted pooled "
            "estimate; a large gap between a diamond and its bar is a composition effect.",
        ]
        if ctx["excluded_groups"]:
            lines.append(
                f"Groups left out of the adjustment (too few papers in one of the two periods): "
                f"{', '.join(ctx['excluded_groups'])}."
            )
        lines.append("")
    else:
        lines += [
            "Pooled over all papers. With a single usable source group no source adjustment is possible, so a "
            "change in the mix of venues over time could contribute to these shifts.",
            "",
        ]
    lines += ["![Change in share per cluster](figures/shift.png)", ""]

    pooled_by = pooled.set_index("cluster")["log2_ratio"]
    head = "| Cluster | Share before → after | log₂ ratio [Bonferroni interval] | Verdict |"
    sep = "|---|---|---|---|"
    if ctx["adjusted"]:
        head, sep = head + " Unadjusted log₂ |", sep + "---:|"
    lines += [head, sep]
    for _, row in shift.iterrows():
        verdict = "—"
        if row["significant"]:
            verdict = "▲ larger share" if row["log2_ratio"] > 0 else "▼ smaller share"
        cells = (
            f"| {_esc(names[int(row['cluster'])])} | {_pct(row['share_baseline'])} → "
            f"{_pct(row['share_recent'])} | {row['log2_ratio']:+.2f} "
            f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}] | {verdict} |"
        )
        if ctx["adjusted"]:
            cells += f" {pooled_by[int(row['cluster'])]:+.2f} |"
        lines.append(cells)

    lines += [
        "",
        "## Share of papers per cluster and year (pooled)",
        "",
        "![Share of papers by cluster and year](figures/share_heatmap.png)",
        "",
        "## Clusters",
        "",
    ]
    for c in ctx["clusters"]:
        lines.append(f"### {_esc(c['label'])}")
        lines.append("")
        if c["summary"]:
            lines += [f"_{_esc(c['summary'])}_ (LLM-written; check against the papers)", ""]
        lines.append(f"{c['n']:,} papers ({_pct(c['share'])}) · keywords: {', '.join(c['keywords'])}")
        lines.append("")
        for rep in c["representatives"]:
            title = _esc(rep["title"])
            lines.append(f"- [{title}]({rep['url']}) ({rep['year']})" if rep["url"] else f"- {title} ({rep['year']})")
        lines.append("")

    emb = f"`{cfg['sbert_model']}`" if cfg["embedder"] == "sbert" else f"TF-IDF + LSA ({cfg['dims']} dims)"
    lines += [
        "## Settings",
        "",
        f"- embedder {emb}; clustering: {ctx['clustering']}"
        + (" (k chosen by silhouette over " + ", ".join(map(str, ctx["k_scan"])) + ")" if ctx["k_scan"] else ""),
        f"- baseline {b[0]}–{b[1]}, recent {r[0]}–{r[1]}; "
        f"{cfg['n_boot']:,} bootstrap draws; seed {cfg['seed']}",
        f"- LLM labels: {'yes (' + str(ctx['llm_model']) + ')' if cfg['llm_labels'] else 'no (keywords only)'}",
        "",
        "## Read this before quoting a number",
        "",
        "- The corpus is whatever the sources contain, not the whole field. Sources that stop being updated "
        "make their topics look like they declined (see the coverage warnings above).",
        "- Shares are relative: when one cluster grows, the others must shrink. Intervals reflect sampling of "
        "papers only, not the choice of embedder, clustering or sources.",
        "- Clusters are a convenient partition, not ground truth; rerun with another method before reading a "
        "story into one cluster.",
        "- LLM names and summaries, when present, are unverified paraphrases of the keywords and titles.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
