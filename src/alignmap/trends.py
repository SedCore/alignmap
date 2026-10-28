"""Which subfields grew or shrank between two periods?

Everything is computed on *shares* (fraction of a period's papers), not raw counts:
publication volume grew by an order of magnitude over these years, so counts would make every
cluster look like it is growing.

Two estimates are provided:

``shift_table``           pooled over all papers.
``shift_table_adjusted``  source-adjusted: cluster shares are computed *within* each source group
                          (forums, papers, blogs ...) and combined with fixed weights, so a change in
                          the mix of venues (say, more arXiv papers and fewer forum posts) cannot
                          masquerade as a change in topics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def parse_year_range(text: str) -> tuple[int, int]:
    """'2021-2022' -> (2021, 2022); '2022' -> (2022, 2022)."""
    parts = str(text).replace("..", "-").split("-")
    if len(parts) == 1:
        return int(parts[0]), int(parts[0])
    if len(parts) == 2:
        lo, hi = int(parts[0]), int(parts[1])
        if lo > hi:
            raise ValueError(f"Year range {text!r} is reversed")
        return lo, hi
    raise ValueError(f"Cannot parse year range {text!r}")


def share_by_year(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """Rows = clusters, columns = years, values = fraction of that year's papers."""
    counts = pd.crosstab(df[cluster_col], df["year"])
    return counts.div(counts.sum(axis=0), axis=1)


def _table(clusters, nb, nr, share_b, share_r, boots, family_alpha) -> pd.DataFrame:
    """Assemble the result table: log2 ratio, Bonferroni-adjusted bootstrap interval, flag."""
    K = len(clusters)
    a = family_alpha / K
    lo, hi = np.percentile(boots, [100 * a / 2, 100 * (1 - a / 2)], axis=0)
    out = pd.DataFrame(
        {
            "cluster": clusters,
            "n_baseline": nb,
            "n_recent": nr,
            "share_baseline": share_b,
            "share_recent": share_r,
            "log2_ratio": np.log2(share_r / share_b),
            "ci_low": lo,
            "ci_high": hi,
        }
    )
    out["significant"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    return out.sort_values("log2_ratio", ascending=False).reset_index(drop=True)


def shift_table(
    df: pd.DataFrame,
    baseline: tuple[int, int],
    recent: tuple[int, int],
    *,
    cluster_col: str = "cluster",
    alpha: float = 0.5,
    n_boot: int = 5000,
    family_alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Log2 ratio of each cluster's share, recent vs. baseline, with bootstrap intervals.

    * Shares use additive smoothing (``alpha`` pseudo-documents per cluster) so a cluster
      that is absent from one period gets a finite, conservative ratio.
    * Intervals come from a multinomial bootstrap of each period's cluster counts.
    * ``family_alpha`` is Bonferroni-split over the K clusters, so ``significant`` controls
      the chance of *any* false flag at roughly ``family_alpha`` (about the sampling of
      papers only - not the choice of terms, embedder or clustering).
    """
    base = df[df["year"].between(*baseline)]
    rec = df[df["year"].between(*recent)]
    if base.empty or rec.empty:
        raise ValueError(
            f"Need papers in both periods (baseline {baseline}: {len(base)}, "
            f"recent {recent}: {len(rec)}). Available years: {sorted(df['year'].unique())}"
        )

    clusters = np.sort(df[cluster_col].unique())
    K = len(clusters)
    nb = base[cluster_col].value_counts().reindex(clusters, fill_value=0).to_numpy()
    nr = rec[cluster_col].value_counts().reindex(clusters, fill_value=0).to_numpy()

    def smooth(c: np.ndarray) -> np.ndarray:
        return (c + alpha) / (c.sum(axis=-1, keepdims=True) + alpha * K)

    rng = np.random.default_rng(seed)
    boots = np.log2(
        smooth(rng.multinomial(nr.sum(), nr / nr.sum(), size=n_boot))
        / smooth(rng.multinomial(nb.sum(), nb / nb.sum(), size=n_boot))
    )
    return _table(clusters, nb, nr, smooth(nb), smooth(nr), boots, family_alpha)


def usable_groups(
    df: pd.DataFrame,
    baseline: tuple[int, int],
    recent: tuple[int, int],
    *,
    group_col: str = "group",
    min_docs: int = 30,
) -> list[str]:
    """Source groups with at least ``min_docs`` papers in BOTH periods."""
    b = df[df["year"].between(*baseline)][group_col].value_counts()
    r = df[df["year"].between(*recent)][group_col].value_counts()
    return sorted(g for g in set(b.index) & set(r.index) if b[g] >= min_docs and r[g] >= min_docs)


def shift_table_adjusted(
    df: pd.DataFrame,
    baseline: tuple[int, int],
    recent: tuple[int, int],
    *,
    cluster_col: str = "cluster",
    group_col: str = "group",
    min_docs: int = 30,
    alpha: float = 0.5,
    n_boot: int = 5000,
    family_alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Source-adjusted version of :func:`shift_table`.

    Within each usable source group g and period, cluster shares p_g are computed (smoothed). The
    adjusted share is sum_g w_g * p_g with the SAME weights w_g in both periods (g's share of all
    papers in the two periods combined), so only within-source change contributes. Groups with fewer
    than ``min_docs`` papers in either period are left out; the bootstrap resamples each
    (group, period) cell separately.
    """
    groups = usable_groups(df, baseline, recent, group_col=group_col, min_docs=min_docs)
    if len(groups) < 2:
        raise ValueError(
            f"Source adjustment needs >= 2 groups with {min_docs}+ papers in both periods; got {groups}"
        )
    sub = df[df[group_col].isin(groups)]
    clusters = np.sort(df[cluster_col].unique())
    K = len(clusters)

    cells: dict[tuple[str, str], np.ndarray] = {}
    for g in groups:
        for name, (lo, hi) in (("b", baseline), ("r", recent)):
            s = sub[(sub[group_col] == g) & sub["year"].between(lo, hi)]
            cells[(g, name)] = s[cluster_col].value_counts().reindex(clusters, fill_value=0).to_numpy()

    w = np.array([cells[(g, "b")].sum() + cells[(g, "r")].sum() for g in groups], dtype=float)
    w /= w.sum()

    def adjusted(per_group: list[np.ndarray]) -> np.ndarray:
        return sum(
            wi * (c + alpha) / (c.sum(axis=-1, keepdims=True) + alpha * K) for wi, c in zip(w, per_group)
        )

    rng = np.random.default_rng(seed)

    def draw(period: str) -> list[np.ndarray]:
        out = []
        for g in groups:
            c = cells[(g, period)]
            out.append(rng.multinomial(c.sum(), c / c.sum(), size=n_boot))
        return out

    share_b = adjusted([cells[(g, "b")] for g in groups])
    share_r = adjusted([cells[(g, "r")] for g in groups])
    boots = np.log2(adjusted(draw("r")) / adjusted(draw("b")))
    nb = sum(cells[(g, "b")] for g in groups)
    nr = sum(cells[(g, "r")] for g in groups)
    return _table(clusters, nb, nr, share_b, share_r, boots, family_alpha)
