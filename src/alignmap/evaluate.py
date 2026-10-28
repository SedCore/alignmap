"""Does a clustering agree with what the community itself says the topics are?

LessWrong, the Alignment Forum and the EA Forum let authors and moderators attach topic tags to
posts ("Interpretability", "AI Governance", ...). Those tags are human judgements that no
pipeline in this repository ever sees, so agreement with them is an independent yardstick for
comparing embedders and clustering methods on the same posts.

Tags are weak supervision, not ground truth: they are multi-label, unevenly applied, and
coarser than some clusters. The numbers are for comparing pipelines on identical data, not for
absolute claims.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure

# Very generic tags carry no topic information. Edit freely; the tags actually used are always
# printed in the report, so you can check them.
GENERIC_TAGS = frozenset(
    {"AI", "Rationality", "World Modeling", "World Optimization", "Practical", "Community", "Site Meta",
     "Personal Blog", "Open Threads", "Meetups & Local Communities"}
)


def tag_labels(
    df: pd.DataFrame, *, top: int = 30, exclude: Iterable[str] = GENERIC_TAGS, min_count: int = 20
) -> tuple[pd.Series, list[str]]:
    """Reduce each post's tag list to ONE topic label.

    Among the ``top`` most frequent non-generic tags (with at least ``min_count`` posts), a post
    gets its least frequent one (its most specific topic). Posts with none of them get NaN.
    Returns the label Series (aligned with ``df``) and the tags used.
    """
    exclude = set(exclude)
    counts = Counter(t for tags in df["tags"] for t in set(tags) if t not in exclude)
    used = [t for t, c in counts.most_common(top) if c >= min_count]
    rank = {t: counts[t] for t in used}

    def pick(tags: list[str]):
        cands = [t for t in tags if t in rank]
        return min(cands, key=lambda t: (rank[t], t)) if cands else np.nan

    return df["tags"].map(pick), used


def agreement(pred: np.ndarray, truth: pd.Series) -> dict:
    """Agreement between cluster labels and tag labels on posts that have both.

    ``coverage`` is the fraction of tagged posts that the clustering placed in a cluster (HDBSCAN
    noise, label -1, is excluded from the agreement scores, so read the two together).
    """
    pred = np.asarray(pred)
    tagged = truth.notna().to_numpy()
    used = tagged & (pred >= 0)
    out = {"n_tagged": int(tagged.sum()), "n_evaluated": int(used.sum())}
    out["coverage"] = float(used.sum() / tagged.sum()) if tagged.sum() else float("nan")
    t, p = truth.to_numpy()[used], pred[used]
    if used.sum() < 2 or len(set(t)) < 2 or len(set(p)) < 2:
        out.update(ami=float("nan"), homogeneity=float("nan"), completeness=float("nan"), v_measure=float("nan"))
        return out
    h, c, v = homogeneity_completeness_v_measure(t, p)
    out.update(
        ami=float(adjusted_mutual_info_score(t, p)),
        homogeneity=float(h),
        completeness=float(c),
        v_measure=float(v),
    )
    return out
