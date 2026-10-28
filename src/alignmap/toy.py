"""Synthetic corpora with KNOWN structure, used for tests and ``alignmap demo``.

Documents are word salad drawn from four topic vocabularies (plus 12% cross-topic noise) and carry a
human-style ``tags`` list with the planted topic. They say NOTHING about the real alignment
literature, and every output built from them is bannered as synthetic.

``make_toy_records``      topic shares change over time:
                              interpretability   10% -> 40%   (grows)
                              agent foundations  35% ->  5%   (shrinks)
                              rlhf               30% -> 30%   (flat)
                              governance         25% -> 25%   (flat)

``make_composition_toy``  two venues ("forum" and "papers") with different topic mixes, and a venue
                          mix that drifts from 20% to 70% papers. Within each venue only two topics
                          really change (agent foundations falls, governance rises); interpretability
                          and RLHF are constant. A pooled analysis therefore shows a SPURIOUS rise of
                          interpretability, which a source-adjusted analysis must not.
"""

from __future__ import annotations

import numpy as np

SOURCE = "synthetic-toy"
FORUM, PAPERS = "synthetic-forum", "synthetic-papers"

TAG_NAMES = {
    "interpretability": "Interpretability",
    "agent_foundations": "Agent Foundations",
    "rlhf": "RLHF",
    "governance": "AI Governance",
}

TOPICS = {
    "interpretability": {
        "start": 0.10,
        "end": 0.40,
        "vocab": "interpretability circuits features activations sparse autoencoder probing attention "
        "heads neurons superposition residual stream dictionary steering logit lens ablation causal "
        "mechanistic".split(),
    },
    "agent_foundations": {
        "start": 0.35,
        "end": 0.05,
        "vocab": "agent foundations decision theory embedded agency logical uncertainty counterfactual "
        "utility fixed point reflective consistency bounded rationality cooperation mathematical proof "
        "formal".split(),
    },
    "rlhf": {
        "start": 0.30,
        "end": 0.30,
        "vocab": "rlhf reward preference human feedback annotators policy optimization overoptimization kl "
        "penalty finetuning instruction helpfulness ranking bradley terry sycophancy dpo".split(),
    },
    "governance": {
        "start": 0.25,
        "end": 0.25,
        "vocab": "governance regulation compute standards audit institutions international coordination "
        "licensing evaluation assessment frameworks incident reporting verification treaty export "
        "liability".split(),
    },
}

GENERIC = (
    "we propose study results approach method paper show analysis problem work new using based "
    "framework evidence systems case discuss present findings future research".split()
)


def _text(rng, name: str, noise: float) -> str:
    others = [w for n in TOPICS if n != name for w in TOPICS[n]["vocab"]]
    words = rng.choice(TOPICS[name]["vocab"], size=40).tolist()
    words = [rng.choice(others).item() if rng.random() < noise else w for w in words]
    words += rng.choice(GENERIC, size=25).tolist()
    rng.shuffle(words)
    return " ".join(words).capitalize() + "."


def _date(rng, year: int) -> str:
    return f"{year}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}T12:00:00Z"


def _record(i: int, source: str, name: str, text: str, date: str) -> dict:
    return {
        "id": f"toy:{i:05d}",
        "source": source,
        "title": f"Synthetic document {i:05d}",
        "authors": ["A. Nonymous"],
        "text": text,
        "url": "",
        "date_published": date,
        "tags": [TAG_NAMES[name], "AI"],
        "_topic": name,
    }


def make_toy_records(
    n_per_year: int = 150, years=range(2021, 2027), seed: int = 0, noise: float = 0.12
) -> list[dict]:
    rng = np.random.default_rng(seed)
    years = list(years)
    names = list(TOPICS)
    records: list[dict] = []
    i = 0
    for y in years:
        frac = (y - years[0]) / max(1, years[-1] - years[0])
        p = np.array([(1 - frac) * TOPICS[n]["start"] + frac * TOPICS[n]["end"] for n in names])
        counts = rng.multinomial(n_per_year, p / p.sum())
        for name, c in zip(names, counts):
            for _ in range(int(c)):
                text = _text(rng, name, noise)
                records.append(_record(i, SOURCE, name, text, _date(rng, y)))
                i += 1
    return records


# topic mix per venue at the first and last year (rows sum to 1)
_MIX = {
    FORUM: {
        "start": {"interpretability": 0.10, "agent_foundations": 0.40, "rlhf": 0.35, "governance": 0.15},
        "end": {"interpretability": 0.10, "agent_foundations": 0.15, "rlhf": 0.35, "governance": 0.40},
    },
    PAPERS: {
        "start": {"interpretability": 0.60, "agent_foundations": 0.05, "rlhf": 0.15, "governance": 0.20},
        "end": {"interpretability": 0.60, "agent_foundations": 0.02, "rlhf": 0.15, "governance": 0.23},
    },
}


def make_composition_toy(
    n_per_year: int = 300, years=range(2021, 2027), seed: int = 0, noise: float = 0.12
) -> list[dict]:
    rng = np.random.default_rng(seed)
    years = list(years)
    names = list(TOPICS)
    records: list[dict] = []
    i = 0
    for y in years:
        frac = (y - years[0]) / max(1, years[-1] - years[0])
        papers_share = 0.2 + 0.5 * frac
        n_papers = int(round(n_per_year * papers_share))
        for source, n in ((FORUM, n_per_year - n_papers), (PAPERS, n_papers)):
            mix = _MIX[source]
            p = np.array([(1 - frac) * mix["start"][t] + frac * mix["end"][t] for t in names])
            for name, c in zip(names, rng.multinomial(n, p / p.sum())):
                for _ in range(int(c)):
                    text = _text(rng, name, noise)
                    records.append(_record(i, source, name, text, _date(rng, y)))
                    i += 1
    return records
