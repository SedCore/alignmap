"""Corpus I/O and cleaning.

Records use the schema of the AI Alignment Research Dataset (ARD,
https://huggingface.co/datasets/StampyAI/alignment-research-dataset):

    id, source, title, authors, text, url, date_published

plus optional source-specific keys (``abstract`` for arXiv; ``tags``, ``karma`` ... for
LessWrong / Alignment Forum / EA Forum). Output of ``alignmap fetch`` (arXiv API) uses the
same schema, so every source can be mixed in one run.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

CORE_KEYS = ("id", "source", "title", "authors", "text", "url", "date_published")
# Extra keys carried into the DataFrame; everything else (full texts, bibliographies...) is dropped
# on purpose so that ~450 MB of JSONL does not become several GB in memory.
KEEP_EXTRA = (
    "categories", "primary_category", "karma", "votes", "words", "comment_count", "modified_at", "_topic",
)
# When the same work appears in several sources (an arXiv paper cross-posted to the Alignment
# Forum, an Alignment Forum post that is also on LessWrong) the most specific source wins.
SOURCE_PRIORITY = {"arxiv": 0, "alignmentforum": 1, "distill": 2, "eaforum": 3, "lesswrong": 4}
# Sources are analysed in groups so that a change in the *mix* of venues is not mistaken for a
# change in topics (see trends.shift_table_adjusted).
SOURCE_GROUPS = {
    "alignmentforum": "forum", "lesswrong": "forum", "eaforum": "forum",
    "arxiv": "papers", "distill": "papers",
    "blogs": "blogs",
    "synthetic-forum": "forum", "synthetic-papers": "papers",
}
MIN_ABSTRACT_CHARS = 100

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def iter_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc


def read_jsonl(path) -> list[dict]:
    return list(iter_jsonl(path))


def iter_records(paths: Iterable[Path]) -> Iterator[dict]:
    for p in paths:
        yield from iter_jsonl(p)


def write_jsonl(records: Iterable[dict], path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def collect_inputs(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files, directories (recursively) and glob patterns into .jsonl files."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(p.rglob("*.jsonl")))
        elif p.is_file():
            found.append(p)
        else:
            matches = sorted(Path(m) for m in glob.glob(str(raw), recursive=True))
            if not matches:
                raise FileNotFoundError(f"No input matches {raw!s}")
            found.extend(m for m in matches if m.is_file())
    if not found:
        raise FileNotFoundError("No .jsonl input files found")
    return found


def _clean(value) -> str:
    return _WS.sub(" ", str(value or "")).strip()


def _norm_title(title: str) -> str:
    return _NON_ALNUM.sub(" ", title.lower()).strip()


def _authors(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(a).strip() for a in value if str(a).strip()]


def _tag_name(tag) -> str:
    if isinstance(tag, dict):  # some exports store tags as {"name": ...} objects
        tag = tag.get("name") or tag.get("tag") or ""
    return str(tag).strip()


def _tags(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        value = [value]
    names = (_tag_name(t) for t in value)
    return [n for n in names if n]


def build_corpus(
    records: Iterable[dict],
    *,
    sources: Iterable[str] | None = None,
    min_chars: int = 200,
    max_chars: int = 3000,
) -> tuple[pd.DataFrame, dict]:
    """Clean raw records into a DataFrame ready for embedding.

    * ``sources`` keeps only those ``source`` values (None keeps all).
    * The embedded body is the abstract when the record has one (arXiv, Distill), otherwise the
      first ``max_chars`` characters of the text.
    * Records with no parsable date, too little text, or duplicated id / title are dropped; for
      duplicates the more specific source wins (see SOURCE_PRIORITY).
    """
    keep_sources = set(sources) if sources else None
    stats: dict = {
        "n_input": 0, "dropped_source": 0, "dropped_no_date": 0, "dropped_short": 0, "dropped_duplicate": 0,
    }
    rows: list[dict] = []

    for order, rec in enumerate(records):
        stats["n_input"] += 1
        source = str(rec.get("source") or "unknown")
        if keep_sources is not None and source not in keep_sources:
            stats["dropped_source"] += 1
            continue

        ts = pd.to_datetime(rec.get("date_published"), utc=True, errors="coerce")
        if pd.isna(ts):
            stats["dropped_no_date"] += 1
            continue

        title = _clean(rec.get("title"))
        abstract = _clean(rec.get("abstract"))
        if len(abstract) >= MIN_ABSTRACT_CHARS:
            body = abstract[:max_chars]
        else:
            body = _clean(str(rec.get("text") or "")[: max_chars * 2])[:max_chars]
        if len(body) < min_chars:
            stats["dropped_short"] += 1
            continue

        row = {k: rec[k] for k in KEEP_EXTRA if k in rec}
        row.update(
            id=str(rec.get("id") or hashlib.sha1((title + body[:200]).encode()).hexdigest()[:16]),
            source=source,
            group=SOURCE_GROUPS.get(source, source),
            title=title,
            authors=_authors(rec.get("authors")),
            tags=_tags(rec.get("tags")),
            text=body,
            url=str(rec.get("url") or ""),
            date=ts,
            year=int(ts.year),
            doc=f"{title}. {body}" if title else body,
            _order=order,
        )
        rows.append(row)

    # De-duplicate, letting the more specific source win.
    rows.sort(key=lambda r: (SOURCE_PRIORITY.get(r["source"], 9), r["_order"]))
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    kept: list[dict] = []
    for r in rows:
        norm = _norm_title(r["title"])
        key = norm if len(norm) >= 15 else ""  # very short titles are not reliable keys
        if r["id"] in seen_ids or (key and key in seen_titles):
            stats["dropped_duplicate"] += 1
            continue
        seen_ids.add(r["id"])
        if key:
            seen_titles.add(key)
        kept.append(r)

    if not kept:
        raise ValueError(f"No usable documents after cleaning (stats: {stats})")

    df = pd.DataFrame(kept).sort_values(["date", "_order"], kind="stable").drop(columns="_order").reset_index(drop=True)
    stats["n_kept"] = int(len(df))
    stats["by_source"] = {str(k): int(v) for k, v in df["source"].value_counts().items()}
    stats["by_group"] = {str(k): int(v) for k, v in df["group"].value_counts().items()}
    stats["by_year"] = {int(k): int(v) for k, v in df["year"].value_counts().sort_index().items()}
    stats["last_date_by_source"] = {str(s): d.date().isoformat() for s, d in df.groupby("source")["date"].max().items()}
    return df, stats


def docs_for_embedding(df: pd.DataFrame, fmt: str = "plain") -> list[str]:
    """Text handed to the embedder. ``specter`` reproduces SPECTER's ``title[SEP]abstract`` input."""
    if fmt == "plain":
        return df["doc"].tolist()
    if fmt == "specter":
        return [f"{t}[SEP]{x}" for t, x in zip(df["title"], df["text"])]
    raise ValueError(f"Unknown doc format {fmt!r} (choose 'plain' or 'specter')")


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = sources, columns = years, values = number of papers kept."""
    return pd.crosstab(df["source"], df["year"])


def coverage_warnings(df: pd.DataFrame, gap_days: int = 180) -> list[str]:
    """Flag sources whose newest paper is much older than the newest paper overall.

    A source that stops being updated makes its topics look like they declined.
    """
    newest = df["date"].max()
    out = []
    for source, last in df.groupby("source")["date"].max().items():
        if (newest - last).days > gap_days:
            out.append(
                f"`{source}` has no papers after {last.date().isoformat()} "
                f"(newest source: {newest.date().isoformat()}); its absence can mimic a decline."
            )
    return out
