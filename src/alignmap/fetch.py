"""Fetch alignment-related papers from the public arXiv API.

Design notes
------------
* Standard library only (urllib + ElementTree) - no scraping dependency.
* One query per calendar year, so every year gets the same term set and the same
  per-year cap; results are cached per (term-set, year) as JSONL.
* Polite by default: >= 3 s between requests (arXiv's published guidance),
  exponential back-off on HTTP 429 / 5xx, descriptive User-Agent.
* Network access is injectable (``http_get``), which is how the tests run offline.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .corpus import write_jsonl

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
    "ax": "http://arxiv.org/schemas/atom",
}

DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CY", "stat.ML"]

# Phrases that already named recognisable subfields by 2022. Keeping the *same* list
# for every year matters: a term that only exists after 2023 would make a subfield look
# "emerging" purely because of how the corpus was queried.
CORE_TERMS = [
    "AI alignment",
    "AI safety",
    "value alignment",
    "scalable oversight",
    "reward hacking",
    "specification gaming",
    "mechanistic interpretability",
    "reinforcement learning from human feedback",
    "corrigibility",
    "goal misgeneralization",
    "inner alignment",
    "deceptive alignment",
    "AI existential risk",
]

# Adds vocabulary that is (mostly) post-2022. Useful for coverage, but see above.
EXTENDED_TERMS = CORE_TERMS + [
    "weak-to-strong generalization",
    "alignment faking",
    "emergent misalignment",
    "AI scheming",
    "AI control",
    "chain-of-thought monitorability",
    "sycophancy",
    "constitutional AI",
    "model organisms of misalignment",
    "automated alignment research",
]

HttpGet = Callable[[str], str]


def terms_tag(terms: Sequence[str]) -> str:
    if list(terms) == CORE_TERMS:
        return "core"
    if list(terms) == EXTENDED_TERMS:
        return "extended"
    return "custom-" + hashlib.sha1("|".join(terms).encode()).hexdigest()[:8]


def build_query(
    terms: Sequence[str],
    start: date,
    end: date,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
) -> str:
    term_q = " OR ".join(f'all:"{t}"' for t in terms)
    cat_q = " OR ".join(f"cat:{c}" for c in categories)
    return (
        f"({term_q}) AND ({cat_q}) AND "
        f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    )


def _url(query: str, start: int, n: int) -> str:
    params = {
        "search_query": query,
        "start": start,
        "max_results": n,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    return ARXIV_API + "?" + urllib.parse.urlencode(params)


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_feed(xml_text: str) -> tuple[list[dict], int]:
    """Parse an arXiv Atom response into (records, totalResults)."""
    root = ET.fromstring(xml_text)
    total = int(root.findtext("os:totalResults", default="0", namespaces=NS) or 0)
    records: list[dict] = []
    for entry in root.findall("a:entry", NS):
        id_url = (entry.findtext("a:id", default="", namespaces=NS) or "").strip()
        if "/api/errors" in id_url or "/abs/" not in id_url:
            continue  # arXiv reports API errors as a pseudo-entry
        arxiv_id = re.sub(r"v\d+$", "", id_url.split("/abs/", 1)[1])
        primary = entry.find("ax:primary_category", NS)
        records.append(
            {
                "id": f"arxiv:{arxiv_id}",
                "source": "arxiv",
                "title": _clean(entry.findtext("a:title", default="", namespaces=NS)),
                "authors": [
                    _clean(a.findtext("a:name", default="", namespaces=NS))
                    for a in entry.findall("a:author", NS)
                ],
                "text": _clean(entry.findtext("a:summary", default="", namespaces=NS)),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "date_published": _clean(entry.findtext("a:published", default="", namespaces=NS)),
                "categories": [c.get("term") for c in entry.findall("a:category", NS)],
                "primary_category": primary.get("term") if primary is not None else None,
            }
        )
    return records, total


def default_http_get(url: str, *, retries: int = 5, timeout: int = 60, sleep=time.sleep) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "alignmap/0.1 (research prototype)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            code = getattr(exc, "code", None)
            retriable = code is None or code == 429 or code >= 500
            if not retriable or attempt == retries - 1:
                raise
            sleep(min(60, 5 * 2**attempt))
    raise RuntimeError("unreachable")


def fetch_window(
    query: str,
    *,
    max_results: int = 3000,
    page_size: int = 100,
    delay: float = 3.1,
    http_get: HttpGet | None = None,
    sleep=time.sleep,
    log=print,
) -> list[dict]:
    """Page through one arXiv query, returning at most ``max_results`` records."""
    http_get = http_get or default_http_get
    out: list[dict] = []
    start = 0
    while len(out) < max_results:
        n = min(page_size, max_results - len(out))
        records, total = parse_feed(http_get(_url(query, start, n)))
        out.extend(records)
        start += len(records)
        log(f"    {len(out)}/{min(total, max_results)}")
        if not records or start >= total:
            break
        sleep(delay)
    return out


def fetch_years(
    years: Sequence[int],
    out_dir,
    terms: Sequence[str],
    *,
    max_per_year: int = 3000,
    refresh: bool = False,
    page_size: int = 100,
    delay: float = 3.1,
    http_get: HttpGet | None = None,
    sleep=time.sleep,
    log=print,
) -> list[Path]:
    """Fetch (or reuse cached) per-year JSONL files; returns their paths.

    The current calendar year is always re-fetched because it is still growing.
    A manifest with the exact query terms and retrieval time is written next to the data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = terms_tag(terms)
    manifest_path = out_dir / f"manifest_{tag}.json"
    manifest = {"tag": tag, "terms": list(terms), "categories": DEFAULT_CATEGORIES, "years": {}}
    if manifest_path.exists():
        try:
            manifest["years"] = json.loads(manifest_path.read_text(encoding="utf-8")).get("years", {})
        except json.JSONDecodeError:
            pass

    this_year = date.today().year
    paths: list[Path] = []
    for y in years:
        path = out_dir / f"arxiv_{tag}_{y}.jsonl"
        if path.exists() and not refresh and y < this_year:
            log(f"{y}: cached ({path.name})")
            paths.append(path)
            continue
        log(f"{y}: fetching from arXiv ...")
        query = build_query(terms, date(y, 1, 1), date(y, 12, 31))
        records = fetch_window(
            query,
            max_results=max_per_year,
            page_size=page_size,
            delay=delay,
            http_get=http_get,
            sleep=sleep,
            log=log,
        )
        write_jsonl(records, path)
        manifest["years"][str(y)] = {
            "n": len(records),
            "capped": len(records) >= max_per_year,
            "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        paths.append(path)
        sleep(delay)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
