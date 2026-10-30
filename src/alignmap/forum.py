"""Fetch recent posts from the LessWrong, Alignment Forum and EA Forum GraphQL APIs.

Why this exists: the public AI Alignment Research Dataset stops in late 2023, so anything about
2024-2026 has to come from the sources themselves. The three forums run the same software
(ForumMagnum) and expose the same public GraphQL endpoint, so one fetcher serves all of them.

Design notes
------------
* Standard library only; network access is injectable (``post``), which is how the tests run offline.
* The posts' plain text comes straight from the API (``contents.plaintextMainText``), so there is no
  HTML or MathJax to strip.
* The three sites do not all run the same version of the schema. Rather than guess, the query
  *repairs itself*: when the server answers ``Cannot query field "x" on type "T"`` the field is
  removed (or replaced by a fallback, e.g. ``html`` for ``plaintextMainText``) and the request is
  repeated. Only ``_id``, ``title`` and ``postedAt`` are indispensable. The fields that were not
  available are written to the manifest.
* Paging is newest-first. Two strategies are supported (``offset``, and ``before`` = date cursor);
  with ``paging="auto"`` a small probe at the start checks which one the server actually honours,
  because a server that silently ignores a paging argument would otherwise return the same page
  forever. ``offset`` is preferred: it is the one verified against a live forum.
* Polite by default: a pause between requests, back-off on HTTP 429 / 5xx, a descriptive User-Agent.
* LessWrong and the EA Forum carry much that is not about AI. By default a post is kept when its
  tags or title mention AI (see ``AI_PATTERN``), or when it is on the Alignment Forum. This filter is
  an assumption that shapes the corpus; ``--all-posts`` switches it off.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .corpus import clean_text

FORUMS = {
    "alignmentforum": "https://www.alignmentforum.org",
    "lesswrong": "https://www.lesswrong.com",
    "eaforum": "https://forum.effectivealtruism.org",
}
USER_AGENT = "alignmap/0.2 (research prototype; +https://github.com/sedcore/alignmap)"

# Tags or titles that make a post "about AI". Deliberately narrow: "existential risk" is left out
# because on the EA Forum it is as often about pandemics or nuclear war as about AI.
AI_PATTERN = re.compile(
    r"\bAIs?\b|alignment|interpretab|corrigib|mesa-?optim|language models?|machine learning|\bML\b|"
    r"\bRLHF\b|goodhart|deceptive|scaling laws?|scalable oversight|superintelligen|\bLLMs?\b|"
    r"\btransformers?\b|neural net|\bGPT|\bAGI\b|\bTAI\b|reward (?:hacking|model)|situational awareness",
    re.I,
)

DEFAULT_FIELDS = {
    "Post": ["_id", "title", "postedAt", "pageUrl", "baseScore", "commentCount", "af", "user", "contents", "tags"],
    "User": ["displayName"],
    "Revision": ["plaintextMainText", "wordCount"],
    "Tag": ["name"],
}
NESTED = {"user": "User", "contents": "Revision", "tags": "Tag"}  # field name -> its GraphQL type
REQUIRED = {"_id", "title", "postedAt"}
FALLBACKS = {"plaintextMainText": ["html", "markdown"]}  # tried in this order if the field is missing

_FIELD_ERROR = re.compile(r'Cannot query field "(\w+)" on type "(\w+)"')

PostFn = Callable[[str, dict], dict]


# --- transport -----------------------------------------------------------------------------------


def default_post(url: str, payload: dict, *, retries: int = 5, timeout: int = 180, sleep=time.sleep) -> dict:
    """POST a GraphQL request and return the parsed JSON body.

    GraphQL validation errors come back as HTTP 400 *with* a JSON body, which the caller needs, so
    that case is returned rather than raised. 429 and 5xx are retried with back-off.
    """
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            if exc.code == 400:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    raise RuntimeError(f"HTTP 400 from {url}: {text[:300]}") from exc
            retriable = exc.code == 429 or exc.code >= 500
            if not retriable or attempt == retries - 1:
                hint = " (the site may be blocking this client; try a longer --delay)" if exc.code == 403 else ""
                raise RuntimeError(f"HTTP {exc.code} from {url}{hint}: {text[:300]}") from exc
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        sleep(min(120, 5 * 2**attempt))
    raise RuntimeError("unreachable")


# --- the self-repairing query --------------------------------------------------------------------


def render_fields(fields: dict[str, list[str]]) -> str:
    parts = []
    for name in fields["Post"]:
        if name in NESTED:
            sub = fields.get(NESTED[name], [])
            if sub:
                parts.append(f"{name}{{{' '.join(sub)}}}")
        else:
            parts.append(name)
    return " ".join(parts)


def render_terms(limit: int, offset: int | None = None, before: str | None = None) -> str:
    terms = f'view: "new", limit: {int(limit)}'
    if offset is not None:
        terms += f", offset: {int(offset)}"
    if before is not None:
        terms += f", before: {json.dumps(before)}"
    return terms


class ForumClient:
    """One forum's GraphQL endpoint, with a query that drops the fields the server does not know."""

    def __init__(self, source: str, post: PostFn | None = None, *, sleep=time.sleep, log=print):
        if source not in FORUMS:
            raise ValueError(f"Unknown forum {source!r}; choose from {', '.join(FORUMS)}")
        self.source = source
        self.origin = FORUMS[source]
        self.url = self.origin + "/graphql"
        self.post = post or default_post
        self.fields = {k: list(v) for k, v in DEFAULT_FIELDS.items()}
        self.dropped: list[str] = []
        self.log = log

    def _repair(self, messages: list[str]) -> bool:
        """Edit the field set after a 'Cannot query field' error. False if nothing could be done."""
        changed = False
        for message in messages:
            m = _FIELD_ERROR.search(message)
            if not m:
                continue
            name, typ = m.group(1), m.group(2)
            if typ == "Post" and name in REQUIRED:
                raise RuntimeError(f"{self.url} has no field {name!r} on Post; this fetcher cannot work with it")
            fields = self.fields.get(typ)
            if fields is None or name not in fields:
                continue
            fields.remove(name)
            self.dropped.append(f"{typ}.{name}")
            for alt in FALLBACKS.get(name, []):
                if f"{typ}.{alt}" not in self.dropped and alt not in fields:
                    fields.append(alt)
                    break
            self.log(f"    [{self.source}] the server has no {typ}.{name}; adjusting the query")
            changed = True
        return changed

    def page(self, limit: int, offset: int | None = None, before: str | None = None) -> list[dict]:
        for _ in range(30):  # every pass removes at least one field, so this always terminates
            query = f"{{ posts(input: {{terms: {{{render_terms(limit, offset, before)}}}}}) {{ results {{ {render_fields(self.fields)} }} }} }}"
            reply = self.post(self.url, {"query": query})
            errors = reply.get("errors") or []
            data = reply.get("data") or {}
            posts = data.get("posts") if isinstance(data, dict) else None
            if errors and not posts:
                messages = [str(e.get("message", e)) if isinstance(e, dict) else str(e) for e in errors]
                if self._repair(messages):
                    continue
                raise RuntimeError(f"{self.url} returned an error: {'; '.join(messages)[:500]}")
            if posts is None:
                raise RuntimeError(f"{self.url} returned no 'posts' data: {str(reply)[:300]}")
            return list(posts.get("results") or [])
        raise RuntimeError(f"Could not build a query that {self.url} accepts (dropped: {self.dropped})")


# --- paging --------------------------------------------------------------------------------------


def _ids(rows: Iterable[dict]) -> list[str]:
    return [r["_id"] for r in rows]


def detect_paging(client: ForumClient) -> str:
    """Return "offset" or "before", whichever the server demonstrably honours (offset is preferred).

    Each strategy is tested by asking for a window that must differ from the newest posts; a server
    that ignores the argument returns the newest posts again and fails the test. A server that
    answers a probe with an error simply fails that probe.

    ``offset`` is preferred because it was checked against the live EA Forum, while a ``before``
    cursor with a full timestamp was refused there (a bare date is accepted, but a date cursor then
    skips the posts that share the day of the last one on a page).
    """
    head = client.page(limit=4)
    if len(head) < 4:
        return "offset"  # almost nothing to page through, so either is fine
    newest = _ids(head)

    try:
        if _ids(client.page(limit=2, offset=1)) == newest[1:3]:
            return "offset"
    except RuntimeError:
        pass

    try:
        older = client.page(limit=2, before=head[2]["postedAt"])
    except RuntimeError:
        older = []
    if older and not set(_ids(older)) & set(newest[:2]) and all(
        str(r["postedAt"]) <= str(head[2]["postedAt"]) for r in older
    ):
        return "before"
    raise RuntimeError(
        f"{client.url} honours neither 'offset' nor 'before' in the 'new' view, so the posts cannot be paged "
        "through. Run the probes by hand and open an issue with the output."
    )


def _parse_time(value) -> datetime | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts.to_pydatetime()


def to_record(source: str, origin: str, row: dict, *, max_text_chars: int) -> dict | None:
    """One API row -> one record in the schema the rest of the package reads."""
    title = re.sub(r"\s+", " ", str(row.get("title") or "")).strip()
    if not title:
        return None
    contents = row.get("contents") or {}
    text = contents.get("plaintextMainText")
    if text is None:  # older schema: only HTML or Markdown, which must be cleaned before it is cut
        raw = contents.get("html") or contents.get("markdown") or ""
        text = clean_text(raw, window=max_text_chars * 6)
    text = str(text)[:max_text_chars]
    tags = [t["name"] for t in (row.get("tags") or []) if isinstance(t, dict) and t.get("name")]
    user = row.get("user") or {}
    authors = [user["displayName"]] if isinstance(user, dict) and user.get("displayName") else []
    rec = {
        "id": f"{source}:{row['_id']}",
        "source": source,
        "title": title,
        "authors": authors,
        "text": text,
        "url": row.get("pageUrl") or f"{origin}/posts/{row['_id']}",
        "date_published": row["postedAt"],
        "tags": tags,
    }
    for key, name in (("karma", "baseScore"), ("comment_count", "commentCount")):
        if row.get(name) is not None:
            rec[key] = row[name]
    if contents.get("wordCount") is not None:
        rec["words"] = contents["wordCount"]
    return rec


def _next_window(current: str, last_ts: datetime | None) -> str:
    """The date window that starts again just after the last post seen (a day of overlap, never a gap)."""
    if last_ts is None:
        raise RuntimeError("cannot move the paging window: no dated post seen yet")
    nxt = (last_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if nxt >= current:  # one day holds more posts than the offset cap allows: give up loudly
        raise RuntimeError(f"more than the server's offset cap of posts on {last_ts:%Y-%m-%d}; lower --page-size")
    return nxt


def is_about_ai(rec: dict, row: dict) -> bool:
    return bool(row.get("af")) or any(AI_PATTERN.search(t) for t in rec["tags"]) or bool(AI_PATTERN.search(rec["title"]))


def fetch_source(
    source: str,
    years: tuple[int, int],
    *,
    post: PostFn | None = None,
    paging: str = "auto",
    page_size: int = 50,
    delay: float = 1.0,
    all_posts: bool = False,
    max_text_chars: int = 15000,
    max_posts: int = 200_000,
    max_skip: int = 1000,
    sleep=time.sleep,
    log=print,
) -> tuple[list[dict], dict]:
    """Page through one forum, newest first, until the posts are older than ``years[0]``.

    In ``offset`` mode the offset is counted inside a date window (``before: "YYYY-MM-DD"``): the
    servers refuse large offsets ("Exceeded maximum value for skip", seen above ~2000 on the
    Alignment Forum), so when the offset would pass ``max_skip`` the window moves to the day after
    the last post seen and the offset restarts at 0. The overlap of up to a day is removed by id.
    A bare date is what the live servers accept in ``before``.

    Returns the kept records and a dict of what happened (counts, strategy, dropped fields).
    """
    lo, hi = years
    start = datetime(lo, 1, 1, tzinfo=timezone.utc)
    end = datetime(hi + 1, 1, 1, tzinfo=timezone.utc)
    client = ForumClient(source, post, sleep=sleep, log=log)
    mode = detect_paging(client) if paging == "auto" else paging
    if mode not in ("before", "offset"):
        raise ValueError(f"paging must be auto, before or offset, not {paging!r}")
    log(f"  {source}: paging by '{mode}', {lo}-{hi}")

    counts: Counter = Counter()
    seen: set[str] = set()
    kept: list[dict] = []
    last_ts: datetime | None = None
    offset = 0
    if mode == "before":
        cursor = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    else:
        cursor = end.strftime("%Y-%m-%d")  # date window for offset paging; starts at the end of the range
    prev_ids: list[str] = []
    while len(seen) < max_posts:
        if mode == "offset" and offset + page_size > max_skip and prev_ids:
            cursor, offset = _next_window(cursor, last_ts), 0
            counts["windows"] += 1
        try:
            rows = client.page(page_size, offset=offset if mode == "offset" else None, before=cursor)
        except RuntimeError as exc:
            if mode == "offset" and "skip" in str(exc).lower() and offset > 0:
                cursor, offset = _next_window(cursor, last_ts), 0  # the server's cap is lower than max_skip
                counts["windows"] += 1
                continue
            raise
        if not rows:
            break
        if _ids(rows) == prev_ids:  # the server ignored the paging argument: same page again
            log(f"  {source}: the same page came back twice; stopping")
            counts["stalled"] += 1
            break
        prev_ids = _ids(rows)
        fresh = [r for r in rows if r.get("_id") not in seen]
        last_ts = _parse_time(rows[-1].get("postedAt")) or last_ts
        reached_start = False
        for row in fresh:
            seen.add(row["_id"])
            ts = _parse_time(row.get("postedAt"))
            if ts is None:
                counts["no_date"] += 1
            elif ts >= end:
                counts["newer_than_range"] += 1
            elif ts < start:
                counts["older_than_range"] += 1
                reached_start = True
            elif source == "alignmentforum" and row.get("af") is False:
                counts["not_af"] += 1
            else:
                rec = to_record(source, client.origin, row, max_text_chars=max_text_chars)
                if rec is None:
                    counts["no_title"] += 1
                elif not all_posts and source != "alignmentforum" and not is_about_ai(rec, row):  # AF is all on topic
                    counts["not_about_ai"] += 1
                else:
                    kept.append(rec)
        offset += len(rows)
        if mode == "before":
            cursor = rows[-1].get("postedAt")  # in offset mode a cursor would skip posts: never send both
        log(f"    {source}: {len(seen):,} posts looked at, {len(kept):,} kept (reached {str(rows[-1].get('postedAt'))[:10]})")
        if reached_start:
            break
        sleep(delay)

    info = {
        "paging": mode, "n_seen": len(seen), "n_kept": len(kept), "counts": dict(counts),
        "dropped_fields": client.dropped, "ai_filter": not all_posts and source != "alignmentforum",
    }
    return kept, info


def fetch_forums(
    sources: Iterable[str],
    years: tuple[int, int],
    out_dir,
    *,
    refresh: bool = False,
    post: PostFn | None = None,
    sleep=time.sleep,
    log=print,
    **kwargs,
) -> list[Path]:
    """Fetch each forum into ``<out_dir>/<source>_<lo>-<hi>.jsonl`` and write a manifest.

    An existing file is reused unless ``refresh`` is set (a range that includes the current year is
    still growing, so re-fetch it when you want the latest posts).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_forums.json"
    manifest: dict = {"years": list(years), "sources": {}}
    if manifest_path.exists():
        try:
            manifest["sources"] = json.loads(manifest_path.read_text(encoding="utf-8")).get("sources", {})
        except json.JSONDecodeError:
            pass

    paths: list[Path] = []
    for source in sources:
        path = out_dir / f"{source}_{years[0]}-{years[1]}.jsonl"
        if path.exists() and not refresh:
            log(f"{source}: cached ({path.name}); use --refresh to fetch again")
            paths.append(path)
            continue
        log(f"{source}: fetching ...")
        records, info = fetch_source(source, years, post=post, sleep=sleep, log=log, **kwargs)
        tmp = path.with_suffix(".jsonl.part")  # never leave a half-written file that looks finished
        with open(tmp, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)
        info["retrieved"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        info["file"] = path.name
        manifest["sources"][source] = info
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log(f"{source}: wrote {len(records):,} posts to {path}")
        paths.append(path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
