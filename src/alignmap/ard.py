"""Download the AI Alignment Research Dataset (ARD), one JSONL file per source.

The dataset (MIT licence) lives at https://huggingface.co/datasets/StampyAI/alignment-research-dataset
as one ``<source>.jsonl`` file per source. We fetch the files directly instead of going through the
``datasets`` library, because the repository uses a loading script that recent versions of
``datasets`` refuse to run. A manifest with the SHA-256 of every file pins the exact data used.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

BASE_URL = "https://huggingface.co/datasets/StampyAI/alignment-research-dataset/resolve/main/{name}.jsonl"
KNOWN_SOURCES = (
    "agentmodels", "agisf", "aisafety.info", "alignmentforum", "arbital", "arxiv", "blogs",
    "distill", "eaforum", "lesswrong", "special_docs", "youtube",
)
# Research writing with dates and authors; leaves out transcripts, FAQs and a mixed curated bag.
DEFAULT_SOURCES = ("alignmentforum", "lesswrong", "eaforum", "arxiv", "blogs", "distill")


def download_ard(
    sources: Sequence[str] = DEFAULT_SOURCES,
    out_dir="data/ard",
    *,
    refresh: bool = False,
    opener=None,
    log=print,
    chunk_size: int = 1 << 20,
) -> list[Path]:
    unknown = [s for s in sources if s not in KNOWN_SOURCES]
    if unknown:
        raise ValueError(f"Unknown ARD source(s) {unknown}; known: {', '.join(KNOWN_SOURCES)}")
    opener = opener or urllib.request.urlopen
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "ard_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    paths: list[Path] = []
    for name in sources:
        dest = out_dir / f"{name}.jsonl"
        paths.append(dest)
        if dest.exists() and not refresh:
            log(f"{name}: already downloaded ({dest.name})")
            continue
        url = BASE_URL.format(name=name)
        log(f"{name}: downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "alignmap/0.2 (research)"})
        tmp = dest.with_suffix(".jsonl.part")
        sha = hashlib.sha256()
        done = 0
        next_report = 25 << 20
        with opener(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            commit = resp.headers.get("X-Repo-Commit")
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    sha.update(chunk)
                    done += len(chunk)
                    if done >= next_report:
                        log(f"    {done >> 20} MB")
                        next_report += 25 << 20
        if total and done != total:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: incomplete download ({done} of {total} bytes); try again")
        tmp.replace(dest)
        manifest[name] = {
            "url": url,
            "bytes": done,
            "sha256": sha.hexdigest(),
            "repo_commit": commit,
            "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        log(f"    {done / 1e6:.1f} MB")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
