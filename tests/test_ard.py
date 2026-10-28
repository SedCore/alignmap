import hashlib
import json

import pytest

from alignmap import ard
from alignmap.ard import download_ard

PAYLOADS = {
    "alignmentforum": b'{"id": "1", "source": "alignmentforum"}\n' * 50,
    "arxiv": b'{"id": "2", "source": "arxiv"}\n' * 30,
}


class FakeResponse:
    def __init__(self, payload, *, declared=None, commit="abc123"):
        self._data = payload
        self._pos = 0
        self.headers = {"Content-Length": str(len(payload) if declared is None else declared), "X-Repo-Commit": commit}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


class FakeOpener:
    def __init__(self, payloads=PAYLOADS, **kw):
        self.payloads, self.kw, self.requested = payloads, kw, []

    def __call__(self, req, timeout=None):
        self.requested.append(req.full_url)
        name = req.full_url.rsplit("/", 1)[-1].removesuffix(".jsonl")
        return FakeResponse(self.payloads[name], **self.kw)


def test_download_writes_files_and_a_manifest_with_checksums(tmp_path):
    opener = FakeOpener()
    paths = download_ard(["alignmentforum", "arxiv"], tmp_path, opener=opener, log=lambda *_: None, chunk_size=64)
    assert [p.name for p in paths] == ["alignmentforum.jsonl", "arxiv.jsonl"]
    assert opener.requested == [ard.BASE_URL.format(name=n) for n in ("alignmentforum", "arxiv")]
    assert (tmp_path / "arxiv.jsonl").read_bytes() == PAYLOADS["arxiv"]
    assert not list(tmp_path.glob("*.part"))

    manifest = json.loads((tmp_path / "ard_manifest.json").read_text())
    entry = manifest["alignmentforum"]
    assert entry["sha256"] == hashlib.sha256(PAYLOADS["alignmentforum"]).hexdigest()
    assert entry["bytes"] == len(PAYLOADS["alignmentforum"]) and entry["repo_commit"] == "abc123"
    assert entry["url"].endswith("/alignmentforum.jsonl") and entry["retrieved"]


def test_existing_files_are_skipped_unless_refresh(tmp_path):
    download_ard(["arxiv"], tmp_path, opener=FakeOpener(), log=lambda *_: None)
    again = FakeOpener()
    download_ard(["arxiv"], tmp_path, opener=again, log=lambda *_: None)
    assert again.requested == []
    download_ard(["arxiv"], tmp_path, opener=again, refresh=True, log=lambda *_: None)
    assert len(again.requested) == 1


def test_manifest_keeps_entries_from_earlier_runs(tmp_path):
    download_ard(["arxiv"], tmp_path, opener=FakeOpener(), log=lambda *_: None)
    download_ard(["alignmentforum"], tmp_path, opener=FakeOpener(), log=lambda *_: None)
    assert set(json.loads((tmp_path / "ard_manifest.json").read_text())) == {"arxiv", "alignmentforum"}


def test_truncated_download_is_rejected_and_leaves_no_file(tmp_path):
    opener = FakeOpener(declared=len(PAYLOADS["arxiv"]) + 1000)
    with pytest.raises(RuntimeError, match="incomplete download"):
        download_ard(["arxiv"], tmp_path, opener=opener, log=lambda *_: None)
    assert not (tmp_path / "arxiv.jsonl").exists() and not list(tmp_path.glob("*.part"))


def test_unknown_source_is_rejected_before_any_request(tmp_path):
    opener = FakeOpener()
    with pytest.raises(ValueError, match="Unknown ARD source"):
        download_ard(["arxiv", "not-a-source"], tmp_path, opener=opener, log=lambda *_: None)
    assert opener.requested == []


def test_default_sources_are_research_writing_with_dates():
    assert set(ard.DEFAULT_SOURCES) <= set(ard.KNOWN_SOURCES)
    assert {"alignmentforum", "lesswrong", "arxiv"} <= set(ard.DEFAULT_SOURCES)
    assert "youtube" not in ard.DEFAULT_SOURCES
