"""arXiv fetching, tested offline against hand-written Atom responses."""

import urllib.parse
from datetime import date

import pytest

from alignmap import fetch
from alignmap.corpus import read_jsonl

ENTRY = """
  <entry>
    <id>http://arxiv.org/abs/{aid}v2</id>
    <updated>2023-01-01T00:00:00Z</updated>
    <published>{published}</published>
    <title>{title}</title>
    <summary>  {summary}
    </summary>
    <author><name>Ada Example</name></author>
    <author><name>Grace Sample</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.CY" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CY" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>"""


def make_feed(entries: list[str], total: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        f"<opensearch:totalResults>{total}</opensearch:totalResults>"
        + "".join(entries)
        + "</feed>"
    )


def entry(i: int, **kw) -> str:
    return ENTRY.format(
        aid=kw.get("aid", f"2301.{i:05d}"),
        published=kw.get("published", "2023-01-11T17:59:13Z"),
        title=kw.get("title", f"Paper {i}"),
        summary=kw.get("summary", f"Abstract {i}."),
    )


def test_parse_feed_extracts_fields():
    xml = make_feed(
        [entry(1, aid="2301.04567", title="Scaling Reward\n   Models:  An Empirical Study")],
        total=1,
    )
    records, total = fetch.parse_feed(xml)
    assert total == 1
    (rec,) = records
    assert rec["id"] == "arxiv:2301.04567"  # version suffix stripped
    assert rec["title"] == "Scaling Reward Models: An Empirical Study"
    assert rec["text"] == "Abstract 1."
    assert rec["authors"] == ["Ada Example", "Grace Sample"]
    assert rec["url"] == "https://arxiv.org/abs/2301.04567"
    assert rec["date_published"] == "2023-01-11T17:59:13Z"
    assert rec["categories"] == ["cs.CY", "cs.AI"]
    assert rec["primary_category"] == "cs.CY"
    assert rec["source"] == "arxiv"


def test_parse_feed_handles_old_style_ids():
    records, _ = fetch.parse_feed(make_feed([entry(1, aid="hep-th/9901001")], total=1))
    assert records[0]["id"] == "arxiv:hep-th/9901001"


def test_parse_feed_skips_api_error_entries():
    error = (
        "<entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_1234</id>"
        "<title>Error</title><summary>bad request</summary></entry>"
    )
    records, _ = fetch.parse_feed(make_feed([error], total=1))
    assert records == []


def test_build_query_has_phrases_categories_and_dates():
    q = fetch.build_query(["AI alignment", "reward hacking"], date(2022, 1, 1), date(2022, 12, 31))
    assert 'all:"AI alignment" OR all:"reward hacking"' in q
    assert "cat:cs.AI" in q and "cat:stat.ML" in q
    assert "submittedDate:[202201010000 TO 202212312359]" in q


def test_terms_tag_distinguishes_term_sets():
    assert fetch.terms_tag(fetch.CORE_TERMS) == "core"
    assert fetch.terms_tag(fetch.EXTENDED_TERMS) == "extended"
    assert fetch.terms_tag(["a"]).startswith("custom-")
    assert fetch.terms_tag(["a"]) != fetch.terms_tag(["b"])


def fake_arxiv(total: int, calls: list):
    """An http_get that serves `total` fake papers, honouring start / max_results."""

    def http_get(url: str) -> str:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        start, n = int(qs["start"][0]), int(qs["max_results"][0])
        calls.append((start, n))
        return make_feed([entry(i) for i in range(start, min(start + n, total))], total)

    return http_get


def test_fetch_window_paginates_and_sleeps_between_pages():
    calls, sleeps = [], []
    out = fetch.fetch_window(
        "q", max_results=100, page_size=2, delay=3.1,
        http_get=fake_arxiv(5, calls), sleep=sleeps.append, log=lambda *_: None,
    )
    assert len(out) == 5
    assert calls == [(0, 2), (2, 2), (4, 2)]
    assert sleeps == [3.1, 3.1]  # between pages, not after the last one


def test_fetch_window_respects_max_results():
    calls = []
    out = fetch.fetch_window(
        "q", max_results=3, page_size=2,
        http_get=fake_arxiv(50, calls), sleep=lambda s: None, log=lambda *_: None,
    )
    assert len(out) == 3
    assert calls == [(0, 2), (2, 1)]


def test_fetch_years_writes_manifest_and_reuses_cache(tmp_path):
    calls = []
    kwargs = dict(
        terms=["AI alignment"], max_per_year=10, page_size=5,
        http_get=fake_arxiv(4, calls), sleep=lambda s: None, log=lambda *_: None,
    )
    paths = fetch.fetch_years([2022], tmp_path, **kwargs)
    assert len(read_jsonl(paths[0])) == 4
    n_calls = len(calls)
    assert n_calls > 0
    assert (tmp_path / f"manifest_{fetch.terms_tag(['AI alignment'])}.json").exists()

    fetch.fetch_years([2022], tmp_path, **kwargs)  # past year + cache present -> no new requests
    assert len(calls) == n_calls

    fetch.fetch_years([2022], tmp_path, refresh=True, **kwargs)
    assert len(calls) > n_calls


def test_http_get_retries_then_gives_up(monkeypatch):
    import urllib.error

    attempts, waits = [], []

    def boom(req, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError(req.full_url, 503, "unavailable", {}, None)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", boom)
    with pytest.raises(urllib.error.HTTPError):
        fetch.default_http_get("http://x", retries=3, sleep=waits.append)
    assert len(attempts) == 3
    assert waits == [5, 10]  # exponential back-off, no sleep after the final failure


def test_http_get_does_not_retry_client_errors(monkeypatch):
    import urllib.error

    attempts = []

    def not_found(req, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError(req.full_url, 404, "nope", {}, None)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", not_found)
    with pytest.raises(urllib.error.HTTPError):
        fetch.default_http_get("http://x", retries=5, sleep=lambda s: None)
    assert len(attempts) == 1
