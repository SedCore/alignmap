"""Forum fetching, tested offline against a fake ForumMagnum GraphQL server."""

import io
import json
import re
import urllib.error

import pandas as pd
import pytest

from alignmap import forum
from alignmap.cli import build_parser, main
from alignmap.corpus import build_corpus, read_jsonl


class FakeForum:
    """Serves ``n`` posts, newest first, one every ``step_days`` days. Even-numbered posts are about AI."""

    def __init__(self, n=120, step_days=10, unsupported=(), honours_offset=True, honours_before=True, max_limit=None,
                 before_errors=False, max_skip=None):
        base = pd.Timestamp("2026-10-20T12:00:00Z")
        self.rows = []
        for i in range(n):
            ai = i % 2 == 0
            ts = base - pd.Timedelta(days=step_days * i)
            self.rows.append(
                {
                    "_id": f"p{i:03d}", "title": f"Post number {i}", "postedAt": ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "pageUrl": f"https://forum.test/posts/p{i:03d}", "baseScore": 10 + i, "commentCount": i,
                    "af": False, "user": {"displayName": f"Author {i}"},
                    "tags": [{"name": "AI Risk"}, {"name": "Interpretability"}] if ai else [{"name": "Gardening"}],
                    "body": f"Body of post {i} about its subject. " * 8,
                }
            )
        self.unsupported = list(unsupported)
        self.honours_offset, self.honours_before, self.max_limit = honours_offset, honours_before, max_limit
        self.before_errors = before_errors
        self.max_skip = max_skip
        self.queries: list[str] = []

    def __call__(self, url, payload):
        q = payload["query"]
        self.queries.append(q)
        bad = [(f, t) for f, t in self.unsupported if re.search(rf"\b{f}\b", q)]
        if bad:
            return {"errors": [{"message": f'Cannot query field "{f}" on type "{t}".'} for f, t in bad]}
        if self.before_errors and re.search(r'before: "\d{4}-\d\d-\d\dT', q):  # live servers refuse a full timestamp
            return {"errors": [{"message": "Invalid value for 'before'"}]}
        limit = int(re.search(r"limit: (\d+)", q).group(1))
        if self.max_limit:
            limit = min(limit, self.max_limit)
        off = re.search(r"offset: (\d+)", q)
        if self.max_skip is not None and off and int(off.group(1)) > self.max_skip:  # as the live forums do
            return {"errors": [{"message": "Exceeded maximum value for skip"}]}
        before = re.search(r'before: "([^"]+)"', q)
        rows = self.rows
        if before and self.honours_before:
            rows = [r for r in rows if r["postedAt"] < before.group(1)]
        start = int(off.group(1)) if off and self.honours_offset else 0
        out = []
        for r in rows[start : start + limit]:
            # answer only with the fields that were asked for, as a real server does
            row = {k: r[k] for k in ("_id", "title", "postedAt", "pageUrl", "baseScore", "commentCount", "af")
                   if re.search(rf"\b{k}\b", q)}
            if "user{" in q:
                row["user"] = {"displayName": r["user"]["displayName"]} if "displayName" in q else {}
            if "tags{" in q:
                row["tags"] = r["tags"] if re.search(r"\bname\b", q) else [{}] * len(r["tags"])
            contents = {}
            if "plaintextMainText" in q:
                contents["plaintextMainText"] = r["body"]
            if re.search(r"\bhtml\b", q):
                contents["html"] = "<style>.mjx-chtml {display: inline-block;}</style><p>" + r["body"] + "</p>"
            if "wordCount" in q:
                contents["wordCount"] = len(r["body"].split())
            row["contents"] = contents
            out.append(row)
        return {"data": {"posts": {"results": out}}}

    def ai_ids_between(self, lo, hi):
        return [
            r["_id"] for r in self.rows
            if lo <= int(r["postedAt"][:4]) <= hi and r["tags"][0]["name"] == "AI Risk"
        ]


def run(fake, source="lesswrong", years=(2025, 2026), **kw):
    kw.setdefault("delay", 0)
    return forum.fetch_source(source, years, post=fake, sleep=lambda s: None, log=lambda *a: None, **kw)


# --- the query -----------------------------------------------------------------------------------


def test_query_asks_for_plain_text_tags_and_dates_and_pages_newest_first():
    fake = FakeForum(n=5)
    run(fake, years=(2020, 2026), paging="offset")
    q = fake.queries[0]
    assert 'view: "new"' in q and "limit: 50" in q and "offset: 0" in q
    assert "contents{plaintextMainText wordCount}" in q and "tags{name}" in q and "postedAt" in q


def test_records_follow_the_schema_the_rest_of_the_package_reads():
    recs, info = run(FakeForum(n=6), years=(2020, 2026), paging="offset")
    r = recs[0]
    assert r["id"] == "lesswrong:p000" and r["source"] == "lesswrong" and r["title"] == "Post number 0"
    assert r["authors"] == ["Author 0"] and r["tags"] == ["AI Risk", "Interpretability"]
    assert r["url"] == "https://forum.test/posts/p000" and r["date_published"].startswith("2026-10-20")
    assert r["karma"] == 10 and r["comment_count"] == 0 and r["words"] > 0
    assert r["text"].startswith("Body of post 0")
    df, _ = build_corpus(recs)  # and the corpus builder accepts it
    assert set(df["source"]) == {"lesswrong"} and set(df["group"]) == {"forum"}
    assert df["tags"].iloc[0] == ["AI Risk", "Interpretability"]


# --- paging --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["offset", "before"])
def test_both_paging_strategies_collect_exactly_the_posts_in_range_without_duplicates(mode):
    fake = FakeForum()
    recs, info = run(fake, paging=mode, page_size=7)
    want = fake.ai_ids_between(2025, 2026)
    assert [r["id"] for r in recs] == [f"lesswrong:{i}" for i in want]
    assert len({r["id"] for r in recs}) == len(recs)
    assert min(r["date_published"] for r in recs) >= "2025-01-01"
    assert info["paging"] == mode and info["counts"]["older_than_range"] > 0
    assert info["counts"]["not_about_ai"] > 0


def test_it_stops_walking_back_once_posts_are_older_than_the_range():
    fake = FakeForum(n=2000)
    run(fake, paging="before", page_size=50)
    assert len(fake.queries) < 10  # ~66 posts in range, not 2000


def test_a_range_ending_before_the_newest_posts_skips_the_newer_ones():
    fake = FakeForum()
    recs, info = run(fake, years=(2024, 2025), paging="offset")
    assert {r["date_published"][:4] for r in recs} <= {"2024", "2025"}
    assert [r["id"] for r in recs] == [f"lesswrong:{i}" for i in fake.ai_ids_between(2024, 2025)]
    assert not info["counts"].get("newer_than_range")  # the date window starts at the end of the range
    assert 'before: "2026-01-01"' in fake.queries[-1]
    recs2, info2 = run(FakeForum(), years=(2024, 2025), paging="before")
    assert [r["id"] for r in recs2] == [r["id"] for r in recs]
    assert not info2["counts"].get("newer_than_range")  # the cursor jumped straight to the end of the range


def test_a_server_that_caps_the_page_size_is_still_walked_to_the_end():
    fake = FakeForum(max_limit=10)
    recs, _ = run(fake, paging="offset", page_size=100)
    assert [r["id"] for r in recs] == [f"lesswrong:{i}" for i in fake.ai_ids_between(2025, 2026)]


def test_auto_prefers_offset_and_falls_back_to_the_date_cursor():
    fake = FakeForum()
    assert run(fake, paging="auto")[1]["paging"] == "offset"
    assert not any(re.search(r'before: "[^"]*T', q) for q in fake.queries)  # no timestamp cursor probe needed
    assert run(FakeForum(honours_offset=False), paging="auto")[1]["paging"] == "before"


def test_auto_survives_a_server_that_answers_the_before_probe_with_an_error():
    fake = FakeForum(honours_offset=False, before_errors=True)
    with pytest.raises(RuntimeError, match="neither 'offset' nor 'before'"):
        run(fake, paging="auto")
    recs, info = run(FakeForum(before_errors=True), paging="auto")  # offset works, so before is never needed
    assert info["paging"] == "offset" and recs


def test_auto_gives_up_clearly_when_the_server_honours_no_paging_argument():
    with pytest.raises(RuntimeError, match="neither 'offset' nor 'before'"):
        run(FakeForum(honours_before=False, honours_offset=False), paging="auto")


def test_an_ignored_paging_argument_cannot_loop_forever():
    recs, info = run(FakeForum(honours_offset=False), paging="offset", page_size=10)
    assert info["counts"]["stalled"] == 1  # the second page was the first again, so the walk stopped
    assert len(recs) <= 10


def test_unknown_paging_mode_is_rejected():
    with pytest.raises(ValueError):
        run(FakeForum(), paging="sideways")


# --- schema differences between the sites --------------------------------------------------------


def test_missing_fields_are_dropped_and_plain_text_falls_back_to_cleaned_html():
    fake = FakeForum(unsupported=[("baseScore", "Post"), ("plaintextMainText", "Revision")])
    recs, info = run(fake, years=(2020, 2026), paging="offset")
    assert recs and "karma" not in recs[0]  # baseScore is gone
    assert set(info["dropped_fields"]) == {"Post.baseScore", "Revision.plaintextMainText"}
    assert recs[0]["text"].startswith("Body of post 0")  # the stylesheet in the HTML was removed
    assert "mjx" not in recs[0]["text"] and "<p>" not in recs[0]["text"]


def test_a_missing_nested_type_leaves_the_rest_of_the_query_intact():
    fake = FakeForum(unsupported=[("displayName", "User"), ("name", "Tag")])
    recs, info = run(fake, years=(2020, 2026), paging="offset", all_posts=True)
    assert recs[0]["authors"] == [] and recs[0]["tags"] == []
    assert recs[0]["text"].startswith("Body of post 0")
    assert "user" not in fake.queries[-1] and "tags" not in fake.queries[-1]


def test_an_indispensable_field_missing_is_a_clear_error():
    with pytest.raises(RuntimeError, match="postedAt"):
        run(FakeForum(unsupported=[("postedAt", "Post")]))


def test_other_graphql_errors_are_reported_with_the_server_message():
    def broken(url, payload):
        return {"errors": [{"message": "rate limit exceeded"}]}

    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        run(broken, paging="offset")


# --- which posts are kept ------------------------------------------------------------------------


def rows_for_filter():
    def row(i, title, tags, af=False):
        return {"_id": f"x{i}", "title": title, "postedAt": f"2025-0{i}-01T00:00:00.000Z", "af": af,
                "tags": [{"name": t} for t in tags], "contents": {"plaintextMainText": "text " * 50}}

    return [
        row(1, "Tagged post", ["AI"]),
        row(2, "How I fixed my sourdough", ["Food"]),
        row(3, "Untagged but obviously about alignment research", []),
        row(4, "An Alignment Forum post with an unrelated title", [], af=True),
        row(5, "Who goes to the meetup", ["Community"]),
    ]


def serve(rows):
    def post(url, payload):
        return {"data": {"posts": {"results": rows if "offset: 0" in payload["query"] else []}}}

    return post


def test_lesswrong_posts_are_kept_when_tags_title_or_the_af_flag_say_they_are_about_ai():
    recs, info = run(serve(rows_for_filter()), paging="offset", years=(2025, 2025))
    assert [r["title"] for r in recs] == [
        "Tagged post", "Untagged but obviously about alignment research",
        "An Alignment Forum post with an unrelated title",
    ]
    assert info["counts"]["not_about_ai"] == 2 and info["ai_filter"] is True


def test_all_posts_switches_the_ai_filter_off():
    recs, info = run(serve(rows_for_filter()), paging="offset", years=(2025, 2025), all_posts=True)
    assert len(recs) == 5 and info["ai_filter"] is False


def test_on_the_alignment_forum_posts_marked_not_af_are_skipped_but_everything_else_is_kept():
    rows = rows_for_filter()
    rows[0]["af"] = False
    rows[1]["af"] = False  # explicitly not an AF post
    rows[2].pop("af")  # flag not reported: keep
    recs, info = run(serve(rows), source="alignmentforum", paging="offset", years=(2025, 2025))
    assert info["counts"]["not_af"] == 3  # x1, x2 and x5 carry af=False
    assert "alignmentforum:x3" in [r["id"] for r in recs] and "alignmentforum:x4" in [r["id"] for r in recs]


def test_ai_pattern_matches_ai_topics_and_not_everyday_words():
    for yes in ("AI", "AI Risk", "Interpretability (ML & AI)", "Inner Alignment", "Language models",
                "Machine Learning", "AI safety", "RLHF", "Goodhart's Law"):
        assert forum.AI_PATTERN.search(yes), yes
    for no in ("Gardening", "Rationality", "Said", "Existential risk", "Community", "Maintenance", "Fair"):
        assert not forum.AI_PATTERN.search(no), no


def test_unknown_forum_is_rejected():
    with pytest.raises(ValueError):
        forum.ForumClient("reddit")


# --- files, cache, manifest, CLI -----------------------------------------------------------------


def test_fetch_forums_writes_one_file_per_source_a_manifest_and_no_part_files(tmp_path):
    fake = FakeForum()
    paths = forum.fetch_forums(
        ["lesswrong", "eaforum"], (2025, 2026), tmp_path, post=fake, delay=0, sleep=lambda s: None, log=lambda *a: None,
    )
    assert [p.name for p in paths] == ["lesswrong_2025-2026.jsonl", "eaforum_2025-2026.jsonl"]
    recs = read_jsonl(paths[1])
    assert recs and all(r["source"] == "eaforum" for r in recs)
    manifest = json.loads((tmp_path / "manifest_forums.json").read_text(encoding="utf-8"))
    assert manifest["years"] == [2025, 2026] and set(manifest["sources"]) == {"lesswrong", "eaforum"}
    assert manifest["sources"]["lesswrong"]["n_kept"] == len(read_jsonl(paths[0]))
    assert manifest["sources"]["lesswrong"]["paging"] in ("before", "offset")
    assert manifest["sources"]["lesswrong"]["ai_filter"] is True and manifest["sources"]["eaforum"]["ai_filter"] is True
    assert not list(tmp_path.glob("*.part"))


def test_existing_files_are_reused_unless_refresh_is_given(tmp_path):
    fake = FakeForum()
    kw = dict(post=fake, delay=0, sleep=lambda s: None, log=lambda *a: None)
    forum.fetch_forums(["lesswrong"], (2025, 2026), tmp_path, **kw)
    n_calls = len(fake.queries)
    forum.fetch_forums(["lesswrong"], (2025, 2026), tmp_path, **kw)
    assert len(fake.queries) == n_calls  # cached: no request
    forum.fetch_forums(["lesswrong"], (2025, 2026), tmp_path, refresh=True, **kw)
    assert len(fake.queries) > n_calls


def test_cli_parses_fetch_forum_with_sensible_defaults():
    args = build_parser().parse_args(["fetch-forum"])
    assert args.sources == list(forum.FORUMS) and args.years == "2021-2026"
    assert args.paging == "auto" and args.delay == 1.0 and args.all_posts is False and args.out == "data/forums"
    args = build_parser().parse_args(["fetch-forum", "--sources", "lesswrong", "--all-posts", "--paging", "offset"])
    assert args.sources == ["lesswrong"] and args.all_posts and args.paging == "offset"


def test_cli_fetch_forum_end_to_end_with_a_fake_server(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(forum, "default_post", FakeForum())
    out = tmp_path / "forums"
    assert main(["fetch-forum", "--sources", "lesswrong", "--years", "2025-2026", "--out", str(out), "--delay", "0"]) == 0
    assert (out / "lesswrong_2025-2026.jsonl").stat().st_size > 0
    assert "Next: alignmap run" in capsys.readouterr().out


def test_cli_reports_a_server_problem_as_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(forum, "default_post", lambda url, payload: {"errors": [{"message": "boom"}]})
    code = main(["fetch-forum", "--sources", "lesswrong", "--out", str(tmp_path), "--delay", "0"])
    assert code == 2 and "boom" in capsys.readouterr().err


# --- transport -----------------------------------------------------------------------------------


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, body):
    return urllib.error.HTTPError("https://x.test/graphql", code, "err", {}, io.BytesIO(body.encode()))


def test_default_post_returns_the_json_body_of_a_graphql_validation_error(monkeypatch):
    body = json.dumps({"errors": [{"message": 'Cannot query field "x" on type "Post".'}]})

    def fake(req, timeout=None):
        raise http_error(400, body)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    reply = forum.default_post("https://x.test/graphql", {"query": "{}"}, sleep=lambda s: None)
    assert "Cannot query field" in reply["errors"][0]["message"]


def test_default_post_retries_server_errors_with_back_off_then_succeeds(monkeypatch):
    calls, waits = [], []

    def fake(req, timeout=None):
        calls.append(req)
        if len(calls) < 3:
            raise http_error(503, "busy")
        return FakeResponse(json.dumps({"data": {"posts": {"results": []}}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    reply = forum.default_post("https://x.test/graphql", {"query": "{}"}, sleep=waits.append)
    assert reply == {"data": {"posts": {"results": []}}} and len(calls) == 3 and waits == [5, 10]
    assert calls[0].get_header("User-agent").startswith("alignmap/") and calls[0].get_method() == "POST"


def test_default_post_explains_a_403_and_does_not_retry_it(monkeypatch):
    calls = []

    def fake(req, timeout=None):
        calls.append(1)
        raise http_error(403, "forbidden")

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(RuntimeError, match="longer --delay"):
        forum.default_post("https://x.test/graphql", {"query": "{}"}, sleep=lambda s: None)
    assert len(calls) == 1


def test_the_alignment_forum_is_not_ai_filtered_and_says_so_in_the_manifest(tmp_path):
    fake = FakeForum()
    forum.fetch_forums(["alignmentforum"], (2025, 2026), tmp_path, post=fake, delay=0, sleep=lambda s: None, log=lambda *a: None)
    info = json.loads((tmp_path / "manifest_forums.json").read_text(encoding="utf-8"))["sources"]["alignmentforum"]
    assert info["ai_filter"] is False and info["counts"].get("not_about_ai", 0) == 0


# --- the offset cap of the live servers ----------------------------------------------------------


def test_offset_paging_moves_its_date_window_before_the_server_cap_and_loses_nothing():
    fake = FakeForum(n=600, step_days=1, max_skip=40)
    recs, info = run(fake, years=(2025, 2026), paging="offset", page_size=10, max_skip=30)
    assert [r["id"] for r in recs] == [f"lesswrong:{i}" for i in fake.ai_ids_between(2025, 2026)]
    assert info["counts"]["windows"] > 5
    assert not any("Exceeded" in str(q) for q in fake.queries)


def test_offset_paging_recovers_when_the_server_cap_is_lower_than_expected():
    fake = FakeForum(n=600, step_days=1, max_skip=25)
    recs, info = run(fake, years=(2025, 2026), paging="offset", page_size=10, max_skip=1000)
    assert [r["id"] for r in recs] == [f"lesswrong:{i}" for i in fake.ai_ids_between(2025, 2026)]
    assert info["counts"]["windows"] > 5


def test_many_posts_on_one_day_is_a_clear_error_not_a_silent_gap():
    fake = FakeForum(n=200, step_days=0.01, max_skip=20)  # ~100 posts a day
    with pytest.raises(RuntimeError, match="offset cap"):
        run(fake, years=(2020, 2026), paging="offset", page_size=10, max_skip=20)
