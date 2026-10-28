import json

import pandas as pd
import pytest

from alignmap.corpus import (
    build_corpus,
    collect_inputs,
    coverage_table,
    coverage_warnings,
    docs_for_embedding,
    iter_records,
    read_jsonl,
    write_jsonl,
)

LONG = "This paper studies how to make language models follow human intent. " * 5


def rec(i, **kw):
    base = {
        "id": f"id{i}",
        "source": "arxiv",
        "title": f"A reasonably long unique title number {i}",
        "authors": ["Ada"],
        "text": LONG,
        "url": f"https://example.org/{i}",
        "date_published": "2023-05-01T00:00:00Z",
    }
    base.update(kw)
    return base


def test_build_corpus_basic_columns():
    df, stats = build_corpus([rec(1), rec(2, date_published="2021-02-03")])
    assert list(df["year"]) == [2021, 2023]  # sorted by date
    assert stats["n_kept"] == 2 and stats["by_year"] == {2021: 1, 2023: 1}
    assert df.loc[0, "doc"].startswith(df.loc[0, "title"])
    assert {"id", "source", "group", "title", "authors", "tags", "text", "url", "date", "year", "doc"} <= set(df.columns)


def test_drops_missing_dates_short_text_and_duplicates():
    records = [
        rec(1),
        rec(2, date_published=None),
        rec(3, date_published="not a date"),
        rec(4, text="too short"),
        rec(1),  # same id
        rec(5, title="A reasonably long unique title number 1"),  # same title as rec(1)
        rec(6),
    ]
    df, stats = build_corpus(records)
    assert sorted(df["id"]) == ["id1", "id6"]
    assert stats["dropped_no_date"] == 2
    assert stats["dropped_short"] == 1
    assert stats["dropped_duplicate"] == 2


def test_short_titles_are_not_used_as_duplicate_keys():
    df, _ = build_corpus([rec(1, title="Intro"), rec(2, title="Intro")])
    assert len(df) == 2


def test_authors_and_extra_keys_are_normalised_and_kept():
    df, _ = build_corpus([rec(1, authors="Solo Author", categories=["cs.AI"])])
    assert df.loc[0, "authors"] == ["Solo Author"]
    assert df.loc[0, "categories"] == ["cs.AI"]


def test_text_is_truncated_to_max_chars():
    df, _ = build_corpus([rec(1, text="word " * 5000)], max_chars=300)
    assert len(df.loc[0, "text"]) <= 300


def test_all_records_dropped_is_an_error():
    with pytest.raises(ValueError, match="No usable documents"):
        build_corpus([rec(1, date_published=None)])


def test_abstract_is_preferred_over_the_full_text_prefix():
    abstract = "We show that sparse autoencoders recover interpretable features from a language model. " * 3
    df, _ = build_corpus([rec(1, abstract=abstract, text="Introduction. " + "unrelated body " * 100)])
    assert df.loc[0, "text"].startswith("We show that sparse autoencoders")
    assert "unrelated body" not in df.loc[0, "text"]


def test_a_very_short_abstract_falls_back_to_the_text():
    df, _ = build_corpus([rec(1, abstract="Too short.")])
    assert df.loc[0, "text"].startswith("This paper studies")


def test_sources_filter_keeps_only_the_requested_sources_and_counts_the_rest():
    records = [rec(1, source="arxiv"), rec(2, source="lesswrong"), rec(3, source="alignmentforum")]
    df, stats = build_corpus(records, sources=["lesswrong", "alignmentforum"])
    assert sorted(df["source"]) == ["alignmentforum", "lesswrong"]
    assert stats["dropped_source"] == 1 and stats["n_input"] == 3


def test_most_specific_source_wins_when_a_work_appears_twice():
    same_title = "Language models can be steered with activation additions"
    records = [
        rec(1, source="lesswrong", id="lw-1", title=same_title),
        rec(2, source="alignmentforum", id="af-1", title=same_title),
        rec(3, source="arxiv", id="ax-1", title=same_title),
    ]
    df, stats = build_corpus(records)
    assert list(df["id"]) == ["ax-1"] and stats["dropped_duplicate"] == 2


def test_cross_post_to_forum_beats_lesswrong_copy():
    same_title = "A longer title shared between two forum copies"
    df, _ = build_corpus(
        [rec(1, source="lesswrong", id="lw", title=same_title), rec(2, source="alignmentforum", id="af", title=same_title)]
    )
    assert list(df["id"]) == ["af"]


def test_tags_are_normalised_to_a_list_of_strings():
    df, _ = build_corpus(
        [rec(1, source="lesswrong", tags=["Interpretability", " ", "AI"]), rec(2, tags="RLHF"), rec(3)]
    )
    by_id = df.set_index("id")["tags"]
    assert by_id["id1"] == ["Interpretability", "AI"]
    assert by_id["id2"] == ["RLHF"]
    assert by_id["id3"] == []


def test_tags_given_as_objects_are_reduced_to_their_names():
    df, _ = build_corpus([rec(1, tags=[{"name": "Interpretability", "id": 7}, {"name": ""}, "RLHF"])])
    assert df.loc[0, "tags"] == ["Interpretability", "RLHF"]


def test_source_groups():
    records = [
        rec(1, source="arxiv"), rec(2, source="alignmentforum"), rec(3, source="lesswrong"),
        rec(4, source="eaforum"), rec(5, source="blogs"), rec(6, source="something-new"),
    ]
    df, stats = build_corpus(records)
    group = dict(zip(df["source"], df["group"]))
    assert group["arxiv"] == "papers"
    assert group["alignmentforum"] == group["lesswrong"] == group["eaforum"] == "forum"
    assert group["blogs"] == "blogs"
    assert group["something-new"] == "something-new"  # unknown sources are their own group
    assert stats["by_group"]["forum"] == 3


def test_docs_for_embedding_formats():
    df, _ = build_corpus([rec(1)])
    assert docs_for_embedding(df, "plain") == df["doc"].tolist()
    assert docs_for_embedding(df, "specter")[0] == f"{df.loc[0, 'title']}[SEP]{df.loc[0, 'text']}"
    with pytest.raises(ValueError, match="Unknown doc format"):
        docs_for_embedding(df, "nope")


def test_coverage_table_and_staleness_warning():
    records = [
        rec(1, source="arxiv", date_published="2021-03-01"),
        rec(2, source="arxiv", date_published="2025-03-01"),
        rec(3, source="lesswrong", date_published="2021-04-01"),
        rec(4, source="lesswrong", date_published="2022-06-01"),
    ]
    df, stats = build_corpus(records)
    table = coverage_table(df)
    assert table.loc["arxiv", 2025] == 1 and table.loc["lesswrong", 2021] == 1
    assert stats["last_date_by_source"]["lesswrong"] == "2022-06-01"
    warnings = coverage_warnings(df)
    assert len(warnings) == 1 and "lesswrong" in warnings[0] and "2022-06-01" in warnings[0]
    # a source that is only a little behind the newest one is not flagged
    fresh, _ = build_corpus([rec(1, date_published="2025-03-01"), rec(2, source="lesswrong", date_published="2025-01-15")])
    assert coverage_warnings(fresh) == []


def test_jsonl_roundtrip_and_error_location(tmp_path):
    p = tmp_path / "a.jsonl"
    assert write_jsonl([rec(1), rec(2)], p) == 2
    assert [r["id"] for r in read_jsonl(p)] == ["id1", "id2"]
    p.write_text(json.dumps(rec(1)) + "\n{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"a\.jsonl:2"):
        read_jsonl(p)


def test_iter_records_streams_several_files_in_order(tmp_path):
    write_jsonl([rec(1), rec(2)], tmp_path / "a.jsonl")
    write_jsonl([rec(3)], tmp_path / "b.jsonl")
    stream = iter_records([tmp_path / "a.jsonl", tmp_path / "b.jsonl"])
    assert iter(stream) is stream  # a generator: files are read lazily, not loaded all at once
    assert [r["id"] for r in stream] == ["id1", "id2", "id3"]


def test_collect_inputs_accepts_files_dirs_and_globs(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("a.jsonl", "sub/b.jsonl"):
        write_jsonl([rec(1)], tmp_path / name)
    (tmp_path / "manifest.json").write_text("{}")
    assert len(collect_inputs([tmp_path])) == 2
    assert len(collect_inputs([tmp_path / "a.jsonl"])) == 1
    assert len(collect_inputs([str(tmp_path / "*.jsonl")])) == 1
    with pytest.raises(FileNotFoundError):
        collect_inputs([tmp_path / "missing" / "*.jsonl"])


def test_build_corpus_accepts_a_generator():
    df, stats = build_corpus(r for r in [rec(1), rec(2)])
    assert isinstance(df, pd.DataFrame) and stats["n_kept"] == 2
