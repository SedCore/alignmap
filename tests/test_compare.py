import pandas as pd
import pytest

from alignmap.compare import compare_arms, write_compare
from alignmap.corpus import build_corpus
from alignmap.pipeline import RunConfig
from alignmap.presets import PRESETS, apply_preset

QUIET = dict(log=lambda *_: None)


def test_presets_describe_the_documented_configurations():
    cfg = apply_preset(RunConfig(), "2022-style")
    assert cfg.embedder == "sbert" and cfg.sbert_model == "sentence-transformers/allenai-specter"
    assert cfg.doc_format == "specter" and cfg.umap_neighbors == 250 and cfg.k == 5
    assert cfg.reducer == "umap" and cfg.clusterer == "kmeans"
    modern = apply_preset(RunConfig(), "modern-fast")
    assert modern.clusterer == "hdbscan" and modern.k == RunConfig().k  # k is simply not used by HDBSCAN
    assert set(PRESETS) == {"lsa-baseline", "2022-style", "modern-fast", "modern-strong"}


def test_apply_preset_does_not_mutate_and_rejects_unknown_names():
    base = RunConfig()
    apply_preset(base, "2022-style")
    assert base.embedder == "tfidf"
    with pytest.raises(ValueError, match="Unknown preset"):
        apply_preset(base, "gpt-magic")


def test_every_preset_field_is_a_real_config_field():
    fields = set(RunConfig.__dataclass_fields__)
    for name, overrides in PRESETS.items():
        assert set(overrides) <= fields, name


def test_comparison_prefers_the_right_number_of_clusters(toy_df, tmp_path):
    arms = {"k4": RunConfig(k=4), "k2": RunConfig(k=2)}
    table, used = compare_arms(toy_df, arms, cache_dir=tmp_path, n_rounds=3, **QUIET)
    row = table.set_index("arm")
    assert row.loc["k4", "n_clusters"] == 4 and row.loc["k2", "n_clusters"] == 2
    assert row.loc["k4", "tags_ami"] > 0.9 > row.loc["k2", "tags_ami"]
    assert row.loc["k4", "stability_ari"] > 0.9
    assert (row["noise_fraction"] == 0).all() and (row["seconds"] >= 0).all()
    assert len(used) == 4
    assert len(list(tmp_path.glob("emb_*.npy"))) == 1  # both arms share one cached embedding


def test_comparison_of_umap_hdbscan_reports_noise_and_finds_the_topics(toy_df):
    pytest.importorskip("umap")
    arms = {"umap-hdbscan": RunConfig(reducer="umap", clusterer="hdbscan", min_cluster_size=0.03)}
    table, _ = compare_arms(toy_df, arms, n_rounds=2, **QUIET)
    row = table.iloc[0]
    assert row["n_clusters"] == 4 and row["tags_ami"] > 0.9
    assert 0 <= row["noise_fraction"] < 0.15 and row["tags_coverage"] > 0.85


def test_comparison_without_tags_still_works_and_has_no_agreement_columns(toy_records, tmp_path):
    untagged = [{k: v for k, v in r.items() if k != "tags"} for r in toy_records]
    df, stats = build_corpus(untagged)
    table, used = compare_arms(df, {"k4": RunConfig(k=4)}, n_rounds=2, **QUIET)
    assert used == [] and not [c for c in table.columns if c.startswith("tags_")]
    write_compare(table, used, tmp_path / "compare.md", n_docs=len(df), sources=stats["by_source"])
    assert "Tag agreement" not in (tmp_path / "compare.md").read_text(encoding="utf-8")


def test_write_compare_produces_a_readable_markdown_table_and_a_csv(toy_df, tmp_path):
    table, used = compare_arms(toy_df, {"k4": RunConfig(k=4)}, n_rounds=2, **QUIET)
    write_compare(table, used, tmp_path / "out" / "compare.md", n_docs=len(toy_df), sources={"synthetic-toy": len(toy_df)})
    md = (tmp_path / "out" / "compare.md").read_text(encoding="utf-8")
    assert "# Method comparison" in md and "Tag agreement (AMI)" in md and "How to read this" in md
    assert f"{len(toy_df):,} papers from synthetic-toy" in md
    assert all(f"`{t}`" in md for t in used)
    csv = pd.read_csv(tmp_path / "out" / "compare.csv")
    assert csv.loc[0, "arm"] == "k4" and csv.loc[0, "tags_ami"] > 0.9
