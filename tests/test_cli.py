import json
from types import SimpleNamespace

import pytest

from alignmap.cli import _config, build_parser, main
from alignmap.corpus import write_jsonl
from alignmap.pipeline import RunConfig, run_pipeline
from alignmap.toy import make_composition_toy


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_demo_end_to_end_writes_every_artifact(tmp_path):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out)]) == 0
    for name in (
        "report.md", "run.json", "clusters.json", "assignments.csv", "shift_pooled.csv", "shift_adjusted.csv",
        "share_by_year.csv", "coverage.csv", "figures/shift.png", "figures/share_heatmap.png", "toy_corpus.jsonl",
    ):
        assert (out / name).stat().st_size > 0, name

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DEMO DATA" in report  # synthetic output must never pass for findings
    assert "▲ larger share" in report and "▼ smaller share" in report
    assert "source-adjusted" in report.lower()

    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["n_clusters"] == 4 and run["config"]["seed"] == 0
    assert run["stability_ari"] > 0.9 and run["source_adjusted"] is True
    assert run["tag_agreement"]["ami"] > 0.9
    assert {"python", "numpy", "scikit-learn"} <= run.keys()


def test_trend_demo_has_one_source_so_no_adjusted_table(tmp_path):
    out = tmp_path / "demo"
    assert main(["demo", "--kind", "trend", "--out", str(out)]) == 0
    assert (out / "shift_pooled.csv").exists() and not (out / "shift_adjusted.csv").exists()
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["source_adjusted"] is False


def test_run_on_jsonl_file_records_input_hash(tmp_path, toy_records):
    src = tmp_path / "corpus.jsonl"
    write_jsonl(toy_records, src)
    out = tmp_path / "out"
    assert main(["run", "--input", str(src), "--out", str(out), "--k", "auto", "--k-range", "2", "6", "2"]) == 0
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert len(run["provenance"]["inputs"][0]["sha256"]) == 64
    assert run["k_scan"] and run["n_clusters"] in (2, 4, 6)


def test_run_can_restrict_sources(tmp_path, composition_records):
    src = tmp_path / "corpus.jsonl"
    write_jsonl(composition_records, src)
    out = tmp_path / "out"
    assert main(["run", "--input", str(src), "--out", str(out), "--k", "3", "--sources", "synthetic-forum"]) == 0
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert set(run["corpus"]["by_source"]) == {"synthetic-forum"}
    assert run["corpus"]["dropped_source"] > 0
    assert run["source_adjusted"] is False  # one venue left: nothing to adjust over


def test_same_seed_gives_identical_results(tmp_path, toy_records):
    cfg = RunConfig(k=4, n_boot=500, seed=3)
    a = run_pipeline(toy_records, tmp_path / "a", cfg)
    b = run_pipeline(toy_records, tmp_path / "b", cfg)
    assert (a["df"]["cluster"] == b["df"]["cluster"]).all()
    assert a["shift"].equals(b["shift"])


def test_pipeline_with_llm_labels_uses_injected_client(tmp_path, toy_records):
    class Client:
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            return SimpleNamespace(content=[SimpleNamespace(text='{"name": "Named by LLM", "summary": "A summary."}')])

    cfg = RunConfig(k=4, n_boot=200, llm_labels=True, llm_model="m")
    result = run_pipeline(toy_records, tmp_path, cfg, llm_client=Client())
    assert all(c["name"] == "Named by LLM" and c["llm"] for c in result["clusters"])
    assert "LLM-written" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_stale_source_is_flagged_in_the_report(tmp_path):
    records = [r for r in make_composition_toy(n_per_year=200, seed=1)
               if not (r["source"] == "synthetic-forum" and r["date_published"] >= "2024")]
    run_pipeline(records, tmp_path, RunConfig(k=4, n_boot=200, min_group_docs=20))
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "synthetic-forum" in report and "no papers after" in report


def test_missing_period_gives_helpful_error(tmp_path, toy_records, capsys):
    src = tmp_path / "c.jsonl"
    write_jsonl(toy_records, src)
    code = main(["run", "--input", str(src), "--out", str(tmp_path / "o"), "--recent", "2030-2031"])
    assert code == 2
    assert "recent period 2030-2031" in capsys.readouterr().err


def test_missing_input_is_a_clean_error(tmp_path, capsys):
    assert main(["run", "--input", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "o")]) == 2
    assert "error:" in capsys.readouterr().err


def test_sbert_model_flag_reaches_the_config():
    assert _config(parse("run", "--input", "x", "--sbert-model", "foo/bar")).sbert_model == "foo/bar"
    assert _config(parse("run", "--input", "x")).sbert_model == "BAAI/bge-small-en-v1.5"


def test_explicit_flags_override_a_preset_which_overrides_the_defaults():
    cfg = _config(parse("run", "--input", "x", "--preset", "2022-style", "--k", "7", "--max-seq-length", "256"))
    assert cfg.k == 7 and cfg.max_seq_length == 256  # explicit flags win
    assert cfg.umap_neighbors == 250 and cfg.sbert_model == "sentence-transformers/allenai-specter"  # preset
    assert cfg.n_boot == RunConfig().n_boot  # untouched defaults stay
    cfg = _config(parse("run", "--input", "x", "--preset", "modern-fast", "--min-cluster-size", "0.05"))
    assert cfg.clusterer == "hdbscan" and cfg.min_cluster_size == 0.05


def test_config_parsing_of_k_ranges_periods_and_lists():
    cfg = _config(parse("run", "--input", "x", "--k", "auto", "--k-range", "4", "12", "4", "--baseline", "2020-2021",
                        "--recent", "2026", "--sources", "arxiv", "lesswrong", "--tag-exclude", "AI", "Other"))
    assert cfg.k is None and cfg.k_range == (4, 12, 4)
    assert cfg.baseline == (2020, 2021) and cfg.recent == (2026, 2026)
    assert cfg.sources == ("arxiv", "lesswrong") and cfg.tag_exclude == ("AI", "Other")


def test_compare_runs_arms_from_a_json_file_offline(tmp_path, toy_records):
    src = tmp_path / "corpus.jsonl"
    write_jsonl(toy_records, src)
    configs = tmp_path / "arms.json"
    configs.write_text(json.dumps({"lsa-k4": {"k": 4}, "lsa-k2": {"k": 2}}), encoding="utf-8")
    out = tmp_path / "cmp"
    assert main(["compare", "--input", str(src), "--configs", str(configs), "--out", str(out),
                 "--stability-rounds", "2", "--cache-dir", str(tmp_path / "cache")]) == 0
    md = (out / "compare.md").read_text(encoding="utf-8")
    assert "lsa-k4" in md and "lsa-k2" in md and "Tag agreement (AMI)" in md
    assert (out / "compare.csv").stat().st_size > 0


def test_compare_defaults_to_three_presets():
    assert parse("compare", "--input", "x").presets == ["lsa-baseline", "2022-style", "modern-fast"]


def test_download_rejects_unknown_sources():
    with pytest.raises(SystemExit):
        parse("download", "--sources", "not-a-source")


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        main([])
