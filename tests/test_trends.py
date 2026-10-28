import numpy as np
import pandas as pd
import pytest

from alignmap.trends import parse_year_range, share_by_year, shift_table, shift_table_adjusted, usable_groups

from conftest import cluster_of_topic


def frame(counts_by_period):
    """counts_by_period: {year: [n_cluster0, n_cluster1, ...]}"""
    rows = []
    for year, counts in counts_by_period.items():
        for c, n in enumerate(counts):
            rows += [{"year": year, "cluster": c}] * n
    return pd.DataFrame(rows)


def test_parse_year_range():
    assert parse_year_range("2021-2022") == (2021, 2022)
    assert parse_year_range("2024") == (2024, 2024)
    with pytest.raises(ValueError):
        parse_year_range("2025-2021")
    with pytest.raises(ValueError):
        parse_year_range("2020-2021-2022")


def test_share_by_year_columns_sum_to_one():
    df = frame({2021: [30, 70], 2022: [50, 50]})
    s = share_by_year(df)
    assert np.allclose(s.sum(axis=0), 1.0)
    assert s.loc[0, 2021] == pytest.approx(0.3)


def test_known_shift_has_the_right_log_ratio_and_sign():
    # cluster 0: 80% -> 20% of papers (log2 ratio = -2); cluster 1: 20% -> 80% (= +2)
    df = frame({2021: [800, 200], 2025: [200, 800]})
    t = shift_table(df, (2021, 2021), (2025, 2025), n_boot=2000, seed=1).set_index("cluster")
    assert t.loc[0, "log2_ratio"] == pytest.approx(-2.0, abs=0.02)
    assert t.loc[1, "log2_ratio"] == pytest.approx(2.0, abs=0.02)
    assert bool(t.loc[0, "significant"]) and bool(t.loc[1, "significant"])
    assert t.loc[1, "ci_low"] < t.loc[1, "log2_ratio"] < t.loc[1, "ci_high"]


def test_no_change_is_not_flagged():
    df = frame({2021: [500, 500], 2025: [500, 500]})
    t = shift_table(df, (2021, 2021), (2025, 2025), n_boot=2000, seed=1)
    assert not t["significant"].any()
    assert (t["ci_low"] < 0).all() and (t["ci_high"] > 0).all()


def test_small_samples_give_wide_intervals_and_no_false_flag():
    df = frame({2021: [6, 4], 2025: [4, 6]})  # same direction as a real shift, but n = 10
    t = shift_table(df, (2021, 2021), (2025, 2025), n_boot=2000, seed=1)
    assert not t["significant"].any()


def test_cluster_absent_from_baseline_gets_a_finite_ratio():
    df = frame({2021: [100, 0], 2025: [60, 40]})
    t = shift_table(df, (2021, 2021), (2025, 2025), n_boot=1000, seed=1).set_index("cluster")
    assert np.isfinite(t.loc[1, "log2_ratio"]) and t.loc[1, "log2_ratio"] > 3
    assert bool(t.loc[1, "significant"])


def test_bootstrap_is_deterministic_given_a_seed():
    df = frame({2021: [70, 30], 2025: [40, 60]})
    a = shift_table(df, (2021, 2021), (2025, 2025), n_boot=500, seed=7)
    b = shift_table(df, (2021, 2021), (2025, 2025), n_boot=500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_missing_period_is_a_clear_error():
    df = frame({2021: [10, 10]})
    with pytest.raises(ValueError, match="both periods"):
        shift_table(df, (2021, 2021), (2025, 2025))


def test_pipeline_ground_truth_on_toy_corpus(toy_clustered):
    """The end-to-end check: planted growth/decline must be recovered, flat topics must not be flagged."""
    df, _ = toy_clustered
    t = shift_table(df, (2021, 2022), (2025, 2026), n_boot=4000, seed=0).set_index("cluster")
    cluster_of = {topic: df.loc[df["_topic"] == topic, "cluster"].mode().iloc[0] for topic in df["_topic"].unique()}

    growing, shrinking = t.loc[cluster_of["interpretability"]], t.loc[cluster_of["agent_foundations"]]
    assert growing["log2_ratio"] > 1 and bool(growing["significant"])
    assert shrinking["log2_ratio"] < -1.5 and bool(shrinking["significant"])
    for flat in ("rlhf", "governance"):
        assert not bool(t.loc[cluster_of[flat], "significant"]), flat


def grouped(cells):
    """cells: {(group, year): [n_cluster0, n_cluster1, ...]}"""
    rows = []
    for (group, year), counts in cells.items():
        for c, n in enumerate(counts):
            rows += [{"group": group, "year": year, "cluster": c}] * n
    return pd.DataFrame(rows)


def test_usable_groups_needs_enough_papers_in_both_periods():
    df = grouped({("forum", 2021): [40, 40], ("forum", 2025): [40, 40],
                  ("papers", 2021): [40, 40], ("papers", 2025): [5, 5],  # too few recently
                  ("blogs", 2025): [50, 50]})  # absent from the baseline
    assert usable_groups(df, (2021, 2021), (2025, 2025), min_docs=30) == ["forum"]
    assert usable_groups(df, (2021, 2021), (2025, 2025), min_docs=5) == ["forum", "papers"]


def test_adjusted_shift_needs_two_groups():
    df = grouped({("forum", 2021): [50, 50], ("forum", 2025): [50, 50]})
    with pytest.raises(ValueError, match=">= 2 groups"):
        shift_table_adjusted(df, (2021, 2021), (2025, 2025))


def test_adjustment_removes_a_pure_composition_effect():
    """Every venue keeps the same topic mix, but the mix of venues changes; pooled shares move, adjusted don't."""
    # forum shrinks from 1000 to 250 papers while papers grow from 250 to 1000: the pooled share of
    # cluster 0 falls from 68% to 32% purely because of the venue mix
    df = grouped({
        ("forum", 2021): [800, 200], ("forum", 2025): [200, 50],  # forum: 80/20 both times
        ("papers", 2021): [50, 200], ("papers", 2025): [200, 800],  # papers: 20/80 both times
    })
    pooled = shift_table(df, (2021, 2021), (2025, 2025), n_boot=2000, seed=1).set_index("cluster")
    adj = shift_table_adjusted(df, (2021, 2021), (2025, 2025), n_boot=2000, seed=1).set_index("cluster")
    assert pooled.loc[0, "log2_ratio"] < -0.8 and bool(pooled.loc[0, "significant"])  # spurious decline
    assert abs(adj.loc[0, "log2_ratio"]) < 0.05 and not adj["significant"].any()


def test_adjustment_still_detects_a_real_within_group_change():
    cells = {
        ("forum", 2021): [400, 100], ("forum", 2025): [100, 400],  # forum: 80/20 -> 20/80
        ("papers", 2021): [250, 250], ("papers", 2025): [250, 250],  # papers: unchanged
    }
    t = shift_table_adjusted(grouped(cells), (2021, 2021), (2025, 2025), n_boot=2000, seed=1).set_index("cluster")
    assert t.loc[0, "log2_ratio"] < -0.5 and bool(t.loc[0, "significant"])
    assert t.loc[1, "log2_ratio"] > 0.5 and bool(t.loc[1, "significant"])


def test_adjusted_bootstrap_is_deterministic_given_a_seed():
    df = grouped({("a", 2021): [60, 40], ("a", 2025): [40, 60], ("b", 2021): [30, 70], ("b", 2025): [50, 50]})
    x = shift_table_adjusted(df, (2021, 2021), (2025, 2025), n_boot=300, seed=5, min_docs=10)
    y = shift_table_adjusted(df, (2021, 2021), (2025, 2025), n_boot=300, seed=5, min_docs=10)
    pd.testing.assert_frame_equal(x, y)


def test_composition_toy_pooled_analysis_is_fooled_but_adjusted_is_not(composition_clustered):
    """Ground truth: within venues only agent_foundations falls and governance rises. The venue mix
    drifts towards papers, which are interpretability-heavy, so a pooled analysis reports a spurious
    interpretability boom."""
    df = composition_clustered
    cl = cluster_of_topic(df)
    base, recent = (2021, 2022), (2025, 2026)
    pooled = shift_table(df, base, recent, n_boot=3000, seed=0).set_index("cluster")
    adj = shift_table_adjusted(df, base, recent, n_boot=3000, seed=0).set_index("cluster")

    interp = cl["interpretability"]
    assert pooled.loc[interp, "log2_ratio"] > 0.5 and bool(pooled.loc[interp, "significant"])  # spurious
    assert not bool(adj.loc[interp, "significant"]) and abs(adj.loc[interp, "log2_ratio"]) < 0.3
    assert not bool(adj.loc[cl["rlhf"], "significant"])

    af, gov = adj.loc[cl["agent_foundations"]], adj.loc[cl["governance"]]  # real, within-venue changes
    assert af["log2_ratio"] < -0.5 and bool(af["significant"])
    assert gov["log2_ratio"] > 0.3 and bool(gov["significant"])
