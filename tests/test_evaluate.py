import numpy as np
import pandas as pd
import pytest

from alignmap.evaluate import GENERIC_TAGS, agreement, tag_labels
from alignmap.toy import TAG_NAMES


def tagged_frame(tag_lists):
    return pd.DataFrame({"tags": tag_lists})


def test_tag_labels_picks_the_most_specific_frequent_tag_and_skips_generic_ones():
    tags = (
        [["AI", "Interpretability"]] * 40          # 40 posts
        + [["AI", "Interpretability", "Mechanistic Interpretability"]] * 25
        + [["Rationality", "AI Governance"]] * 30
        + [["Personal Blog"]] * 10                  # only generic tags
        + [["Very Rare Tag"]] * 3                   # too rare to be a reference label
        + [[]] * 5
    )
    truth, used = tag_labels(tagged_frame(tags), top=10, min_count=20)
    assert set(used) == {"Interpretability", "Mechanistic Interpretability", "AI Governance"}
    assert not set(used) & set(GENERIC_TAGS)
    # a post with both gets the LESS frequent (more specific) one
    assert truth.iloc[40] == "Mechanistic Interpretability"
    assert truth.iloc[0] == "Interpretability"
    assert truth.iloc[65] == "AI Governance"
    assert truth.iloc[95:].isna().all()  # generic-only, rare-only and untagged posts have no label


def test_tag_labels_respects_top_and_custom_exclusions():
    tags = [["A"]] * 50 + [["B"]] * 40 + [["C"]] * 30
    _, used = tag_labels(tagged_frame(tags), top=2, min_count=1)
    assert used == ["A", "B"]
    _, used = tag_labels(tagged_frame(tags), top=5, min_count=1, exclude=["A"])
    assert used == ["B", "C"]


def test_tag_labels_on_the_toy_corpus_recover_the_planted_topics(toy_df):
    truth, used = tag_labels(toy_df)
    assert set(used) == set(TAG_NAMES.values())  # the generic "AI" tag is ignored
    expected = toy_df["_topic"].map(TAG_NAMES)
    assert (truth == expected).all()


def test_agreement_is_perfect_for_the_true_partition_and_near_zero_for_a_shuffled_one(toy_df):
    truth, _ = tag_labels(toy_df)
    perfect = pd.factorize(toy_df["_topic"])[0]
    got = agreement(perfect, truth)
    assert got["ami"] == pytest.approx(1.0) and got["v_measure"] == pytest.approx(1.0)
    assert got["coverage"] == 1.0 and got["n_tagged"] == got["n_evaluated"] == len(toy_df)

    shuffled = np.random.default_rng(0).permutation(perfect)
    assert abs(agreement(shuffled, truth)["ami"]) < 0.05


def test_agreement_penalises_merging_and_over_splitting_in_different_scores():
    truth = pd.Series(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
    merged = np.array([0] * 200 + [1] * 100)  # a and b merged
    split = np.concatenate([np.arange(300) % 6])  # every topic split into two, at random
    m, s = agreement(merged, truth), agreement(split, truth)
    assert m["homogeneity"] < 1.0 and m["completeness"] == pytest.approx(1.0)
    assert s["homogeneity"] < 0.1 and s["ami"] < 0.1


def test_agreement_excludes_noise_but_reports_how_much_was_left_out():
    truth = pd.Series(["a"] * 50 + ["b"] * 50 + [np.nan] * 20)
    pred = np.array([0] * 40 + [-1] * 10 + [1] * 50 + [0] * 20)
    got = agreement(pred, truth)
    assert got["n_tagged"] == 100 and got["n_evaluated"] == 90
    assert got["coverage"] == pytest.approx(0.9)
    assert got["ami"] == pytest.approx(1.0)  # what was clustered, was clustered perfectly


def test_agreement_is_nan_when_it_cannot_be_computed():
    truth = pd.Series(["a"] * 10 + ["b"] * 10)
    got = agreement(np.zeros(20, dtype=int), truth)  # one cluster only
    assert np.isnan(got["ami"]) and got["coverage"] == 1.0
    got = agreement(np.full(20, -1), truth)
    assert np.isnan(got["ami"]) and got["coverage"] == 0.0
