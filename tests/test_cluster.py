import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from alignmap.cluster import (
    ClusterSpec,
    cluster_reduced,
    ctfidf_keywords,
    fit_hdbscan,
    fit_kmeans,
    representative_docs,
    resolve_min_cluster_size,
    run_clustering,
    sampled_silhouette,
    select_k,
    stability,
)
from alignmap.toy import TOPICS


def kmeans4(Zs, seed):
    return fit_kmeans(Zs, 4, seed=seed, n_init=3).labels_


def blobs(seed=0, n=200, n_noise=20):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]])
    X = np.vstack([rng.normal(c, 0.7, size=(n, 2)) for c in centres])
    truth = np.repeat([0, 1, 2], n)
    scatter = rng.uniform(-30, 45, size=(n_noise, 2))
    scatter = scatter[np.min(np.linalg.norm(scatter[:, None] - centres[None], axis=2), axis=1) > 6]
    return np.vstack([X, scatter]), np.concatenate([truth, -np.ones(len(scatter), dtype=int)])


def test_kmeans_recovers_the_planted_topics(toy_clustered):
    df, _ = toy_clustered
    assert adjusted_rand_score(df["_topic"], df["cluster"]) > 0.9


def test_select_k_finds_the_true_number_of_topics(toy_Z):
    best, scores = select_k(toy_Z, range(2, 9), seed=0)
    assert best == 4
    assert set(scores) == set(range(2, 9))


def test_select_k_ignores_impossible_values_and_errors_if_none_valid(toy_Z):
    best, scores = select_k(toy_Z[:30], [1, 3, 500], seed=0)
    assert best == 3 and list(scores) == [3]
    with pytest.raises(ValueError, match="No valid k"):
        select_k(toy_Z[:5], [10, 20])


def test_silhouette_is_high_for_separable_topics_and_nan_for_one_cluster(toy_Z, toy_clustered):
    _, km = toy_clustered
    assert sampled_silhouette(toy_Z, km.labels_) > 0.1
    assert np.isnan(sampled_silhouette(toy_Z, np.zeros(len(toy_Z), dtype=int)))


def test_silhouette_ignores_noise_points():
    X, truth = blobs()
    assert sampled_silhouette(X, truth) > 0.8  # noise (-1) must not drag the score down
    assert np.isnan(sampled_silhouette(X, np.full(len(X), -1)))


def test_stability_is_high_for_real_structure_and_lower_for_noise(toy_Z):
    assert stability(toy_Z, kmeans4, n_rounds=4, seed=0) > 0.9
    noise = np.random.default_rng(0).normal(size=(600, 32))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    assert stability(noise, kmeans4, n_rounds=4, seed=0) < 0.6


def test_stability_leaves_noise_points_out_and_is_nan_if_nothing_is_comparable():
    X, _ = blobs()
    spec = ClusterSpec(clusterer="hdbscan", min_cluster_size=30)
    assert stability(X, lambda Zs, s: run_clustering(Zs, spec, s), n_rounds=3, seed=0) > 0.95
    assert np.isnan(stability(X, lambda Zs, s: np.full(len(Zs), -1), n_rounds=3, seed=0))


def test_resolve_min_cluster_size():
    assert resolve_min_cluster_size(0.01, 1000) == 10
    assert resolve_min_cluster_size(0.0001, 100) == 5  # never below 5
    assert resolve_min_cluster_size(25, 1000) == 25  # >= 1 is an absolute count


def test_hdbscan_finds_the_clusters_and_labels_scatter_as_noise():
    X, truth = blobs()
    labels = fit_hdbscan(X, 30)
    assert len(set(labels[labels >= 0])) == 3
    assert adjusted_rand_score(truth[truth >= 0], labels[truth >= 0]) > 0.99
    far = truth < 0
    assert far.any() and (labels[far] == -1).mean() > 0.8


def test_cluster_reduced_dispatch_and_errors():
    X, _ = blobs()
    assert set(cluster_reduced(X, ClusterSpec(clusterer="kmeans", k=3), seed=0)) == {0, 1, 2}
    assert -1 in set(cluster_reduced(X, ClusterSpec(clusterer="hdbscan", min_cluster_size=30), seed=0))
    with pytest.raises(ValueError, match="resolved"):
        cluster_reduced(X, ClusterSpec(clusterer="kmeans", k=None))
    with pytest.raises(ValueError, match="Unknown clusterer"):
        cluster_reduced(X, ClusterSpec(clusterer="nope"))
    with pytest.raises(ValueError, match="Unknown reducer"):
        run_clustering(X, ClusterSpec(reducer="nope"))


def test_spec_describe_is_readable():
    assert ClusterSpec(k=5).describe() == "no reduction → k-means (k=5)"
    assert "UMAP(5d, n_neighbors=250)" in ClusterSpec(reducer="umap", umap_neighbors=250, k=5).describe()
    assert "HDBSCAN" in ClusterSpec(clusterer="hdbscan", min_cluster_size=0.01).describe()


def test_umap_then_hdbscan_recovers_topics_without_being_told_k(toy_Z, toy_df):
    pytest.importorskip("umap")
    spec = ClusterSpec(reducer="umap", umap_components=5, umap_neighbors=30, clusterer="hdbscan", min_cluster_size=0.03)
    labels = run_clustering(toy_Z, spec, seed=0)
    assert len(set(labels[labels >= 0])) == 4
    assert (labels < 0).mean() < 0.15
    kept = labels >= 0
    assert adjusted_rand_score(toy_df["_topic"][kept], labels[kept]) > 0.9


def test_umap_is_reproducible_for_a_fixed_seed(toy_Z):
    pytest.importorskip("umap")
    from alignmap.cluster import reduce_umap

    a = reduce_umap(toy_Z[:300], n_components=3, n_neighbors=15, seed=1)
    b = reduce_umap(toy_Z[:300], n_components=3, n_neighbors=15, seed=1)
    assert a.shape == (300, 3)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_ctfidf_keywords_name_each_topic(toy_clustered):
    df, _ = toy_clustered
    kws = ctfidf_keywords(df["doc"].tolist(), df["cluster"].to_numpy(), top_n=6)
    for topic, spec in TOPICS.items():
        cluster = df.loc[df["_topic"] == topic, "cluster"].mode().iloc[0]
        tokens = [tok for kw in kws[cluster] for tok in kw.split()]
        in_topic = sum(tok in spec["vocab"] for tok in tokens)
        # keywords must come from the planted topic's vocabulary, not generic filler or other topics
        assert in_topic / len(tokens) >= 0.8, (topic, kws[cluster])


def test_ctfidf_ignores_noise_documents():
    docs = ["reward model training"] * 6 + ["circuit analysis features"] * 6 + ["unrelated lonely words"] * 4
    labels = np.array([0] * 6 + [1] * 6 + [-1] * 4)
    kws = ctfidf_keywords(docs, labels, top_n=3, min_df=1, max_df=1.0)
    assert set(kws) == {0, 1}
    assert not any("lonely" in k for ks in kws.values() for k in ks)


def test_ctfidf_skips_redundant_overlapping_terms():
    docs = ["reward model training"] * 6 + ["circuit analysis features"] * 6
    labels = np.array([0] * 6 + [1] * 6)
    kws = ctfidf_keywords(docs, labels, top_n=5, min_df=1, max_df=1.0)
    token_sets = [set(k.split()) for k in kws[0]]
    for i, a in enumerate(token_sets):
        for b in token_sets[i + 1 :]:
            assert not (a <= b or b <= a)


def test_representative_docs_are_closest_to_the_centre(toy_Z, toy_clustered):
    df, km = toy_clustered
    reps = representative_docs(toy_Z, km.labels_, km.cluster_centers_, n=3)
    for c, idx in reps.items():
        assert len(idx) == 3
        assert all(km.labels_[i] == c for i in idx)
        unit = km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c])
        member_sims = toy_Z[km.labels_ == c] @ unit
        assert toy_Z[idx[0]] @ unit == pytest.approx(member_sims.max())
        sims = [toy_Z[i] @ unit for i in idx]
        assert sims == sorted(sims, reverse=True)  # ordered best-first
