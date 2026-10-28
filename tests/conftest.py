import numpy as np
import pytest

from alignmap.cluster import fit_kmeans
from alignmap.corpus import build_corpus
from alignmap.embed import embed_tfidf_lsa
from alignmap.toy import make_composition_toy, make_toy_records


@pytest.fixture(scope="session")
def toy_records():
    return make_toy_records(n_per_year=150, seed=0)


@pytest.fixture(scope="session")
def toy_df(toy_records):
    df, _ = build_corpus(toy_records)
    return df


@pytest.fixture(scope="session")
def toy_Z(toy_df):
    return embed_tfidf_lsa(toy_df["doc"].tolist(), dims=64, seed=0)


@pytest.fixture(scope="session")
def toy_clustered(toy_df, toy_Z):
    """toy corpus with k=4 clusters assigned (cluster ids are arbitrary)."""
    km = fit_kmeans(toy_Z, 4, seed=0)
    df = toy_df.copy()
    df["cluster"] = km.labels_
    return df, km


@pytest.fixture(scope="session")
def composition_records():
    return make_composition_toy(n_per_year=300, seed=0)


@pytest.fixture(scope="session")
def composition_clustered(composition_records):
    """Two venues whose mix drifts; clusters from k-means (k=4) on LSA vectors."""
    df, _ = build_corpus(composition_records)
    Z = embed_tfidf_lsa(df["doc"].tolist(), dims=64, seed=0)
    df = df.copy()
    df["cluster"] = fit_kmeans(Z, 4, seed=0).labels_
    return df


def cluster_of_topic(df):
    """Which k-means cluster id holds each planted topic."""
    return {t: int(df.loc[df["_topic"] == t, "cluster"].mode().iloc[0]) for t in df["_topic"].unique()}
