import numpy as np
import pytest

import alignmap.embed as embed_mod
from alignmap.embed import embed, embed_tfidf_lsa


def docs():
    return [f"reward model preference {i}" if i % 2 else f"circuit feature activation {i}" for i in range(40)]


def test_tfidf_embeddings_are_unit_norm_and_deterministic():
    a = embed_tfidf_lsa(docs(), dims=8, seed=0, min_df=1)
    b = embed_tfidf_lsa(docs(), dims=8, seed=0, min_df=1)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(a, b)


def test_tiny_vocabulary_is_a_clear_error():
    with pytest.raises(ValueError, match="too small or too uniform"):
        embed_tfidf_lsa(["alpha beta"] * 4, dims=4, min_df=1, max_df=1.0, ngram_range=(1, 1))


def test_unknown_embedder_is_a_clear_error():
    with pytest.raises(ValueError, match="Unknown embedder"):
        embed(docs(), "word2vec")


def test_cache_is_used_on_the_second_call_and_keyed_by_documents_and_parameters(tmp_path, monkeypatch):
    calls = []
    real = embed_mod.embed_tfidf_lsa

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(embed_mod, "embed_tfidf_lsa", counting)
    d = docs()
    first = embed(d, "tfidf", dims=5, cache_dir=tmp_path)
    second = embed(d, "tfidf", dims=5, cache_dir=tmp_path)
    assert len(calls) == 1 and len(list(tmp_path.glob("emb_*.npy"))) == 1
    np.testing.assert_array_equal(first, second)

    embed(d, "tfidf", dims=6, cache_dir=tmp_path)  # other parameters: new entry
    embed(d[:-1], "tfidf", dims=5, cache_dir=tmp_path)  # other documents: new entry
    assert len(calls) == 3 and len(list(tmp_path.glob("emb_*.npy"))) == 3


def test_sbert_cache_key_depends_on_model_and_input_format(tmp_path, monkeypatch):
    seen = []

    def fake_sbert(d, *, model_name, max_seq_length, doc_prefix, **_):
        seen.append((model_name, max_seq_length, doc_prefix))
        return np.ones((len(d), 3)) / np.sqrt(3)

    monkeypatch.setattr(embed_mod, "embed_sbert", fake_sbert)
    d = docs()
    embed(d, "sbert", sbert_model="m1", max_seq_length=512, cache_dir=tmp_path)
    embed(d, "sbert", sbert_model="m1", max_seq_length=512, cache_dir=tmp_path)  # cached
    embed(d, "sbert", sbert_model="m2", max_seq_length=512, cache_dir=tmp_path)
    embed(d, "sbert", sbert_model="m1", max_seq_length=256, cache_dir=tmp_path)
    embed(d, "sbert", sbert_model="m1", max_seq_length=512, doc_prefix="passage: ", cache_dir=tmp_path)
    assert seen == [("m1", 512, ""), ("m2", 512, ""), ("m1", 256, ""), ("m1", 512, "passage: ")]
