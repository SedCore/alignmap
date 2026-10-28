from types import SimpleNamespace

from alignmap.labeling import _parse, llm_label_clusters

BATCH = [
    {"cluster": 0, "keywords": ["reward", "preference", "feedback"], "titles": ["T1", "T2"]},
    {"cluster": 1, "keywords": ["circuits", "features"], "titles": ["T3"]},
]


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create(...)."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=[SimpleNamespace(text=reply)])


def test_valid_json_is_used():
    client = FakeClient(['{"name": "Learning from human feedback", "summary": "RLHF methods."}'] * 2)
    out = llm_label_clusters(BATCH, client=client, model="some-model")
    assert out[0] == {"name": "Learning from human feedback", "summary": "RLHF methods.", "llm": True}
    assert client.calls[0]["model"] == "some-model"
    prompt = client.calls[0]["messages"][0]["content"]
    assert "reward, preference, feedback" in prompt and "- T1" in prompt


def test_json_inside_code_fence_is_accepted():
    assert _parse('```json\n{"name": "Circuits", "summary": "s"}\n```')["name"] == "Circuits"


def test_invalid_replies_and_api_errors_fall_back_to_keywords():
    client = FakeClient(["I think this is about rewards.", RuntimeError("rate limited")])
    out = llm_label_clusters(BATCH, client=client)
    assert out[0] == {"name": "reward, preference, feedback", "summary": "", "llm": False}
    assert out[1]["name"] == "circuits, features" and out[1]["llm"] is False


def test_empty_name_is_rejected():
    assert _parse('{"name": "  ", "summary": "x"}') is None
    assert _parse("no json here") is None


def test_model_comes_from_env(monkeypatch):
    monkeypatch.setenv("ALIGNMAP_LLM_MODEL", "env-model")
    client = FakeClient(['{"name": "n", "summary": "s"}'] * 2)
    llm_label_clusters(BATCH, client=client)
    assert client.calls[0]["model"] == "env-model"
