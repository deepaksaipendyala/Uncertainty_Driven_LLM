from types import SimpleNamespace

from token_self_repair.llm.openai_provider import OpenAIProvider


class DummyCompletions:
    def __init__(self, response):
        self._response = response
        self.captured_kwargs = None

    def create(self, **kwargs):
        self.captured_kwargs = kwargs
        return self._response


class DummyClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=DummyCompletions(response))


def _make_logprob_entry(token: str, logprob: float, *top_alts):
    top = [SimpleNamespace(token=alt_token, logprob=alt_prob) for alt_token, alt_prob in top_alts]
    return SimpleNamespace(token=token, logprob=logprob, top_logprobs=top)


def _make_response(entries):
    logprobs = SimpleNamespace(content=entries)
    choice = SimpleNamespace(logprobs=logprobs)
    return SimpleNamespace(choices=[choice])


def test_openai_provider_emits_fixed_width_logits():
    entries = [
        _make_logprob_entry("Hello", -0.05, ("Hi", -1.1), ("Hey", -2.0)),
        _make_logprob_entry(" world", -0.1, (" earth", -1.5)),
    ]
    response = _make_response(entries)
    provider = OpenAIProvider(model="stub", client=DummyClient(response), top_logprobs=3)

    tokens = list(provider.generate("Hello?", max_tokens=4))

    assert [tok.token for tok in tokens] == ["Hello", " world"]
    assert all(len(tok.logits) == 3 for tok in tokens)
    # Padding should push very negative values when there are not enough alternatives
    assert tokens[1].logits[-1] == -100.0

    captured = provider.client.chat.completions.captured_kwargs
    assert captured["logprobs"] is True
    assert captured["top_logprobs"] == 3
