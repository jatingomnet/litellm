import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))
import litellm


@pytest.mark.skipif(
    not os.environ.get("FASTROUTER_API_KEY"),
    reason="FASTROUTER_API_KEY not set",
)
def test_completion_fastrouter_basic():
    litellm._turn_on_debug()
    resp = litellm.completion(
        model="fastrouter/openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply with just the word: pong"}],
    )
    assert resp.choices[0].message.content is not None
    response_cost = (
        getattr(resp, "_hidden_params", {})
        .get("additional_headers", {})
        .get("llm_provider-x-litellm-response-cost")
    )
    assert response_cost is not None, "FastRouter usage.cost should propagate to hidden_params"
    assert float(response_cost) >= 0
