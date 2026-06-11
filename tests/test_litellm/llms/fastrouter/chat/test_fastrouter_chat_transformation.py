import os
import sys
from unittest.mock import Mock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.fastrouter.chat.transformation import FastRouterConfig
from litellm.llms.fastrouter.common_utils import FastRouterException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.utils import Choices, Message, ModelResponse, Usage


def _mock_response_with_usage(usage: dict) -> httpx.Response:
    mock = Mock(spec=httpx.Response)
    mock.json.return_value = {
        "id": "fr-123",
        "model": "anthropic/claude-sonnet-4.5",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "usage": usage,
    }
    mock.headers = {}
    return mock


def _empty_model_response() -> ModelResponse:
    return ModelResponse(
        id="fr-123",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="Hi", role="assistant"),
            )
        ],
        created=1234567890,
        model="anthropic/claude-sonnet-4.5",
        object="chat.completion",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def _call_transform_response(raw_response: httpx.Response) -> ModelResponse:
    config = FastRouterConfig()
    model_response = _empty_model_response()
    with patch.object(
        OpenAIGPTConfig, "transform_response", return_value=model_response
    ):
        return config.transform_response(
            model="fastrouter/anthropic/claude-sonnet-4.5",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=Mock(),
            request_data={},
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )


def test_transform_response_extracts_cost_from_usage():
    raw = _mock_response_with_usage(
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.00042}
    )

    result = _call_transform_response(raw)

    assert (
        result._hidden_params["additional_headers"][
            "llm_provider-x-litellm-response-cost"
        ]
        == 0.00042
    )


def test_transform_response_no_cost_does_not_set_header():
    raw = _mock_response_with_usage(
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    )

    result = _call_transform_response(raw)

    additional_headers = getattr(result, "_hidden_params", {}).get(
        "additional_headers", {}
    )
    assert "llm_provider-x-litellm-response-cost" not in additional_headers


def test_transform_response_invalid_json_does_not_raise():
    bad_response = Mock(spec=httpx.Response)
    bad_response.json.side_effect = ValueError("not json")
    bad_response.headers = {}

    result = _call_transform_response(bad_response)

    additional_headers = getattr(result, "_hidden_params", {}).get(
        "additional_headers", {}
    )
    assert "llm_provider-x-litellm-response-cost" not in additional_headers


def test_get_error_class_returns_fastrouter_exception():
    config = FastRouterConfig()
    exc = config.get_error_class(
        error_message="boom", status_code=429, headers={"x-test": "1"}
    )
    assert isinstance(exc, FastRouterException)
    assert exc.status_code == 429


@pytest.mark.parametrize(
    "input_model,expected_model",
    [
        ("fastrouter/openai/gpt-4o", "openai/gpt-4o"),
        ("fastrouter/anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.5"),
        ("fastrouter/auto", "fastrouter/auto"),
    ],
)
def test_get_llm_provider_prefix_strip(input_model: str, expected_model: str):
    model, provider, _, _ = litellm.get_llm_provider(model=input_model)
    assert provider == "fastrouter"
    assert model == expected_model


def test_fastrouter_config_registered_on_module():
    assert hasattr(litellm, "FastRouterConfig")
    assert litellm.FastRouterConfig is FastRouterConfig
