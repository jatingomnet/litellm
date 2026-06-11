"""
Support for FastRouter's OpenAI-compatible `/v1/chat/completions` endpoint.

FastRouter is an OpenAI-compatible LLM aggregator; calls are dispatched via the
shared OpenAI chat pipeline with a thin response hook to surface the `usage.cost`
field that FastRouter returns on every completion.

Docs: https://docs.fastrouter.ai/api-reference/chat-completions
"""

from typing import Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import FastRouterException


class FastRouterConfig(OpenAIGPTConfig):
    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        model_response = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

        try:
            usage = (raw_response.json() or {}).get("usage") or {}
            cost = usage.get("cost")
            if cost is not None:
                if not hasattr(model_response, "_hidden_params"):
                    model_response._hidden_params = {}
                additional_headers = model_response._hidden_params.setdefault(
                    "additional_headers", {}
                )
                additional_headers["llm_provider-x-litellm-response-cost"] = float(cost)
        except Exception:
            pass

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return FastRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
