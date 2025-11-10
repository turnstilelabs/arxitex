import dataclasses
import json
import time

import httpx
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from together import AsyncTogether, Together

from .json_extractor import JSONExtractor
from .metrics import TokenUsage, log_response_usage, log_usage
from .prompt import Prompt
from .prompt_cache import get_prompt_result, save_prompt_result
from .registry import (
    DEFAULT_ASYNC_MODEL,
    DEFAULT_MODEL,
    Provider,
    is_supported_model,
    list_supported_models,
    provider_for_model,
)
from .retry_utils import retry_async, retry_sync

timeout = httpx.Timeout(30.0, connect=5.0)


@retry_sync
def run_openai(prompt, model, output_class):
    client = OpenAI()
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=output_class,
    )
    # usage logging (best-effort)
    try:
        log_response_usage(
            response, model=model, provider="openai", context="llms.run_openai"
        )
    except Exception:
        pass
    return response.choices[0].message.parsed


@retry_sync
def run_together(prompt, model, output_class):
    client = Together()
    combined_prompt = f"{prompt.system}\n{prompt.user}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.6,
    )
    content = resp.choices[0].message.content
    # usage logging (best-effort)
    try:
        log_response_usage(
            resp, model=model, provider="together", context="llms.run_together"
        )
    except Exception:
        pass
    logger.info(f"Raw response: {content}")

    json_extractor = JSONExtractor()
    return json_extractor.extract_json(content, output_class)


def _run_prompt(prompt: Prompt, model: str, output_class):
    logger.info("Run LLM prompt: " + json.dumps(dataclasses.asdict(prompt), indent=4))
    start_time = time.time()

    if not model:
        logger.warning(f"No model specified, defaulting to {DEFAULT_MODEL}")
        model = DEFAULT_MODEL

    if not is_supported_model(model):
        supported = list_supported_models()
        raise ValueError(
            f"Unsupported model: {model}. Supported models: {json.dumps(supported)}"
        )

    provider = provider_for_model(model)
    logger.info(f"LLM model: {model}")

    if provider == Provider.OPENAI:
        result = run_openai(prompt, model, output_class)
    else:
        result = run_together(prompt, model, output_class)

    logger.info(f"LLM Output: {result}")
    logger.info(f"Got LLM response: {time.time() - start_time:.1f} seconds")
    return result


def execute_prompt(
    prompt: Prompt, output_class: str, model: str = "gpt-4o-2024-08-06"
) -> str:
    cache_hit = get_prompt_result(prompt, model)
    if cache_hit is not None:
        logger.info("Prompt cache hit")
        try:
            prov = provider_for_model(model).value
        except Exception:
            prov = "unknown"
        log_usage(
            TokenUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                model=model,
                provider=prov,
                cached=True,
                context="llms.execute_prompt.cache",
            )
        )
        if issubclass(output_class, BaseModel):
            return output_class.model_validate(cache_hit)
        elif hasattr(output_class, "from_dict"):
            return output_class.from_dict(cache_hit)
        else:
            return output_class(cache_hit)
    result = _run_prompt(prompt, model, output_class)
    save_prompt_result(prompt, model, result)
    return result


# --- ASYNCHRONOUS FUNCTIONS  ---


@retry_async
async def arun_openai(prompt, model, output_class):
    client = AsyncOpenAI()
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]

    response = await client.beta.chat.completions.parse(
        model=model, messages=messages, response_format=output_class, timeout=timeout
    )
    # usage logging (best-effort)
    try:
        log_response_usage(
            response, model=model, provider="openai", context="llms.arun_openai"
        )
    except Exception:
        pass
    return response.choices[0].message.parsed


@retry_async
async def arun_together(prompt, model, output_class):
    client = AsyncTogether()
    combined_prompt = f"{prompt.system}\n{prompt.user}"
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.6,
    )
    content = resp.choices[0].message.content
    # usage logging (best-effort)
    try:
        log_response_usage(
            resp, model=model, provider="together", context="llms.arun_together"
        )
    except Exception:
        pass
    logger.info(f"Raw response: {content}")

    json_extractor = JSONExtractor()
    return json_extractor.extract_json(content, output_class)


async def _arun_prompt(prompt: Prompt, model: str, output_class):
    if not model:
        logger.warning(f"No model specified, defaulting to {DEFAULT_ASYNC_MODEL}")
        model = DEFAULT_ASYNC_MODEL

    if not is_supported_model(model):
        supported = list_supported_models()
        raise ValueError(
            f"Unsupported model: {model}. Supported models: {json.dumps(supported)}"
        )

    provider = provider_for_model(model)

    if provider == Provider.OPENAI:
        result = await arun_openai(prompt, model, output_class)
    else:
        result = await arun_together(prompt, model, output_class)

    logger.info(f"LLM Output: {result}")
    return result


async def aexecute_prompt(
    prompt: Prompt, output_class: str, model: str = "gpt-5-mini-2025-08-07"
) -> str:
    cache_hit = get_prompt_result(prompt, model)
    if cache_hit is not None:
        logger.info("Prompt cache hit")
        try:
            prov = provider_for_model(model).value
        except Exception:
            prov = "unknown"
        log_usage(
            TokenUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                model=model,
                provider=prov,
                cached=True,
                context="llms.aexecute_prompt.cache",
            )
        )
        if issubclass(output_class, BaseModel):
            return output_class.model_validate(cache_hit)
        elif hasattr(output_class, "from_dict"):
            return output_class.from_dict(cache_hit)
        else:
            return output_class(cache_hit)

    result = await _arun_prompt(prompt, model, output_class)
    save_prompt_result(prompt, model, result)
    return result
