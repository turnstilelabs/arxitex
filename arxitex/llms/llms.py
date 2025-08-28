import dataclasses
import json
import time
import httpx
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from together import AsyncTogether, Together

from .json_extractor import JSONExtractor
from .prompt import Prompt
from .prompt_cache import get_prompt_result, save_prompt_result

timeout = httpx.Timeout(30.0, connect=5.0)

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
    return response.choices[0].message.parsed

def run_together(prompt, model, output_class):
    client = Together()
    combined_prompt = f"{prompt.system}\n{prompt.user}"
    response = (
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=0.6,
        )
        .choices[0]
        .message.content
    )
    logger.info(f"Raw response: {response}")

    json_extractor = JSONExtractor()
    return json_extractor.extract_json(response, output_class)

def _run_prompt(prompt: Prompt, model: str, output_class):
    logger.info("Run LLM prompt: " + json.dumps(dataclasses.asdict(prompt), indent=4))
    start_time = time.time()

    openai_models = [
        "gpt-4.1-mini-2025-04-14",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07"
    ]

    together_models = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    ]

    model_runners = {
        **{m: run_openai for m in openai_models},
        **{m: run_together for m in together_models},
    }

    if not model:
        logger.warning("No model specified, defaulting to gpt-4o-2024-08-06")
        model = "gpt-4o-2024-08-06"

    if model not in model_runners:
        raise ValueError(f"Unsupported model: {model}")

    logger.info(f"LLM model: {model}")
    
    result = model_runners[model](prompt, model, output_class)
    
    logger.info(f"LLM Output: {result}")
    logger.info(f"Got LLM response: {time.time() - start_time:.1f} seconds")
    return result

def execute_prompt(
    prompt: Prompt, output_class: str, model: str = "gpt-4o-2024-08-06"
) -> str:
    cache_hit = get_prompt_result(prompt, model)
    if cache_hit is not None:
        logger.info("Prompt cache hit")
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

async def arun_openai(prompt, model, output_class):
    client = AsyncOpenAI()
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=output_class,
        timeout=timeout
    )
    return response.choices[0].message.parsed

async def arun_together(prompt, model, output_class):
    client = AsyncTogether()
    combined_prompt = f"{prompt.system}\n{prompt.user}"
    response = (
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=0.6,
        )
    ).choices[0].message.content
    logger.info(f"Raw response: {response}")

    json_extractor = JSONExtractor()
    return json_extractor.extract_json(response, output_class)

async def _arun_prompt(prompt: Prompt, model: str, output_class):
    #logger.info("Run async LLM prompt: " + json.dumps(dataclasses.asdict(prompt), indent=2))
    start_time = time.time()

    openai_models = [
        "gpt-4.1-2025-04-14",
        "gpt-5-mini-2025-08-07",
        "gpt-5-2025-08-07",
        "gpt-5-nano-2025-08-07"
    ]

    together_models = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-V3.1",
        "openai/gpt-oss-120b"
    ]

    model_runners = {
        **{m: arun_openai for m in openai_models},
        **{m: arun_together for m in together_models},
    }

    if not model:
        logger.warning("No model specified, defaulting to gpt-4o-2024-08-06")
        model = "gpt-4o-2024-08-06"

    if model not in model_runners:
        raise ValueError(f"Unsupported model: {model}")    
    result = await model_runners[model](prompt, model, output_class)
    
    logger.info(f"LLM Output: {result}")
    return result


async def aexecute_prompt(
    prompt: Prompt, output_class: str, model: str = "gpt-5-mini-2025-08-07"
) -> str:
    cache_hit = get_prompt_result(prompt, model)
    if cache_hit is not None:
        logger.info("Prompt cache hit")
        if issubclass(output_class, BaseModel):
            return output_class.model_validate(cache_hit)
        elif hasattr(output_class, "from_dict"):
            return output_class.from_dict(cache_hit)
        else:
            return output_class(cache_hit)
            
    result = await _arun_prompt(prompt, model, output_class)
    save_prompt_result(prompt, model, result)
    return result