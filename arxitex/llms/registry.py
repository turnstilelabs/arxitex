import os
from enum import Enum
from typing import Dict, List, Set


class Provider(str, Enum):
    OPENAI = "openai"
    TOGETHER = "together"


OPENAI_MODELS: Set[str] = {
    "gpt-4o-2024-08-06",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-2025-04-14",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-5-2025-08-07",
}

TOGETHER_MODELS: Set[str] = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-V3.1",
    "openai/gpt-oss-120b",
}


DEFAULT_MODEL = os.environ.get("ARXITEX_DEFAULT_MODEL", "gpt-4o-2024-08-06")
DEFAULT_ASYNC_MODEL = os.environ.get(
    "ARXITEX_DEFAULT_ASYNC_MODEL", "gpt-5-mini-2025-08-07"
)
JSON_EXTRACTION_MODEL = os.environ.get(
    "ARXITEX_JSON_EXTRACTION_MODEL", "gpt-5-nano-2025-08-07"
)

ALLOW_UNLISTED = os.environ.get("ARXITEX_ALLOW_UNLISTED_MODELS", "false").lower() in (
    "1",
    "true",
    "yes",
)


def is_supported_model(model: str) -> bool:
    if ALLOW_UNLISTED:
        return True
    return model in OPENAI_MODELS or model in TOGETHER_MODELS


def provider_for_model(model: str) -> Provider:
    if model in OPENAI_MODELS:
        return Provider.OPENAI
    if model in TOGETHER_MODELS:
        return Provider.TOGETHER
    raise ValueError(f"Unknown model provider for {model}")


def list_supported_models() -> Dict[str, List[str]]:
    return {
        "openai": sorted(OPENAI_MODELS),
        "together": sorted(TOGETHER_MODELS),
    }
