import hashlib
import json
import logging
import os
from typing import Any

from pydantic import BaseModel

from .prompt import Prompt

PROMPTS_CACHE_PATH = os.environ.get("PROMPTS_CACHE_PATH") or "./prompts_cache"
if not PROMPTS_CACHE_PATH:
    raise EnvironmentError("PROMPTS_CACHE_PATH is not set.")


def serialize_object(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _hash_prompt(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _file_path(prompt: Prompt, model: str) -> str:
    h = _hash_prompt(prompt.system + prompt.user + model)
    p = f"{PROMPTS_CACHE_PATH}/{prompt.id}-{h}.json"
    logging.info(f"Prompt cache file path: {p}")
    return p


def get_prompt_result(prompt: Prompt, model: str) -> str | None:
    file_path = _file_path(prompt, model)
    if os.path.exists(file_path):
        with open(file_path) as file:
            return json.load(file)
    else:
        return None


def save_prompt_result(prompt: Prompt, model: str, data: Any) -> None:
    file_path = _file_path(prompt, model)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        if isinstance(data, list):
            result_dicts = [
                (
                    d.to_dict()
                    if hasattr(d, "to_dict")
                    else d.dict() if hasattr(d, "dict") else d
                )
                for d in data
            ]
            json.dump(result_dicts, f, indent=2)
        else:
            result_dict = (
                data.to_dict()
                if hasattr(data, "to_dict")
                else data.dict() if hasattr(data, "dict") else data
            )
            json.dump(result_dict, f, indent=2)
