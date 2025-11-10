import json
import re
from typing import Optional, Type, TypeVar

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .metrics import log_response_usage
from .registry import JSON_EXTRACTION_MODEL
from .retry_utils import retry_sync

T = TypeVar("T", bound=BaseModel)


def extract_after_think(text: str) -> str:
    marker = "</think>"
    index = text.find(marker)
    if index != -1:
        return text[index + len(marker) :].strip()  # noqa E23
    return text


class JSONExtractor:
    """This is a poor man fix to get structure outputs for calling together.ai models"""

    def __init__(self, model: str | None = None, client: OpenAI | None = None):
        self.client = client or OpenAI()
        self.model = model or JSON_EXTRACTION_MODEL

    def extract_json(self, text: str, output_class: Type[T]) -> Optional[T]:
        """
        Extract and validate JSON using a multi-step approach.

        Args:
            text (str): Source text to extract JSON from
            output_class (Type[BaseModel]): Target Pydantic model
            max_retries (int): Number of extraction attempts

        Returns:
            Optional[BaseModel]: Validated model instance
        """
        text = extract_after_think(text)
        local_extraction = self._local_extract_json(text, output_class)
        if local_extraction:
            logger.info(f"Successfully extracted JSON with regex: {local_extraction}")
            return local_extraction

        try:
            llm_extraction = self._llm_extract_json(text, output_class)
            if llm_extraction:
                logger.info(f"Successfully extracted JSON with LLM: {llm_extraction}")
                return llm_extraction
        except Exception as e:
            logger.error(f"Failed to extract any JSON in data with error: {e}")

        return None

    def _local_extract_json(self, text: str, output_class: Type[T]) -> Optional[T]:
        """
        Attempt local JSON extraction using regex and parsing.

        Args:
            text (str): Source text
            output_class (Type[BaseModel]): Target Pydantic model

        Returns:
            Optional[BaseModel]: Validated model instance
        """
        json_patterns = [
            r"```(?:json)?\s*({.*?})\s*```",  # Code block JSON
            r"({[^{}]+})",  # Basic JSON object
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            for match in matches:
                try:
                    parsed_json = json.loads(match)
                    return output_class.model_validate(parsed_json)
                except (json.JSONDecodeError, ValidationError):
                    continue

        return None

    @retry_sync
    def _llm_extract_json(self, text: str, output_class: Type[T]) -> Optional[T]:
        messages = [
            {"role": "system", "content": generate_extraction_prompt(output_class)},
            {"role": "user", "content": text},
        ]

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=output_class,
            )
            try:
                log_response_usage(
                    response,
                    model=self.model,
                    provider="openai",
                    context="json_extractor._llm_extract_json",
                )
            except Exception:
                pass
            return response.choices[0].message.parsed

        except Exception as e:
            logger.error(f"OpenAI JSON Extraction failed with error: {e}")
            return None


def generate_extraction_prompt(output_class: Type[T]) -> str:
    schema = output_class.model_json_schema()

    return f"""JSON Extraction Instructions:

### Schema Requirements:
{json.dumps(schema, indent=2)}

### Extraction Guidelines:
1. Extract ONLY the JSON object matching the exact schema
2. Use ONLY information explicitly present in the source text
3. For missing fields:
   - Use null for optional fields
   - Skip or null out values if no clear information exists
4. Maintain precise data types
5. Ensure 100% schema compliance

### Critical Constraints:
- NO hallucinated or invented information
- STRICTLY follow the schema structure
- Prioritize accuracy over completeness
"""
