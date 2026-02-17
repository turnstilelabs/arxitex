from pydantic import BaseModel, Field


class SemanticTag(BaseModel):
    semantic_tag: str = Field(..., min_length=8, max_length=200)
