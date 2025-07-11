from pydantic import BaseModel


class StupidLemmaDetectionResult(BaseModel):
    is_technical_result: int
    mathlib_ready: int
    
    technical_result_reason: str
    mathlib_reason: str
    key_concepts: list[str]