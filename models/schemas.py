from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import BaseModel, model_validator
from core.config import MIN_CONFIDENCE, MAX_QUERY_LENGTH
from core.constants import LOW_QUALITY_PATTERNS, GARBAGE_PATTERNS, MEDICAL_DISCLAIMER

class MessageDto(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: Optional[str] = None
    text: Optional[str] = None
    history: Optional[List[MessageDto]] = None

    @property
    def query(self) -> str:
        return (self.question or self.text or "").strip()

    @model_validator(mode="after")
    def validate_query(self) -> "AskRequest":
        q = self.query
        if not q:
            raise ValueError("Request must include a non-empty 'question' or 'text' field.")
        if len(q) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds {MAX_QUERY_LENGTH} characters.")
        return self

class MatchResult(BaseModel):
    question: str
    answer: str
    confidence: float
    category: Optional[str] = None

class AskResponse(BaseModel):
    query: str
    reply: str
    model_used: str
    matches: List[MatchResult]
    is_medical: bool
    found_in_database: bool
    low_confidence: bool
    language: str
    disclaimer: str = MEDICAL_DISCLAIMER

class ImageAnalysisResponse(BaseModel):
    status: str
    analysis: Optional[str] = None
    model_used: Optional[str] = None
    disclaimer: str = MEDICAL_DISCLAIMER

@dataclass
class KnowledgeMatch:
    question: str
    answer: str
    confidence: float
    category: Optional[str] = None

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE

    @property
    def is_low_quality(self) -> bool:
        al = self.answer.lower()
        return any(p.lower() in al for p in LOW_QUALITY_PATTERNS)

    @property
    def is_garbage(self) -> bool:
        a = self.answer.strip()
        if len(a) < 12: return True
        al = a.lower()
        return any(p.lower() in al for p in GARBAGE_PATTERNS)

@dataclass
class QueryContext:
    raw_query: str
    language: str
    matches: List[KnowledgeMatch] = field(default_factory=list)
    history: Optional[List[MessageDto]] = None

    @property
    def has_reliable_matches(self) -> bool:
        return any(m.is_reliable for m in self.matches)

    @property
    def best_confidence(self) -> float:
        return self.matches[0].confidence if self.matches else 0.0
