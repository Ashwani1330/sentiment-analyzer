from pydantic import BaseModel, Field
from typing import List, Optional, Any

class AnalyzeIn(BaseModel):
    review_text: str = Field(..., min_length=1)
    include_baseline: bool = True
    #return_evidence: bool = True

class Evidence(BaseModel):
    text: str
    similarity: float
    source_document_id: Optional[int] = None

class AspectOut(BaseModel):
    aspect: str
    sentiment: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    #evidence: Optional[List[Evidence]] = None

class AnalyzeOut(BaseModel):
    rag_results: List[AspectOut]
    baseline_results: Optional[List[AspectOut]] = None
    processing_time_ms: int
