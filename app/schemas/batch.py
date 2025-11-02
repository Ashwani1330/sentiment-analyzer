from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field

Sentiment = Literal["positive", "negative", "neutral"]

class BatchItem(BaseModel):
    id: str = Field(..., description="Your external id for the review")
    text: str = Field(..., description="The raw review text")

class GoldLabel(BaseModel):
    id: str
    labels: List[Dict[str, Sentiment]]  # [{ "aspect": "...", "sentiment": "positive" }]

class BatchAnalyzeRequest(BaseModel):
    items: List[BatchItem]
    include_baseline: bool = False
    return_evidence: bool = False
    gold: Optional[List[GoldLabel]] = None  # optional evaluation payload
    max_concurrency: int = 6                # cap parallelism

class AspectPrediction(BaseModel):
    aspect: str
    sentiment: Sentiment
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    #evidence: Optional[list] = None  # passthrough from your pipeline if requested

class ReviewResult(BaseModel):
    id: str
    rag: List[AspectPrediction]
    baseline: Optional[List[AspectPrediction]] = None

class Counts(BaseModel):
    positive: int = 0
    negative: int = 0
    neutral: int = 0

class Aggregate(BaseModel):
    rag_counts: Counts
    baseline_counts: Optional[Counts] = None

class MetricsBlock(BaseModel):
    precision: float
    recall: float
    f1: float
    accuracy: Optional[float] = None
    support: Optional[int] = None

class EvalMetrics(BaseModel):
    micro: MetricsBlock
    macro: MetricsBlock
    per_label: Dict[Sentiment, MetricsBlock]

class BatchAnalyzeResponse(BaseModel):
    results: List[ReviewResult]
    aggregate: Aggregate
    metrics: Optional[Dict[str, EvalMetrics]] = None  # {"rag": {...}, "baseline": {...}}
    processing_time_ms: int
