# app/api/routers/analyze.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.pipelines.analyzer import analyze_review
from time import perf_counter

router = APIRouter(prefix="/api/v1")

class AnalyzeIn(BaseModel):
    review_text: str
    include_baseline: bool = True
   # return_evidence: bool = True

@router.post("/analyze")
async def analyze(body: AnalyzeIn):
    t0 = perf_counter()
    out = await analyze_review(body.review_text, body.include_baseline)
    out["processing_time_ms"] = int((perf_counter()-t0)*1000)
    return out
