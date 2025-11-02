from __future__ import annotations
import asyncio
import time
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional, Literal

from app.schemas.batch import (
    BatchAnalyzeRequest, BatchAnalyzeResponse, ReviewResult, AspectPrediction,
    Aggregate, Counts
)
from app.pipelines.analyzer import analyze_review
from app.utils.metrics import compute_label_metrics

router = APIRouter(prefix="/api/v1", tags=["analyze"])

Sentiment = Literal["positive", "negative", "neutral"]
LABELS: List[Sentiment] = ["positive", "negative", "neutral"]

def _empty_counts() -> Counts:
    return Counts(positive=0, negative=0, neutral=0)

def _tally(preds: List[AspectPrediction], counts: Counts):
    for p in preds:
        s = p.sentiment
        if s == "positive":
            counts.positive += 1
        elif s == "negative":
            counts.negative += 1
        elif s == "neutral":
            counts.neutral += 1

@router.post("/batch-analyze", response_model=BatchAnalyzeResponse)
async def batch_analyze(req: BatchAnalyzeRequest):
    """
    Batch analyze multiple reviews.
    - Reuses analyze_review() per item (RAG + optional baseline)
    - Aggregates counts
    - If gold is provided, computes accuracy/F1 (micro/macro) for RAG (and baseline if requested)
    """
    if not req.items:
        raise HTTPException(status_code=400, detail="items must be a non-empty array")

    start = time.perf_counter()
    max_c = max(1, min(req.max_concurrency, 2)) 
    sem = asyncio.Semaphore(max_c)

    async def _process_one(item) -> ReviewResult:
        async with sem:
            result = await analyze_review(
                review=item.text,
                include_baseline=req.include_baseline
            )
            # normalize from your pipeline shape to our schema
            rag_preds = [
                AspectPrediction(
                    aspect=r["aspect"],
                    sentiment=r["sentiment"],
                    confidence=r.get("confidence"),
                    reasoning=r.get("reasoning"),
                    #evidence=r.get("evidence") if req.return_evidence else None
                )
                for r in result.get("rag_results", [])
            ]
            base_preds = None
            if req.include_baseline:
                base_preds = [
                    AspectPrediction(
                        aspect=r["aspect"],
                        sentiment=r["sentiment"],
                        confidence=r.get("confidence"),
                        reasoning=r.get("reasoning")
                    )
                    for r in result.get("baseline_results", [])
                ]
            return ReviewResult(id=item.id, rag=rag_preds, baseline=base_preds)

    # run in parallel
    results: List[ReviewResult] = await asyncio.gather(*[_process_one(it) for it in req.items])

    # aggregates
    rag_counts = _empty_counts()
    base_counts = _empty_counts() if req.include_baseline else None

    for r in results:
        _tally(r.rag, rag_counts)
        if req.include_baseline and r.baseline:
            _tally(r.baseline, base_counts)  # type: ignore[arg-type]

    aggregate = Aggregate(rag_counts=rag_counts, baseline_counts=base_counts)

    # optional evaluation
    metrics_out = None
    if req.gold:
        # adapt predictions into expected shape for metrics util
        rag_items = []
        base_items = []

        for r in results:
            rag_items.append({
                "id": r.id,
                "labels": [{"aspect": p.aspect, "sentiment": p.sentiment} for p in r.rag]
            })
            if req.include_baseline and r.baseline:
                base_items.append({
                    "id": r.id,
                    "labels": [{"aspect": p.aspect, "sentiment": p.sentiment} for p in r.baseline]
                })

        gold_items = [{"id": g.id, "labels": g.labels} for g in req.gold]
        metrics_out = {"rag": compute_label_metrics(rag_items, gold_items)}
        if req.include_baseline:
            metrics_out["baseline"] = compute_label_metrics(base_items, gold_items)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return BatchAnalyzeResponse(
        results=results,
        aggregate=aggregate,
        metrics=metrics_out,
        processing_time_ms=elapsed_ms
    )
