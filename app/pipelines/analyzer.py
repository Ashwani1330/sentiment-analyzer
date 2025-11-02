from app.services.aspect_extract import extract_aspects
from app.services.classify_baseline import classify_baseline
from app.services.classify_rag import classify_rag

async def analyze_review(review: str, include_baseline: bool = True):
    aspects = await extract_aspects(review)
    rag_results, base_results = [], []
    for a in aspects:
        r = await classify_rag(review, a["aspect"], a.get("context"))
        r.pop("evidence", None)  
        rag_results.append({"aspect": a["aspect"], **r})
        if include_baseline:
            b = await classify_baseline(review, a["aspect"])
            b.pop("evidence", None)
            base_results.append({"aspect": a["aspect"], **b})
    return {"rag_results": rag_results, "baseline_results": base_results or None}
