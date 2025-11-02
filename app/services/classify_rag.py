from app.services.embeddings import embed_text
from app.services.retrieval import knn_by_text_embedding
from app.services.llm import chat_json

RAG_TMPL = """You are great at aspect-based sentiment with sarcasm and implicit aspects.
Here are similar examples:
{evidence} 
Now analyze:
Review: {review}
Aspect: {aspect}
Context: {context}
Return strict JSON: {{"sentiment":"...", "confidence":0.0, "reasoning":"..."}}"""

async def classify_rag(review: str, aspect: str, context: str | None, k: int = 3) -> dict:
    q = f"aspect: {aspect}; context: {context or ''}"
    q_vec = await embed_text(q)
    hits = await knn_by_text_embedding(q_vec, k=k)
    ev = "\n".join([f"- {h['text']}  (sim={h['similarity']:.2f})" for h in hits])
    result = await chat_json(RAG_TMPL.format(evidence=ev, review=review, aspect=aspect, context=context or ""))
    result["evidence"] = hits
    return result
