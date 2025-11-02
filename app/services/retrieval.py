# app/services/retrieval.py
from sqlalchemy import text
from app.db.engine import SessionLocal
from app.models.document import EMBED_DIM

def _to_pgvector(vec):
    return "[" + ",".join(str(float(x)) for x in vec) + "]"

async def knn_by_text_embedding(query_vec: list[float], k: int = 3):
    sql = text("""
    SELECT id, text, aspect_hint, sentiment_label,
           1 - (embedding <-> CAST(:q AS vector(1536))) AS similarity
    FROM documents
    ORDER BY embedding <-> CAST(:q AS vector(1536))
    LIMIT :k
""")
    async with SessionLocal() as s:
        rows = (await s.execute(sql, {"q": _to_pgvector(query_vec), "k": int(k)})).mappings().all()
        return [dict(r) for r in rows]
