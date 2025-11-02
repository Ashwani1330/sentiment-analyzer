import json, asyncio
from sqlalchemy import text
from app.db.engine import SessionLocal
from app.services.embeddings import embed_text

SEED_PATH = "data/seed_examples.json"

def to_pgvector(vec):
    return "[" + ",".join(str(float(x)) for x in vec) + "]"

async def main():
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    inserted = 0
    async with SessionLocal() as s:
        for it in items:
            embed_input = ((it.get("aspect_hint") or "") + " " + it["text"]).strip()
            vec = await embed_text(embed_input)        # list[float] length 1536
            pgvec = to_pgvector(vec)                   # string "[...]"

            await s.execute(
                text("""
                    INSERT INTO documents
                        (text, aspect_hint, sentiment_label, category, flags, embedding)
                    VALUES
                        (:t, :a, :sl, :c, CAST(:fl AS JSONB), CAST(:e AS vector(1536)))
                """),
                {
                    "t": it["text"],
                    "a": it.get("aspect_hint"),
                    "sl": it.get("sentiment_label"),
                    "c": it.get("category"),
                    "fl": json.dumps(it.get("flags", {})),
                    "e": pgvec,   # <-- bind the vector string!
                }
            )
            inserted += 1

        await s.commit()

    print(f"Inserted {inserted} documents from {SEED_PATH}.")

if __name__ == "__main__":
    asyncio.run(main())
