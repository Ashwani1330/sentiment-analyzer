# app/services/aspect_extract.py
from openai import OpenAI
from app.core.config import settings
import json

_client = OpenAI(api_key=settings.OPENAI_API_KEY)

SYSTEM = "Extract product aspects from the review and return strict JSON."
USER_TMPL = """Review: {review}
Return JSON: {{"aspects":[{{"aspect":"...", "span":[start,end], "context":"..."}}]}}"""

async def extract_aspects(review: str) -> list[dict]:
    msg = USER_TMPL.format(review=review)
    resp = _client.chat.completions.create(
    model=settings.OPENAI_CHAT_MODEL,   # optional: use your env model
    messages=[{"role":"system","content":SYSTEM},{"role":"user","content":msg}],
    temperature=0,
    response_format={"type": "json_object"}   # <-- add this line
)
    data = json.loads(resp.choices[0].message.content)
    return data.get("aspects", [])

