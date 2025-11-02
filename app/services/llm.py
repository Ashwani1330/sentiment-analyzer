import json,os
import anyio
from openai import AsyncOpenAI
from app.core.config import settings

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

SYSTEM_JSON = (
    "You MUST reply with a single valid JSON object. No text before/after. "
    "If uncertain, return {\"verdict\":\"cannot_evaluate\"}."
)

def _chat_sync(msg: str) -> dict:
    resp = _client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[{"role": "system", "content": "Return strict JSON."},
                  {"role": "user", "content": msg}],
        temperature=0
    )
    return resp.choices[0].message.content

async def chat_json(user_prompt: str) -> dict:
    resp = await _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_JSON},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},   # <-- key line
    )
    content = resp.choices[0].message.content or "{}"  # <-- guard
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # last-resort fallback so your route doesn't 500
        return {"verdict": "cannot_evaluate", "error": "non_json_response"}