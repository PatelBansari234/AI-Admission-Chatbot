"""Gemini LLM provider for both grounded RAG response synthesis/streaming
and optional multilingual intent understanding.
"""

import json
import logging
import re
from functools import lru_cache

from backend.config import GEMINI_API_KEY, GEMINI_MODEL, MAX_NEW_TOKENS

logger = logging.getLogger(__name__)

_ACTIVE_MODEL: str | None = None
_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = """You are a fast multilingual admission intent extractor.
You analyze student messages in English, Hindi, Gujarati, or Hinglish (including informal or broken spelling).
Extract the information and respond with ONLY a JSON object:
{
  "language": "English" | "Hindi" | "Gujarati" | "Hinglish",
  "intent": "application" | "fee" | "eligibility" | "documents" | "process" | "source_query" | "general",
  "course": "BCA" | "BSc-Physics" | "BSc-Chemistry" | "BSc-Mathematics" | "BSc-Botany" | "BSc-Zoology" | "BSc-Biochemistry" | "BSc-Biotechnology" | "BA-English" | "BA-Economics" | "BA-Psychology" | "BCom-BPS" | "BCom" | "MSc-Big Data Analytics" | "MSc-Physics" | "MSc-Chemistry" | "MSc-Mathematics" | "MSc-Biotechnology" | "MA-English" | null,
  "qualification": "Science" | "Commerce" | "Arts" | null,
  "query": "clean English search keywords"
}

Rules:
- NEVER answer the question or output facts/fees/dates.
- Map course abbreviations/colloquialisms to their standard name (e.g. "bca" -> "BCA", "physics" -> "BSc-Physics", "12 sci" -> qualification="Science").
- Output ONLY the JSON object.
"""


_COURSE_MAP = {
    "physics": "BSc-Physics",
    "bsc physics": "BSc-Physics",
    "b.sc physics": "BSc-Physics",
    "bsc-physics": "BSc-Physics",
    "chemistry": "BSc-Chemistry",
    "bsc chemistry": "BSc-Chemistry",
    "b.sc chemistry": "BSc-Chemistry",
    "bsc-chemistry": "BSc-Chemistry",
    "mathematics": "BSc-Mathematics",
    "maths": "BSc-Mathematics",
    "bsc mathematics": "BSc-Mathematics",
    "bsc-mathematics": "BSc-Mathematics",
    "botany": "BSc-Botany",
    "bsc botany": "BSc-Botany",
    "bsc-botany": "BSc-Botany",
    "zoology": "BSc-Zoology",
    "bsc zoology": "BSc-Zoology",
    "bsc-zoology": "BSc-Zoology",
    "biochemistry": "BSc-Biochemistry",
    "bsc biochemistry": "BSc-Biochemistry",
    "bsc-biochemistry": "BSc-Biochemistry",
    "biotechnology": "BSc-Biotechnology",
    "biotech": "BSc-Biotechnology",
    "bsc biotechnology": "BSc-Biotechnology",
    "bsc-biotechnology": "BSc-Biotechnology",
    "bca": "BCA",
    "b.c.a": "BCA",
    "bcom": "BCom",
    "b.com": "BCom",
    "bcom bps": "BCom-BPS",
    "english": "BA-English",
    "ba english": "BA-English",
    "economics": "BA-Economics",
    "ba economics": "BA-Economics",
    "psychology": "BA-Psychology",
    "ba psychology": "BA-Psychology",
    "msc big data analytics": "MSc-Big Data Analytics",
    "msc data science": "MSc-Big Data Analytics",
    "msc physics": "MSc-Physics",
    "msc chemistry": "MSc-Chemistry",
    "msc mathematics": "MSc-Mathematics",
    "msc biotechnology": "MSc-Biotechnology",
    "ma english": "MA-English",
}


@lru_cache(maxsize=1)
def _get_client():
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.warning(f"Could not initialize Gemini client: {e}")
        return None


def understand_multilingual(question: str, recent_turns: list[dict] | None = None, context: dict | None = None) -> dict | None:
    """Uses Gemini to parse student intent and multilingual content.
    Fails open (returns None) on any network, parsing, or configuration error."""
    client = _get_client()
    if client is None:
        return None

    norm_question = (question or "").strip()
    if not norm_question:
        return None

    # Context snippet
    convo_lines = []
    for turn in (recent_turns or [])[-4:]:
        role = "Student" if turn.get("role") == "user" else "Assistant"
        convo_lines.append(f"{role}: {turn.get('content', '').strip()}")
    context_str = "\n".join(convo_lines) if convo_lines else "(none)"

    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Student Message: {norm_question}"
    )

    try:
        from google.genai import types
        # Gemini only needs a tiny JSON payload (language, intent, course, confidence).
        # 150 tokens is ample for this compact output.
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=150,
            response_mime_type="application/json",
        )
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )
        if not resp or not resp.text:
            return None

        match = _JSON_RE.search(resp.text)
        if not match:
            return None

        parsed = json.loads(match.group(0))

        def _clean(val):
            if not isinstance(val, str):
                return None
            val = val.strip()
            return val if val and val.lower() != "null" else None

        course = _clean(parsed.get("course"))
        qualification = _clean(parsed.get("qualification"))
        intent = _clean(parsed.get("intent")) or "general"
        language = _clean(parsed.get("language")) or "English"
        query = _clean(parsed.get("query")) or norm_question

        if course:
            normalized_key = course.lower().replace("-", " ").strip()
            course = _COURSE_MAP.get(normalized_key, _COURSE_MAP.get(course.lower(), course))

        needs_lookup = bool(course or qualification or intent in ("fee", "eligibility", "application"))

        # Return only the compact fields required for multilingual understanding.
        return {
            "language": language,
            "intent": intent,
            "course": course,
            "confidence": parsed.get("confidence"),
        }
    except Exception as e:
        logger.warning(f"Gemini multilingual understanding failed, falling back to local: {e}")
        return None


def _candidate_models() -> list[str]:
    global _ACTIVE_MODEL
    candidates = []
    if _ACTIVE_MODEL:
        candidates.append(_ACTIVE_MODEL)
    if GEMINI_MODEL and GEMINI_MODEL not in candidates:
        candidates.append(GEMINI_MODEL)
    for m in _FALLBACK_MODELS:
        if m not in candidates:
            candidates.append(m)
    return candidates


def _prepare_gemini_request(messages: list[dict], max_new_tokens: int):
    from google.genai import types

    system_parts = []
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        text = (msg.get("content") or "").strip()
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
        else:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        max_output_tokens=max_new_tokens,
    )
    return contents, config


def generate(messages: list[dict], max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate a grounded admission response using Gemini API."""
    global _ACTIVE_MODEL
    client = _get_client()
    if client is None:
        from backend.rag.prompt import NO_CONTEXT_ANSWER
        return NO_CONTEXT_ANSWER

    contents, config = _prepare_gemini_request(messages, max_new_tokens)
    last_err = None
    for model_name in _candidate_models():
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            _ACTIVE_MODEL = model_name
            if resp and resp.text:
                return resp.text.strip()
        except Exception as exc:
            last_err = exc
            logger.warning(f"Gemini generate failed with model {model_name}: {exc}")
            continue

    logger.error(f"All Gemini models failed in generate(): {last_err}")
    from backend.rag.prompt import NO_CONTEXT_ANSWER
    return NO_CONTEXT_ANSWER


def stream(messages: list[dict], max_new_tokens: int = MAX_NEW_TOKENS):
    """Stream grounded admission response tokens from Gemini API."""
    global _ACTIVE_MODEL
    client = _get_client()
    if client is None:
        from backend.rag.prompt import NO_CONTEXT_ANSWER
        yield NO_CONTEXT_ANSWER
        return

    contents, config = _prepare_gemini_request(messages, max_new_tokens)
    last_err = None
    for model_name in _candidate_models():
        try:
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
            yielded_any = False
            for chunk in response_stream:
                _ACTIVE_MODEL = model_name
                piece = getattr(chunk, "text", None)
                if piece:
                    yielded_any = True
                    yield piece
            if yielded_any:
                return
        except Exception as exc:
            last_err = exc
            logger.warning(f"Gemini stream failed with model {model_name}: {exc}")
            continue

    logger.error(f"All Gemini models failed in stream(): {last_err}")
    from backend.rag.prompt import NO_CONTEXT_ANSWER
    yield NO_CONTEXT_ANSWER

