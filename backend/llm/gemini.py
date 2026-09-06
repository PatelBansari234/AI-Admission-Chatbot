"""Optional multilingual language-understanding provider using Gemini.
Extracts {language, intent, course, qualification, query} from English, Hindi, Gujarati,
and Hinglish queries without generating college facts.
"""

import json
import logging
import re
from functools import lru_cache

from backend.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

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
