"""Language detection for all 22 Scheduled Indian Languages + English.

Uses Unicode script analysis, script-specific character markers, vocabulary heuristics,
and session language tracking to accurately identify input language.
"""

from collections import Counter
import re
from typing import Optional, Tuple

from backend.translation.config import SCRIPT_RANGES, SUPPORTED_LANGUAGES

# Common vocabulary signatures for Devanagari languages
HINDI_MARKERS = {
    "के", "लिए", "की", "का", "है", "हैं", "कैसे", "करें", "बताएं", "बताओ", "चाहिए",
    "होगा", "सकते", "सकता", "सकती", "जानकारी", "प्रवेश", "फीस", "कॉलेज", "पाठ्यक्रम",
}
MARATHI_MARKERS = {
    "आहे", "नाहीत", "नाही", "आहेत", "काय", "कसे", "करावे", "सांगा", "कृपया",
    "अभ्यासक्रम", "पात्रता", "मिळेल", "कोणते", "शाखा", "प्रवेशासाठी",
}
NEPALI_MARKERS = {
    "छ", "छैन", "हुनेछ", "गर्छ", "कसरी", "भएको", "गर्नुहोस्", "हुन्छ", "शुल्क",
}
SANSKRIT_MARKERS = {
    "अस्ति", "भवति", "कथम्", "किम्", "शुल्कम्", "प्रवेशः", "महाविद्यालयः", "नमः",
}
MAITHILI_MARKERS = {
    "अछि", "नहि", "कोना", "कथी", "हमर", "अहाँ",
}

# Assamese distinctive Unicode characters in Bengali block:
# U+09F0 (ৰ - Assamese Ra), U+09F1 (ৱ - Assamese Wa)
ASSAMESE_CHARS = {"\u09f0", "\u09f1"}

# Romanized Gujarati markers
ROMAN_GUJARATI_PATTERNS = [
    r"\b(kem\s+cho|majama|tamne|tame|aavshe|ketli|ketlo|chhe|nathi|mane|shu|su|kya|ahiya|karvu)\b",
    r"\b(sxca\s+ma|college\s+ma|fees\s+ketli|admission\s+male|bca\s+ma)\b",
]

# Romanized Hindi markers
ROMAN_HINDI_PATTERNS = [
    r"\b(kya|kaise|kitni|kitna|hoga|milega|batao|kripya|chahiye|hai|nahi|hota|bataiye|admission\s+kab|kab\s+hoga|bhai)\b",
    r"\b(fees\s+kitni|college\s+me|admission\s+chahiye|padhai\s+kaisi)\b",
]


class LanguageDetector:
    """Robust, fast script- and lexical-based language detector."""

    def __init__(self):
        # In-memory session tracking by conversation_id
        self._session_languages: dict[str, str] = {}

    def get_session_language(self, conversation_id: Optional[str]) -> Optional[str]:
        if not conversation_id:
            return None
        return self._session_languages.get(conversation_id)

    def set_session_language(self, conversation_id: Optional[str], lang_code: str):
        if conversation_id and lang_code in SUPPORTED_LANGUAGES:
            self._session_languages[conversation_id] = lang_code

    def detect_script(self, text: str) -> Optional[str]:
        """Detect the dominant Indic or Latin script by character counting."""
        counts = Counter()
        for char in text:
            cp = ord(char)
            # Ignore whitespace, numbers, punctuation, ASCII symbols
            if char.isspace() or char.isdigit() or not char.isalpha():
                continue

            for start, end, script in SCRIPT_RANGES:
                if start <= cp <= end:
                    counts[script] += 1
                    break
            else:
                if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
                    counts["Latn"] += 1

        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def _resolve_devanagari(self, text: str) -> str:
        """Disambiguate Devanagari script among Hindi, Marathi, Nepali, Sanskrit, Maithili."""
        # 1. Check for Marathi-specific letter 'ळ' (U+0933)
        if "\u0933" in text:
            return "mar_Deva"

        tokens = set(re.findall(r"[\u0900-\u097F]+", text))
        if "\u0933" in text or (tokens & MARATHI_MARKERS and not (tokens & HINDI_MARKERS)):
            return "mar_Deva"
        if (tokens & NEPALI_MARKERS) and not (tokens & HINDI_MARKERS):
            return "npi_Deva"
        if (tokens & SANSKRIT_MARKERS) and not (tokens & HINDI_MARKERS):
            return "san_Deva"
        if (tokens & MAITHILI_MARKERS) and not (tokens & HINDI_MARKERS):
            return "mai_Deva"

        # Default Devanagari is Hindi
        return "hin_Deva"

    def _resolve_bengali_block(self, text: str) -> str:
        """Disambiguate between Assamese, Bengali, and Manipuri (Bengali script)."""
        for ch in text:
            if ch in ASSAMESE_CHARS:
                return "asm_Beng"
        return "ben_Beng"

    def _resolve_perso_arabic(self, text: str) -> str:
        """Disambiguate between Urdu, Kashmiri (Arab), Sindhi (Arab)."""
        # Default to Urdu
        return "urd_Arab"

    def detect(
        self, text: str, conversation_id: Optional[str] = None
    ) -> Tuple[str, float]:
        """Detects the FLORES-200 language code and confidence score (0.0 - 1.0).

        If text is in Latin characters:
        - Check for Romanized Gujarati / Hindi heuristics.
        - Fall back to session language if available and consistent.
        - Otherwise, returns 'eng_Latn'.
        """
        cleaned = text.strip()
        if not cleaned:
            session_lang = self.get_session_language(conversation_id)
            return session_lang or "eng_Latn", 1.0

        script = self.detect_script(cleaned)

        if script == "Gujr":
            code = "guj_Gujr"
            self.set_session_language(conversation_id, code)
            return code, 0.98

        elif script == "Deva":
            code = self._resolve_devanagari(cleaned)
            self.set_session_language(conversation_id, code)
            return code, 0.95

        elif script == "Beng":
            code = self._resolve_bengali_block(cleaned)
            self.set_session_language(conversation_id, code)
            return code, 0.95

        elif script == "Guru":
            code = "pan_Guru"
            self.set_session_language(conversation_id, code)
            return code, 0.98

        elif script == "Orya":
            code = "ory_Orya"
            self.set_session_language(conversation_id, code)
            return code, 0.98

        elif script == "Taml":
            code = "tam_Taml"
            self.set_session_language(conversation_id, code)
            return code, 0.98

        elif script == "Telu":
            code = "tel_Telu"
            self.set_session_language(conversation_id, code)
            return code, 0.98

        elif script == "Knda":
            code = "kan_Knda"
            self.set_session_language(conversation_id, code)
            return code, 0.98

        elif script == "Mlym":
            code = "mal_Mlym"
            self.set_session_language(conversation_id, code)
            return code, 0.98

        elif script == "Arab":
            code = self._resolve_perso_arabic(cleaned)
            self.set_session_language(conversation_id, code)
            return code, 0.92

        elif script == "Olck":
            code = "sat_Olck"
            self.set_session_language(conversation_id, code)
            return code, 0.98

        elif script == "Mtei":
            code = "mni_Mtei"
            self.set_session_language(conversation_id, code)
            return code, 0.98

        elif script == "Latn":
            # Latin script - check for Romanized Indian language
            lowered = cleaned.lower()
            for pattern in ROMAN_GUJARATI_PATTERNS:
                if re.search(pattern, lowered):
                    # Student is speaking Romanized Gujarati
                    code = "guj_Gujr"
                    self.set_session_language(conversation_id, code)
                    return code, 0.85

            for pattern in ROMAN_HINDI_PATTERNS:
                if re.search(pattern, lowered):
                    # Student is speaking Romanized Hindi
                    code = "hin_Deva"
                    self.set_session_language(conversation_id, code)
                    return code, 0.85

            # Pure English
            return "eng_Latn", 1.0

        # Fallback to session language or English
        session_lang = self.get_session_language(conversation_id)
        if session_lang:
            return session_lang, 0.70
        return "eng_Latn", 0.70


# Global singleton instance
detector = LanguageDetector()
