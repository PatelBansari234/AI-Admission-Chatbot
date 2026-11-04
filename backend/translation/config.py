"""Configuration and metadata for IndicTrans2 22 Scheduled Indian Languages support.

Lists all 22 languages recognized under the Eighth Schedule of the Indian Constitution,
their IndicTrans2 FLORES codes, native script names, ISO-639 codes, and runtime settings.
"""

import os
from typing import NamedTuple


class LanguageInfo(NamedTuple):
    code: str              # IndicTrans2 FLORES code e.g. "hin_Deva"
    name: str              # English name e.g. "Hindi"
    native_name: str       # Native script name e.g. "हिन्दी"
    script: str            # Script code e.g. "Deva"
    iso_code: str          # ISO 639-1 / 639-2 code e.g. "hi"
    is_scheduled: bool     # True for 22 8th Schedule languages


# All 22 Scheduled Indian Languages + English
SUPPORTED_LANGUAGES: dict[str, LanguageInfo] = {
    "eng_Latn": LanguageInfo("eng_Latn", "English", "English", "Latn", "en", False),
    # 22 Scheduled Languages:
    "asm_Beng": LanguageInfo("asm_Beng", "Assamese", "অসমীয়া", "Beng", "as", True),
    "ben_Beng": LanguageInfo("ben_Beng", "Bengali", "বাংলা", "Beng", "bn", True),
    "brx_Deva": LanguageInfo("brx_Deva", "Bodo", "बर'", "Deva", "brx", True),
    "doi_Deva": LanguageInfo("doi_Deva", "Dogri", "डोगरी", "Deva", "doi", True),
    "guj_Gujr": LanguageInfo("guj_Gujr", "Gujarati", "ગુજરાતી", "Gujr", "gu", True),
    "hin_Deva": LanguageInfo("hin_Deva", "Hindi", "हिन्दी", "Deva", "hi", True),
    "kan_Knda": LanguageInfo("kan_Knda", "Kannada", "ಕನ್ನಡ", "Knda", "kn", True),
    "kas_Arab": LanguageInfo("kas_Arab", "Kashmiri (Perso-Arabic)", "کٲشُر", "Arab", "ks", True),
    "kas_Deva": LanguageInfo("kas_Deva", "Kashmiri (Devanagari)", "कॉशुर", "Deva", "ks", True),
    "gom_Deva": LanguageInfo("gom_Deva", "Konkani", "कोंकणी", "Deva", "kok", True),
    "mai_Deva": LanguageInfo("mai_Deva", "Maithili", "मैथिली", "Deva", "mai", True),
    "mal_Mlym": LanguageInfo("mal_Mlym", "Malayalam", "മലയാളം", "Mlym", "ml", True),
    "mar_Deva": LanguageInfo("mar_Deva", "Marathi", "मराठी", "Deva", "mr", True),
    "mni_Beng": LanguageInfo("mni_Beng", "Manipuri (Bengali)", "মৈতৈলোন্", "Beng", "mni", True),
    "mni_Mtei": LanguageInfo("mni_Mtei", "Manipuri (Meitei)", "ꯃꯩꯇꯩꯂꯣꯟ", "Mtei", "mni", True),
    "npi_Deva": LanguageInfo("npi_Deva", "Nepali", "नेपाली", "Deva", "ne", True),
    "ory_Orya": LanguageInfo("ory_Orya", "Odia", "ଓଡ଼ିଆ", "Orya", "or", True),
    "pan_Guru": LanguageInfo("pan_Guru", "Punjabi", "ਪੰਜਾਬੀ", "Guru", "pa", True),
    "san_Deva": LanguageInfo("san_Deva", "Sanskrit", "संस्कृतम्", "Deva", "sa", True),
    "sat_Olck": LanguageInfo("sat_Olck", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ", "Olck", "sat", True),
    "snd_Arab": LanguageInfo("snd_Arab", "Sindhi (Arabic)", "سنڌي", "Arab", "sd", True),
    "snd_Deva": LanguageInfo("snd_Deva", "Sindhi (Devanagari)", "सिन्धी", "Deva", "sd", True),
    "tam_Taml": LanguageInfo("tam_Taml", "Tamil", "தமிழ்", "Taml", "ta", True),
    "tel_Telu": LanguageInfo("tel_Telu", "Telugu", "తెలుగు", "Telu", "te", True),
    "urd_Arab": LanguageInfo("urd_Arab", "Urdu", "اردو", "Arab", "ur", True),
}

# Unicode Script Ranges -> IndicTrans2 Script Codes
SCRIPT_RANGES = [
    (0x0A80, 0x0AFF, "Gujr"),  # Gujarati
    (0x0900, 0x097F, "Deva"),  # Devanagari (Hindi, Marathi, Nepali, Sanskrit, Maithili, Bodo, Dogri, Konkani)
    (0x0980, 0x09FF, "Beng"),  # Bengali / Assamese
    (0x0A00, 0x0A7F, "Guru"),  # Gurmukhi (Punjabi)
    (0x0B00, 0x0B7F, "Orya"),  # Odia
    (0x0B80, 0x0BFF, "Taml"),  # Tamil
    (0x0C00, 0x0C7F, "Telu"),  # Telugu
    (0x0C80, 0x0CFF, "Knda"),  # Kannada
    (0x0D00, 0x0D7F, "Mlym"),  # Malayalam
    (0x0600, 0x06FF, "Arab"),  # Perso-Arabic (Urdu, Kashmiri, Sindhi)
    (0x1C50, 0x1C7F, "Olck"),  # Ol Chiki (Santali)
    (0xABC0, 0xABFF, "Mtei"),  # Meitei Mayek (Manipuri)
]

# Feature flags & settings from environment
TRANSLATION_ENABLED = os.getenv("INDIC_TRANSLATION_ENABLED", "true").lower() in ("true", "1", "yes")
TRANSLATION_DEVICE = os.getenv("INDIC_TRANSLATION_DEVICE", "auto")  # 'cuda' or 'cpu'
TRANSLATION_MAX_INPUT_LEN = int(os.getenv("INDIC_TRANSLATION_MAX_INPUT_LEN", "512"))
TRANSLATION_MAX_OUTPUT_LEN = int(os.getenv("INDIC_TRANSLATION_MAX_OUTPUT_LEN", "512"))
TRANSLATION_BEAM_SIZE = int(os.getenv("INDIC_TRANSLATION_BEAM_SIZE", "1"))  # 1 for fast greedy, 4 for high-precision
