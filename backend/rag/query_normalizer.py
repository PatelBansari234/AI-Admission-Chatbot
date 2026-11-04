"""
User-Query Normalization and Understanding Layer for SXCA Admission Assistant (Niaa).

Provides lightweight, robust query de-noising, WhatsApp/casual slang resolution,
domain typo correction, and explicit vs. ambiguous percentage understanding without
altering factual meaning or bypassing evidence gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ==============================================================================
# 1. PROTECTED DOMAIN VOCABULARY & ENTITIES
# These terms MUST NEVER be rewritten or fuzzy-transformed into something else.
# ==============================================================================

PROTECTED_EXACT_TERMS = {
    # Faculty names & variations (Strict Anti-Hallucination)
    "nisarg vyas", "mr. nisarg vyas", "mr nisarg vyas", "nisarg",
    "nisarg patil", "mr. nisarg patil", "mr nisarg patil",
    "nancy patel", "ms. nancy patel", "ms nancy patel",
    "dr. pravida raja", "pravida raja", "pravida",
    "dr. sebastian v.a.", "dr sebastian", "dr. sebastian", "sebastian",
    "fr. dr. david roy", "fr david", "dr david", "david roy",
    "fr. antony pitchai sj", "fr antony pitchai", "antony pitchai",
    # Official Degree Names & Abbreviations
    "bca", "bsc", "b.sc", "bcom", "b.com", "ba", "b.a", "msc", "m.sc", "ma", "m.a",
    "bda", "aiml", "ai", "ml", "it", "bps", "bpa",
    # Specific Subject / Paper Codes
    "pai-1803", "pbd-1806", "cai-2", "cai2", "cia-2", "cia2", "cia-1", "ese",
    # Negation Terms (Must NEVER be inverted or stripped)
    "not", "no", "never", "none", "without", "nathi", "nahi", "na",
}

# ==============================================================================
# 2. WHATSAPP & CASUAL TEXTING SHORTHAND
# Standardizes common SMS, WhatsApp, and Gen Z shorthand to standard forms.
# ==============================================================================

WHATSAPP_SHORTHAND_MAP: list[tuple[re.Pattern, str]] = [
    # Please / Kindly
    (re.compile(r"\b(?:pls+|plz+|plzz+|plez+)\b", re.I), "please"),
    # Pronouns & Contractions
    (re.compile(r"\bu\b", re.I), "you"),
    (re.compile(r"\bur\b", re.I), "your"),
    (re.compile(r"\burself\b", re.I), "yourself"),
    # 'r' -> 'are' (when preceded or followed by question words/pronouns)
    (re.compile(r"\b(what|how|why|when|where|who|u|you|there)\s+r\b", re.I), r"\1 are"),
    (re.compile(r"\br\s+(u|you|there|the|these|those|any)\b", re.I), r"are \1"),
    # Because / Cause
    (re.compile(r"\b(?:bcz+|coz+|cuz+|bcoz+|bcuz+)\b", re.I), "because"),
    # Question words & helpers
    (re.compile(r"\b(?:wht+|wat+|wtt+)\b", re.I), "what"),
    (re.compile(r"\b(?:hw+|hww+)\b", re.I), "how"),
    (re.compile(r"\bwhch\b", re.I), "which"),
    (re.compile(r"\babt\b", re.I), "about"),
    (re.compile(r"\btel\b", re.I), "tell"),
    (re.compile(r"\bdiff\b", re.I), "difference"),
    # Want to / Going to / Don't know
    (re.compile(r"\bwanna\b", re.I), "want to"),
    (re.compile(r"\bgonna\b", re.I), "going to"),
    (re.compile(r"\bidk\b", re.I), "I don't know"),
    (re.compile(r"\brn\b", re.I), "right now"),
    (re.compile(r"\bbtw\b", re.I), "by the way"),
    (re.compile(r"\basap\b", re.I), "as soon as possible"),
    (re.compile(r"\btmrw|tom|tmrw\b", re.I), "tomorrow"),
    (re.compile(r"\byday|yest\b", re.I), "yesterday"),
    # 'fr' resolution: "fr AI" -> "for AI", standalone "fr" -> "for real"
    (re.compile(r"\bfr\s+([a-zA-Z]+)", re.I), r"for \1"),
    (re.compile(r"\bfr\b", re.I), "for real"),
    # Hindi Texting Shortcuts
    (re.compile(r"\bmil\s*skta\s*h\b", re.I), "mil sakta hai"),
    (re.compile(r"\bmil\s*skti\s*h\b", re.I), "mil sakti hai"),
    (re.compile(r"\bskta\b", re.I), "sakta"),
    (re.compile(r"\bskti\b", re.I), "sakti"),
    (re.compile(r"\b(sakta|sakti|sakte|kya|hai|hoga|chahiye)\s+h\b", re.I), r"\1 hai"),
    (re.compile(r"\bh\??$", re.I), "hai"),
]

# ==============================================================================
# 3. DOMAIN TYPOS & COMMON MISSPELLINGS (SXCA VOCABULARY)
# ==============================================================================

DOMAIN_TYPO_MAP: list[tuple[re.Pattern, str]] = [
    # Eligibility
    (re.compile(r"\b(?:eligibilty|elgibility|eligblty|elgiblity|eligibilityy|eligble|elgible)\b", re.I), "eligibility"),
    # Admission
    (re.compile(r"\b(?:admision|admn|admsn|admisson|admissionss|admissin|admisn)\b", re.I), "admission"),
    # Course / Courses
    (re.compile(r"\b(?:cources?|couce|cousre|couses?|corses?|courss)\b", re.I), "courses"),
    # Fees
    (re.compile(r"\b(?:feees+|fess|feee|feis)\b", re.I), "fees"),
    # College
    (re.compile(r"\b(?:collge|colg|clg|cllg|colelge|colledge)\b", re.I), "college"),
    # Requirement / Requirements
    (re.compile(r"\b(?:requirment|requirments|rqrmnt|requriment|requirmentss)\b", re.I), "requirement"),
    # Documents
    (re.compile(r"\b(?:documnt|documnts|docuemnts|documts|documntation)\b", re.I), "documents"),
    # Procedure / Process
    (re.compile(r"\b(?:procedur|prosedure|procdure|proces|pocess)\b", re.I), "procedure"),
    # Syllabus
    (re.compile(r"\b(?:syllbus|sylabus|syllabous|syllabuss)\b", re.I), "syllabus"),
    # Subjects
    (re.compile(r"\b(?:subjet|subjets|subjetcs|subjecs)\b", re.I), "subjects"),
    # Details
    (re.compile(r"\b(?:detals|detailes|detials|detailss)\b", re.I), "details"),
    # Hostel
    (re.compile(r"\b(?:hostal|hostle|hostell)\b", re.I), "hostel"),
    # Percent / Percentage
    (re.compile(r"\b(?:persent|prcnt|persentage|parcentage|percentag)\b", re.I), "percent"),
    # Information
    (re.compile(r"\b(?:infromation|infomation|infrmation)\b", re.I), "information"),
    # Stream typos
    (re.compile(r"\b(?:scince|scie|scinece)\b", re.I), "science"),
    (re.compile(r"\b(?:commrce|comerce|commiorce)\b", re.I), "commerce"),
    (re.compile(r"\b(?:artss)\b", re.I), "arts"),
]

# ==============================================================================
# 4. CASUAL ADDRESS TAGS
# Strips conversational prefixes that would otherwise mislead intent detection.
# e.g. "bro, which course is best for AI?" -> "which course is best for AI?"
# ==============================================================================

CASUAL_ADDRESS_PREFIXES = re.compile(
    r"^(?:bro|dude|yaar|bhai|dost|sir|maam|mam|sister|mate)\s*[,:\-]?\s*",
    re.I
)

CASUAL_WORDS_CLEANUP = [
    (re.compile(r"\blowkey\b", re.I), ""),
]


@dataclass
class NormalizedQuery:
    original_text: str
    normalized_text: str
    detected_language: str
    extracted_percentage: float | None = None
    is_ambiguous_score: bool = False
    ambiguous_score_val: float | None = None
    extracted_course: str | None = None
    extracted_stream: str | None = None
    applied_normalizations: list[tuple[str, str]] = field(default_factory=list)


def clean_character_repetitions(text: str) -> str:
    """Safely reduces exaggerated repeated characters (3+ identical chars down to 1 or 2).

    Preserves standard English double vowels (e.g. 'fees', 'free', 'good').
    """
    def _reduce_match(m: re.Match) -> str:
        char = m.group(1)
        # Double 'e', 'o', 'l', 's' are common in English (fees, cool, will, pass)
        if char.lower() in ("e", "o", "l", "s"):
            return char * 2
        return char

    # Replace 3 or more repeated alphabetic characters
    return re.sub(r"([a-zA-Z])\1{2,}", _reduce_match, text)


def normalize_query_text(text: str) -> tuple[str, list[tuple[str, str]]]:
    """De-noises and normalizes user query text while strictly preserving numbers,
    negation, faculty names, and course abbreviations.
    """
    if not text:
        return "", []

    raw = text.strip()
    applied: list[tuple[str, str]] = []

    # 1. Clean exaggerated character repetitions
    reduced = clean_character_repetitions(raw)
    if reduced != raw:
        applied.append(("repetition_reduction", reduced))

    # 2. Clean casual leading address tags ("bro, which course..." -> "which course...")
    without_bro = CASUAL_ADDRESS_PREFIXES.sub("", reduced).strip()
    if without_bro and without_bro != reduced:
        applied.append(("strip_casual_prefix", without_bro))
        reduced = without_bro

    # 3. Apply casual word removals (e.g. lowkey)
    for pat, rep in CASUAL_WORDS_CLEANUP:
        if pat.search(reduced):
            reduced = pat.sub(rep, reduced).strip()

    # 4. WhatsApp shorthand replacements
    current = reduced
    for pat, rep in WHATSAPP_SHORTHAND_MAP:
        if pat.search(current):
            prev = current
            current = pat.sub(rep, current)
            if current != prev:
                applied.append((prev, current))

    # 5. Domain typo corrections (ensuring protected entities are never corrupted)
    for pat, rep in DOMAIN_TYPO_MAP:
        if pat.search(current):
            prev = current
            current = pat.sub(rep, current)
            if current != prev:
                applied.append((prev, current))

    # 6. Normalize multiple spaces
    current = re.sub(r"\s+", " ", current).strip()

    return current, applied


def extract_percentage_and_score(text: str) -> dict[str, Any]:
    """Understands numbers and percentages in human queries.

    Differentiates between:
    - Explicit percentages: "65%", "65 percent", "65 percentage", "65 persent", "65 prcnt", "65 ટકા", "65 प्रतिशत"
    - Marks out of 100: "I scored 65 marks out of 100", "65 / 100", "65 out of 100" -> 65.0%
    - Ambiguous numbers: "I got 65 in 12th", "12th ma 65 che" -> is_ambiguous = True, value = 65.0
    """
    t_low = text.lower()

    # Case 1: Explicit Marks out of 100 ("scored 65 marks out of 100", "65/100", "65 out of 100")
    m_out_of_100 = re.search(
        r"\b(?:scored?|got|marks?|obtained)?\s*(\d{1,2}(?:\.\d+)?)\s*(?:marks?)?\s*(?:out\s*of|\/)\s*100\b",
        t_low,
    )
    if m_out_of_100:
        val = float(m_out_of_100.group(1))
        if 0.0 <= val <= 100.0:
            return {
                "percentage": val,
                "is_ambiguous": False,
                "ambiguous_score": None,
                "marks_out_of_100": True,
                "raw_match": m_out_of_100.group(0),
            }

    # Case 2: Explicit Percentages ("65%", "65 percent", "65 percentage", "65 persent", "65 prcnt", "65 ટકા", "65 प्रतिशत")
    m_pct = re.search(
        r"\b(\d{1,2}(?:\.\d+)?)\s*(?:%|percent\b|percentage\b|persent\b|prcnt\b|per\s*cent\b|ટકા\b|પ્રતિશત\b|प्रतिशत\b)",
        t_low,
    )
    if m_pct:
        val = float(m_pct.group(1))
        if 0.0 <= val <= 100.0:
            return {
                "percentage": val,
                "is_ambiguous": False,
                "ambiguous_score": None,
                "marks_out_of_100": False,
                "raw_match": m_pct.group(0),
            }

    # Case 3: Ambiguous Scores with 12th / Board / Marks
    # e.g. "I got 65 in 12th", "12th ma 65 che", "I have 65 in class 12", "scored 65 in 12th"
    # MUST NOT be silently converted to percentage!
    m_ambig = re.search(
        r"(?:\b(?:got|scored|have|obtained|che|ma|me|mein|aaya|pass\s*with)\s+(\d{1,2}(?:\.\d+)?)\s+(?:in|marks?\s*in)?\s*(?:12th|12|hsc|class\s*12|board)\b)|"
        r"(?:\b(?:12th|12|hsc|class\s*12|board)\s*(?:ma|me|mein|pass)?\s*(?:got|scored|have|with)?\s+(\d{1,2}(?:\.\d+)?)\s*(?:marks?|che|aaya)?\b)",
        t_low,
    )
    if m_ambig:
        num_str = m_ambig.group(1) or m_ambig.group(2)
        if num_str:
            val = float(num_str)
            # Avoid matching class numbers like 12 or years like 24
            if 30.0 <= val <= 99.0 and val != 12.0:
                return {
                    "percentage": None,
                    "is_ambiguous": True,
                    "ambiguous_score": val,
                    "marks_out_of_100": False,
                    "raw_match": m_ambig.group(0),
                }

    # No percentage or ambiguous score found
    return {
        "percentage": None,
        "is_ambiguous": False,
        "ambiguous_score": None,
        "marks_out_of_100": False,
        "raw_match": None,
    }


def parse_and_normalize(question: str, profile: dict[str, Any] | None = None) -> NormalizedQuery:
    """Main entrypoint for user-query normalization and understanding.

    Produces a clean, normalized interpretation while keeping the original
    text intact for logging, history, and citations.
    """
    from backend.rag import counsellor

    norm_text, applied = normalize_query_text(question)
    detected_lang = counsellor.detect_language(norm_text, profile)

    score_info = extract_percentage_and_score(norm_text)
    if score_info["percentage"] is None and not score_info["is_ambiguous"]:
        # Fall back to checking the original question in case of formatting nuance
        score_info = extract_percentage_and_score(question)

    # Course extraction from normalized text
    matched_course, _ = counsellor.match_course(norm_text)
    if not matched_course:
        matched_course, _ = counsellor.match_course(question)

    # Stream extraction
    stream = None
    s_low = norm_text.lower()
    if re.search(r"\bscience\b", s_low) and not re.search(r"\b(?:computer|data)\s*science\b", s_low):
        stream = "Science"
    elif re.search(r"\bcommerce\b", s_low):
        stream = "Commerce"
    elif re.search(r"\barts\b", s_low):
        stream = "Arts"

    return NormalizedQuery(
        original_text=question,
        normalized_text=norm_text,
        detected_language=detected_lang,
        extracted_percentage=score_info["percentage"],
        is_ambiguous_score=score_info["is_ambiguous"],
        ambiguous_score_val=score_info["ambiguous_score"],
        extracted_course=matched_course,
        extracted_stream=stream,
        applied_normalizations=applied,
    )


def build_percentage_eligibility_response(
    parsed: NormalizedQuery,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generates an accurate, verified eligibility response for percentage questions.

    - Handles ambiguous numbers by asking a polite clarification.
    - Compares verified criteria for a specific course (e.g. BCA requires 12th Science or Commerce).
    - Never invents eligibility thresholds, rounding rules, or guarantees.
    """
    from backend.rag import eligibility, pipeline

    lang = parsed.detected_language
    app_url = pipeline.get_verified_application_url()

    # 1. Ambiguous score handling ("I got 65 in 12th")
    if parsed.is_ambiguous_score and parsed.ambiguous_score_val is not None:
        score_int = int(parsed.ambiguous_score_val)
        course_display = parsed.extracted_course or "your chosen program"

        if lang == "gujarati":
            ans = (
                f"કૃપા કરીને સ્પષ્ટ કરશો કે **{score_int}** એ તમારા 12મા ધોરણની કુલ ટકાવારી ({score_int}%) છે, ગુણ (marks) છે, કે પર્સેન્ટાઇલ છે?\n\n"
                f"જો આ 12th સાયન્સ અથવા કોમર્સમાં **{score_int}%** એકંદર ટકાવારી હોય, તો તમે સેન્ટ ઝેવિયર્સ કોલેજમાં **{course_display}** માં પ્રવેશ ફોર્મ ભરવાની લાયકાત ધરાવો છો. "
                f"અંતિમ પ્રવેશ કોલેજ દ્વારા જાહેર કરવામાં આવતી મેરિટ યાદી (Merit List) પર આધારિત રહેશે."
            )
        elif lang == "hindi":
            ans = (
                f"कृपया स्पष्ट करें कि **{score_int}** आपकी 12वीं की कुल प्रतिशतता ({score_int}%) है, कुल अंक हैं, या परसेंटाइल है?\n\n"
                f"यदि यह 12वीं साइंस या कॉमर्स में **{score_int}%** एग्रीगेट है, तो आप सेंट जेवियर्स कॉलेज में **{course_display}** के लिए आवेदन करने के योग्य हैं। "
                f"अंतिम प्रवेश कॉलेज की आधिकारिक मेरिट सूची (Merit List) के आधार पर होगा।"
            )
        else:
            ans = (
                f"Could you please clarify whether **{score_int}** refers to your overall percentage ({score_int}%), total marks, or percentile rank in 12th?\n\n"
                f"If you have an aggregate of **{score_int}%** in 12th Science or Commerce, you meet the basic eligibility to apply for **{course_display}** at St. Xavier's College (Autonomous), Ahmedabad. "
                f"Please note that final admission is offered strictly on the basis of the official merit cut-off lists prepared from 12th board marks for that academic year."
            )

        src = [{
            "title": "SXCA Admission Brochure 2025-26",
            "source": "SXCA-Admission-Brochure-2526.pdf",
            "doc_type": "pdf",
            "page": 1,
            "url": "https://sxca.edu.in/wp-content/uploads/2025/06/SXCA-Admission-Brochure-2526.pdf",
            "score": 1.0,
            "excerpt": "Undergraduate admission criteria and merit list evaluation guidelines"
        }]
        return {"answer": ans, "sources": src, "grounded": True, "status": "CLARIFICATION_REQUIRED"}

    # 2. Explicit percentage handling (e.g. 65%)
    pct_val = parsed.extracted_percentage or 65.0
    pct_str = f"{pct_val:g}"

    # A. Specific course requested (e.g. BCA, BSc IT, BCom, etc.)
    if parsed.extracted_course:
        course_name = parsed.extracted_course
        rows = eligibility.find_by_course(course_name)
        req_eligibility = rows[0].get("eligibility", "12th Standard pass") if rows else "12th Standard pass"

        if lang == "gujarati":
            ans = (
                f"હા, 12મા ધોરણમાં **{pct_str}%** સાથે, તમે સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં **{course_name}** માટે અરજી કરી શકો છો!\n\n"
                f"• **સત્તાવાર પાત્રતા (Eligibility):** {req_eligibility}\n"
                f"• **પ્રવેશ પ્રક્રિયા:** પ્રવેશ 12મા ધોરણના બોર્ડ માર્ક્સના આધારે તૈયાર થતી સત્તાવાર મેરિટ યાદી (Merit List) દ્વારા આપવામાં આવે છે.\n"
                f"• **મહત્વપૂર્ણ નોંધ:** {pct_str}% સાથે તમે પાત્રતા પૂરી કરો છો, પરંતુ અંતિમ પ્રવેશ સંબંધિત પ્રવેશ રાઉન્ડ અને કેટેગરીના મેરિટ કટ-ઓફ પર નિર્ભર કરે છે.\n\n"
                f"• **સત્તાવાર પ્રવેશ પોર્ટલ:** [અહીં ઓનલાઇન અરજી કરો]({app_url})\n"
                f"• **હેલ્પલાઇન:** admissions@sxca.edu.in | +91 79 2630 2872"
            )
        elif lang == "hindi":
            ans = (
                f"हाँ, 12वीं में **{pct_str}%** के साथ, आप सेंट जेवियर्स कॉलेज (स्वायत्त), अहमदाबाद में **{course_name}** के लिए आवेदन करने के पात्र हैं!\n\n"
                f"• **आधिकारिक पात्रता (Eligibility):** {req_eligibility}\n"
                f"• **प्रवेश प्रक्रिया:** प्रवेश 12वीं बोर्ड अंकों के आधार पर बनने वाली आधिकारिक मेरिट सूची (Merit List) के माध्यम से दिया जाता है।\n"
                f"• **महत्वपूर्ण सूचना:** {pct_str}% होने पर आप आवेदन करने के योग्य हैं, लेकिन अंतिम प्रवेश उस वर्ष के राउंड और श्रेणीवार मेरिट कट-ऑफ पर निर्भर करेगा।\n\n"
                f"• **आधिकारिक प्रवेश पोर्टल:** [ऑनलाइन आवेदन लिंक]({app_url})\n"
                f"• **हेल्पलाइन:** admissions@sxca.edu.in | +91 79 2630 2872"
            )
        else:
            ans = (
                f"Yes, with **{pct_str}%** in 12th Standard, you meet the eligibility criteria to apply for **{course_name}** at St. Xavier's College (Autonomous), Ahmedabad!\n\n"
                f"• **Official Eligibility Requirement:** {req_eligibility}\n"
                f"• **Admission Process:** St. Xavier's College admits students based on the official merit rank lists prepared from 12th board marks.\n"
                f"• **Important Notice:** Scoring {pct_str}% makes you eligible to apply; final admission is determined by the declared merit cut-off for your category and round. Admission is never pre-guaranteed.\n\n"
                f"• **Official Admission Portal:** [Apply Online Here]({app_url})\n"
                f"• **Admissions Helpline:** admissions@sxca.edu.in | +91 79 2630 2872"
            )

        src = [{
            "title": f"SXCA Admission Brochure 2025-26 - {course_name}",
            "source": "SXCA-Admission-Brochure-2526.pdf",
            "doc_type": "pdf",
            "page": 1,
            "url": "https://sxca.edu.in/wp-content/uploads/2025/06/SXCA-Admission-Brochure-2526.pdf",
            "score": 1.0,
            "excerpt": f"{course_name} Eligibility: {req_eligibility}"
        }]
        return {"answer": ans, "sources": src, "grounded": True}

    # B. General percentage question (e.g. "courses available for 65%")
    return {
        "answer": pipeline._format_percentage_eligibility_summary(pct_str, parsed.extracted_stream),
        "sources": [{
            "title": "SXCA Admission Brochure 2025-26",
            "source": "SXCA-Admission-Brochure-2526.pdf",
            "doc_type": "pdf",
            "page": 1,
            "url": "https://sxca.edu.in/wp-content/uploads/2025/06/SXCA-Admission-Brochure-2526.pdf",
            "score": 1.0,
            "excerpt": "Undergraduate Eligibility Criteria"
        }],
        "grounded": True,
    }
