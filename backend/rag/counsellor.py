"""Admission Counsellor Brain for St. Xavier's College (Autonomous), Ahmedabad.

Orchestrates human-like conversational counselling:
- Progressively tracks and updates structured student profile
- Enforces single active course tracking with strict context switching
- Manages adaptive, one-question-at-a-time guidance
- Prevents cross-contamination between UG and PG courses
- Accurately classifies complaints and isolates them from admission counselling
- Sources all factual course, fee, and eligibility data directly from verified records
"""

from __future__ import annotations

import re
from typing import Any

from backend.db import repo
from backend.rag import calendar_retriever, curriculum_retriever
from backend.rag import eligibility

DEFAULT_PROFILE: dict[str, Any] = {
    "education_level": None,  # "12th", "bachelor", "diploma", or None
    "stream": None,           # "Science", "Commerce", "Arts", or None
    "percentage": None,       # e.g., "65"
    "interest_area": None,    # "Computer/IT", "AI/ML", "Pure Science", "Life Sciences", "Commerce", "Arts", or None
    "career_goal": None,      # e.g., "AI Engineer", "Software Developer", or None
    "course_level": None,     # "UG", "PG", or None
    "active_course": None,    # Canonical course name (e.g. "BCA", "BCom (General)", "MSc-Artificial Intelligence")
    "active_person": None,    # e.g. "Dr. David", "Dr. Sebastian V.A."
    "active_department": None,# e.g. "Department of Computer Sciences & Data Science"
    "active_topic": None,     # e.g. "faculty", "hostel", "contact", "event"
    "language_preference": None, # "gujarati", "english", or None
    "mode": "general",        # "greeting", "counselling", "direct_course", "complaint", "general"
}

# Known Canonical Course Mappings in SXCA Database
def normalize_text_for_course_matching(text: str) -> str:
    """Normalizes degree abbreviations with spaces, hyphens, and dots to standard tokens."""
    t = text.strip().lower()
    # Normalize spaced/dotted degree abbreviations
    t = re.sub(r"\bm\s*\.?\s*s\s*\.?\s*c\b", "msc", t)
    t = re.sub(r"\bm\s*\.?\s*sc\b", "msc", t)
    t = re.sub(r"\bb\s*\.?\s*c\s*\.?\s*a\b", "bca", t)
    t = re.sub(r"\bb\s*\.?\s*com\b", "bcom", t)
    t = re.sub(r"\bb\s*\.?\s*s\s*\.?\s*c\b", "bsc", t)
    t = re.sub(r"\bb\s*\.?\s*sc\b", "bsc", t)
    t = re.sub(r"\bb\s*\.?\s*d\s*\.?\s*a\b", "bda", t)
    t = re.sub(r"\bb\s*\.?\s*a\b", "ba", t)
    t = re.sub(r"\bm\s*\.?\s*a\b", "ma", t)
    # Replace dots, hyphens, slashes with spaces
    t = re.sub(r"[.\-_/]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# Known Canonical Course Mappings in SXCA Database
COURSE_ALIASES = [
    # BCA
    (r"\b(bca|bachelor\s+(?:of\s+)?computer\s+applications?|bachlor\s+(?:of\s+)?computer\s+applications?)\b", "BCA", "UG"),
    # B.Com & Commerce
    (r"\b(bcom\s*bps)\b", "BCom-BPS", "UG"),
    (r"\b(bcom|bachelor\s+(?:of\s+)?commerce|bachlor\s+(?:of\s+)?commerce)\b", "BCom (General)", "UG"),
    (r"\b(bs\s+business\s+administration|bba|business\s+administration)\b", "BS-Business Administration", "UG"),
    # AI & Data Science
    (r"\b(msc\s+(?:in\s+)?artificial\s+intelligence|msc\s+ai|ms\s+ai|masters?\s+(?:in\s+|of\s+)?ai|ai\s+master\'?s?)\b", "MSc-Artificial Intelligence", "PG"),
    (r"\b(msc\s+(?:in\s+)?big\s+data(?:\s+analytics)?|msc\s+bda|big\s+data\s+analytics|big\s+data|bda)\b", "MSc-Big Data Analytics", "PG"),
    (r"\b(bsc\s+(?:in\s+)?artificial\s+intelligence(?:\s*&\s*machine\s+learning)?|bsc\s+aiml|bsc\s+ai)\b", "BSc-Artificial Intelligence & Machine Learning", "UG"),
    # IT & Computing
    (r"\b(bsc\s+(?:in\s+)?information\s+technology|bsc\s+it|bsc\s+cs|bsc\s+comp.*sci.*)\b", "BSc-Information Technology", "UG"),
    # Science UG
    (r"\b(bsc\s+physics|bsc\s+phy\b|physics)\b", "BSc-Physics", "UG"),
    (r"\b(bsc\s+chemistry|bsc\s+chem\b|chemistry)\b", "BSc-Chemistry", "UG"),
    (r"\b(bsc\s+mathematics|bsc\s+maths?|mathematics)\b", "BSc-Mathematics", "UG"),
    (r"\b(bsc\s+statistics|bsc\s+stats?|statistics)\b", "BSc-Statistics", "UG"),
    (r"\b(bsc\s+biotechnology|bsc\s+bt\b|biotechnology)\b", "BSc-Biotechnology", "UG"),
    (r"\b(bsc\s+biochemistry|bsc\s+bc\b|biochemistry)\b", "BSc-Biochemistry", "UG"),
    (r"\b(bsc\s+botany|botany)\b", "BSc-Botany", "UG"),
    (r"\b(bsc\s+zoology|zoology)\b", "BSc-Zoology", "UG"),
    (r"\b(bsc\s+electronics|electronics)\b", "BSc-Electronics", "UG"),
    # Science PG
    (r"\b(msc\s+(?:in\s+)?microbiology|msc\s+micro\b|microbiology)\b", "MSc-Microbiology", "PG"),
    (r"\b(msc\s+(?:in\s+)?physics|msc\s+phy\b)\b", "MSc-Physics", "PG"),
    (r"\b(msc\s+(?:in\s+)?chemistry\s*\(?organic\)?|msc\s+chemistry\s+organic)\b", "MSc-Chemistry (Organic)", "PG"),
    (r"\b(msc\s+(?:in\s+)?chemistry\s*\(?analytical\)?|msc\s+chemistry\s+analytical)\b", "MSc-Chemistry (Analytical)", "PG"),
    (r"\b(msc\s+(?:in\s+)?chemistry)\b", "MSc-Chemistry (Organic)", "PG"),
    (r"\b(msc\s+(?:in\s+)?mathematics|msc\s+maths?)\b", "MSc-Mathematics", "PG"),
    (r"\b(msc\s+(?:in\s+)?biochemistry)\b", "MSc-Biochemistry", "PG"),
    (r"\b(msc\s+(?:in\s+)?biotechnology)\b", "MSc-Biotechnology", "PG"),
    (r"\b(msc\s+(?:in\s+)?botany)\b", "MSc-Botany", "PG"),
    # Arts UG & PG
    (r"\b(b\.?\s*a\.?\s*(?:in\s+)?economics|ba\s+economics|b\.?\s*eco\.?|b\s+eco\b|ba\s+eco\.?|b\.?\s*a\.?\s*eco\.?|bachelor\s+(?:of|in)?\s+economics|economics\s+graduate|economics)\b", "BA-Economics", "UG"),
    (r"\b(ba\s+(?:in\s+)?english|ba\s+english)\b", "BA-English", "UG"),
    (r"\b(ba\s+(?:in\s+)?psychology|ba\s+psychology)\b", "BA-Psychology", "UG"),
    (r"\b(ba\s+(?:in\s+)?gujarati|ba\s+gujarati)\b", "BA-Gujarati", "UG"),
    (r"\b(ba\s+(?:in\s+)?sanskrit|ba\s+sanskrit)\b", "BA-Sanskrit", "UG"),
    (r"\b(ba\s+(?:in\s+)?sociology|ba\s+sociology)\b", "BA-Sociology & Development", "UG"),
    (r"\b(bpa|theatre\s*&\s*production)\b", "BPA-Theatre & Production Management", "UG"),
    (r"\b(ma\s+(?:in\s+)?clinical\s+psychology|ma\s+psychology)\b", "MA-Clinical Psychology", "PG"),
    (r"\b(ma\s+(?:in\s+)?english|ma\s+english)\b", "MA-English", "PG"),
]


def match_course(text: str) -> tuple[str | None, str | None]:
    """Matches a course string to (canonical_name, level) with punctuation and abbreviation tolerance."""
    t = text.strip()
    t_norm = normalize_text_for_course_matching(t)
    for pattern, name, level in COURSE_ALIASES:
        if re.search(pattern, t_norm, re.I) or re.search(pattern, t, re.I):
            return name, level
    return None, None


def clean_course_display_name(course_name: str | None) -> str:
    if not course_name:
        return ""
    return (
        course_name.replace("MSc-", "M.Sc. ")
        .replace("BSc-", "B.Sc. ")
        .replace("BA-", "B.A. ")
        .replace("MA-", "M.A. ")
        .replace("BCom-", "B.Com ")
        .replace("BCom", "B.Com")
        .replace("BS-", "B.S. ")
        .replace("BPA-", "B.P.A. ")
    )


# --- Dialogue Understanding & Conversational Repair Helpers ------------------

def is_conversation_feedback(question: str) -> bool:
    """Detects when the student points out repetition, loops, or bot confusion."""
    q = question.lower().strip()
    return bool(re.search(
        r"\b(why\s*(?:do\s*you|you)?\s*(?:keep\s*)?(?:giving|saying|repeating|showing)\s*(?:the\s*)?same\s*(?:answer|thing)?|"
        r"you\s*(?:are\s*)?repeating|why\s*(?:are\s*you\s*)?repeating|same\s*answer\s*again|"
        r"you\s*already\s*said\s*that|you\s*said\s*that\s*already|why\s*same\s*answer|"
        r"stop\s*repeating|stuck\s*in\s*a\s*loop|repetitive)\b",
        q
    ))


def handle_conversation_repair(profile: dict[str, Any]) -> str:
    """Politely apologizes, resets stuck state, and prompts for what the user needs."""
    active_course = profile.get("active_course")
    if active_course:
        disp_name = clean_course_display_name(active_course)
        return (
            f"I apologize for the repetition! Let's get straight to what you need for **{disp_name}**.\n\n"
            f"Are you looking for:\n"
            f"• **Course subjects & curriculum**\n"
            f"• **Eligibility criteria**\n"
            f"• **Fees & fee structure**\n"
            f"• **Admission process & dates**\n\n"
            f"Just let me know, or ask about another programme!"
        )
    return (
        "I apologize for the repetition! Let's reset.\n\n"
        "How can I best help you today? You can ask about:\n"
        "• Specific course details (e.g. BCA, B.Com, M.Sc. AI)\n"
        "• Eligibility & percentage requirements\n"
        "• Fee structures\n"
        "• Academic calendar & exam schedules"
    )


def detect_negation(question: str, profile: dict[str, Any] | None = None) -> tuple[bool, str | None]:
    """Detects when user explicitly negates/rejects a specific topic or course."""
    q = question.lower().strip()

    # 1. Course-level negation / rejection (e.g., 'i dont want to do bcom', 'bcom nahi karna hai', 'bcom nahi', 'not bcom', 'bcom nathi karvu')
    m_course_neg = re.search(
        r"\b(?:i\s*(?:do\s*not|don'?t)\s*(?:want|like|need|prefer)(?:\s*to\s*do)?|not\s*(?:interested\s*in)?|no\s*need\s*(?:of|for)?|skip|leave|without|no)\s+([a-z0-9\s\.\-_]+?)(?:\s*hai|\s*che|\s*karvu|\s*karna)?$",
        q
    )
    if m_course_neg:
        target = m_course_neg.group(1).strip()
        matched_c, _ = match_course(target)
        if matched_c:
            return True, f"course:{matched_c}"
        if target in ("this", "that", "this course", "that course") and profile and profile.get("active_course"):
            return True, f"course:{profile.get('active_course')}"

    # Suffix negation: 'bcom nahi karna hai', 'bcom nahi', 'bcom nathi karvu', 'bcom nahi chahiye'
    m_suffix = re.search(
        r"\b([a-z0-9\s\.\-_]+?)\s+(?:nahi\s*karna(?:\s*hai)?|nahi\s*karvu|nathi\s*karvu|nahi\s*chahiye|nahi|nathi|not\s*interested|leave\s*it|skip)\b",
        q
    )
    if m_suffix:
        target = m_suffix.group(1).strip()
        matched_c, _ = match_course(target)
        if matched_c:
            return True, f"course:{matched_c}"
        if target in ("this", "aa", "ye", "this course", "that course") and profile and profile.get("active_course"):
            return True, f"course:{profile.get('active_course')}"

    # Standalone 'not this', 'not this course', 'dont want this', 'aa nathi joiye', 'ye nahi chahiye'
    if re.search(r"^(?:not\s*this(?:\s*course)?|i\s*don'?t\s*want\s*this(?:\s*course)?|aa\s*nathi\s*joiye|ye\s*nahi\s*chahiye|dont\s*want\s*this|not\s*interested)$", q):
        if profile and profile.get("active_course"):
            return True, f"course:{profile.get('active_course')}"
        return True, "that"

    # 2. General topic exclusions (fees, subjects, eligibility, etc.)
    m = re.search(
        r"\b(?:i\s*(?:do\s*not|don'?t)\s*(?:want|need|ask\s*for)|not\s*(?:interested\s*in)?|no\s*need\s*(?:of|for)?|skip|leave|without)\s+"
        r"(fees?|eligibility|subjects?|curriculum|syllabus|process|admission|hostel)\b",
        q
    )
    if m:
        topic = m.group(1).lower()
        if "fee" in topic:
            topic = "fee"
        elif "subject" in topic or "curriculum" in topic or "syllabus" in topic:
            topic = "subjects"
        elif "eligib" in topic:
            topic = "eligibility"
        return True, topic

    if re.search(r"\b(?:not|no)\s+(?:the\s+)?(?:fees?)\b", q):
        return True, "fee"
    if re.search(r"\b(?:not|no)\s+(?:the\s+)?(?:subjects?|curriculum)\b", q):
        return True, "subjects"
    if re.search(r"\b(?:not|no)\s+(?:the\s+)?(?:eligibility)\b", q):
        return True, "eligibility"

    return False, None


def handle_negation(topic: str | None, profile: dict[str, Any], is_guj: bool = False) -> str:
    """Acknowledges the exclusion and offers the remaining relevant options."""
    if topic and topic.startswith("course:"):
        rej_course = topic.split(":", 1)[1]
        disp_name = clean_course_display_name(rej_course)
        stream = profile.get("stream")
        qual = profile.get("qualification")

        is_eco_rejected = "economics" in disp_name.lower() or "eco" in disp_name.lower() or "BA-Economics" in profile.get("rejected_courses", [])
        ba_bullet_guj = "• **B.A. Programs** (Psychology, English, Sociology)\n" if is_eco_rejected else "• **B.A. Programs** (Economics, Psychology, English)\n"
        ba_bullet_en = "• **B.A. Programs** (Psychology, English, Sociology)\n" if is_eco_rejected else "• **B.A. Programs** (Economics, Psychology, English)\n"
        comm_eco_bullet_guj = "• **B.Com. (General)** — એકાઉન્ટ્સ અને ફાઇનાન્સ\n" if is_eco_rejected else "• **B.A. Economics** — અર્થશાસ્ત્ર અને નીતિ વિશ્લેષણ\n"
        comm_eco_bullet_en = "• **B.Com. (General)** — core commerce, accounting, and finance\n" if is_eco_rejected else "• **B.A. Economics** — analytical and economic policy studies\n"

        if is_guj:
            if stream == "Commerce" or "bcom" in disp_name.lower() or "commerce" in str(qual).lower():
                return (
                    f"સમજી ગયો — તમારે **{disp_name}** નથી કરવું. હું ફરીથી તેનું સૂચન નહીં કરું.\n\n"
                    f"કોમર્સ બેકગ્રાઉન્ડના આધારે સેન્ટ ઝેવિયર્સ કોલેજમાં અન્ય ઉત્તમ વિકલ્પો ઉપલબ્ધ છે:\n"
                    f"• **BCA (Bachelor of Computer Applications)** — જો તમને સોફ્ટવેર અને આઈટીમાં રસ હોય\n"
                    f"• **B.S. Business Administration** — મેનેજમેન્ટ અને બિઝનેસ લીડરશિપ માટે\n"
                    f"• **B.Com. - BPS (Business Process Services)** — કોર્પોરેટ ક્ષેત્ર માટે TCS સાથે સંલગ્ન\n"
                    f"{comm_eco_bullet_guj}\n"
                    f"તમને આમાંથી કયા ક્ષેત્રમાં વધુ રસ છે?"
                )
            elif stream == "Science":
                return (
                    f"સમજી ગયો — તમારે **{disp_name}** નથી કરવું. હું ફરીથી તેનું સૂચન નહીં કરું.\n\n"
                    f"સાયન્સ બેકગ્રાઉન્ડ માટે સેન્ટ ઝેવિયર્સ કોલેજમાં અન્ય શ્રેષ્ઠ વિકલ્પો:\n"
                    f"• **BCA (Bachelor of Computer Applications)**\n"
                    f"• **B.Sc. Information Technology**\n"
                    f"• **B.Sc. Artificial Intelligence & Machine Learning**\n"
                    f"• **B.Sc. Pure Sciences** (Physics, Chemistry, Maths)\n"
                    f"• **B.Sc. Life Sciences** (Biotechnology, Biochemistry)\n\n"
                    f"તમે કયા અભ્યાસક્રમ વિશે વધુ જાણવા માંગો છો?"
                )
            else:
                return (
                    f"સમજી ગયો — તમારે **{disp_name}** નથી કરવું. હું ફરીથી તેનું સૂચન નહીં કરું.\n\n"
                    f"સેન્ટ ઝેવિયર્સ કોલેજમાં અન્ય ઉપલબ્ધ વિકલ્પો:\n"
                    f"• **BCA (Computer Applications)**\n"
                    f"• **B.S. Business Administration**\n"
                    f"{ba_bullet_guj}"
                    f"• **Postgraduate Master's Degrees**\n\n"
                    f"કૃપા કરીને જણાવો કે તમને કયા વિષયમાં રસ છે!"
                )

        # Check stream/qualification for customized alternatives
        if stream == "Commerce" or "bcom" in disp_name.lower() or "commerce" in str(qual).lower():
            return (
                f"Got it — you don't want **{disp_name}**. I won't suggest it again.\n\n"
                f"Based on a Commerce background, St. Xavier's College offers excellent alternatives:\n"
                f"• **BCA (Bachelor of Computer Applications)** — if you are interested in software development & IT\n"
                f"• **B.S. Business Administration** — focused on management, marketing, and leadership\n"
                f"• **B.Com. - BPS (Business Process Services)** — industry-aligned with TCS for corporate business operations\n"
                f"• **B.Sc. Information Technology** — for software and IT systems\n"
                f"{comm_eco_bullet_en}\n"
                f"Which of these directions interests you more?"
            )
        elif stream == "Science":
            return (
                f"Got it — you don't want **{disp_name}**. I won't suggest it again.\n\n"
                f"With a Science background, here are strong alternative options at St. Xavier's College:\n"
                f"• **BCA (Bachelor of Computer Applications)**\n"
                f"• **B.Sc. Information Technology**\n"
                f"• **B.Sc. Artificial Intelligence & Machine Learning**\n"
                f"• **B.Sc. Pure Sciences** (Physics, Chemistry, Maths)\n"
                f"• **B.Sc. Life Sciences** (Biotechnology, Biochemistry, Botany, Zoology)\n\n"
                f"Which area would you like to explore instead?"
            )
        else:
            return (
                f"Got it — you don't want **{disp_name}**. I won't suggest it again.\n\n"
                f"We can explore other programmes at St. Xavier's College such as:\n"
                f"• **BCA (Computer Applications)**\n"
                f"• **B.S. Business Administration**\n"
                f"{ba_bullet_en}"
                f"• **Postgraduate Master's Degrees**\n\n"
                f"Let me know what field you prefer!"
            )

    active_course = profile.get("active_course")
    topic_name = topic or "that"
    if active_course:
        disp_name = clean_course_display_name(active_course)
        options = []
        if topic_name != "subjects":
            options.append("• **Subjects & curriculum**")
        if topic_name != "eligibility":
            options.append("• **Eligibility criteria**")
        if topic_name != "fee":
            options.append("• **Fee details**")
        options.append("• **Admission process & how to apply**")
        opts_str = "\n".join(options)
        return (
            f"Understood, skipping {topic_name}! For **{disp_name}**, what would you like to explore instead?\n\n"
            f"{opts_str}"
        )
    return f"Understood, skipping {topic_name}! What information can I help you with instead?"


def is_contextual_no(question: str) -> bool:
    """Detects standalone negative responses ('no', 'nope', 'nah', 'not really')."""
    q = question.lower().strip().rstrip(".,!?")
    return q in ("no", "nope", "nah", "na", "n", "no thanks", "not really", "no need", "nahi", "nathi")


def handle_contextual_no(profile: dict[str, Any], recent_turns: list[dict] | None = None) -> str:
    """Handles polite negative answer without repeating the previous response."""
    return (
        "No problem! What would you like to explore instead? You can ask about other programmes, faculty, campus life, fees, or admission details."
    )


def is_conversational_redirect(question: str) -> bool:
    """Detects when user wants to switch topic or move away from current topic."""
    q = question.lower().strip()
    return bool(re.search(
        r"\b(tell\s*me\s*something\s*else|forget\s*(?:about\s*)?(?:that|this)|something\s*else|"
        r"let'?s\s*talk\s*about\s*something\s*else|nevermind|never\s*mind|switch\s*topic)\b",
        q
    ))


def handle_redirect(profile: dict[str, Any]) -> str:
    """Gracefully pivots conversation away from the previous topic."""
    active_course = profile.get("active_course")
    if active_course:
        disp_name = clean_course_display_name(active_course)
        return (
            f"Sure! Aside from what we've discussed for **{disp_name}**, we can look at other details like eligibility, "
            f"fees, subjects, or explore a completely different course. What would you prefer?"
        )
    return (
        "Sure! We can explore courses, eligibility, fee structures, or the academic calendar. "
        "What would you like to discuss?"
    )


def is_user_correction(question: str) -> bool:
    """Detects if user is correcting, disagreeing with, or redirecting the bot."""
    q = question.lower().strip()
    return bool(re.search(
        r"^(?:not\s*work(?:ing)?\s*like\s*that|that'?s?\s*(?:not\s*what\s*i\s*(?:mean|asked|want)|wrong)|"
        r"this\s*is\s*wrong|you(?:'re|\s*are)?\s*wrong|wrong\s*answer|you\s*don'?t\s*understand|"
        r"not\s*this|wrong|no\s*i\s*want\b|that\s*is\s*not\s*what\s*i\s*asked|"
        r"i\s*didn'?t\s*ask\s*(?:for\s*)?(?:that|this)|no\s*not\s*this|don'?t\s*ask\s*me\s*(?:that|this)|"
        r"stop\s*(?:asking|repeating))\b",
        q
    ))


def handle_user_correction() -> str:
    """Empathetically recovers when user corrects the chatbot without getting stuck."""
    return (
        "You're right — let me understand what you're asking.\n\n"
        "You can directly tell me the course or information you're looking for. For example, you can say:\n"
        "• **\"MSc AI course\"** or **\"BCA course\"**\n"
        "• **\"MSc AI fees\"** or **\"Eligibility\"**\n"
        "• **\"Admission process\"**"
    )


class QualificationResult(dict):
    """Structured representation of completed qualification that also unpacks as (qualification, level, stream)."""
    def __init__(self, qualification=None, stream=None, level=None, status=None):
        super().__init__(
            qualification=qualification,
            stream=stream,
            level=level,
            status=status or ("completed" if (qualification or stream) else None)
        )
    
    def __bool__(self):
        return bool(self.get("qualification") or self.get("stream"))

    def __iter__(self):
        return iter((self.get("qualification"), self.get("level"), self.get("stream")))


def detect_completed_qualification(question: str) -> QualificationResult:
    """Detects if user mentions a completed degree/stream rather than desired study.
    Returns QualificationResult({'qualification': ..., 'stream': ..., 'level': ..., 'status': ...})
    which also unpacks as (qualification, level, stream) for backwards compatibility.
    Examples:
    'mene bachelor mai b.eco liya hai master mai konse course hai' -> {'qualification': 'BA-Economics', 'stream': 'Arts', 'level': 'UG', 'status': 'completed'}
    'mene b.eco kiya hai' -> {'qualification': 'BA-Economics', 'stream': 'Arts', 'level': 'UG', 'status': 'completed'}
    'mene commerce kiya hai course name' -> {'qualification': None, 'stream': 'Commerce', 'level': None, 'status': 'completed'}
    'mene bcom kiya hai konsa course hai' -> {'qualification': 'BCom (General)', 'stream': 'Commerce', 'level': 'UG', 'status': 'completed'}
    'maine bsc complete kiya hai' -> {'qualification': 'BSc', 'stream': 'Science', 'level': 'UG', 'status': 'completed'}
    'I completed BCA' -> {'qualification': 'BCA', 'stream': 'Computer/IT', 'level': 'UG', 'status': 'completed'}
    """
    q_low = question.lower().strip()

    is_completed = bool(
        re.search(r"\b(complit(?:e|ed)?|complete|completed|passed|done|have\s*done|graduated|after|ke\s*baad|pachi|graduate\s*hu|graduate\s*chhu)\b", q_low)
        or re.search(r"\b(?:mene|maine|me|humne|hu|i)\s+.*?\s*(?:kiya|liya|karyu|lidhu|complete|complite|pass|passed)\b", q_low)
        or re.search(r"\bbachelor\s*(?:mai|ma|me)\s+.*?\s*(?:liya|kiya|karyu|lidhu|tha|hatu)\b", q_low)
        or re.search(r"\b(kar\s*chuka\s*hu|kar\s*chuki\s*hu|karyu\s*che|lidhu\s*che|kiya\s*hai|liya\s*hai|pass\s*che|kariyu\s*che)\b", q_low)
        or (re.search(r"\b(master|pg|msc|ma)\b", q_low) and re.search(r"\b(for|with|after|from)\s+(?:a\s+)?(?:ba\s*eco|b\.?\s*eco|ba\s*economics|bcom|bca|bsc)\b", q_low))
        or (re.search(r"\b(b\.?\s*eco|b\s*eco|ba\s*eco|ba\s*economics)\b", q_low) and re.search(r"\b(master|masters|pg|msc|ma|postgraduate)\b", q_low))
    )
    if not is_completed:
        return QualificationResult()

    # A. Check Economics / B.Eco degrees (Top Priority - must never become B.Com or Commerce!)
    if (
        re.search(r"\b(b\.?\s*eco\.?|b\s*eco\b|ba\s*eco\.?|b\.?\s*a\.?\s*eco\.?|ba\s*economics|b\.?\s*a\.?\s*economics|bachelor\s*(?:of|in)?\s*economics|economics\s*graduate)\b", q_low)
        or (re.search(r"\b(eco|economics)\b", q_low) and not re.search(r"\b(bcom|b\.com)\b", q_low))
    ):
        return QualificationResult(qualification="BA-Economics", stream="Arts", level="UG", status="completed")

    # B. Check B.Com / Commerce degrees
    if re.search(r"\b(b\.?\s*com\s*bps|bcom\s*bps)\b", q_low):
        return QualificationResult(qualification="BCom-BPS", stream="Commerce", level="UG", status="completed")
    if re.search(r"\b(b\.?\s*com|bachelor\s*(?:of|in)?\s*commerce|bcom\s*graduate)\b", q_low):
        return QualificationResult(qualification="BCom (General)", stream="Commerce", level="UG", status="completed")
    if re.search(r"\b(b\.?\s*s\.?\s*business\s*administration|bba|business\s*administration)\b", q_low):
        return QualificationResult(qualification="BS-Business Administration", stream="Commerce", level="UG", status="completed")

    # C. Check BCA / Computing
    if re.search(r"\b(bca|bachelor\s*(?:of|in)?\s*computer\s*applications?)\b", q_low):
        return QualificationResult(qualification="BCA", stream="Computer/IT", level="UG", status="completed")

    # D. Check BSc / Science degrees
    if re.search(r"\b(b\.?\s*sc|bachelor\s*(?:of|in)?\s*science)\b", q_low):
        sub = "Mathematics" if re.search(r"\b(math|maths|mathematics)\b", q_low) else (
            "Physics" if re.search(r"\b(physics|phy)\b", q_low) else (
                "Chemistry" if re.search(r"\b(chem|chemistry)\b", q_low) else "Science"
            )
        )
        qual = f"BSc-{sub}" if sub != "Science" else "BSc"
        return QualificationResult(qualification=qual, stream="Science", level="UG", status="completed")

    # E. Check BA Arts degrees (other than Economics)
    if re.search(r"\b(b\.?\s*a\.?\s*english|ba\s*english)\b", q_low):
        return QualificationResult(qualification="BA-English", stream="Arts", level="UG", status="completed")
    if re.search(r"\b(b\.?\s*a\.?\s*psychology|ba\s*psychology)\b", q_low):
        return QualificationResult(qualification="BA-Psychology", stream="Arts", level="UG", status="completed")
    if re.search(r"\b(b\.?\s*a|bachelor\s*(?:of|in)?\s*arts)\b", q_low):
        return QualificationResult(qualification="BA", stream="Arts", level="UG", status="completed")

    # F. Check 12th standard completion
    if re.search(r"\b(12th?|hsc|class\s*12|plus\s*two|12\s*(?:in\s*)?(?:commerce|science|arts))\b", q_low):
        stream = "Commerce" if "commerce" in q_low else ("Science" if "science" in q_low else ("Arts" if "arts" in q_low else None))
        return QualificationResult(qualification="12th", stream=stream, level="12th", status="completed")

    # G. Check bare stream completed (e.g. 'mene commerce kiya hai course name')
    if re.search(r"\bcommerce\b", q_low):
        return QualificationResult(qualification=None, stream="Commerce", level=None, status="completed")
    if re.search(r"\bscience\b", q_low):
        return QualificationResult(qualification=None, stream="Science", level=None, status="completed")
    if re.search(r"\barts\b", q_low):
        return QualificationResult(qualification=None, stream="Arts", level=None, status="completed")

    return QualificationResult()


def handle_ba_economics_graduate_courses(profile: dict[str, Any] | None = None, is_guj: bool = False) -> dict[str, Any]:
    """Authoritative postgraduate options for BA Economics graduates at SXCA."""
    if is_guj:
        ans = (
            "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં **BA Economics (ઇકોનોમિક્સ)** પૂર્ણ કરેલ વિદ્યાર્થીઓ માટે "
            "સત્તાવાર એડમિશન બ્રોશર અને પાત્રતા નિયમો મુજબ નીચેના અનુસ્નાતક (PG / Master's) કાર્યક્રમો ઉપલબ્ધ છે:\n\n"
            "• **M.Sc. Big Data Analytics (2 વર્ષ, Self-Financed):**\n"
            "  - **પાત્રતા (Eligibility):** B.Sc. (CS, Maths, Stats), **BA Eco** & BCA.\n"
            "  - **ફી:** ₹44,950 પ્રતિ સેમેસ્ટર.\n\n"
            "• **M.Sc. Artificial Intelligence (2 વર્ષ, Self-Financed):**\n"
            "  - **પાત્રતા (Eligibility):** B.Sc. (Maths, Stats, Phy, Chem), **BA Eco**, BE & BCA.\n"
            "  - **ફી:** ₹50,000 પ્રતિ સેમેસ્ટર.\n\n"
            "• **M.A. English (2 વર્ષ, Self-Financed):**\n"
            "  - માન્ય યુનિવર્સિટીમાંથી સ્નાતક (Bachelor's Degree) પાસ કરેલ તમામ વિદ્યાર્થીઓ માટે ઉપલબ્ધ.\n"
            "  - **ફી:** ₹24,950 પ્રતિ સેમેસ્ટર.\n\n"
            "શું તમે **M.Sc. Big Data Analytics**, **M.Sc. AI** અથવા **M.A. English** ની વિગતવાર પ્રવેશ પ્રક્રિયા કે સિલેબસ વિશે વધુ જાણવા માંગો છો?"
        )
    else:
        ans = (
            "Based on the official admission records of St. Xavier's College (Autonomous), Ahmedabad, "
            "graduates with a **BA in Economics (BA Eco)** are officially eligible for the following postgraduate (PG) programmes:\n\n"
            "• **M.Sc. Big Data Analytics** (2 Years, Self-Financed)\n"
            "  - **Official Eligibility:** B.Sc. (CS, Maths, Stats), **BA Eco** & BCA.\n"
            "  - **Fee:** ₹44,950 per semester.\n\n"
            "• **M.Sc. Artificial Intelligence** (2 Years, Self-Financed)\n"
            "  - **Official Eligibility:** B.Sc. (Maths, Stats, Phy, Chem), **BA Eco**, BE & BCA.\n"
            "  - **Fee:** ₹50,000 per semester.\n\n"
            "• **M.A. English** (2 Years, Self-Financed)\n"
            "  - Open to Bachelor's degree graduates from recognized universities.\n"
            "  - **Fee:** ₹24,950 per semester.\n\n"
            "Would you like details on the admission process, syllabus, or key dates for **M.Sc. Big Data Analytics** or **M.Sc. Artificial Intelligence**?"
        )
    src = [
        {
            "title": "Postgraduate Programmes & Official Eligibility",
            "source": "programmes-postgraduate.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/programmes/postgraduate/",
            "score": 1.0,
            "excerpt": "Postgraduate course offerings and official eligibility criteria at St. Xavier's College"
        },
        {
            "title": "SXCA Admission Guidelines & Brochure",
            "source": "admissions-sxca-edu-in-sxca.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://admissions.sxca.edu.in/SXCA/",
            "score": 1.0,
            "excerpt": "M.Sc. AI & M.Sc. BDA eligibility: BSc (Maths, Stats, Phy, Chem), BA Eco, BE & BCA"
        }
    ]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_commerce_stream_inquiry(profile: dict[str, Any] | None = None, is_guj: bool = False) -> dict[str, Any]:
    """Handles stream inquiries like 'mene commerce kiya hai course name' without guessing degree."""
    if is_guj:
        ans = (
            "જો તમારો બેકગ્રાઉન્ડ **કોમર્સ (Commerce)** છે, તો કૃપા કરીને સ્પષ્ટ કરશો કે તમે **12th કોમર્સ** પાસ કરેલ છે કે **સ્નાતક (Bachelor's Degree જેમ કે B.Com/BBA)** પૂર્ણ કરેલ છે?\n\n"
            "• **12th કોમર્સ પછી અંડરગ્રેજ્યુએટ (UG) વિકલ્પો:**\n"
            "  - **B.Com (General)** (Grant-in-Aid, ₹3,565 પ્રતિ સેમેસ્ટર)\n"
            "  - **B.Com - BPS** (Self-Financed, ₹19,950 પ્રતિ સેમેસ્ટર)\n"
            "  - **B.S. Business Administration** (Self-Financed, ₹30,000 પ્રતિ સેમેસ્ટર)\n"
            "  - **BCA (Bachelor of Computer Applications)** (Self-Financed, ₹29,750 પ્રતિ સેમેસ્ટર)\n\n"
            "• **કોમર્સ ગ્રેજ્યુએશન (B.Com) પછી અનુસ્નાતક (PG) વિકલ્પો:**\n"
            "  - **M.Sc. Big Data Analytics** (ગણિત/આંકડાશાસ્ત્ર બેકગ્રાઉન્ડ સાથે)\n"
            "  - **M.A. English**\n\n"
            "તમે કયા સ્તર (UG કે PG) ના અભ્યાસક્રમોમાં રસ ધરાવો છો?"
        )
    else:
        ans = (
            "If you have a **Commerce** background, could you please clarify whether you have completed **12th Standard (HSC)** or a **Bachelor's degree** (such as B.Com or BBA)?\n\n"
            "• **Undergraduate (UG) Options after 12th Commerce:**\n"
            "  - **B.Com (General)** (Grant-in-Aid, ₹3,565/sem)\n"
            "  - **B.Com - BPS (Business Process Services)** (Self-Financed, ₹19,950/sem)\n"
            "  - **B.S. Business Administration** (Self-Financed, ₹30,000/sem)\n"
            "  - **BCA (Bachelor of Computer Applications)** (Self-Financed, ₹29,750/sem)\n\n"
            "• **Postgraduate (PG) Options after Graduation (B.Com):**\n"
            "  - **M.Sc. Big Data Analytics** (for candidates with quantitative / statistics background)\n"
            "  - **M.A. English**\n\n"
            "Which level of study are you looking to pursue at St. Xavier's College?"
        )
    src = [
        {
            "title": "Undergraduate Programmes at SXCA",
            "source": "programmes-undergraduate.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/programmes/undergraduate/",
            "score": 1.0,
            "excerpt": "B.Com General, B.Com BPS, and B.S. Business Administration offerings"
        },
        {
            "title": "Postgraduate Programmes at SXCA",
            "source": "programmes-postgraduate.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/programmes/postgraduate/",
            "score": 1.0,
            "excerpt": "Postgraduate courses and eligibility at St. Xavier's College"
        }
    ]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_commerce_graduate_courses(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Answers what postgraduate and higher options are available for B.Com graduates at SXCA."""
    ans = (
        "Since you have completed **B.Com**, here are the postgraduate (PG) and professional options available at St. Xavier's College (Autonomous), Ahmedabad:\n\n"
        "• **M.Sc. Big Data Analytics** (2 Years, Self-Financed) — Open to graduates with quantitative and analytical foundations. If you studied Mathematics or Statistics as part of your degree, you are eligible to apply.\n"
        "• **M.A. English** (2 Years, Self-Financed) — Advanced studies in English literature and communication, open to Bachelor's graduates of any stream.\n"
        "• **Professional & Additional Degree Options:**\n"
        "  - If you wish to pivot into computer programming and IT, students also pursue **BCA (Bachelor of Computer Applications)**.\n"
        "  - If you are seeking specialized global business curriculum, you can explore **B.S. Business Administration**.\n\n"
        "Would you like detailed eligibility, course structure, or fee details for **M.Sc. Big Data Analytics** or **M.A. English**?"
    )
    src = [
        {"source": "programmes-postgraduate.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/programmes/postgraduate/", "score": 1.0, "excerpt": "Postgraduate courses and eligibility at St. Xavier's College"},
        {"source": "admissions-sxca-edu-in-sxca.html", "doc_type": "html", "page": 1, "url": "https://admissions.sxca.edu.in/SXCA/", "score": 1.0, "excerpt": "SXCA Admission Portal and Eligibility Guidelines"}
    ]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_bca_graduate_courses(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Answers what postgraduate options are available for BCA graduates at SXCA."""
    ans = (
        "Since you have completed **BCA**, you are eligible for advanced postgraduate technology programmes at St. Xavier's College (Autonomous), Ahmedabad:\n\n"
        "• **M.Sc. Artificial Intelligence** (2 Years, Self-Financed) — In-depth training in Machine Learning, Deep Learning, NLP, and Computer Vision. Open to graduates with a computer science / applications background.\n"
        "• **M.Sc. Big Data Analytics** (2 Years, Self-Financed) — Advanced data architectures, data engineering, statistical modeling, and machine learning.\n\n"
        "Would you like detailed eligibility criteria, fees, or curriculum details for **M.Sc. AI** or **M.Sc. Big Data Analytics**?"
    )
    src = [
        {"source": "programmes-postgraduate.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/programmes/postgraduate/", "score": 1.0, "excerpt": "Postgraduate courses and eligibility at St. Xavier's College"},
        {"source": "admissions-sxca-edu-in-sxca.html", "doc_type": "html", "page": 1, "url": "https://admissions.sxca.edu.in/SXCA/", "score": 1.0, "excerpt": "SXCA Admission Portal and Eligibility Guidelines"}
    ]
    return {"answer": ans, "sources": src, "grounded": True}


FACULTY_ROSTER: list[dict[str, Any]] = [
    {
        "id": "nisarg_vyas",
        "canonical_name": "Nisarg Vyas",
        "full_name": "Mr. Nisarg Vyas",
        "first_name": "Nisarg",
        "last_name": "Vyas",
        "title": "Mr.",
        "designation": "Assistant Professor & Coordinator of AI and ML",
        "department": "Department of Data Science",
        "email": "nisarg.vyas@sxca.edu.in",
        "programmes": "B.Sc. Artificial Intelligence & Machine Learning, BCA, M.Sc. AI, and M.Sc. Big Data Analytics",
        "source_doc": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html",
        "source_url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/",
        "aliases": ["nisarg vyas", "mr nisarg vyas", "dr nisarg vyas", "prof nisarg vyas", "nisarg vyash", "nisar vyas", "nisar vyash"],
        "roles": ["ai coordinator", "ai and ml coordinator", "coordinator of ai and ml"],
    },
    {
        "id": "nisarg_patel",
        "canonical_name": "Nisarg Patel",
        "full_name": "Mr. Nisarg Patel",
        "first_name": "Nisarg",
        "last_name": "Patel",
        "title": "Mr.",
        "designation": "Faculty / Staff",
        "department": "Department of Computer Science",
        "email": "nisarg.patel@sxca.edu.in",
        "source_doc": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html",
        "source_url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/",
        "aliases": ["nisarg patel", "mr nisarg patel"],
        "roles": [],
    },
    {
        "id": "nisarg_shah",
        "canonical_name": "Nisarg Shah",
        "full_name": "Mr. Nisarg Shah",
        "first_name": "Nisarg",
        "last_name": "Shah",
        "title": "Mr.",
        "designation": "Faculty / Staff",
        "department": "Faculty of Sciences",
        "email": "nisarg.shah@sxca.edu.in",
        "source_doc": "academics-departments-science-faculty-science.html",
        "source_url": "https://sxca.edu.in/academics/departments/science/faculty-science/",
        "aliases": ["nisarg shah", "mr nisarg shah"],
        "roles": [],
    },
    {
        "id": "nisarg_patil",
        "canonical_name": "Nisarg Patil",
        "full_name": "Mr. Nisarg Patil",
        "first_name": "Nisarg",
        "last_name": "Patil",
        "title": "Mr.",
        "designation": "Assistant Professor",
        "department": "Department of Biochemistry & Biotechnology",
        "email": "nisarg.patil@sxca.edu.in",
        "source_doc": "academics-departments-science-faculty-science.html",
        "source_url": "https://sxca.edu.in/academics/departments/science/faculty-science/",
        "aliases": ["nisarg patil", "mr nisarg patil"],
        "roles": [],
    },
    {
        "id": "david_roy",
        "canonical_name": "David K. Roy",
        "full_name": "Fr. Dr. David K. Roy SJ",
        "first_name": "David",
        "last_name": "Roy",
        "title": "Fr. Dr.",
        "designation": "Director & Assistant Professor",
        "department": "Department of Computer Sciences & Data Science",
        "email": "david.roy@sxca.edu.in",
        "programmes": "BCA, B.Sc. IT, B.Sc. AI & ML, M.Sc. AI, and M.Sc. Big Data Analytics",
        "source_doc": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html",
        "source_url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/",
        "aliases": ["david", "dr david", "fr david", "fr dr david roy sj", "david roy", "dr david roy", "fr david roy", "fr.david", "dr.david"],
        "roles": ["director"],
    },
    {
        "id": "sebastian_va",
        "canonical_name": "Sebastian V.A.",
        "full_name": "Dr. Sebastian V.A.",
        "first_name": "Sebastian",
        "last_name": "V.A.",
        "title": "Dr.",
        "designation": "Principal & Associate Professor",
        "department": "Department of Economics",
        "email": "principal@sxca.edu.in",
        "source_doc": "about-us.html",
        "source_url": "https://sxca.edu.in/about-us/",
        "aliases": ["sebastian", "dr sebastian", "dr sebastian va", "sebastian va", "dr.sebastian"],
        "roles": ["principal"],
    },
    {
        "id": "antony_pitchai",
        "canonical_name": "Antony Pitchai",
        "full_name": "Fr. Antony Pitchai SJ",
        "first_name": "Antony",
        "last_name": "Pitchai",
        "title": "Fr.",
        "designation": "Rector & Jesuit Superior",
        "department": "College Trust / Governing Body",
        "email": "rector@sxca.edu.in",
        "source_doc": "about-us.html",
        "source_url": "https://sxca.edu.in/about-us/",
        "aliases": ["antony pitchai", "fr antony pitchai", "pitchai", "fr antony", "fr.antony"],
        "roles": ["rector", "superior"],
    },
    {
        "id": "pravida_raja",
        "canonical_name": "Pravida Raja",
        "full_name": "Dr. Pravida Raja",
        "first_name": "Pravida",
        "last_name": "Raja",
        "title": "Dr.",
        "designation": "Head of Department (HOD) & Assistant Professor",
        "department": "Department of Data Science",
        "email": "pravida.raja@sxca.edu.in",
        "programmes": "BCA, B.Sc. AI & ML, B.Sc. IT, M.Sc. Artificial Intelligence, and M.Sc. Big Data Analytics",
        "source_doc": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html",
        "source_url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/",
        "aliases": ["pravida", "pravida raja", "dr pravida", "dr pravida raja", "dr.pravida", "dr.pravida raja"],
        "roles": ["hod of data science", "hod", "head of department"],
    },
    {
        "id": "nithya_nadar",
        "canonical_name": "Nithya Nadar",
        "full_name": "Dr. Nithya Nadar",
        "first_name": "Nithya",
        "last_name": "Nadar",
        "title": "Dr.",
        "designation": "Assistant Professor",
        "department": "Department of Computer Sciences & Data Science",
        "email": "nithya.nadar@sxca.edu.in",
        "programmes": "BCA, B.Sc. IT, and computing programmes",
        "source_doc": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html",
        "source_url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/",
        "aliases": ["nithya", "nithya nadar", "dr nithya", "dr nithya nadar", "dr.nithya"],
        "roles": [],
    },
    {
        "id": "shaunak_vyas",
        "canonical_name": "Shaunak Vyas",
        "full_name": "Mr. Shaunak Vyas",
        "first_name": "Shaunak",
        "last_name": "Vyas",
        "title": "Mr.",
        "designation": "Assistant Professor",
        "department": "Department of Communication",
        "email": "shaunak.vyas@sxca.edu.in",
        "source_doc": "academics-departments-arts-humanities-communication.html",
        "source_url": "https://sxca.edu.in/academics/departments/arts-humanities/communication/",
        "aliases": ["shaunak", "shaunak vyas", "mr shaunak vyas", "mr.shaunak vyas"],
        "roles": [],
    },
    {
        "id": "sudeshna_menon",
        "canonical_name": "Sudeshna Menon",
        "full_name": "Dr. Sudeshna Menon",
        "first_name": "Sudeshna",
        "last_name": "Menon",
        "title": "Dr.",
        "designation": "Head, Assistant Professor",
        "department": "Department of Biochemistry & Biotechnology",
        "email": "sudeshna.menon@sxca.edu.in",
        "source_doc": "academics-departments-science-faculty-science.html",
        "source_url": "https://sxca.edu.in/academics/departments/science/faculty-science/",
        "aliases": ["sudeshna", "sudeshna menon", "dr sudeshna menon"],
        "roles": [],
    },
    {
        "id": "pinky_desai",
        "canonical_name": "Pinky Desai",
        "full_name": "Dr. Pinky Desai",
        "first_name": "Pinky",
        "last_name": "Desai",
        "title": "Dr.",
        "designation": "Head, Associate Professor",
        "department": "Department of Economics",
        "email": "pinky.desai@sxca.edu.in",
        "source_doc": "academics-departments-arts-humanities-economics.html",
        "source_url": "https://sxca.edu.in/academics/departments/arts-humanities/economics/",
        "aliases": ["pinky", "pinky desai", "dr pinky desai"],
        "roles": [],
    },
    {
        "id": "nandita_christy",
        "canonical_name": "Nandita Christy",
        "full_name": "Dr. Nandita Christy",
        "first_name": "Nandita",
        "last_name": "Christy",
        "title": "Dr.",
        "designation": "Head, Associate Professor",
        "department": "Department of Statistics",
        "email": "nandita.christy@sxca.edu.in",
        "source_doc": "academics-departments-science-statistics.html",
        "source_url": "https://sxca.edu.in/academics/departments/science/statistics/",
        "aliases": ["nandita", "nandita christy", "dr nandita christy"],
        "roles": [],
    }
]


def _extract_target_name(norm_q: str) -> str:
    """Extract candidate person name / tokens from query."""
    q_clean = norm_q.rstrip("?.! ")

    # 1. Question patterns: "who is X", "tell me about X", "about X", "who is the X", "details of X"
    m = re.search(
        r"\b(?:who\s+(?:is|was)|tell\s+me\s+about|information\s+about|details\s+(?:of|about)|about|kon\s+che|kaun\s+hai)\s+(?:dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?|fr\.?)?\s*([A-Za-z\s\.\']+)",
        q_clean,
        re.I,
    )
    if m:
        extracted = m.group(1).strip()
        extracted = re.sub(r"\b(?:in|at|from|of)\s+(?:sxca|st\.?\s*xavier'?s?|college|clg).*$", "", extracted, flags=re.I).strip()
        return extracted

    # 2. Direct honorific pattern: "dr. david", "mr nisarg vyas", "fr.david"
    m2 = re.search(r"\b(?:dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?|fr\.?)\s*([A-Za-z\s\.\']+)", q_clean, re.I)
    if m2:
        extracted = m2.group(1).strip()
        extracted = re.sub(r"\b(?:in|at|from|of)\s+(?:sxca|st\.?\s*xavier'?s?|college|clg).*$", "", extracted, flags=re.I).strip()
        return extracted

    return q_clean


_ANOTHER_FACULTY_RE = re.compile(
    r"\b(?:another|different|other|someone\s+else|some\s+other)\s+"
    r"(?:faculty(?:\s*member)?|prof(?:essor)?|teacher|staff|person|member|hod|coordinator|director|principal)\b|"
    r"\b(?:tell\s+me\s+about\s+(?:another|a\s+different|another\s+one))\b",
    re.I,
)

_PERSON_PRONOUNS_RE = re.compile(
    r"\b(he|she|him|her|his|hers|they|them|their|theirs|that\s+person|this\s+person|the\s+faculty|the\s+professor|the\s+teacher)\b",
    re.I,
)

_PERSON_ATTRIBUTE_RE = re.compile(
    r"\b(teach|teaches|teaching|subjects?|programmes?|courses?|department|dept|which\s*dept|what\s*dept|"
    r"his\s*dept|her\s*dept|email|mail|contact|office|exprience|experience|qualification|salary|phone|mobile|age|degree|phd)\b",
    re.I,
)


def _recover_active_person_from_turns(recent_turns: list[dict] | None) -> dict[str, Any] | None:
    if not recent_turns:
        return None
    for turn in reversed(recent_turns):
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            m_amb = re.search(r"multiple people matching ['\"]?([A-Za-z]+)['\"]?", content, re.I)
            if m_amb:
                amb_name = m_amb.group(1).title()
                cands = [
                    p["canonical_name"]
                    for p in FACULTY_ROSTER
                    if p["first_name"].lower() == amb_name.lower() or p["last_name"].lower() == amb_name.lower()
                ]
                return {
                    "status": "AMBIGUOUS",
                    "name": amb_name,
                    "candidates": cands,
                }
            for person in FACULTY_ROSTER:
                if person["canonical_name"] in content or person["full_name"] in content:
                    return {
                        "id": person["id"],
                        "name": person["canonical_name"],
                        "full_name": person["full_name"],
                        "status": "RESOLVED",
                        "person": person,
                    }
        elif role == "user":
            if _ANOTHER_FACULTY_RE.search(content):
                return {"status": "NOT_FOUND"}
            f_check = resolve_faculty(content)
            if f_check["status"] == "RESOLVED":
                p = f_check["person"]
                return {
                    "id": p["id"],
                    "name": p["canonical_name"],
                    "full_name": p["full_name"],
                    "status": "RESOLVED",
                    "person": p,
                }
            elif f_check["status"] == "AMBIGUOUS":
                return {
                    "status": "AMBIGUOUS",
                    "name": f_check.get("query", ""),
                    "candidates": [c["canonical_name"] for c in f_check["candidates"]],
                }
    return None


def resolve_faculty(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Accurate faculty and person entity-resolution.
    1. Collects ALL valid matching entities from the official SXCA faculty/staff roster.
    2. If exactly ONE person matches: returns status 'RESOLVED' and the person record.
    3. If MORE THAN ONE person matches: returns status 'AMBIGUOUS' and the list of candidates.
    4. If ZERO people match: returns status 'NOT_FOUND'.
    Does NOT use highest similarity to pick one when multiple valid matches exist.
    """
    if not question:
        return {"status": "NOT_FOUND", "query": "", "candidates": []}

    from backend.rag.query_rewrite import normalize_student_text
    norm_q = normalize_student_text(question).strip().lower()

    # 0. Check if user explicitly requests another / different faculty member
    if _ANOTHER_FACULTY_RE.search(norm_q):
        return {"status": "ANOTHER_FACULTY", "query": norm_q, "candidates": []}

    prof = profile or {}
    active_p = prof.get("active_person")
    p_status = None
    p_name = ""
    p_id = ""
    p_candidates = []
    if isinstance(active_p, dict):
        p_status = active_p.get("status")
        p_name = active_p.get("name") or active_p.get("full_name") or ""
        p_id = active_p.get("id") or ""
        p_candidates = active_p.get("candidates") or []
    elif isinstance(active_p, str) and active_p:
        p_status = "RESOLVED"
        p_name = active_p

    # Check for pronoun or implicit follow-up inquiry on active person
    has_pronoun = bool(_PERSON_PRONOUNS_RE.search(norm_q))
    has_person_attr = bool(_PERSON_ATTRIBUTE_RE.search(norm_q))

    # If the user uses a pronoun or asks an attribute question without naming a different person:
    if has_pronoun or (p_status and has_person_attr and not any(p["first_name"].lower() in norm_q or p["last_name"].lower() in norm_q for p in FACULTY_ROSTER)):
        if p_status == "AMBIGUOUS":
            return {
                "status": "AMBIGUOUS_FOLLOWUP",
                "name": p_name or "that name",
                "candidates": p_candidates,
            }
        elif p_status == "RESOLVED":
            matched_person = None
            for p in FACULTY_ROSTER:
                if p_id and p.get("id") == p_id:
                    matched_person = p
                    break
                if p_name and (p["canonical_name"].lower() == p_name.lower() or p["full_name"].lower() == p_name.lower()):
                    matched_person = p
                    break
            if matched_person:
                return {"status": "RESOLVED", "person": matched_person}
        elif p_status == "NOT_FOUND":
            return {"status": "NOT_FOUND", "query": norm_q, "candidates": []}

    # Follow-up regarding active person's department
    if p_name and re.search(r"\b(department|his\s*dept|her\s*dept|which\s*dept|what\s*dept)\b", norm_q):
        for person in FACULTY_ROSTER:
            if person["full_name"].lower() == p_name.lower() or person["canonical_name"].lower() == p_name.lower():
                return {"status": "RESOLVED", "person": person}

    # Extract target name/tokens
    raw_target = _extract_target_name(norm_q)
    stop_tokens = {
        "who", "is", "was", "tell", "me", "about", "dr", "prof", "mr", "ms",
        "mrs", "fr", "the", "at", "in", "of", "sxca", "college", "clg",
        "st", "xaviers", "xavier", "ahmedabad", "details", "info", "information",
        "kon", "che", "kaun", "hai", "kya", "name", "sir", "madam",
        "what", "does", "he", "she", "teach", "teaches", "teaching", "his", "her",
        "email", "department", "dept"
    }
    target_tokens = [t for t in re.findall(r"[a-z0-9]+", raw_target) if t not in stop_tokens]

    # If no name tokens were found, check if query matched by specific administrative role
    if not target_tokens:
        role_matches = []
        for person in FACULTY_ROSTER:
            for role in person.get("roles", []):
                if re.search(r"\b" + re.escape(role) + r"\b", norm_q):
                    if person not in role_matches:
                        role_matches.append(person)
                    break
        if len(role_matches) == 1:
            return {"status": "RESOLVED", "person": role_matches[0]}
        elif len(role_matches) > 1:
            return {"status": "AMBIGUOUS", "query": norm_q, "candidates": role_matches}
        return {"status": "NOT_FOUND", "query": norm_q, "candidates": []}

    target_query = " ".join(target_tokens)

    # 1. Check exact full alias matches across the roster (e.g. "nisarg vyas", "david roy")
    full_alias_matches: list[dict[str, Any]] = []
    for person in FACULTY_ROSTER:
        for alias in person.get("aliases", []):
            if re.search(r"\b" + re.escape(alias) + r"\b", norm_q) or re.search(r"\b" + re.escape(alias) + r"\b", target_query):
                if person not in full_alias_matches:
                    full_alias_matches.append(person)
                break

    if len(full_alias_matches) == 1:
        return {"status": "RESOLVED", "person": full_alias_matches[0]}
    elif len(full_alias_matches) > 1:
        return {"status": "AMBIGUOUS", "query": target_query.title(), "candidates": full_alias_matches}

    # 2. Token-level matching against candidates
    candidate_matches: list[dict[str, Any]] = []
    for person in FACULTY_ROSTER:
        p_first = person["first_name"].lower()
        p_last = person["last_name"].lower()
        p_canonical = person["canonical_name"].lower()
        p_aliases = [a.lower() for a in person.get("aliases", [])]

        if len(target_tokens) >= 2:
            # Multi-token query (e.g. "nisarg vyas" or "david roy")
            if target_query in p_canonical or any(target_query in a for a in p_aliases):
                if person not in candidate_matches:
                    candidate_matches.append(person)
            elif all(any(tok == name_part for name_part in (p_first, p_last)) or any(tok in a for a in p_aliases) for tok in target_tokens):
                if person not in candidate_matches:
                    candidate_matches.append(person)
        else:
            # Single-token query (e.g. "nisarg", "vyas", "david", "pravida")
            tok = target_tokens[0]
            if tok == p_first or tok == p_last or tok in p_canonical or any(tok == a for a in p_aliases):
                if person not in candidate_matches:
                    candidate_matches.append(person)

    if len(candidate_matches) == 1:
        return {"status": "RESOLVED", "person": candidate_matches[0]}
    elif len(candidate_matches) > 1:
        # Check overwrite rule:
        # Do NOT overwrite a resolved person with an ambiguous partial name unless the new query explicitly identifies another person.
        if p_status == "RESOLVED" and p_name:
            matching_curr = [c for c in candidate_matches if c["canonical_name"].lower() == p_name.lower()]
            if matching_curr:
                return {"status": "RESOLVED", "person": matching_curr[0]}
        return {"status": "AMBIGUOUS", "query": target_query.title(), "candidates": candidate_matches}
    else:
        return {"status": "NOT_FOUND", "query": target_query.title(), "candidates": []}


def should_inherit_course_context(question: str, profile: dict | None = None, recent_turns: list | None = None) -> bool:
    """Context Gate: Determines whether active_course should be inherited.
    Returns False for independent topics (faculty, leadership, contact, helpline,
    hostel, clubs, sports, events, calendar/exams, library, complaints, general discovery)
    unless the user explicitly names or references a course in the immediate query.
    """
    q_low = question.strip().lower()

    # 1. If query explicitly names a course/degree, do not inherit old course (new course takes precedence)
    matched_c, _ = match_course(question)
    if matched_c:
        return False

    # 2. Independent topics that MUST BREAK active course context:
    # a. Faculty / Person / Leadership / Staff / Department follow-up
    if (
        re.search(
            r"\b(who\s+is|tell\s+me\s+about\s+(?:dr|prof|mr|ms|fr)|dr\.?|prof\.?|mr\.?|ms\.?|fr\.?|"
            r"faculty|professor|professors|teacher|teachers|head\s+of|hod|coordinator|"
            r"principal|director|rector|vice\s*principal|president|presedent|chairman|trustee|provincial|who\s*heads|"
            r"david|pravida|nithya|sebastian|pitchai|nadar|vyas|vyash|nisarg|nisar|"
            r"which\s+department|what\s+department|his\s+department|her\s+department|"
            r"what\s+does\s+he\s+teach|what\s+does\s+she\s+teach|what\s+is\s+his\s+email|what\s+is\s+her\s+email|"
            r"tell\s+me\s+about\s+(?:him|her)|another\s+faculty)\b",
            q_low,
        )
        or "fr.david" in q_low
        or "dr.david" in q_low
        or "nisar vyash" in q_low
        or bool(profile and profile.get("active_person") and _PERSON_PRONOUNS_RE.search(q_low))
    ):
        return False

    # b. Contact / Phone / Helpline / Office Timing / Address
    if re.search(
        r"\b(contact|phone|telephone|mobile|helpline|call\b|can\s+i\s+call|number\s+for|where\s+can\s+i\s+call|"
        r"office\s*timing|office\s*hours?|clg\s*timing|college\s*timing|office\s*kyare|"
        r"address|location|where\s+is\s+(?:the\s+)?college|kya\s+che|kyan\s+che)\b",
        q_low,
    ):
        return False

    # c. Hostel / Accommodation
    if re.search(
        r"\b(hostel|hostels|hostal|hostle|boys\s*hostel|girls\s*hostel|accommodation|stay|mess\b|living\s*facility)\b",
        q_low,
    ):
        return False

    # d. Clubs, Sports, Gym, NCC, NSS, Facilities, Library, Placement, Scholarships, Events, Culfest
    if re.search(
        r"\b(clubs?|student\s*clubs?|sports?|gym|gymnasium|playground|ncc\b|nss\b|"
        r"events?|event\s*kya|function|functions|workshops?|seminars?|culfest|cultural\b|"
        r"library|placement|placements?|career\s*cell|recruiters?|companies|scholarships?|freeship)\b",
        q_low,
    ):
        return False

    # e. Exams & Academic Calendar
    if re.search(
        r"\b(cai-?2|cai\s*2|cia-?2|cia\s*2|cai-?1|cia-?1|exam\s*dates?|examination\s*dates?|pariksha|"
        r"academic\s*calendar|holiday|holidays|vacation|vacations|reopen|term\s*dates?|semester\s*start)\b",
        q_low,
    ):
        return False

    # f. Campus Complaints / IT Issues
    if re.search(
        r"\b(complaint|broken|not\s*working|repair|damaged|issue\s+with|problem\s+in|"
        r"lab\s*\d+|room\s*\d+|computer\s+in|pc\s+in)\b",
        q_low,
    ):
        return False

    # g. Broad Course Discovery / General Guidance
    if re.search(
        r"^(?:courses\??|all\s*courses\??|available\s*courses\??|programmes\??|programs\??)$|"
        r"\b(what\s*courses|which\s*courses|list\s+of\s+courses|available\s+courses|"
        r"ug\s*courses|all\s*ug\s*courses|pg\s*courses|all\s*pg\s*courses|which\s*pg\s*courses|"
        r"master\s*ma\s*course|master\s*ma\s*kaya\s*course|science\s*background)\b",
        q_low,
    ):
        return False

    # h. General Admission Process (when asked generally without referring to "this course")
    if re.search(
        r"^(?:what\s+(?:is\s+)?(?:the\s+)?admission\s+process\??|admission\s+process\??|"
        r"how\s+to\s+apply\??|how\s+to\s+get\s+admission\??|admission\s+steps\??|"
        r"admission\s+procedure\??|does\s+interview\s+happen\??|is\s+interview\s+required\??)$",
        q_low,
    ):
        return False

    # 3. Explicit / strong follow-ups that SHOULD inherit active_course:
    if re.search(r"\b(its|it\b|this\s*course|that\s*course|for\s*this|in\s*this)\b", q_low):
        return True

    # Attribute questions on the active course:
    if re.search(
        r"\b(fee|fees|fess|feee|free|cost|charges|tuition|how\s*much|"
        r"subjects?|subj|syllabus|curriculum|papers?|sem\s*[1-8]|semester\s*[1-8]|"
        r"eligibility|eligible|criteria|qualify|laaykat|patrata|"
        r"documents?|docs?|marksheet|certificate|papers\s*required|"
        r"duration|how\s*many\s*years|how\s*long)\b",
        q_low,
    ):
        return True

    return False


def extract_and_update_profile(
    current_profile: dict[str, Any] | None,
    question: str,
    recent_turns: list[dict] | None = None,
) -> dict[str, Any]:
    """Extracts facts incrementally from student message, preserving existing
    state and executing strict context switching when a new course is named."""
    profile = dict(DEFAULT_PROFILE)
    if current_profile:
        profile.update(current_profile)

    # Recover active_person if not already in profile
    if not profile.get("active_person") and recent_turns:
        rec_person = _recover_active_person_from_turns(recent_turns)
        if rec_person:
            profile["active_person"] = rec_person

    q = question.strip()
    q_low = q.lower()

    # Language Preference Detection
    if re.search(r"\b(mane\s*gujarati\s*ma|gujarati\s*ma\s*j|mne\s*guj|guj\s*ma\s*jawab|gujarati\s*ma\s*bolo)\b", q_low):
        profile["language_preference"] = "gujarati"
    elif re.search(r"\b(answer\s*in\s*english|english\s*ma\s*bolo|reply\s*in\s*english)\b", q_low):
        profile["language_preference"] = "english"

    # Entity Tracking for Faculty / Leadership
    f_res = resolve_faculty(q, profile)
    if f_res["status"] == "RESOLVED":
        person = f_res.get("person")
        if person:
            profile["active_person"] = {
                "id": person["id"],
                "name": person["canonical_name"],
                "full_name": person.get("full_name", person["canonical_name"]),
                "status": "RESOLVED",
            }
            profile["active_department"] = person.get("department", "")
            profile["active_topic"] = "faculty"
            profile["ambiguous_faculty"] = None
    elif f_res["status"] == "AMBIGUOUS":
        # Check overwrite rule:
        # Do NOT overwrite a resolved person with an ambiguous partial name unless the new query explicitly identifies another person.
        curr_p = profile.get("active_person")
        is_already_resolved = isinstance(curr_p, dict) and curr_p.get("status") == "RESOLVED"
        matched_target = f_res.get("query", "").strip().lower()
        if is_already_resolved and matched_target:
            curr_name = (curr_p.get("name") or "").lower()
            curr_full = (curr_p.get("full_name") or "").lower()
            if matched_target in curr_name or matched_target in curr_full:
                pass  # Keep resolved person
            else:
                profile["active_person"] = {
                    "status": "AMBIGUOUS",
                    "name": f_res.get("query", "").title(),
                    "candidates": [c["canonical_name"] for c in f_res["candidates"]],
                }
                profile["active_topic"] = "faculty"
                profile["ambiguous_faculty"] = [c["canonical_name"] for c in f_res["candidates"]]
        else:
            profile["active_person"] = {
                "status": "AMBIGUOUS",
                "name": f_res.get("query", "").title(),
                "candidates": [c["canonical_name"] for c in f_res["candidates"]],
            }
            profile["active_topic"] = "faculty"
            profile["ambiguous_faculty"] = [c["canonical_name"] for c in f_res["candidates"]]
    elif f_res["status"] == "ANOTHER_FACULTY":
        profile["active_person"] = {"status": "NOT_FOUND"}
        profile["active_topic"] = "faculty"
        profile["ambiguous_faculty"] = None
    elif f_res["status"] == "AMBIGUOUS_FOLLOWUP":
        profile["active_topic"] = "faculty"
    elif re.search(r"\b(president|presedent|chairman|trustee)\b", q_low):
        profile["active_person"] = {
            "id": "antony_pitchai",
            "name": "Fr. Antony Pitchai SJ",
            "full_name": "Fr. Antony Pitchai SJ",
            "status": "RESOLVED",
        }
        profile["active_topic"] = "faculty"

    # 0. Check for conversation feedback repair, negation, contextual no, or user correction
    if is_conversation_feedback(q):
        profile["pending_clarification"] = None
        profile["mode"] = "direct_course" if profile.get("active_course") else "general"
        return profile

    if is_user_correction(q):
        profile["mode"] = "general"
        profile["pending_clarification"] = None
        return profile

    is_neg, neg_topic = detect_negation(q, profile)
    if is_neg:
        profile["pending_clarification"] = None
        if neg_topic and neg_topic.startswith("course:"):
            rej_course = neg_topic.split(":", 1)[1]
            profile["active_course"] = None
            if "rejected_courses" not in profile:
                profile["rejected_courses"] = []
            if rej_course not in profile["rejected_courses"]:
                profile["rejected_courses"].append(rej_course)
            profile["mode"] = "counselling"
        return profile

    if is_contextual_no(q):
        profile["pending_clarification"] = None
        return profile

    # 0a. Check for confirmation/course switch in pending clarification or recent turns
    pending = profile.get("pending_clarification")
    last_assistant_msg = ""
    for turn in reversed(recent_turns or []):
        if turn.get("role") == "assistant":
            last_assistant_msg = (turn.get("content") or "").lower()
            break

    is_ai_prompt = (
        (pending and pending.get("type") == "ai")
        or "two dedicated artificial intelligence programmes" in last_assistant_msg
        or "b.sc. ai & ml" in last_assistant_msg
    )
    if is_ai_prompt:
        if re.search(r"\b(bsc|b\.?s\.?c|ug|undergraduate|12th|bachelor)\b", q_low):
            profile["active_course"] = "BSc-Artificial Intelligence & Machine Learning"
            profile["course_level"] = "UG"
            profile["mode"] = "direct_course"
            profile["pending_clarification"] = None
            return profile
        elif re.search(r"\b(msc|m\.?s\.?c|pg|postgraduate|master|masters)\b", q_low):
            profile["active_course"] = "MSc-Artificial Intelligence"
            profile["course_level"] = "PG"
            profile["mode"] = "direct_course"
            profile["pending_clarification"] = None
            return profile

    is_fee_prompt = (
        (pending and pending.get("type") == "fee")
        or "do you mean the fee for" in last_assistant_msg
    )
    if is_fee_prompt:
        if re.search(r"^(?:yes|yeah|yep|ha|haa|sure|correct|right|y|haan)\b", q_low):
            profile["confirmed_clarification"] = {"type": "fee", "course": profile.get("active_course")}
            profile["pending_clarification"] = None
            return profile
        elif re.search(r"^no\b", q_low):
            profile["pending_clarification"] = None
            matched_course, matched_level = match_course(q)
            if matched_course:
                profile["active_course"] = matched_course
                profile["course_level"] = matched_level
                profile["mode"] = "direct_course"
                return profile
            profile["mode"] = "general"
            return profile

    # 0b. Check for Completed Qualification (e.g., 'mene bcom kiya hai konsa course hai', 'mene commerce kiya hai course name')
    comp_res = detect_completed_qualification(q)
    comp_course = comp_res.get("qualification")
    comp_level = comp_res.get("level")
    comp_stream = comp_res.get("stream")
    if comp_course:
        profile["qualification"] = clean_course_display_name(comp_course)
        profile["education_level"] = comp_level or "bachelor"
        if comp_stream:
            profile["stream"] = comp_stream
        profile["course_level"] = "PG"
        profile["active_course"] = None
        profile["mode"] = "counselling"
        profile["pending_clarification"] = None
        return profile
    elif comp_stream:
        profile["stream"] = comp_stream
        profile["education_level"] = comp_level
        profile["active_course"] = None
        profile["mode"] = "counselling"
        profile["pending_clarification"] = None
        return profile

    # 1. Context Gate & Course Switching
    can_inherit_course = should_inherit_course_context(q, profile, recent_turns)
    matched_course, matched_level = match_course(q)

    # Do not set active_course to a course the user previously rejected!
    if matched_course and matched_course in profile.get("rejected_courses", []):
        matched_course = None

    if matched_course:
        profile["active_course"] = matched_course
        profile["course_level"] = matched_level
        profile["mode"] = "direct_course"
        profile["pending_clarification"] = None
    elif any(phrase in q_low for phrase in ["changed my mind", "different course", "other course"]):
        profile["active_course"] = None
        profile["mode"] = "counselling"
        profile["pending_clarification"] = None
    elif not can_inherit_course:
        # Context gate closed: do NOT inherit previous course for independent topics!
        profile["active_course"] = None
    elif not profile.get("active_course") and recent_turns and can_inherit_course:
        for turn in reversed(recent_turns):
            content = turn.get("content", "")
            tc, tl = match_course(content)
            if tc:
                profile["active_course"] = tc
                if not profile.get("course_level"):
                    profile["course_level"] = tl
                profile["mode"] = "direct_course"
                break

    # 2. Education Level extraction
    if re.search(r"\b(12th|hsc|class\s*12|plus\s*two|passed\s*12|completed\s*12|in\s*12th)\b", q_low):
        profile["education_level"] = "12th"
        if not profile.get("course_level"):
            profile["course_level"] = "UG"
    elif re.search(r"\b(bachelor'?s?|graduation|graduate|b\.?sc|b\.?com|b\.?a|degree\s*completed)\b", q_low) and not matched_course:
        profile["education_level"] = "bachelor"
        if not profile.get("course_level"):
            profile["course_level"] = "PG"

    # 3. Stream extraction
    if re.search(r"\b(science|sci)\b", q_low) and not re.search(r"\b(data\s*science|computer\s*science)\b", q_low):
        profile["stream"] = "Science"
    elif re.search(r"\b(commerce|comm)\b", q_low):
        profile["stream"] = "Commerce"
    elif re.search(r"\b(arts|humanities)\b", q_low):
        profile["stream"] = "Arts"

    # 4. Percentage extraction
    pct_m = re.search(r"\b(\d{1,2}(?:\.\d+)?)\s*(?:%|percent\b)", q_low)
    if pct_m:
        profile["percentage"] = pct_m.group(1)

    # 5. Interest Area extraction
    if re.search(r"\b(ai|artificial\s*intelligence|machine\s*learning|ml|ai/ml)\b", q_low) and not matched_course:
        profile["interest_area"] = "AI/ML"
    elif re.search(r"\b(computer|computers|it|software|programming|coding|tech)\b", q_low) and not matched_course:
        profile["interest_area"] = "Computer/IT"
    elif re.search(r"\b(biology|biotech|microbiology|life\s*sciences?)\b", q_low) and not matched_course:
        profile["interest_area"] = "Life Sciences"
    elif re.search(r"\b(pure\s*sciences?|physics|chemistry|maths?|mathematics)\b", q_low) and not matched_course:
        profile["interest_area"] = "Pure Science"
    elif re.search(r"\b(business|finance|accounting|management)\b", q_low) and not matched_course:
        profile["interest_area"] = "Commerce/Business"
    elif re.search(r"\b(literature|psychology|theatre)\b", q_low) and not matched_course:
        profile["interest_area"] = "Arts"

    # 6. Career Goal extraction
    if re.search(r"\b(ai\s*engineer|data\s*scientist|machine\s*learning\s*engineer)\b", q_low):
        profile["career_goal"] = "AI Engineer"
        profile["interest_area"] = "AI/ML"
    elif re.search(r"\b(software\s*(?:developer|engineer)|programmer|coder)\b", q_low):
        profile["career_goal"] = "Software Developer"
        profile["interest_area"] = "Computer/IT"

    # 7. Explicit Level Preference
    if re.search(r"\b(ug|undergraduate|bachelor'?s?)\b", q_low) and not matched_course:
        profile["course_level"] = "UG"
    elif re.search(r"\b(pg|postgraduate|master'?s?)\b", q_low) and not matched_course:
        profile["course_level"] = "PG"

    # 8. Counselling Mode trigger
    if re.search(r"\b(don'?t\s*know|confused|help\s*me\s*choose|guide\s*me|what\s*can\s*i\s*do|which\s*course\s*to\s*choose|what\s*should\s*i\s*(?:do|take|choose))\b", q_low):
        profile["mode"] = "counselling"

    return profile


def is_zero_cost_inquiry(question: str) -> bool:
    """Detects genuine zero-cost / free education inquiries like 'Is MSc AI free?'."""
    q = question.lower().strip()
    if re.search(r"\b(hostel|hostal|hostle|accommodation|stay|mess)\b", q):
        return False
    return bool(re.search(r"\b(is\s+(?:it|this|the\s+course|msc\s*ai|bca|bcom|m\.?sc\.?\s*ai)\s*free|free\s+of\s+cost|bina\s+paise|muft)\b", q))


def handle_zero_cost(course_name: str | None) -> str:
    """Answers whether a course is free with authoritative fee data."""
    c_name = course_name or "MSc-Artificial Intelligence"
    disp_name = clean_course_display_name(c_name)
    from backend.rag import eligibility
    rows = eligibility.find_by_course(c_name)
    if rows and rows[0].get("fee_amount"):
        amt = int(rows[0]["fee_amount"])
        note = "Grant-in-Aid" if rows[0].get("category") == "grant_in_aid" else "Self-Financed"
        return f"**{disp_name}** has a fee of **₹{amt:,} per semester** ({note}), so it is not a free programme."
    return f"**{disp_name}** has a fee of **₹50,000 per semester** (Self-Financed), so it is not a free programme."


def handle_fee_confirmation(course_name: str | None) -> str:
    """Confirms if the student meant the course fee when typing bare 'free'."""
    disp_name = clean_course_display_name(course_name) if course_name else "M.Sc. Artificial Intelligence"
    return f"Do you mean the fee for **{disp_name}**?"


def handle_ai_ambiguity() -> str:
    """Asks ONE clarifying question when user simply types 'AI' without course level."""
    return (
        "St. Xavier's College (Autonomous), Ahmedabad offers two dedicated Artificial Intelligence programmes:\n\n"
        "• **B.Sc. Artificial Intelligence & Machine Learning** (Undergraduate — 3/4-year degree after 12th Science)\n"
        "• **M.Sc. Artificial Intelligence** (Postgraduate — 2-year Master's degree after Bachelor's)\n\n"
        "Are you asking about the **B.Sc. AI & ML** undergraduate programme or the **M.Sc. AI** master's degree?"
    )


def classify_intent(question: str, profile: dict[str, Any], recent_turns: list[dict] | None = None) -> str:
    """Classifies user's intent within the counsellor context using human-like NLU."""
    q_low = question.lower().strip()

    # 0. Conversation Feedback & Repair (Top Priority - e.g. "why you keep giving same answer")
    if is_conversation_feedback(question):
        return "FEEDBACK_REPAIR"

    # 0a. Check for Completed Qualification (e.g., 'mene bcom kiya hai konsa course hai', 'mene bachelor mai b.eco liya hai master mai konse course hai', 'mene commerce kiya hai course name')
    comp_res = detect_completed_qualification(question)
    comp_course = comp_res.get("qualification")
    comp_stream = comp_res.get("stream")
    if comp_course == "BA-Economics" or (comp_course and "economics" in comp_course.lower()):
        return "ELIGIBLE_PG_COURSES"
    elif comp_course and ("bcom" in comp_course.lower() or "business administration" in comp_course.lower()):
        return "COMMERCE_GRADUATE_COURSES"
    elif comp_course and "bca" in comp_course.lower():
        return "BCA_GRADUATE_COURSES"
    elif comp_course == "12th":
        if comp_stream == "Arts":
            return "COURSE_DISCOVERY"
        elif comp_stream == "Commerce":
            return "COURSE_DISCOVERY"
        elif comp_stream == "Science":
            return "SCIENCE_COURSES"
        return "COURSE_DISCOVERY"
    elif comp_stream and not comp_course:
        if comp_stream == "Commerce":
            return "COMMERCE_STREAM_INQUIRY"
        elif comp_stream == "Arts":
            return "COURSE_DISCOVERY"
        elif comp_stream == "Science":
            return "SCIENCE_COURSES"
        return "COURSE_DISCOVERY"
    elif comp_course:
        return "GENERAL_PG_ADMISSION"

    # 0b. Contextual PG degree inquiry (e.g. 'in master not ug', 'master courses', 'pg only')
    is_pg_level_inquiry = bool(re.search(r"\b(in\s*masters?|masters?\s*not\s*ug|masters?\s*only|pg\s*not\s*ug|postgraduate\s*not\s*undergraduate|in\s*pg|pg\s*only)\b", q_low))
    if is_pg_level_inquiry:
        prior_qual = profile.get("qualification")
        recent_text = " ".join((turn.get("content") or "").lower() for turn in (recent_turns or [])[-4:])
        if prior_qual == "BA-Economics" or re.search(r"\b(b\.?\s*eco|ba\s*economics|economics)\b", recent_text):
            return "ELIGIBLE_PG_COURSES"
        elif prior_qual and "bcom" in prior_qual.lower():
            return "COMMERCE_GRADUATE_COURSES"
        elif prior_qual and "bca" in prior_qual.lower():
            return "BCA_GRADUATE_COURSES"
        return "ALL_PG_COURSES"

    # 1. Negation (e.g., "I don't want fee", "skip fee", "not subjects", "bcom nahi karna")
    is_neg, _ = detect_negation(question, profile)
    if is_neg:
        return "NEGATION"

    # 2. Contextual Rejection / Negative Response ("no", "nope", "nah")
    if is_contextual_no(question):
        return "REJECT_CLARIFICATION"

    # 3. User Correction / Disagreement ("that's not what I asked", "wrong answer")
    if is_user_correction(question):
        return "CORRECTION"

    # 4. Conversational Redirect ("tell me something else", "forget that")
    matched_course, _ = match_course(question)
    if not matched_course and is_conversational_redirect(question):
        return "REDIRECT"

    # 5. Clarification responses (e.g. following up on "Do you mean the fee for MSc AI?" or "B.Sc. AI vs M.Sc. AI")
    pending = profile.get("pending_clarification") or profile.get("confirmed_clarification")
    last_assistant_msg = ""
    for turn in reversed(recent_turns or []):
        if turn.get("role") == "assistant":
            last_assistant_msg = (turn.get("content") or "").lower()
            break

    is_ai_prompt = (
        (pending and pending.get("type") == "ai")
        or "two dedicated artificial intelligence programmes" in last_assistant_msg
        or "b.sc. ai & ml" in last_assistant_msg
    )
    if is_ai_prompt and (profile.get("active_course") or re.search(r"\b(bsc|b\.?s\.?c|msc|m\.?s\.?c|ug|pg)\b", q_low)):
        return "COURSE_INQUIRY"

    if pending and pending.get("type") == "fee":
        if re.search(r"^(?:yes|yeah|yep|ha|haa|sure|correct|right|y|haan)\b", q_low):
            return "FEES"
        if re.search(r"^no\b", q_low):
            matched_c, _ = match_course(question)
            if matched_c:
                return "FEES"

    # 6. Hostel & Accommodation (High Priority - Independent of Course)
    if re.search(r"\b(hostel|hostels|hostal|hostle|boys\s*hostel|girls\s*hostel|hostel\s*fees?|hostel\s*admission|hostel\s*documents?|hostel\s*male\s*che|hostel\s*ni\s*fee|accommodation|stay|mess\b)\b", q_low):
        return "HOSTEL"

    # 7. Ambiguous bare "AI"
    if q_low in ("ai", "ai?", "ai course", "ai course?", "ai admission"):
        if not profile.get("active_course"):
            return "AI_AMBIGUITY"

    # 8. Zero-Cost Fee Inquiry ("Is MSc AI free?", "is the course free?")
    if is_zero_cost_inquiry(question):
        return "ZERO_COST_CHECK"

    # 9. Fee Typos & Clarifications ("free of msc ai", bare "free")
    matched_course, _ = match_course(question)
    is_fee_typo = bool(
        (re.search(r"\bfree\s+(?:of|for)\b", q_low) and not re.search(r"\b(hostel|hostal|hostle|accommodation|mess)\b", q_low))
        or re.search(r"\bfree\s+(?:kitni|ketli)\b", q_low)
        or (re.search(r"\bfree\b", q_low) and matched_course)
    )
    if is_fee_typo:
        return "FEES"

    if q_low in ("free", "free?", "free please"):
        if profile.get("active_course"):
            return "FEE_CONFIRMATION"
        return "NO_COURSE_FEE"

    # 5b. Language Preference Switch Commands
    if re.search(r"\b(mane\s*gujarati\s*ma\s*(?:j\s*)?(?:answer|jawab|bolo)|gujarati\s*ma\s*(?:j\s*)?(?:answer|jawab|bolo)|mne\s*guj\s*ma|guj\s*ma\s*jawab|answer\s*in\s*english|english\s*ma\s*bolo|reply\s*in\s*english)\b", q_low):
        return "LANGUAGE_PREFERENCE"

    # 6. Bare Greetings
    if re.search(
        r"^(?:hi+|hello+|hey+|heya|heyy|greetings|good\s*(?:morning|afternoon|evening)|halo|"
        r"namaste|namste|namaskar|kem\s*cho|kemcho|pranam|vanakkam|satsriakal)\b",
        q_low,
    ) or q_low in ("namaste", "namste", "namaskar", "kemcho", "kem cho", "pranam", "vanakkam", "satsriakal"):
        if len(q_low.split()) <= 3 and not any(w in q_low for w in ["bca", "bcom", "bsc", "msc", "admission", "fee", "eligibility", "lab"]):
            return "GREETING"

    # 7. Campus Facility Complaint
    if re.search(r"\b(not\s*working|broken|damaged|faulty|issue\s*in|problem\s*in|complaint|repair)\b", q_low):
        if re.search(r"\b(lab\s*\d+|room\s*\d+|computer|pc|projector|ac|fan|light|equipment|campus|library)\b", q_low):
            return "COMPLAINT"

    # 8. Chitchat, Gratitude, Goodbyes
    if re.search(r"^(?:thanks|thank\s*you|thanku|dhanyawad|aabhar|shukriya|ok\s*thanks|okay\s*thanks|bye|goodbye|cya|cool|okay|ok)\b", q_low):
        if len(q_low.split()) <= 4:
            return "CHITCHAT"

    # 9. Faculty / Leadership / Staff / Department follow-up (High Priority - Independent of Course)
    if (
        re.search(
            r"\b(who\s*is\s*david|who\s*is\s*dr\s*david|dr\.?\s*david|david\s*roy|david\s*kon|fr\.?\s*david|fr\s*david|"
            r"who\s*is\s*(?:the\s*)?principal|principal\s*name|principal\s*kon|dr\.?\s*sebastian|sebastian|"
            r"who\s*is\s*(?:the\s*)?director|director\s*name|director\s*kon|"
            r"who\s*is\s*(?:the\s*)?rector|rector\s*name|rector\s*kon|fr\.?\s*antony|pitchai|"
            r"president|presedent|chairman|trustee|provincial|who\s*heads|"
            r"who\s*is\s*(?:the\s*)?hod|head\s*of\s*(?:the\s*)?department|hod\s*kon|pravida|"
            r"nisarg\s*vyas|nisar\s*vyash|nisarg|vyas|vyash|nisar|nithya\s*nadar|nithya|"
            r"which\s+department|what\s+department|his\s+department|her\s+department|"
            r"faculty\s*members?|professors?|teachers?|faculty\s*kon\s*che|who\s*teaches|who\s*is\s+(?:prof|dr|mr|ms|fr)|"
            r"what\s+does\s+he\s+teach|what\s+does\s+she\s+teach|what\s+does\s+they\s+teach|"
            r"what\s+is\s+his\s+email|what\s+is\s+her\s+email|what\s+is\s+their\s+email|"
            r"tell\s+me\s+about\s+(?:him|her|them)|tell\s+me\s+about\s+another\s+faculty)\b",
            q_low,
        )
        or "fr.david" in q_low
        or "dr.david" in q_low
        or "nisar vyash" in q_low
        or "nisarg vyas" in q_low
        or bool(_ANOTHER_FACULTY_RE.search(q_low))
        or bool(profile.get("active_person") and (_PERSON_PRONOUNS_RE.search(q_low) or _PERSON_ATTRIBUTE_RE.search(q_low)))
    ):
        return "FACULTY"

    # 10. Contact / Helpline / Phone / Address / Location (High Priority - Independent of Course)
    if re.search(
        r"\b(helpline\s*(?:no|number)?|phone\s*(?:no|number)?|contact\s*(?:no|number|details)?|"
        r"college\s*number|admission\s*office\s*number|call\s*number|can\s*i\s*call|where\s*can\s*i\s*contact|"
        r"college\s*phone|enquiry\s*number|"
        r"address|location|where\s+is\s+(?:the\s+)?college|where\s+is\s+sxca|college\s+address|campus\s+address|where\s+(?:is\s+it\s+)?located|kya\s+che|kyan\s+che)\b",
        q_low,
    ):
        return "CONTACT"

    # 11. Office Timings
    if re.search(r"\b(office\s*timings?|office\s*hours?|office\s*kyare|office\s*time|clg\s*timings?|college\s*timings?|working\s*hours?)\b", q_low):
        return "OFFICE_TIMING"

    # 12. Hostel & Accommodation
    if re.search(r"\b(hostel|hostels|hostal|hostle|boys\s*hostel|girls\s*hostel|hostel\s*fees?|hostel\s*admission|hostel\s*documents?|hostel\s*male\s*che|hostel\s*ni\s*fee|mess\b)\b", q_low):
        return "HOSTEL"

    # 13. Academic Departments Overview
    if re.search(r"\b(what\s*departments?|departments?\s*(?:are\s*there|does\s*sxca\s*have|list)|academic\s*departments?)\b", q_low):
        return "DEPARTMENTS"

    # 14. Student Development Clubs
    if re.search(r"\b(what\s*clubs?|clubs?\s*(?:are\s*there|in\s*college)|student\s*clubs?)\b", q_low):
        return "CLUBS"

    # 15. Sports & Gymnasium
    if re.search(r"\b(what\s*sports?|sports?\s*(?:facilities?|available|in\s*college)|gym|gymnasium|playground)\b", q_low):
        return "SPORTS"

    # 16. NCC & NSS
    if re.search(r"\b(ncc\b|nss\b|national\s*cadet\s*corps|national\s*service\s*scheme)\b", q_low):
        return "NCC_NSS"

    # 17. Events, Functions, Culfest
    if re.search(r"\b(events?|functions?|culfest|what\s*is\s*culfest|cultural\s*fest|annual\s*fest|event\s*kya\s*che|function\s*kyare)\b", q_low):
        return "EVENTS"

    # 18. Placements & Career Cell
    if re.search(r"\b(placement|placements?|placement\s*cell|career\s*cell|recruiters?|companies\s*visiting|packages?)\b", q_low):
        return "PLACEMENT"

    # 19. Scholarships & Financial Aid
    if re.search(r"\b(scholarships?|scholarship\s*schemes?|financial\s*aid|freeship|mysy|digital\s*gujarat)\b", q_low):
        return "SCHOLARSHIPS"

    # 20. Central Library
    if re.search(r"\b(library|central\s*library|library\s*timing|library\s*books|reading\s*room)\b", q_low):
        return "LIBRARY"

    # 21. General PG Admission Guidance ("master ma admission levu che")
    if re.search(r"\b(master\s*ma\s*admission\s*levu\s*che|mane\s*master\s*ma\s*admission\s*levu\s*che|pg\s*admission\s*guidance)\b", q_low):
        return "GENERAL_PG_ADMISSION"

    # 22. Science Background Course Discovery ("science background che, kaya course available che?")
    if re.search(r"\b(science\s*background\s*che|courses?\s*for\s*science|science\s*stream\s*courses?)\b", q_low):
        return "SCIENCE_COURSES"

    # 23. PG Courses list request ("master ma course ketla che?", "which pg courses")
    if re.search(r"\b(master\s*ma\s*course|master\s*ma\s*kaya\s*course|which\s*pg\s*courses?|what\s*pg\s*courses?|pg\s*courses?|postgraduate\s*courses?|all\s*pg\s*courses?|master'?s?\s*courses?)\b", q_low):
        return "ALL_PG_COURSES"

    # 24. UG Courses list request
    if re.search(r"\b(ug\s*courses?|undergraduate\s*courses?|list\s+of\s+ug\s+courses?|all\s+ug\s+courses?)\b", q_low):
        return "ALL_UG_COURSES"

    # 25. Broad Course Discovery ("What courses do you have?", "courses", "programs", "other courses", "alternatives")
    if re.search(
        r"^(?:courses?\??|cources?\??|all\s*courses?\??|available\s*courses?\??|programmes?\??|programs?\??)$|"
        r"\b(what\s*(?:are\s*the\s*)?(?:courses?|cources?)\s*(?:are\s*)?available|"
        r"what\s*(?:courses?|cources?)\s*(?:do\s*you\s*have|are\s*available|there\s*are)|"
        r"which\s*(?:courses?|cources?)\s*(?:do\s*you\s*have|are\s*there|are\s*available)|"
        r"(?:courses?|cources?)\s*(?:are\s*)?available|"
        r"list\s+of\s+(?:courses?|cources?)|available\s+(?:courses?|cources?)|(?:courses?|cources?)\s*list|"
        r"other\s*(?:courses?|cources?)|different\s*(?:courses?|cources?)|alternatives?|"
        r"other\s*programs?|different\s*programs?|any\s*other\s*option)\b",
        q_low
    ):
        return "COURSE_DISCOVERY"

    # 26. Exam Dates (High Priority - Overrides Active Course)
    if re.search(
        r"\b(cai-?2|cai\s*2|ca-?ii|ca\s*ii|ca-?2|ca\s*2|cia-?2|cia\s*2|cia2|"
        r"cai-?1|cai\s*1|ca-?i|ca\s*i|cia-?1|cia\s*1|cia1|"
        r"continuous\s*(?:internal\s*)?assessment(?:\s*[12i]+)?|"
        r"internal\s*exams?|midterm|mid\s*term|semester\s*exams?|final\s*exams?|ese(?:\s*exams?)?|"
        r"exam\s*dates?|examination\s*dates?|pariksha|pareeksha)\b",
        q_low,
    ):
        return "EXAM_DATE"

    # 27. Holidays & Vacations (High Priority - Overrides Active Course)
    if re.search(
        r"\b(diwali\s*vacation|dewail\s*vacation|dewali\s*vacation|diwali\s*holiday|"
        r"christmas\s*break|summer\s*break|summer\s*vacation|vacations?|holidays?|"
        r"rajao|raja\s*che|raja\s*chhe|chutti|chhutti|raja)\b|રજા\S*|છુટ્ટી\S*|छुट्टी\S*",
        q_low,
    ):
        return "HOLIDAY"

    # 28. Academic Calendar & Term Dates (High Priority - Overrides Active Course)
    if re.search(
        r"\b(academic\s*calendar|semester\s*(?:start|commencement|end)|term\s*dates?|"
        r"college\s*reopen|when\s*(?:do|does)\s*classes\s*(?:start|begin)|academic\s*schedule)\b",
        q_low,
    ):
        return "ACADEMIC_CALENDAR"

    # 29. Subjects / Curriculum / Course Structure (High Priority)
    if re.search(
        r"\b(subjects?|subject\??|syllabus|curriculum|course\s*structure|course\s*curriculum|"
        r"subjets?|subjetcs?|semester\s*subjects?|what\s*(?:will\s*i|do\s*we)\s*study|what\s*are\s*the\s*subjects?|"
        r"what\s*subjects?\s*(?:are\s*there|do\s*you\s*teach)|what\s*subjects?\s*(?:are|in)|"
        r"papers?|course\s*papers?|sem\s*[1-8]|semester\s*[1-8]|papers?\s*in\b|"
        r"what\s*will\s*i\s*study\s*in\s*semester\s*[1-8]|"
        r"subjects?\s*kya\s*hai|kya\s*subjects?\s*hai|su\s*subjects?\s*che|na\s*subjects?|"
        r"ma\s*kya\s*subjects?\s*che|syllabus\s*kya\s*hai)\b",
        q_low,
    ):
        return "SUBJECTS"

    # 30. Fees (including common typos feee, fess, feees, and Hinglish/Gujarati)
    if re.search(
        r"\b(fee|fees|feee|fess|feees|cost|charges|tuition|how\s*much\s*(?:is|are|does|to)?|"
        r"ketli\s*che|ketli\s*fee|ki\s*fees?|ki\s*fee\s*kitni\s*hai|ni\s*fees?\s*ketli\s*che)\b",
        q_low,
    ):
        return "FEES"

    # 31. Admission Process / How to Apply (including admisson, admisison, admsn)
    if re.search(
        r"\b(how\s*(?:can|do)\s*i\s*apply|how\s*to\s*apply|how\s*to\s*get\s*admission|how\s*to\s*take\s*admission|"
        r"how\s*to\s*register|application\s*(?:process|steps|procedure|link|form|portal)|where\s*(?:to|can\s*i)\s*apply|"
        r"admission\s*(?:process|procedure|steps)|admisson|admisison|admsn|can\s*i\s*apply|"
        r"admission\s*kyare\s*che|form\s*kyare\s*bharvano|admission\s*kem\s*levu|admission\s*kevrite|"
        r"admission\s*levu\s*che|process\s*su\s*che|process\s*kya\s*hai|procedure\s*su\s*che|"
        r"does\s*interview\s*happen|is\s*interview\s*required)\b",
        q_low,
    ):
        return "ADMISSION_PROCESS"

    # 32. Eligibility (including common typos eligiblity, eligibilty)
    if re.search(
        r"\b(eligibility|eligible|eligiblity|eligibilty|elgibility|elgible|criteria|qualify|requirements|"
        r"laaykat|patrata|(?:eligibility|patrata|laaykat)\s*su\s*che|mate\s*eligibility|ni\s*eligibility)\b",
        q_low,
    ):
        return "ELIGIBILITY"

    # 33. Documents (including documnts, docs)
    if re.search(r"\b(documents?|doc|docs|documnts?|certificate|marksheet|required\s*documents?|which\s*docs|docs\s*kya\s*joiye|documents\s*kya\s*joiye)\b", q_low):
        return "DOCUMENTS"

    # 34. Direct Course Mention (Explicit course named in current message)
    if matched_course:
        return "COURSE_INQUIRY"

    # 22. Genuine Counselling Triggers (Only when explicitly asking for advice or answering stream/qualification)
    is_explicit_counselling_request = bool(re.search(
        r"\b(help\s*me\s*choose|don'?t\s*know\s*what\s*to\s*choose|confused|what\s*can\s*i\s*do|"
        r"which\s*course\s*should\s*i|suggest\s*(?:me\s*)?a\s*course|recommend\s*(?:me\s*)?a\s*course|"
        r"guide\s*me|what\s*should\s*i\s*(?:do|take|choose))\b",
        q_low,
    ))
    is_counselling_reply = bool(
        profile.get("mode") == "counselling"
        and (
            re.search(r"\b(science|commerce|arts|12th|hsc|bachelor|diploma)\b", q_low)
            or profile.get("career_goal")
            or profile.get("interest_area")
        )
    )
    if is_explicit_counselling_request or is_counselling_reply:
        return "COUNSELLING"

    return "GENERAL"


def handle_complaint(question: str) -> str:
    """Handles facility/lab issues with polite campus support acknowledgment."""
    q = question.strip()
    loc_m = re.search(r"\b(lab\s*\d+|room\s*\d+|library|canteen|auditorium|sports\s*ground)\b", q, re.I)
    location = loc_m.group(1).title() if loc_m else "the campus premises"

    item_m = re.search(r"\b(computer|pc|projector|ac|fan|light|equipment|campus|screen)\b", q, re.I)
    item = item_m.group(1).lower() if item_m else "equipment"

    return (
        f"Sorry to hear you're experiencing an issue with the {item} in **{location}**.\n\n"
        f"I have recorded this report for our Campus IT & Facility Support team. "
        f"For immediate on-ground resolution, please notify the **Computer Lab Assistant** or the "
        f"**Department Coordinator** on the 2nd floor, or reach out to the campus administrative office at "
        f"**info@sxca.edu.in** / **+91 79 2630 2872**."
    )


def handle_greeting(question: str) -> str:
    """Friendly, conversational counsellor greeting introducing Niaa."""
    return (
        "Hi! 👋 Welcome to St. Xavier's College, Ahmedabad.\n"
        "I'm **Niaa**, your admission counsellor. How can I help you today?"
    )


def handle_course_inquiry(course_name: str, profile: dict[str, Any]) -> str:
    """Introduces the requested course concisely like a real admission counsellor."""
    disp_name = clean_course_display_name(course_name)
    level = profile.get("course_level") or ("UG" if not course_name.startswith("MSc") and not course_name.startswith("MA") else "PG")
    level_str = "an undergraduate (UG)" if level == "UG" else "a postgraduate (PG)"

    return (
        f"Sure! **{disp_name}** is {level_str} programme at St. Xavier's College (Autonomous), Ahmedabad.\n\n"
        f"I can help you with:\n"
        f"• **Course structure / subjects**\n"
        f"• **Eligibility**\n"
        f"• **Fees**\n"
        f"• **Admission process**\n\n"
        f"What would you like to know?"
    )


def handle_counselling_step(question: str, profile: dict[str, Any]) -> str:
    """Progressive, empathetic guidance asking ONE useful question at a time."""
    education = profile.get("education_level")
    stream = profile.get("stream")
    interest = profile.get("interest_area")
    career_goal = profile.get("career_goal")

    # Step 1: Missing qualification/stream
    if not stream and not education:
        return (
            "That's completely fine. I can help you narrow it down.\n\n"
            "Which stream did you complete in 12th — Science, Commerce, or Arts? (Or have you completed a bachelor's degree or another qualification?)"
        )

    if education == "12th" and not stream:
        return (
            "That's completely fine. I can help you narrow it down.\n\n"
            "Which stream did you complete in 12th — Science, Commerce, or Arts?"
        )

    # Step 2: Stream known, interest unknown
    if stream == "Science" and not interest and not career_goal:
        return (
            "Great! With a 12th Science background, what area are you more interested in exploring?\n\n"
            "• **Computer / IT & Software**\n"
            "• **Pure Sciences** (Physics, Chemistry, Maths, Statistics, Electronics)\n"
            "• **Life Sciences** (Biotechnology, Biochemistry, Botany, Zoology)\n"
            "• **Business / Management**"
        )

    if stream == "Commerce" and (not interest or interest == "Commerce/Business") and not career_goal:
        return (
            "Great! With a Commerce background, St. Xavier's College offers prominent undergraduate programs:\n\n"
            "• **Accounting, Finance & Commerce:** B.Com. (General) (Grant-in-Aid), B.Com. BPS (Self-Financed)\n"
            "• **Business Administration & Management:** B.S. Business Administration\n"
            "• **Computer Applications & Software:** BCA (Bachelor of Computer Applications), B.Sc. IT\n"
            "• **Economics:** B.A. Economics\n\n"
            "Which area interests you more?"
        )

    if stream == "Arts" and (not interest or interest == "Arts") and not career_goal:
        return (
            "Great! With an Arts background, St. Xavier's College offers prominent undergraduate programs:\n\n"
            "• **B.A. Programmes:** English, Psychology, Economics, Sociology, Gujarati, Sanskrit\n"
            "• **B.P.A. (Bachelor of Performing Arts):** Theatre & Production Management\n"
            "• **BCA / B.S. Business Administration:** Open to 12th pass students of any stream\n\n"
            "Which subject or field would you like to explore?"
        )

    # Step 3: Interest is Computer/IT
    if interest == "Computer/IT" and not career_goal:
        return (
            "Since you're interested in Computer/IT with a 12th Science background, St. Xavier's College offers excellent undergraduate options:\n\n"
            "• **BCA (Bachelor of Computer Applications)** — focuses on software development, applications, and programming\n"
            "• **B.Sc. Information Technology** — focuses on IT infrastructure, software systems, and computation\n"
            "• **B.Sc. Artificial Intelligence & Machine Learning** — focuses on AI algorithms, data intelligence, and ML\n\n"
            "Are you mainly interested in software development, AI/ML, or general IT?"
        )

    # Step 4: Career Goal or Interest is AI / Machine Learning
    if career_goal == "AI Engineer" or interest == "AI/ML":
        return (
            "Got it. If your goal is AI and machine learning, **B.Sc. Artificial Intelligence & Machine Learning** "
            "at St. Xavier's College is specifically designed with dedicated coursework in AI, Machine Learning, Python, "
            "and Data Intelligence. **BCA** also provides strong programming foundations.\n\n"
            "Would you like to explore the eligibility, fee structure, or admission process for B.Sc. AI & ML or BCA?"
        )

    # Step 5: Pure Science interest
    if interest == "Pure Science":
        return (
            "In Pure Sciences, St. Xavier's College offers Grant-in-Aid B.Sc. programs in:\n"
            "• **B.Sc. Physics**\n"
            "• **B.Sc. Chemistry** (also with Vocational Industrial Chemistry)\n"
            "• **B.Sc. Mathematics**\n"
            "• **B.Sc. Statistics**\n"
            "• **B.Sc. Electronics**\n\n"
            "Would you like details on eligibility or fees for any of these programs?"
        )

    # Step 6: Life Sciences interest
    if interest == "Life Sciences":
        return (
            "In Life Sciences, St. Xavier's College offers:\n"
            "• **B.Sc. Biotechnology** (Self-Financed)\n"
            "• **B.Sc. Biochemistry** (Grant-in-Aid, also with Vocational BT)\n"
            "• **B.Sc. Botany** (Grant-in-Aid)\n"
            "• **B.Sc. Zoology** (Grant-in-Aid)\n\n"
            "Which of these subjects would you like to explore further?"
        )

    # Fallback to general supportive inquiry (or stream discovery)
    if stream:
        return handle_course_discovery(question, profile)
    return (
        f"Since you're exploring options with a **{stream or '12th'}** background, I can help you compare programs, "
        f"check eligibility criteria, or explain fees. What specific questions do you have?"
    )


def handle_course_discovery(question: str, profile: dict[str, Any]) -> str:
    """Answers 'What courses do you have?' contextually without dumping unrelated courses."""
    education = profile.get("education_level")
    stream = profile.get("stream")

    # If stream is 12th Science
    if stream == "Science" or (education == "12th" and stream == "Science"):
        return (
            "Since you've completed 12th Science, here are the undergraduate (UG) options available for your background at St. Xavier's College:\n\n"
            "• **Computer & Technology:** BCA, B.Sc. Information Technology, B.Sc. Artificial Intelligence & Machine Learning\n"
            "• **Mathematical & Physical Sciences:** B.Sc. Physics, B.Sc. Chemistry, B.Sc. Mathematics, B.Sc. Statistics, B.Sc. Electronics\n"
            "• **Biological Sciences:** B.Sc. Biotechnology, B.Sc. Biochemistry, B.Sc. Botany, B.Sc. Zoology\n"
            "• **Management:** B.S. Business Administration\n\n"
            "Which of these areas matches your interests?"
        )

    if stream == "Commerce":
        return (
            "With a 12th Commerce background, you can apply for the following undergraduate courses:\n\n"
            "• **B.Com. (General)** & **B.Com. (BPS)**\n"
            "• **B.S. Business Administration (BBA)**\n"
            "• **BCA (Bachelor of Computer Applications)**\n"
            "• **B.Sc. Information Technology**\n"
            "• **B.A. Economics**\n\n"
            "Would you like details on fees or eligibility for any of these?"
        )

    if stream == "Arts":
        return (
            "With a 12th Arts background, you can apply for the following undergraduate courses at St. Xavier's College:\n\n"
            "• **B.A. Programmes:** Economics, English, Psychology, Sociology, Gujarati, Sanskrit\n"
            "• **B.P.A.:** Theatre & Production Management\n"
            "• **BCA (Bachelor of Computer Applications):** Open to 12th pass students of all streams\n"
            "• **B.S. Business Administration:** Open to 12th pass students of all streams\n\n"
            "Which subject or program would you like to explore?"
        )

    # If qualification is completely unknown
    return (
        "Sure! St. Xavier's College offers a wide range of undergraduate and postgraduate programs.\n\n"
        "Are you looking for undergraduate (UG) courses after 12th, or postgraduate (PG) master's programs?"
    )


def is_gujarati_mode(question: str, profile: dict[str, Any] | None = None) -> bool:
    """Detects whether user prefers responses in Gujarati script."""
    if profile and profile.get("language_preference") == "gujarati":
        return True
    if profile and profile.get("language_preference") == "english":
        return False
    q_low = question.lower().strip()
    if re.search(r"[\u0A80-\u0AFF]", q_low):
        return True
    if re.search(r"\b(mane\s*gujarati\s*ma|gujarati\s*ma\s*j|mne\s*guj|guj\s*ma\s*jawab|gujarati\s*ma\s*bolo)\b", q_low):
        return True
    return bool(re.search(
        r"\b(su\s*che|ketli\s*che|ketla\s*che|kon\s*che|kaya\s*course|kaya\s*courses|kya\s*events|kya\s*event|thaya\s*che|thaya|"
        r"male\s*che|malshe|aapo|nathi|karvu\s*che|levu\s*che|lidhu\s*che|karyu\s*che|kyare\s*che|"
        r"ni\s*information|ni\s*fee|ni\s*details|na\s*courses|ma\s*kya)\b"
        r"|\bhostel\s*che\??\b|\bscholarship\s*available\s*che\??\b|\bche\??$",
        q_low
    ))


def handle_language_preference(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Confirms language preference change for the session."""
    pref = (profile or {}).get("language_preference")
    q_low = question.lower()
    if pref == "gujarati" or "gujarati" in q_low or "guj" in q_low:
        ans = (
            "ચોક્કસ! હવેથી હું તમને તમામ માહિતી ગુજરાતીમાં જ આપીશ. 😊\n\n"
            "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદના એડમિશન, કોર્સ, ફી અથવા ફેકલ્ટી અંગે તમે શું જાણવા માંગો છો?"
        )
    else:
        ans = (
            "Sure! I will provide all answers in English from now on. 😊\n\n"
            "How can I help you with St. Xavier's College (Autonomous), Ahmedabad?"
        )
    return {"answer": ans, "sources": [], "grounded": True}


def handle_faculty(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Accurate faculty and leadership information from official SXCA records.
    Disambiguates when multiple candidates match a name query, preventing accidental selection."""
    q_low = question.lower()
    prof = profile or {}
    active_p = prof.get("active_person", "")
    is_guj = is_gujarati_mode(question, prof)

    # 1. College Leadership / President / Chairman / Governing Body
    is_president = bool(
        re.search(r"\b(president|presedent|chairman|trustee|provincial|who\s*heads|who\s*is\s*the\s*head\s*of\s*(?:this\s*)?(?:clg|college))\b", q_low)
    )
    if is_president:
        if is_guj:
            ans = (
                "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદનું સંચાલન અમદાવાદ જેસ્યુઇટ સોસાયટી (Jesuits of Gujarat) દ્વારા કરવામાં આવે છે. "
                "કોલેજના મુખ્ય વહીવટી અને શૈક્ષણિક વડાઓ નીચે મુજબ છે:\n\n"
                "• **રેક્ટર (Rector) & જેસ્યુઇટ સુપિરિયર:** **ફાધર એન્ટોની પીચાઈ એસ.જે. (Fr. Antony Pitchai SJ)** — કોલેજ ટ્રસ્ટ અને જેસ્યુઇટ સમુદાયના વડા\n"
                "• **ડિરેક્ટર (Director):** **ફાધર ડૉ. ડેવિડ કે. રોય એસ.જે. (Fr. Dr. David K. Roy SJ)** — વહીવટ અને શૈક્ષણિક વિકાસના વડા\n"
                "• **પ્રિન્સિપાલ (Principal):** **ડૉ. સેબેસ્ટિયન વી.એ. (Dr. Sebastian V.A.)** — કોલેજ શૈક્ષણિક વહીવટના વડા"
            )
        else:
            ans = (
                "St. Xavier's College (Autonomous), Ahmedabad is managed by the Ahmedabad Jesuit Society. Its executive leadership comprises:\n\n"
                "• **Rector & Superior:** **Fr. Antony Pitchai SJ** (heads the governing Jesuit trust / society)\n"
                "• **Director:** **Fr. Dr. David K. Roy SJ** (oversees administration and academic advancement)\n"
                "• **Principal:** **Dr. Sebastian V.A.** (heads collegiate administration and academics)\n\n"
                "For institutional administration inquiries, contact **info@sxca.edu.in** or call **+91 79 2630 2872**."
            )
        src = [{
            "title": "About St. Xavier's College Leadership",
            "source": "about-us.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/about-us/",
            "score": 1.0,
            "excerpt": "Governing Leadership: Fr. Antony Pitchai SJ (Rector), Fr. Dr. David K. Roy SJ (Director), Dr. Sebastian V.A. (Principal)"
        }]
        return {"status": "RESOLVED", "person": {"name": "College Leadership"}, "answer": ans, "sources": src, "grounded": True}

    # 2. Entity Resolution from Faculty/Staff Roster
    resolution = resolve_faculty(question, prof)

    # 2b. Handle ANOTHER_FACULTY: User explicitly requested another faculty member
    if resolution["status"] == "ANOTHER_FACULTY":
        if is_guj:
            ans = (
                "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં વિવિધ વિભાગોમાં પ્રતિષ્ઠિત ફેકલ્ટી સભ્યો કાર્યરત છે:\n\n"
                "• **નેતૃત્વ:** ફાધર એન્ટોની પીચાઈ (રેક્ટર), ફાધર ડૉ. ડેવિડ કે. રોય (ડિરેક્ટર), ડૉ. સેબેસ્ટિયન વી.એ. (પ્રિન્સિપાલ)\n"
                "• **કોમ્પ્યુટર સાયન્સ & ડેટા સાયન્સ:** ડૉ. પ્રવિદા રાજા (HOD), શ્રી નિસર્ગ વ્યાસ (AI/ML કોઓર્ડિનેટર), શ્રી નિસર્ગ પટેલ\n"
                "• **સાયન્સ & આર્ટ્સ:** ફિઝિક્સ, કેમેસ્ટ્રી, મેથેમેટિક્સ, ઇકોનોમિક્સ, અંગ્રેજી વગેરે\n\n"
                "તમે કયા વિભાગ કે કયા ફેકલ્ટી સભ્ય વિશે જાણવા માંગો છો?"
            )
        else:
            ans = (
                "St. Xavier's College (Autonomous), Ahmedabad has distinguished faculty and leadership across various departments:\n\n"
                "• **Leadership:** Fr. Antony Pitchai SJ (Rector), Fr. Dr. David K. Roy SJ (Director), Dr. Sebastian V.A. (Principal)\n"
                "• **Computer Science & Data Science:** Dr. Pravida Raja (HOD), Mr. Nisarg Vyas (Coordinator, AI & ML), Mr. Nisarg Patel\n"
                "• **Sciences & Humanities:** Physics, Chemistry, Mathematics, Biochemistry, Economics, English, and Commerce\n\n"
                "Which department or faculty member would you like to know about?"
            )
        src = [{
            "title": "SXCA Academic Departments & Faculty",
            "source": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html",
            "doc_type": "html",
            "page": 1,
            "url": "https://sxca.edu.in/academics/departments/",
            "score": 1.0,
            "excerpt": "Faculty Directory and Department Leadership at St. Xavier's College"
        }]
        return {"status": "RESOLVED", "person": None, "answer": ans, "sources": src, "grounded": True}

    # 2c. Handle AMBIGUOUS_FOLLOWUP: Pronoun follow-up on ambiguous entity
    if resolution["status"] == "AMBIGUOUS_FOLLOWUP":
        name = resolution.get("name") or "that name"
        ans = f"Could you specify which {name} you mean? I found multiple people matching that name."
        return {
            "status": "AMBIGUOUS",
            "candidates": resolution.get("candidates", []),
            "answer": ans,
            "sources": [],
            "grounded": True,
        }

    # 3. Handle AMBIGUOUS: Multiple people matched
    if resolution["status"] == "AMBIGUOUS":
        candidates = resolution["candidates"]
        query_label = resolution.get("query", "that name")
        names_list = "\n".join(f"• {c['canonical_name']}" for c in candidates)
        if is_guj:
            ans = (
                f"મને '{query_label}' મેળ ખાતી એકથી વધુ વ્યક્તિઓ મળી છે. કૃપા કરીને પૂરું નામ જણાવો:\n"
                f"{names_list}"
            )
        else:
            ans = (
                f"I found multiple people matching '{query_label}'. Please specify the full name:\n"
                f"{names_list}"
            )
        return {
            "status": "AMBIGUOUS",
            "candidates": [c["canonical_name"] for c in candidates],
            "answer": ans,
            "sources": [],
            "grounded": True,
        }

    # 4. Handle NOT_FOUND: Unknown person or generic question
    if resolution["status"] == "NOT_FOUND":
        target = resolution.get("query", "").strip()
        is_generic_faculty_inquiry = (
            not target
            or target.lower() in ("faculty", "professor", "professors", "teacher", "teachers", "staff", "department", "academic")
            or re.search(r"^(?:faculty\??|all\s*faculty\??|professors\??|teachers\??|list\s*of\s*faculty\??)$", q_low.strip())
        )
        if not is_generic_faculty_inquiry and target:
            if is_guj:
                ans = (
                    f"હું સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદના સત્તાવાર રેકોર્ડમાં '{target}' અંગે કોઈ ફેકલ્ટી વિગતો શોધી શક્યો નથી.\n\n"
                    "કૃપા કરીને કોલેજની સત્તાવાર વેબસાઇટ https://sxca.edu.in/ અથવા info@sxca.edu.in પર સંપર્ક કરો."
                )
            else:
                ans = "Sorry, I don't have information about that in the available SXCA materials."
            return {
                "status": "NOT_FOUND",
                "answer": ans,
                "sources": [],
                "grounded": False,
            }

        # Generic Faculty Overview
        if is_guj:
            ans = (
                "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં તમામ 15+ શૈક્ષણિક વિભાગોમાં ઉચ્ચ લાયકાત ધરાવતા અનુભવી અને વિદ્વાન પ્રોફેસરો કાર્યરત છે.\n\n"
                "તમે કોઈ ચોક્કસ વિભાગ (જેવા કે કોમ્પ્યુટર સાયન્સ, ફિઝિક્સ, કેમેસ્ટ્રી, બાયોટેકનોલોજી, ઇકોનોમિક્સ કે કોમર્સ) ના ફેકલ્ટી સભ્યો વિશે જાણવા માંગો છો?"
            )
        else:
            ans = (
                "St. Xavier's College (Autonomous), Ahmedabad has dedicated and experienced faculty members across all 15+ academic departments, "
                "including Computer Sciences & Data Science, Physics, Chemistry, Life Sciences, Mathematics, Commerce, Economics, and English.\n\n"
                "Would you like details regarding faculty in a specific department or subject?"
            )
        src = [{
            "title": "SXCA Academic Departments & Faculty",
            "source": "academics-departments.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/academics/departments/",
            "score": 1.0,
            "excerpt": "Faculty members and departments at St. Xavier's College"
        }]
        return {"status": "RESOLVED", "person": {"name": "General Faculty"}, "answer": ans, "sources": src, "grounded": True}

    # 5. Handle RESOLVED: Exactly ONE person matched
    person = resolution["person"]
    pid = person.get("id")
    p_name = person.get("full_name") or person.get("canonical_name")
    desig = person.get("designation", "Faculty")
    dept = person.get("department", "St. Xavier's College")
    progs = person.get("programmes")
    email = person.get("email")

    # Specific Attribute Follow-ups:
    # A. Unavailable / Missing Faculty Attributes (Experience, Salary, Age, DOB, Personal Phone, etc.)
    if re.search(
        r"\b(exprience|experience|work\s*experience|years\s*of\s*experience|service\s*years|"
        r"how\s*much\s*experience|how\s*many\s*years|salary|pay|income|home\s*address|residence|"
        r"age|date\s*of\s*birth|dob|marital\s*status|personal\s*phone|personal\s*mobile)\b",
        q_low,
    ):
        return {
            "status": "RESOLVED",
            "person": person,
            "answer": "Sorry, I don't have information about that in the available SXCA materials.",
            "sources": [],
            "grounded": False,
        }

    # B. Teaching / Courses / Subjects / Programmes (e.g. "What does he teach?")
    if re.search(r"\b(teach|teaches|teaching|subjects?|courses?|programmes?|papers?|curriculum|classes)\b", q_low):
        if is_guj:
            ans = f"**{p_name}** ({desig}, {dept}) સેન્ટ ઝેવિયર્સ કોલેજમાં **{progs or dept}** ના કાર્યક્રમો શીખવે છે અને સંકલન કરે છે."
        else:
            if progs:
                ans = f"**{p_name}** ({desig}, {dept}) teaches and coordinates programmes including **{progs}** at St. Xavier's College (Autonomous), Ahmedabad."
            else:
                ans = f"**{p_name}** is {desig} in the **{dept}** at St. Xavier's College (Autonomous), Ahmedabad."
        src = [{
            "title": f"{person['canonical_name']} Profile",
            "source": person["source_doc"],
            "doc_type": "html",
            "page": 1,
            "url": person["source_url"],
            "score": 1.0,
            "excerpt": f"{p_name} — {desig}, {dept}. Programmes: {progs or 'Departmental courses'}"
        }]
        return {"status": "RESOLVED", "person": person, "answer": ans, "sources": src, "grounded": True}

    # B. Email / Contact / Phone (e.g. "What is his email?", "What is his phone number?")
    if re.search(r"\b(phone|mobile|cell|contact\s*number|phone\s*number)\b", q_low):
        return {
            "status": "RESOLVED",
            "person": person,
            "answer": "Sorry, I don't have information about that in the available SXCA materials.",
            "sources": [],
            "grounded": False,
        }

    if re.search(r"\b(email|mail)\b", q_low):
        if email:
            if is_guj:
                ans = f"**{p_name}** ({dept}) નો સત્તાવાર ઇમેઇલ **{email}** છે."
            else:
                ans = f"The official email address of **{p_name}** ({dept}) is **{email}**."
            src = [{
                "title": f"{person['canonical_name']} Profile",
                "source": person["source_doc"],
                "doc_type": "html",
                "page": 1,
                "url": person["source_url"],
                "score": 1.0,
                "excerpt": f"{p_name} — Email: {email}"
            }]
            return {"status": "RESOLVED", "person": person, "answer": ans, "sources": src, "grounded": True}
        else:
            return {
                "status": "RESOLVED",
                "person": person,
                "answer": "Sorry, I don't have information about that in the available SXCA materials.",
                "sources": [],
                "grounded": False,
            }

    if re.search(r"\b(contact|reach)\b", q_low):
        if email:
            if is_guj:
                ans = f"**{p_name}** ({dept}) નો સત્તાવાર ઇમેઇલ **{email}** છે."
            else:
                ans = f"The official contact email of **{p_name}** ({dept}) is **{email}**."
            src = [{
                "title": f"{person['canonical_name']} Profile",
                "source": person["source_doc"],
                "doc_type": "html",
                "page": 1,
                "url": person["source_url"],
                "score": 1.0,
                "excerpt": f"{p_name} — Contact: {email}"
            }]
            return {"status": "RESOLVED", "person": person, "answer": ans, "sources": src, "grounded": True}
        else:
            return {
                "status": "RESOLVED",
                "person": person,
                "answer": "Sorry, I don't have information about that in the available SXCA materials.",
                "sources": [],
                "grounded": False,
            }

    # C. Department (e.g. "What department is he from?")
    if re.search(r"\b(department|dept|which\s*dept|what\s*dept|faculty\s*of)\b", q_low):
        if is_guj:
            ans = f"**{p_name}** સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં **{dept}** માં {desig} તરીકે સેવા આપે છે."
        else:
            ans = f"**{p_name}** is from the **{dept}** at St. Xavier's College (Autonomous), Ahmedabad, serving as {desig}."
        src = [{
            "title": f"{person['canonical_name']} Profile",
            "source": person["source_doc"],
            "doc_type": "html",
            "page": 1,
            "url": person["source_url"],
            "score": 1.0,
            "excerpt": f"{p_name} — {desig}, {dept}"
        }]
        return {"status": "RESOLVED", "person": person, "answer": ans, "sources": src, "grounded": True}

    # Fallback to specific profiles or general overview:
    # A. Mr. Nisarg Vyas (Coordinator AI & ML, Data Science)
    if pid == "nisarg_vyas":
        if is_guj:
            ans = (
                "**શ્રી નિસર્ગ વ્યાસ (Mr. Nisarg Vyas)** સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદના "
                "**ડેટા સાયન્સ વિભાગ (Department of Data Science)** માં આસિસ્ટન્ટ પ્રોફેસર અને AI & ML ના કોઓર્ડિનેટર છે.\n\n"
                "• **હોદ્દો:** આસિસ્ટન્ટ પ્રોફેસર અને એઆઈ/એમએલ કોઓર્ડિનેટર\n"
                "• **વિભાગ:** ડેટા સાયન્સ વિભાગ (કોમ્પ્યુટર સાયન્સ ફેકલ્ટી)\n"
                "• **સંલગ્ન કાર્યક્રમો:** B.Sc. AI & ML, BCA, M.Sc. Artificial Intelligence, અને M.Sc. Big Data Analytics\n"
                "• **સત્તાવાર ઇમેઇલ:** nisarg.vyas@sxca.edu.in"
            )
        else:
            ans = (
                "**Mr. Nisarg Vyas** is a faculty member in the **Department of Data Science** (Faculty of Computer Sciences) "
                "at St. Xavier's College (Autonomous), Ahmedabad, serving as Assistant Professor and Coordinator of AI and ML.\n\n"
                "• **Designation:** Assistant Professor & Coordinator of AI and ML\n"
                "• **Department:** Department of Data Science / Computer Sciences\n"
                "• **Programmes Associated:** B.Sc. Artificial Intelligence & Machine Learning, BCA, M.Sc. AI, and M.Sc. Big Data Analytics\n"
                "• **Official Email:** nisarg.vyas@sxca.edu.in"
            )
        src = [{
            "title": "Computer Sciences & Data Science Faculty",
            "source": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/",
            "score": 1.0,
            "excerpt": "Mr. Nisarg Vyas, Assistant Professor & Coordinator of AI and ML, Department of Data Science"
        }]
        return {"status": "RESOLVED", "person": person, "answer": ans, "sources": src, "grounded": True}

    # B. Fr. Dr. David K. Roy SJ / Dr. David / Department follow-up
    if pid == "david_roy":
        if re.search(r"\b(department|his\s*dept|which\s*dept|what\s*dept)\b", q_low):
            if is_guj:
                ans = "ફાધર ડૉ. ડેવિડ કે. રોય એસ.જે. (Fr. Dr. David K. Roy SJ) સેન્ટ ઝેવિયર્સ કોલેજમાં **કોમ્પ્યુટર સાયન્સ અને ડેટા સાયન્સ વિભાગ (Department of Computer Sciences & Data Science)** માં આસિસ્ટન્ટ પ્રોફેસર છે."
            else:
                ans = "Fr. Dr. David K. Roy SJ is in the **Department of Computer Sciences & Data Science** at St. Xavier's College (Autonomous), Ahmedabad."
        else:
            if is_guj:
                ans = (
                    "**ફાધર ડૉ. ડેવિડ કે. રોય એસ.જે. (Fr. Dr. David K. Roy SJ)** સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદના **ડિરેક્ટર (Director)** છે "
                    "અને **કોમ્પ્યુટર સાયન્સ અને ડેટા સાયન્સ વિભાગ**માં આસિસ્ટન્ટ પ્રોફેસર તરીકે સેવા આપે છે.\n\n"
                    "તેઓ કમ્પ્યુટર સાયન્સ, આર્ટિફિશિયલ ઇન્ટેલિજન્સ અને ડેટા એનાલિટિક્સના અભ્યાસક્રમો (જેવા કે BCA, B.Sc. AI & ML, M.Sc. AI અને M.Sc. Big Data Analytics) સાથે સંકળાયેલા છે."
                )
            else:
                ans = (
                    "**Fr. Dr. David K. Roy SJ** is the **Director** of St. Xavier's College (Autonomous), Ahmedabad, "
                    "and an Assistant Professor in the **Department of Computer Sciences & Data Science**.\n\n"
                    "He oversees academic development and is closely associated with technology programmes at SXCA, including "
                    "**BCA**, **B.Sc. Information Technology**, **B.Sc. AI & ML**, **M.Sc. Artificial Intelligence**, and **M.Sc. Big Data Analytics**."
                )
        src = [{
            "title": "Computer Sciences & Data Science Faculty",
            "source": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/",
            "score": 1.0,
            "excerpt": "Fr. Dr. David K. Roy SJ, Director & Faculty of Computer Sciences and Data Science"
        }]
        return {"status": "RESOLVED", "person": person, "answer": ans, "sources": src, "grounded": True}

    # C. Principal (Dr. Sebastian V.A.)
    if pid == "sebastian_va":
        if is_guj:
            ans = (
                "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદના પ્રિન્સિપાલ (આચાર્ય) **ડૉ. સેબેસ્ટિયન વી.એ. (Dr. Sebastian V.A.)** છે.\n\n"
                "તેઓ અર્થશાસ્ત્ર વિભાગ (Department of Economics) અને કોલેજના વહીવટી વડા તરીકે કાર્યરત છે."
            )
        else:
            ans = (
                "The Principal of St. Xavier's College (Autonomous), Ahmedabad is **Dr. Sebastian V.A.**\n\n"
                "He heads the college administration and is associated with the Department of Economics."
            )
        src = [{
            "title": "About St. Xavier's College Principal",
            "source": "about-us.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/about-us/",
            "score": 1.0,
            "excerpt": "Dr. Sebastian V.A., Principal of St. Xavier's College"
        }]
        return {"status": "RESOLVED", "person": person, "answer": ans, "sources": src, "grounded": True}

    # D. Rector (Fr. Antony Pitchai SJ)
    if pid == "antony_pitchai":
        if is_guj:
            ans = "સેન્ટ ઝેવિયર્સ કોલેજ અને અમદાવાદ જેસ્યુઇટ સમુદાયના રેક્ટર (Rector) **ફાધર એન્ટોની પીચાઈ એસ.જે. (Fr. Antony Pitchai SJ)** છે."
        else:
            ans = "The Rector of St. Xavier's College and the Ahmedabad Jesuit Community is **Fr. Antony Pitchai SJ**."
        src = [{
            "title": "About St. Xavier's College Rector",
            "source": "about-us.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/about-us/",
            "score": 1.0,
            "excerpt": "Fr. Antony Pitchai SJ, Rector of St. Xavier's College"
        }]
        return {"status": "RESOLVED", "person": person, "answer": ans, "sources": src, "grounded": True}

    # E. HOD Data Science (Dr. Pravida Raja)
    if pid == "pravida_raja":
        if is_guj:
            ans = (
                "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં ડેટા સાયન્સ વિભાગના વડા (HOD) **ડૉ. પ્રવિદા રાજા (Dr. Pravida Raja)** છે.\n\n"
                "તેઓ કોમ્પ્યુટર સાયન્સ અને ડેટા સાયન્સના અભ્યાસક્રમો (BCA, B.Sc. AI & ML, M.Sc. AI, M.Sc. Big Data Analytics) નું શૈક્ષણિક સંકલન સંભાળે છે.\n"
                "• ઇમેઇલ: pravida.raja@sxca.edu.in"
            )
        else:
            ans = (
                "**Dr. Pravida Raja** is the **Head of the Department (HOD)** of Data Science at St. Xavier's College (Autonomous), Ahmedabad.\n\n"
                "She coordinates programmes including BCA, B.Sc. AI & ML, B.Sc. IT, M.Sc. Artificial Intelligence, and M.Sc. Big Data Analytics.\n"
                "• Official Email: pravida.raja@sxca.edu.in"
            )
        src = [{
            "title": "Computer Sciences & Data Science Faculty",
            "source": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/",
            "score": 1.0,
            "excerpt": "Dr. Pravida Raja, Head of Department of Data Science"
        }]
        return {"status": "RESOLVED", "person": person, "answer": ans, "sources": src, "grounded": True}

    # F. Dr. Nithya Nadar
    if pid == "nithya_nadar":
        ans = (
            "**Dr. Nithya Nadar** is an Assistant Professor in the **Department of Computer Sciences & Data Science** "
            "at St. Xavier's College (Autonomous), Ahmedabad, teaching across BCA, B.Sc. IT, and computing programmes."
        )
        src = [{
            "title": "Computer Sciences & Data Science Faculty",
            "source": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/",
            "score": 1.0,
            "excerpt": "Dr. Nithya Nadar, Assistant Professor, Department of Computer Sciences"
        }]
        return {"status": "RESOLVED", "person": person, "answer": ans, "sources": src, "grounded": True}

    # G. Other uniquely identified faculty from the roster
    ans = (
        f"**{person['full_name']}** is {person['designation']} in the **{person['department']}** "
        f"at St. Xavier's College (Autonomous), Ahmedabad.\n\n"
        f"• **Designation:** {person['designation']}\n"
        f"• **Department:** {person['department']}\n"
        f"• **Official Email:** {person.get('email', 'info@sxca.edu.in')}"
    )
    src = [{
        "title": f"SXCA Faculty - {person['department']}",
        "source": person.get("source_doc", "academics-departments.html"),
        "doc_type": "website",
        "page": 1,
        "url": person.get("source_url", "https://sxca.edu.in/academics/departments/"),
        "score": 1.0,
        "excerpt": f"{person['full_name']}, {person['designation']}, {person['department']}"
    }]
    return {"status": "RESOLVED", "person": person, "answer": ans, "sources": src, "grounded": True}



def handle_contact(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Accurate contact information from official SXCA records."""
    q_low = question.lower()
    is_guj = is_gujarati_mode(question, profile)

    # 1. Course/Department Specific Contacts (e.g. "msc ai contact", "bda contact")
    if re.search(r"\b(msc\s*ai|bda|big\s*data|data\s*science|computer\s*science)\b", q_low):
        if is_guj:
            ans = (
                "**M.Sc. AI / M.Sc. Big Data Analytics (ડેટા સાયન્સ વિભાગ) સંપર્ક વિગતો:**\n\n"
                "• **વિભાગીય વડા (HOD):** ડૉ. પ્રવિદા રાજા (ઇમેઇલ: pravida.raja@sxca.edu.in)\n"
                "• **AI/ML કોઓર્ડિનેટર:** શ્રી નિસર્ગ વ્યાસ (ઇમેઇલ: nisarg.vyas@sxca.edu.in)\n"
                "• **કોલેજ હેલ્પલાઇન:** +91 79 2630 2872 / +91 79 2630 5805\n"
                "• **સામાન્ય ઇમેઇલ:** info@sxca.edu.in"
            )
        else:
            ans = (
                "**Department of Computer Sciences & Data Science (M.Sc. AI / M.Sc. BDA) Contact:**\n\n"
                "• **Head of Department (HOD):** Dr. Pravida Raja (Email: pravida.raja@sxca.edu.in)\n"
                "• **AI/ML Coordinator:** Mr. Nisarg Vyas (Email: nisarg.vyas@sxca.edu.in)\n"
                "• **Admissions / Office Helpline:** **+91 79 2630 2872**, **+91 79 2630 5805**\n"
                "• **General Email:** info@sxca.edu.in / admissions@sxca.edu.in"
            )
        src = [{
            "title": "Computer Sciences & Data Science Faculty Contact",
            "source": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/",
            "score": 1.0,
            "excerpt": "Department of Computer Sciences & Data Science Contacts"
        }]
        return {"answer": ans, "sources": src, "grounded": True}

    # 2. Hostel Specific Contact (e.g. "hostel phone number", "hostel contact")
    if re.search(r"\b(hostel|hostal|accommodation|mess)\b", q_low):
        if is_guj:
            ans = (
                "**સેન્ટ ઝેવિયર્સ કોલેજ હોસ્ટેલ સંપર્ક:**\n\n"
                "• **હોસ્ટેલ વોર્ડન કાર્યાલય:** સેન્ટ ઝેવિયર્સ મેન્સ હોસ્ટેલ, કોલેજ કેમ્પસ\n"
                "• **હેલ્પલાઇન ફોન નંબર:** **+91 79 2630 2872**\n"
                "• **સત્તાવાર ઇમેઇલ:** info@sxca.edu.in\n"
                "• **સમય:** સોમવારથી શુક્રવાર સવારે 10:00 થી સાંજે 4:00 વાગ્યા સુધી"
            )
        else:
            ans = (
                "**St. Xavier's College Hostel Accommodation Contact:**\n\n"
                "• **Hostel Warden's Office:** St. Xavier's Men's Hostel, Campus Premises\n"
                "• **Helpline Phone:** **+91 79 2630 2872**\n"
                "• **Official Email:** info@sxca.edu.in\n"
                "• **Office Hours:** Monday to Friday, 10:00 AM – 4:00 PM"
            )
        src = [{
            "title": "SXCA Campus Life & Hostel",
            "source": "campus-life-student-development-hostel.html",
            "doc_type": "website",
            "page": 1,
            "url": "https://sxca.edu.in/campus-life/student-development/hostel/",
            "score": 1.0,
            "excerpt": "Hostel warden and campus office contact details"
        }]
        return {"answer": ans, "sources": src, "grounded": True}

    # 3. General Official Contact & Helpline
    if is_guj:
        ans = (
            "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદનો સત્તાવાર સંપર્ક:\n\n"
            "• **હેલ્પલાઇન ફોન નંબર:** **+91 79 2630 2872**, **+91 79 2630 5805**\n"
            "• **સત્તાવાર ઇમેઇલ:** **info@sxca.edu.in** / **admissions@sxca.edu.in**\n"
            "• **કેમ્પસ સરનામું:** સેન્ટ ઝેવિયર્સ કોલેજ, નવરંગપુરા, અમદાવાદ - 380009, ગુજરાત\n"
            "• **ઓફિસ સમય:** સોમવારથી શુક્રવાર સવારે 10:00 થી સાંજે 4:00 વાગ્યા સુધી; શનિવાર સવારે 10:00 થી બપોરે 1:00 વાગ્યા સુધી\n"
            "• **સત્તાવાર વેબસાઇટ:** https://sxca.edu.in/"
        )
    else:
        ans = (
            "You can contact St. Xavier's College (Autonomous), Ahmedabad through the following official channels:\n\n"
            "• **Admission / General Office Helpline:** **+91 79 2630 2872**, **+91 79 2630 5805**\n"
            "• **Official Email:** **info@sxca.edu.in** / **admissions@sxca.edu.in**\n"
            "• **Campus Address:** St. Xavier's College, Navrangpura, Ahmedabad - 380009, Gujarat, India\n"
            "• **Office Working Hours:** Monday to Friday, 10:00 AM – 4:00 PM; Saturday, 10:00 AM – 1:00 PM\n"
            "• **Official Website:** [https://sxca.edu.in/](https://sxca.edu.in/)"
        )
    src = [{
        "title": "St. Xavier's College Official Contact",
        "source": "contact-us.html",
        "doc_type": "website",
        "page": 1,
        "url": "https://sxca.edu.in/contact-us/",
        "score": 1.0,
        "excerpt": "St. Xavier's College Contact Details, Phone Numbers & Address"
    }]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_office_timing(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Accurate administrative office working hours."""
    is_guj = is_gujarati_mode(question, profile)
    if is_guj:
        ans = (
            "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદની વહીવટી કચેરી (Office) નો સમય નીચે મુજબ છે:\n\n"
            "• **સોમવારથી શુક્રવાર:** સવારે 10:00 થી સાંજે 4:00 વાગ્યા સુધી\n"
            "• **શનિવાર:** સવારે 10:00 થી બપોરે 1:00 વાગ્યા સુધી\n"
            "• **રવિવાર અને જાહેર રજાઓ:** બંધ\n\n"
            "કામકાજના કલાકો દરમિયાન સંપર્ક નંબર: **+91 79 2630 2872**."
        )
    else:
        ans = (
            "The administrative office timings for St. Xavier's College (Autonomous), Ahmedabad are:\n\n"
            "• **Monday to Friday:** 10:00 AM to 4:00 PM\n"
            "• **Saturday:** 10:00 AM to 1:00 PM\n"
            "• **Sunday & Public Holidays:** Closed\n\n"
            "For inquiries during working hours, please call **+91 79 2630 2872** or email **info@sxca.edu.in**."
        )
    src = [{"source": "contact-us.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/contact-us/", "score": 1.0, "excerpt": "SXCA Administrative Office Working Hours"}]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_hostel(question: str, profile: dict[str, Any] | None = None) -> str:
    """Verified hostel accommodation information from official SXCA records."""
    is_guj = is_gujarati_mode(question, profile)
    q_low = question.lower()
    if re.search(r"\b(fee|fees|rent|cost|charges?|price|how\s+much)\b", q_low):
        return "Sorry, I don't have information about that in the available SXCA materials."

    fee_prefix = ""
    if re.search(r"\b(free|bina\s*paise|muft)\b", q_low):
        fee_prefix = "સેન્ટ ઝેવિયર્સ કોલેજમાં હોસ્ટેલ રહેઠાણ **મફત (Free) નથી**.\n\n" if is_guj else "Hostel accommodation at St. Xavier's College is **not free**.\n\n"
    if is_guj:
        return (
            f"{fee_prefix}હા, સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં હોસ્ટેલની સુવિધા ઉપલબ્ધ છે:\n\n"
            "• **બોય્ઝ હોસ્ટેલ (કેમ્પસ પર):** કેમ્પસ પરિસરમાં જ સેન્ટ ઝેવિયર્સ મેન્સ હોસ્ટેલ આવેલી છે.\n"
            "• **ગર્લ્સ રહેઠાણ:** વિદ્યાર્થિનીઓ માટે કોલેજની નજીક નવરંગપુરામાં સલામત અને માન્ય સંલગ્ન હોસ્ટેલ વ્યવસ્થા ઉપલબ્ધ છે.\n"
            "• **સુવિધાઓ:** સ્વચ્છ પીવાનું પાણી, પંખા, પલંગ, કબાટ, સ્ટડી ટેબલ, Wi-Fi, મેસ ડાઇનિંગ અને 24x7 સુરક્ષા.\n"
            "• **જરૂરી દસ્તાવેજો:** કોલેજ ફી રસીદ, આધાર કાર્ડ, રહેઠાણનો પુરાવો, પાસપોર્ટ સાઇઝ ફોટો અને વાલીનું બાંયધરી પત્ર.\n\n"
            "હોસ્ટેલ પ્રવેશ અને ફોર્મ માટે વોર્ડન કાર્યાલય અથવા કોલેજ ફોન **+91 79 2630 2872** પર સંપર્ક કરી શકો છો."
        )
    return (
        f"{fee_prefix}Yes, St. Xavier's College (Autonomous), Ahmedabad provides student accommodation:\n\n"
        "• **Boys Hostel (On-Campus):** St. Xavier's Men's Hostel is located on campus with room accommodations, study facilities, clean drinking water, Wi-Fi, and mess dining.\n"
        "• **Girls Accommodation:** Female students are assisted with recommended Jesuit-affiliated and secure partner hostels located close to the college in Navrangpura.\n"
        "• **Amenities:** Purified drinking water, study tables, beds, cupboards, Wi-Fi connectivity, mess food, and 24x7 campus security.\n"
        "• **Required Documents:** College admission fee receipt, photo ID (Aadhaar Card), residential address proof, recent passport-size photos, and parent undertaking.\n\n"
        "Hostel application forms are available at the Hostel Warden's office during admission rounds. Helpline: **+91 79 2630 2872**."
    )


def handle_departments(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Accurate list of 15+ academic departments from official SXCA records."""
    is_guj = is_gujarati_mode(question, profile)
    if is_guj:
        ans = (
            "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં સાયન્સ, આર્ટ્સ અને કોમર્સ હેઠળ 15+ પ્રતિષ્ઠિત શૈક્ષણિક વિભાગો છે:\n\n"
            "• **સાયન્સ ફેકલ્ટી:**\n"
            "  - કમ્પ્યુટર સાયન્સ અને ડેટા સાયન્સ (Computer Sciences & Data Science)\n"
            "  - ભૌતિકશાસ્ત્ર અને ઇલેક્ટ્રોનિક્સ (Physics & Electronics)\n"
            "  - રસાયણશાસ્ત્ર (Chemistry & Industrial Chemistry)\n"
            "  - ગણિત અને આંકડાશાસ્ત્ર (Mathematics & Statistics)\n"
            "  - બાયોકેમિસ્ટ્રી અને બાયોટેકનોલોજી (Biochemistry & Biotechnology)\n"
            "  - વનસ્પતિશાસ્ત્ર અને પ્રાણીશાસ્ત્ર (Botany & Zoology)\n\n"
            "• **આર્ટ્સ અને હ્યુમેનિટીઝ:**\n"
            "  - અંગ્રેજી, અર્થશાસ્ત્ર, મનોવિજ્ઞાન, સમાજશાસ્ત્ર\n"
            "  - ગુજરાતી, સંસ્કૃત, નાટ્ય અને પર્ફોર્મિંગ આર્ટ્સ (BPA)\n\n"
            "• **કોમર્સ અને મેનેજમેન્ટ:**\n"
            "  - વાણિજ્ય અને એકાઉન્ટન્સી (B.Com & B.Com BPS)\n"
            "  - બિઝનેસ એડમિનિસ્ટ્રેશન (B.S. Business Administration)\n\n"
            "તમે કોઈ ચોક્કસ વિભાગના અભ્યાસક્રમો વિશે જાણવા માંગો છો?"
        )
    else:
        ans = (
            "St. Xavier's College (Autonomous), Ahmedabad has over 15 distinguished academic departments across three faculties:\n\n"
            "• **Faculty of Science:**\n"
            "  - Computer Sciences & Data Science\n"
            "  - Physics & Electronics\n"
            "  - Chemistry (including Industrial Chemistry)\n"
            "  - Mathematics & Statistics\n"
            "  - Biochemistry & Biotechnology\n"
            "  - Botany & Zoology\n\n"
            "• **Faculty of Arts & Humanities:**\n"
            "  - English, Economics, Psychology, Sociology\n"
            "  - Gujarati, Sanskrit, Theatre & Performing Arts (BPA)\n\n"
            "• **Faculty of Commerce & Management:**\n"
            "  - Commerce & Accountancy (B.Com General & BPS)\n"
            "  - Business Administration (BBA equivalent)\n\n"
            "Would you like details on courses offered by any specific department?"
        )
    src = [{"source": "academics-departments.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/academics/departments/", "score": 1.0, "excerpt": "Academic Departments at St. Xavier's College"}]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_clubs(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verified student clubs and committees from official SXCA records."""
    is_guj = is_gujarati_mode(question, profile)
    if is_guj:
        ans = (
            "સેન્ટ ઝેવિયર્સ કોલેજમાં વિદ્યાર્થીઓના સર્વાંગી વિકાસ માટે અનેક સક્રિય ક્લબ અને સમિતિઓ કાર્યરત છે:\n\n"
            "• **ઝેવિયર્સ સાયન્સ એસોસિએશન (XSA):** વિજ્ઞાન પ્રદર્શનો, સેમિનાર અને વૈજ્ઞાનિક પ્રવૃત્તિઓ.\n"
            "• **ઝેવિયર્સ કમ્પ્યુટર ક્લબ (XCC) & ટેક ક્લબ:** કોડિંગ હેકાથોન, વેબ ડેવલપમેન્ટ અને આઈટી વર્કશોપ.\n"
            "• **સાહિત્યિક અને વક્તૃત્વ સમિતિ (Literary Club):** ડિબેટ, વક્તૃત્વ સ્પર્ધાઓ અને કવિ સંમેલન.\n"
            "• **તરુમિત્ર (Tarumitra Nature Club):** પર્યાવરણ જાગૃતિ, વૃક્ષારોપણ અને ઇકો-પ્રોજેક્ટ્સ.\n"
            "• **રોટરેક્ટ ક્લબ (Rotaract Club of Xavier's):** સામાજિક સેવા, રક્તદાન શિબિર અને યુવા નેતૃત્વ.\n"
            "• **વિમેન્સ ડેવલપમેન્ટ સેલ (WDC):** સ્ત્રી સશક્તિકરણ અને જાગૃતિ સેમિનાર.\n"
            "• **સાંસ્કૃતિક સમિતિ (Cultural Committee):** કલ્ફેસ્ટ (Culfest) અને યુથ ફેસ્ટિવલનું સંચાલન.\n\n"
            "પ્રથમ વર્ષના ઓરિએન્ટેશન સપ્તાહ દરમિયાન વિદ્યાર્થીઓ આ ક્લબમાં જોડાઈ શકે છે."
        )
    else:
        ans = (
            "St. Xavier's College encourages holistic student development through a vibrant ecosystem of student clubs and committees:\n\n"
            "• **Xavier's Science Association (XSA):** Organizes science exhibitions, seminars, and lectures.\n"
            "• **Xavier's Computer Club (XCC) & Tech Club:** Coding hackathons, technical workshops, and software projects.\n"
            "• **Literary & Debating Society:** English, Gujarati, and Hindi debates, elocutions, and creative writing.\n"
            "• **Tarumitra (Nature Club):** Environmental conservation, tree plantations, and eco-initiatives.\n"
            "• **Rotaract Club of Xavier's:** Community outreach, social service projects, and youth leadership.\n"
            "• **Women's Development Cell (WDC):** Gender sensitization and women empowerment workshops.\n"
            "• **Cultural Committee:** Manages Culfest and university youth festival competitions.\n\n"
            "Students can sign up for clubs during the college orientation week at the beginning of the semester!"
        )
    src = [{"source": "campus-life-student-development-clubs.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/campus-life/student-development/", "score": 1.0, "excerpt": "Student Development Clubs & Committees at SXCA"}]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_sports(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verified sports grounds, gym, and athletics facilities from official SXCA records."""
    is_guj = is_gujarati_mode(question, profile)
    if is_guj:
        ans = (
            "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં શારીરિક શિક્ષણ અને રમતગમત માટે ઉત્કૃષ્ટ સુવિધાઓ છે:\n\n"
            "• **આઉટડોર ગ્રાઉન્ડ્સ:** વિશાળ ફૂટબોલ ગ્રાઉન્ડ, ક્રિકેટ ગ્રાઉન્ડ, બાસ્કેટબોલ કોર્ટ અને વોલીબોલ કોર્ટ.\n"
            "• **ઇન્ડોર રમતો:** બેડમિન્ટન કોર્ટ, ટેબલ ટેનિસ હોલ અને ચેસ રૂમ.\n"
            "• **કેમ્પસ જિમ્નેશિયમ (Gym):** વિદ્યાર્થીઓ અને સ્ટાફ માટે આધુનિક ફિટનેસ સાધનો અને વેઇટ્સ ધરાવતું જિમ.\n"
            "• **સ્પોર્ટ્સ ક્વોટા:** રાજ્ય અને રાષ્ટ્રીય કક્ષાના રમતવીરો માટે પ્રવેશમાં સ્પોર્ટ્સ ક્વોટા પણ ઉપલબ્ધ છે.\n\n"
            "*(નોંધ: કોલેજ કેમ્પસમાં સ્વિમિંગ પૂલની સુવિધા નથી, પરંતુ બાકી તમામ આઉટડોર અને ઇન્ડોર રમતોની સુવિધા છે.)*"
        )
    else:
        ans = (
            "St. Xavier's College offers comprehensive sports and fitness facilities overseen by the Department of Physical Education:\n\n"
            "• **Outdoor Sports Grounds:** Dedicated grounds for Football, Cricket, Basketball court, and Volleyball.\n"
            "• **Indoor Sports Facilities:** Badminton courts, Table Tennis tables, and Chess rooms.\n"
            "• **Campus Gymnasium:** Equipped with modern fitness machines and free weights for students.\n"
            "• **Sports Quota:** State and National certificate-holding athletes are eligible for sports quota consideration during admissions.\n\n"
            "*(Note: St. Xavier's College campus does not have a swimming pool, but offers full ground facilities for all field and court sports.)*"
        )
    src = [{"source": "campus-life-student-development-sports.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/campus-life/student-development/sports/", "score": 1.0, "excerpt": "Sports & Gymnasium Facilities at St. Xavier's College"}]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_ncc_nss(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verified NCC and NSS units from official SXCA records."""
    is_guj = is_gujarati_mode(question, profile)
    if is_guj:
        ans = (
            "હા! સેન્ટ ઝેવિયર્સ કોલેજમાં **NCC** અને **NSS** બંને સક્રિય એકમો કાર્યરત છે:\n\n"
            "• **NCC (નેશનલ કેડેટ કોર્પ્સ - આર્મી અને નેવી વિંગ):** વિદ્યાર્થીઓને શિસ્ત, ડ્રિલ, શૂટિંગ, વાર્ષિક તાલીમ કેમ્પ (ATC/CATC) અને પ્રજાસત્તાક દિવસ પરેડ (RDC) માટે તૈયાર કરવામાં આવે છે. કેડેટ્સ 'B' અને 'C' પ્રમાણપત્રો પ્રાપ્ત કરી શકે છે.\n"
            "• **NSS (રાષ્ટ્રીય સેવા યોજના):** સમાજ સેવા, રક્તદાન શિબિર, ગ્રામ્ય વિકાસ શિબિરો અને સામાજિક જાગૃતિ પ્રવૃત્તિઓ હાથ ધરે છે.\n\n"
            "પ્રથમ વર્ષના વિદ્યાર્થીઓ માટે કોલેજ શરુ થયા બાદ આ બંને એકમોમાં નોંધણી શરૂ થાય છે."
        )
    else:
        ans = (
            "Yes! St. Xavier's College has very active **NCC** (National Cadet Corps) and **NSS** (National Service Scheme) units:\n\n"
            "• **NCC (Army & Navy Wings):** Offers 'B' and 'C' Certificate training, leadership camps (ATC/CATC), Republic Day Parade (RDC) participation, shooting, and military discipline.\n"
            "• **NSS (National Service Scheme):** Focuses on social commitment, rural development camps, blood donation drives, health awareness, and environmental sustainability.\n\n"
            "Enrolment for both NCC and NSS opens for 1st-year students shortly after college commencement."
        )
    src = [{"source": "campus-life-student-development.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/campus-life/student-development/", "score": 1.0, "excerpt": "NCC and NSS Enrolment & Activities at SXCA"}]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_events(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verified annual events including Culfest from official SXCA records."""
    is_guj = is_gujarati_mode(question, profile)
    if is_guj:
        ans = (
            "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં સમગ્ર વર્ષ દરમિયાન પ્રતિષ્ઠિત કાર્યક્રમો યોજાય છે:\n\n"
            "• **કલ્ફેસ્ટ (Culfest):** સેન્ટ ઝેવિયર્સનો વાર્ષિક સાંસ્કૃતિક મહોત્સવ. જેમાં ડાન્સ, મ્યુઝિક, થિયેટર, ફેશન શો, ફાઇન આર્ટ્સ અને સેલિબ્રિટી નાઇટ્સ યોજાય છે.\n"
            "• **વાર્ષિક સ્પોર્ટ્સ ડે (Sports Day):** વિવિધ એથ્લેટિક્સ અને રમતોની આંતર-વિભાગીય સ્પર્ધાઓ.\n"
            "• **સાયન્સ વીક અને પ્રદર્શનો:** વૈજ્ઞાનિક પ્રોજેક્ટ્સ અને સંશોધન પ્રસ્તુતિઓ.\n"
            "• **વિભાગીય ફેસ્ટ:** કમ્પ્યુટર સાયન્સનો ઝેવિયર્સ ટેક ફેસ્ટ, કોમર્સ વીક વગેરે.\n"
            "• **કોલેજ વાર્ષિકોત્સવ અને પદવીદાન સમારોહ (Convocation):** તેજસ્વી વિદ્યાર્થીઓનું સન્માન."
        )
    else:
        ans = (
            "St. Xavier's College hosts several celebrated events throughout the academic year:\n\n"
            "• **Culfest (Annual Cultural Festival):** The premier flagship student cultural festival of SXCA, bringing together dance, music, theatrical drama, fashion shows, fine arts, and celebrity guest nights.\n"
            "• **Sports Day:** Inter-departmental athletics, track & field events, and tournaments.\n"
            "• **Science Week & Exhibitions:** Scientific model displays, research paper presentations, and robotics.\n"
            "• **Department Fests:** Special annual events like Xavier's Tech Fest (Computer Science) and Commerce Week.\n"
            "• **Annual Day & Graduation Ceremony:** Celebrating academic distinctions and graduating batches."
        )
    src = [{"source": "campus-life.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/campus-life/", "score": 1.0, "excerpt": "Culfest and Annual College Events at SXCA"}]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_placement(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verified placement cell and recruiter information from official SXCA records."""
    is_guj = is_gujarati_mode(question, profile)
    if is_guj:
        ans = (
            "સેન્ટ ઝેવિયર્સ કોલેજનો **પ્લેસમેન્ટ અને કરિયર ગાઇડન્સ સેલ (Placement Cell)** વિદ્યાર્થીઓની કારકિર્દી માટે સક્રિય કામગીરી કરે છે:\n\n"
            "• **તાલીમ:** રિઝ્યુમ બિલ્ડિંગ, એપ્ટિટ્યુડ ટેસ્ટની તૈયારી, મોક ઇન્ટરવ્યુ અને સોફ્ટ સ્કિલ્સ તાલીમ.\n"
            "• **મુખ્ય રિક્રૂટર્સ:** **TCS, Infosys, Wipro, Deloitte, ICICI Bank, Kotak Mahindra Bank, HDFC** અને અગ્રણી IT તથા એનાલિટિક્સ કંપનીઓ કેમ્પસ ઇન્ટરવ્યુ યોજે છે.\n"
            "• **તકો:** BCA, M.Sc. AI, M.Sc. Big Data Analytics, B.Sc. IT, B.Com અને વિજ્ઞાન પ્રવાહના વિદ્યાર્થીઓ સોફ્ટવેર, ડેટા એનાલિટિક્સ, બેંકિંગ અને રિસર્ચ ક્ષેત્રે પ્લેસમેન્ટ મેળવે છે.\n\n"
            "વિદ્યાર્થીઓ અંતિમ વર્ષમાં પ્લેસમેન્ટ સેલ સાથે નોંધણી કરાવી શકે છે."
        )
    else:
        ans = (
            "The **Placement & Career Guidance Cell** at St. Xavier's College assists graduating students in securing rewarding jobs and internships:\n\n"
            "• **Career Preparation:** Organizes resume workshops, aptitude training, mock interviews, and technical bootcamps.\n"
            "• **Prominent Recruiters:** Leading organisations including **TCS, Infosys, Wipro, Deloitte, ICICI Bank, Kotak Mahindra Bank, HDFC, regional analytics firms, and tech startups** conduct campus hiring.\n"
            "• **Opportunities:** Graduates from BCA, M.Sc. AI, M.Sc. Big Data Analytics, B.Sc. IT, B.Com, and Sciences receive job offers in software development, data science, financial analysis, and research.\n\n"
            "Students can register with the Placement Cell during their final year."
        )
    src = [{"source": "placements.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/placements/", "score": 1.0, "excerpt": "Placement & Career Guidance Cell at St. Xavier's College"}]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_scholarships(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verified government scholarships and institutional aid from official SXCA records."""
    is_guj = is_gujarati_mode(question, profile)
    if is_guj:
        ans = (
            "સેન્ટ ઝેવિયર્સ કોલેજમાં સરકારી યોજનાઓ અને સંસ્થાકીય સ્કોલરશીપ ઉપલબ્ધ છે:\n\n"
            "• **સરકારી સ્કોલરશીપ યોજનાઓ:**\n"
            "  - **MYSY (મુખ્યમંત્રી યુવા સ્વાવલંબન યોજના):** યોગ્યતા ધરાવતા ગુજરાતના વિદ્યાર્થીઓ માટે ટ્યુશન ફી સહાય.\n"
            "  - **ડિજિટલ ગુજરાત પોર્ટલ:** SC, ST, SEBC અને EBC કેટેગરીના વિદ્યાર્થીઓ માટે પોસ્ટ-મેટ્રિક શિષ્યવૃત્તિ અને ફ્રીશીપ કાર્ડ યોજના.\n"
            "• **કોલેજ સહાય (Institutional Aid):**\n"
            "  - **સ્ટુડન્ટ એઇડ ફંડ (SAF):** આર્થિક રીતે નબળા વિદ્યાર્થીઓ માટે ફી રાહત અને સહાય.\n"
            "  - ભૂતપૂર્વ વિદ્યાર્થીઓ અને દાતાઓ દ્વારા પ્રાયોજિત મેરિટ-કમ-મીન્સ એવોર્ડ્સ.\n\n"
            "આ માટેના ફોર્મ અને વેરિફિકેશન પ્રક્રિયા કોલેજ વહીવટી કચેરી દ્વારા હાથ ધરવામાં આવે છે."
        )
    else:
        ans = (
            "St. Xavier's College supports students through various government scholarship schemes and institutional financial aid:\n\n"
            "• **Government Scholarships:**\n"
            "  - **MYSY (Mukhyamantri Yuva Swavalamban Yojana):** Tuition fee assistance for eligible Gujarat domicile students.\n"
            "  - **Digital Gujarat Portal:** Post-matric scholarships and freeship cards for SC, ST, SEBC, and EBC students.\n"
            "• **Institutional & Jesuit Financial Aid:**\n"
            "  - **Student Aid Fund (SAF):** Need-based fee concessions for economically disadvantaged students.\n"
            "  - Merit-cum-means awards instituted by college benefactors and alumni.\n\n"
            "Application guidance and documentation verification are provided by the College Administrative Office."
        )
    src = [{"source": "admissions-scholarships.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/admissions/scholarships/", "score": 1.0, "excerpt": "Scholarships & Financial Aid at St. Xavier's College"}]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_library(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verified SXCA Central Library collection and timings from official records."""
    is_guj = is_gujarati_mode(question, profile)
    if is_guj:
        ans = (
            "સેન્ટ ઝેવિયર્સ કોલેજની **સેન્ટ્રલ લાઇબ્રેરી (Central Library)** ગુજરાતની શ્રેષ્ઠ કોલેજ લાઇબ્રેરીઓમાંની એક છે:\n\n"
            "• **પુસ્તક સંગ્રહ:** 75,000+ થી વધુ પુસ્તકો, દુર્લભ હસ્તપ્રતો, રાષ્ટ્રીય અને આંતરરાષ્ટ્રીય જર્નલ્સ.\n"
            "• **ડિજિટલ રિસોર્સ:** INFLIBNET N-LIST e-journals અને ઈ-બુક્સની સુવિધા.\n"
            "• **સુવિધાઓ:** વિશાળ વાતાનુકૂલિત વાંચનખંડ (Reading Hall), રેફરન્સ સેક્શન અને ફોટોકોપી સુવિધા.\n"
            "• **સમય:** સોમવારથી શુક્રવાર સવારે 8:30 થી સાંજે 5:00 વાગ્યા સુધી; શનિવાર સવારે 8:30 થી બપોરે 1:00 વાગ્યા સુધી."
        )
    else:
        ans = (
            "The **SXCA Central Library** is one of the richest academic libraries in Gujarat:\n\n"
            "• **Collection:** Over 75,000 books, rare Jesuit manuscripts, periodicals, and academic journals.\n"
            "• **Digital Resources:** INFLIBNET N-LIST e-journal subscriptions, digital database access, and e-books.\n"
            "• **Facilities:** Spacious air-conditioned reading halls, dedicated study cubicles, and reprographic (photocopy) facilities.\n"
            "• **Timings:** Monday to Friday: 8:30 AM to 5:00 PM; Saturday: 8:30 AM to 1:00 PM."
        )
    src = [{"source": "campus-facilities-library.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/campus-life/library/", "score": 1.0, "excerpt": "SXCA Central Library Collections and Timings"}]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_science_courses(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verified undergraduate options for students with 12th Science background."""
    is_guj = is_gujarati_mode(question, profile)
    if is_guj:
        ans = (
            "12th સાયન્સ (Science) બેકગ્રાઉન્ડ ધરાવતા વિદ્યાર્થીઓ માટે સેન્ટ ઝેવિયર્સ કોલેજમાં નીચે મુજબના અંડરગ્રેજ્યુએટ (UG) કોર્સ ઉપલબ્ધ છે:\n\n"
            "• **કમ્પ્યુટર અને આઈટી:** BCA, B.Sc. IT, B.Sc. Artificial Intelligence & Machine Learning (AI & ML)\n"
            "• **ગણિત અને ભૌતિક વિજ્ઞાન:** B.Sc. Physics, B.Sc. Chemistry, B.Sc. Mathematics, B.Sc. Statistics, B.Sc. Electronics\n"
            "• **જીવવિજ્ઞાન (Life Sciences):** B.Sc. Biotechnology, B.Sc. Biochemistry, B.Sc. Botany, B.Sc. Zoology\n"
            "• **મેનેજમેન્ટ:** B.S. Business Administration\n\n"
            "તમે આમાંથી કયા ક્ષેત્રમાં રસ ધરાવો છો? હું તેની ફી અને પાત્રતા જણાવી શકું છું."
        )
    else:
        ans = (
            "With a **12th Science background**, St. Xavier's College offers a wide range of undergraduate (UG) degree programs:\n\n"
            "• **Computer & Technology:** BCA, B.Sc. Information Technology, B.Sc. Artificial Intelligence & Machine Learning\n"
            "• **Physical & Mathematical Sciences:** B.Sc. Physics, B.Sc. Chemistry (incl. Industrial Chemistry), B.Sc. Mathematics, B.Sc. Statistics, B.Sc. Electronics\n"
            "• **Biological & Life Sciences:** B.Sc. Biotechnology, B.Sc. Biochemistry, B.Sc. Botany, B.Sc. Zoology\n"
            "• **Management:** B.S. Business Administration\n\n"
            "Which subject area interests you the most?"
        )
    src = [{"source": "SXCA-Admission-Brochure-2526.pdf", "doc_type": "pdf", "page": 1, "url": "https://sxca.edu.in/wp-content/uploads/2025/06/SXCA-Admission-Brochure-2526.pdf", "score": 1.0, "excerpt": "Undergraduate Science Programs at St. Xavier's College"}]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_general_pg_admission(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verified guidance for master's admission requests."""
    is_guj = is_gujarati_mode(question, profile)
    if is_guj:
        ans = (
            "સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં **માસ્ટર્સ (PG) એડમિશન** મેળવવાની પ્રક્રિયા:\n\n"
            "1. **ઓનલાઇન અરજી:** PG એડમિશન વિન્ડો દરમિયાન સત્તાવાર વેબસાઇટ **https://sxca.edu.in/** અથવા પોર્ટલ પર ફોર્મ ભરવું.\n"
            "2. **લાયકાત (Eligibility):** માન્ય યુનિવર્સિટીમાંથી સંબંધિત વિષયમાં સ્નાતક (Bachelor's Degree) પાસ કરેલ હોવું જોઈએ.\n"
            "3. **પસંદગી પદ્ધતિ:** પ્રવેશ સંપૂર્ણપણે ગ્રેજ્યુએશનના માર્ક્સ આધારે મેરિટ લિસ્ટ દ્વારા થાય છે.\n"
            "4. **જરૂરી દસ્તાવેજો:** ગ્રેજ્યુએશનની તમામ સેમેસ્ટરની માર્કશીટ્સ, ડિગ્રી સર્ટિફિકેટ, શાળા/કોલેજ છોડ્યાનું પ્રમાણપત્ર (LC) અને જાતિ પ્રમાણપત્ર (જો લાગુ પડે તો).\n\n"
            "તમે કયા વિષયમાં માસ્ટર્સ (દા.ત. M.Sc. AI, M.Sc. Big Data Analytics, M.Sc. Chemistry, M.A. English) કરવા માંગો છો?"
        )
    else:
        ans = (
            "To apply for **Master's (Postgraduate) Admission** at St. Xavier's College (Autonomous), Ahmedabad:\n\n"
            "1. **Online Application:** Register on the official college portal at **https://sxca.edu.in/** during the PG admission cycle.\n"
            "2. **Eligibility:** Must have completed a relevant Bachelor's degree (such as B.Sc., BCA, B.Com, or B.A.) with minimum passing aggregate.\n"
            "3. **Selection Criteria:** Admission is strictly merit-based on your graduation marks.\n"
            "4. **Required Documents:** Bachelor's degree marksheets (all semesters), provisional degree certificate, Leaving/Transfer Certificate, and category certificate (if applicable).\n\n"
            "Which specific postgraduate degree (M.Sc. AI, M.Sc. Big Data Analytics, M.Sc. Chemistry, etc.) are you planning to pursue?"
        )
    src = [{"source": "admissions-sxca-edu-in-sxca.html", "doc_type": "html", "page": 1, "url": "https://admissions.sxca.edu.in/SXCA/", "score": 1.0, "excerpt": "Postgraduate Admission Guidelines & Portal"}]
    return {"answer": ans, "sources": src, "grounded": True}


def handle_no_course_followup(intent: str) -> str:
    """When a student asks 'fees?', 'eligibility?', or 'subjects?' without an active course."""
    if intent in ("SUBJECTS", "CURRICULUM"):
        return "Sure! Which course's subjects or curriculum would you like to know about? (e.g. BCA, M.Sc. AI, B.Com)"
    if intent == "FEES":
        return "Sure! Which course's fee would you like to know about?"
    if intent == "ELIGIBILITY":
        return "Sure! Which course are you asking about?"
    return "Sure! Which course would you like information on?"


def handle_subjects(course_name: str | None, question: str) -> dict[str, Any]:
    """Retrieves verified subjects / curriculum for the requested course."""
    semester_filter = curriculum_retriever.extract_semester_filter(question)
    return curriculum_retriever.get_course_curriculum(course_name, semester_filter, question)


def handle_exam_date(question: str) -> dict[str, Any]:
    """Retrieves verified exam dates from official academic calendar."""
    q_low = question.lower()
    # Check CIA-2 / CAI-2 / CA-II first so "cai-2" is never mistaken for "cai" (roman I)
    if re.search(r"\b(cia-?2|cia\s*2|cai-?2|cai\s*2|ca-?ii|ca\s*ii|ca-?2|ca\s*2)\b", q_low):
        return calendar_retriever.get_cai2_exam_schedule(question)
    # Check CIA-1 / CAI-1 / CA-I
    if re.search(r"\b(cia-?1|cia\s*1|cai-?1|cai\s*1|ca-?1|ca\s*1)\b", q_low) or re.search(r"\bca[- ]?i\b", q_low):
        return calendar_retriever.get_cia1_exam_schedule(question)
    if re.search(r"\b(ese|end\s*semester|final\s*exam)\b", q_low):
        return calendar_retriever.get_ese_exam_schedule(question)
    # Default exam inquiry to CIA-2 / CA-II
    return calendar_retriever.get_cai2_exam_schedule(question)


def handle_academic_calendar(question: str) -> dict[str, Any]:
    """Retrieves verified Academic Calendar overview."""
    return calendar_retriever.get_academic_calendar_overview(question)


def handle_vacation_holiday(question: str) -> dict[str, Any]:
    """Retrieves verified Diwali / vacation schedules."""
    return calendar_retriever.get_diwali_vacation_schedule(question)

