"""Security Guardrails & Protection Layer for SXCA Admission Chatbot.

Detects and blocks:
- System prompt & developer instruction leakage attempts
- Direct & indirect prompt injection attacks
- Jailbreak attempts (DAN mode, unrestricted AI mode)
- SQL injection payloads
- Abusive / profane language
- Sexual / explicit / inappropriate content
- Violent content & physical threats
- Off-topic questions (sports scores, coding tutorials, recipes, trivia)
- Gibberish / random keyboard mash inputs
"""

import re
from typing import Any

SAFE_REFUSAL_MESSAGE = (
    "I cannot fulfill this request. I am here to assist with St. Xavier's College (Autonomous), Ahmedabad admissions and academic inquiries."
)

RESPONSE_ABUSIVE = (
    "I’m here to help with admission-related questions. Please keep the conversation respectful. How can I help you with SXCA admissions?"
)

RESPONSE_SEXUAL = (
    "I cannot fulfill this request. I am here to assist with St. Xavier's College (Autonomous), Ahmedabad admissions and academic inquiries. How can I help you with SXCA admissions?"
)

RESPONSE_THREAT = (
    "I cannot process requests containing violent or harmful content. I am here to help with St. Xavier's College admissions."
)

RESPONSE_SYSTEM_PROMPT = (
    "I can’t provide internal system instructions or configuration, but I can help you with SXCA admission-related questions."
)

RESPONSE_INJECTION_JAILBREAK = (
    "I cannot fulfill this request. I am here to assist with St. Xavier's College (Autonomous), Ahmedabad admissions and academic inquiries."
)

RESPONSE_SQL_INJECTION = (
    "I cannot fulfill this request. Malicious database query syntax detected."
)

RESPONSE_OFF_TOPIC = (
    "I am an admission counsellor focused exclusively on St. Xavier's College (Autonomous), Ahmedabad (SXCA). "
    "I can help you with courses, eligibility, fees, faculty, and campus details. What would you like to know about SXCA?"
)

RESPONSE_GIBBERISH = (
    "I didn't quite understand that. Could you please rephrase your question or ask about SXCA courses, eligibility, or admissions?"
)


def _normalize_text(text: str) -> str:
    """Normalizes input string by lowercasing, stripping extra whitespace, and collapsing punctuation."""
    if not text:
        return ""
    norm = text.lower()
    norm = re.sub(r"[^\w\s\.-]", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


# --- REGEX PATTERNS ---

# 1. System Prompt & Secrets Leakage
_SYSTEM_PROMPT_PATTERNS = [
    re.compile(r"\b(?:reveal|show|print|output|display|give|get|share|echo|tell\s+me)\s+(?:me\s+)?(?:your|the)?\s*(?:system|developer|hidden|internal)\s+(?:prompt|instructions|rules|message|directives)\b", re.I),
    re.compile(r"\bwhat\s+(?:is|are)\s+your\s+(?:system|developer|hidden)\s+(?:prompt|instructions|rules)\b", re.I),
    re.compile(r"\bprint\s+(?:the\s+)?(?:hidden\s+)?system\s+prompt\b", re.I),
    re.compile(r"\b(?:show|print|cat|read|output|reveal|get|give)\s+(?:me\s+)?(?:the\s+)?(?:\.env|env\s+file|dotenv|secrets\.env|environment\s+variables?|secrets?|api\s+keys?|database\s+passwords?|jwt\s+secret|secret\s+key)\b", re.I),
    re.compile(r"\b(?:database|db|postgres|sql|jwt|secret)\s+(?:password|key|token|credential|string)\b", re.I),
    re.compile(r"\breveal\s+(?:secrets|api\s+keys|environment\s+variables)\b", re.I),
]

# 2. Direct Prompt Injection
_INJECTION_PATTERNS = [
    re.compile(r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system|document|documents)\s*(?:instructions|rules|prompts|directives|guidelines|and\s+tell\s+me|and\s+reveal)?\b", re.I),
    re.compile(r"\b(?:forget|ignore)\s+your\s+(?:rules|instructions|system\s+prompt|role)\b", re.I),
    re.compile(r"\b(?:disregard|override)\s+(?:system\s+prompt|developer\s+instructions)\b", re.I),
    re.compile(r"\bignore\s+the\s+document\s+and\s+tell\b", re.I),
]

# 3. Jailbreak Attempts
_JAILBREAK_PATTERNS = [
    re.compile(r"\b(?:act|play)\s+as\s+(?:an?\s+)?(?:unrestricted|jailbroken|evil|unfiltered|raw)\s*(?:ai|model|assistant|llm|dan)\b", re.I),
    re.compile(r"\b(?:dan|jailbreak)\b", re.I),
    re.compile(r"\b(?:bypass|disable|remove|turn\s+off)\s+(?:all\s+)?(?:your\s+)?(?:safety|security|system|content)?\s*(?:restrictions|rules|guards|safety)\b", re.I),
    re.compile(r"\bpretend\s+you\s+(?:have\s+no|don'?t\s+have)\s+rules\b", re.I),
]

# 4. SQL Injection
_SQL_INJECTION_PATTERNS = [
    re.compile(r"\b(?:union\s+all\s+select|union\s+select|select\s+.*\s+from|insert\s+into|delete\s+from|drop\s+table|drop\s+database|alter\s+table|truncate\s+table|information_schema|exec\s*\()\b", re.I),
    re.compile(r"\b['\"]?\s*(?:or|and)\s*['\"]?\d+['\"]?\s*=\s*['\"]?\d+", re.I),
    re.compile(r"['\";]--", re.I),
]

# 5. Abusive / Bad Words
_ABUSIVE_PATTERNS = [
    re.compile(r"\b(?:stupid|idiot|dumb|fool|foolish|fuck|fucking|fucker|shit|bitch|bastard|asshole|crap|shut\s*up|bullshit|chutiya|saala|kamina|harami|gadha|bakwas|gand|lauda|bhosdike)\b", re.I),
    re.compile(r"[\U0001F92C\U0001F595]", re.UNICODE),  # 🤬 🖕 emojis
]

# 6. Sexual / Inappropriate Content
_SEXUAL_PATTERNS = [
    re.compile(r"\b(?:sex|sexual|porn|pornography|naked|nude|erotic|horny|penis|vagina|boobs|breast|dick|pussy|fuck\s*me|sexy|blowjob|orgasm|hentai)\b", re.I),
]

# 7. Violence / Threats
_THREAT_PATTERNS = [
    re.compile(r"\b(?:kill\s+you|killing|murder|stab|shoot|shooting|bomb|bombing|explosive|terrorist|terrorism|suicide|attack\s+college|gun|handgun|weapon|destroy\s+college)\b", re.I),
]

# 8. Off-Topic Query Indicators
_OFF_TOPIC_PATTERNS = [
    # Sports trivia
    re.compile(r"\b(?:who\s+won|score\s+of|world\s+cup|match\s+result|ipl|champion\s+league|football\s+match|cricket\s+match)\b", re.I),
    # Programming & Software requests
    re.compile(r"\b(?:write\s+(?:a\s+)?(?:python|java|c\+\+|javascript|html|code|script)|snake\s+game|coding|debug\s+my\s+code)\b", re.I),
    # Recipes & Cooking
    re.compile(r"\b(?:recipe|how\s+to\s+cook|how\s+to\s+bake|ingredients\s+for|pizza|cake|burger)\b", re.I),
    # General Trivia & Math Homework
    re.compile(r"\b(?:who\s+is\s+(?:the\s+)?president\s+of|capital\s+of|weather\s+in|tell\s+me\s+a\s+joke|solve\s+\d+\s*[\+\-\*/]\s*\d+)\b", re.I),
]

# Admission / SXCA Domain Keywords (If present, query is NEVER off-topic)
_SXCA_DOMAIN_KEYWORDS = re.compile(
    r"\b(sxca|xavier|xaviers|college|admission|admissions|course|courses|fee|fees|eligibility|eligibility\s*criteria|"
    r"bca|bcom|bsc|msc|ma|ba|bba|bps|aiml|bda|data\s*science|physics|chemistry|maths|statistics|biotechnology|biochemistry|"
    r"botany|zoology|microbiology|psychology|english|economics|gujarati|sociology|hostel|faculty|professor|department|"
    r"principal|director|rector|hod|nisarg|david|pravida|sebastian|antony|campus|library|sports|placement|scholarship|"
    r"exam|cia|cai|semester|holiday|vacation|office|timing|contact|helpline|phone|address|location|hi|hello|namaste|namste)\b",
    re.I,
)


def _is_gibberish(text: str) -> bool:
    """Detects random keyboard mash, non-words, or numeric noise."""
    cleaned = text.strip().lower()
    if not cleaned:
        return False

    words = cleaned.split()
    if len(words) == 1:
        w = words[0]
        # Ignore short valid words or course codes
        if len(w) <= 3 or w in ("bca", "bcom", "bsc", "msc", "bba", "bps", "aiml", "bda", "sxca"):
            return False
        # Repeated character (e.g. "aaaaaaa", "zzzzzz")
        if len(set(w)) == 1:
            return True
        # Keyboard walks / random consonant clusters (e.g. "asdfghjkl", "qwertyuiop", "zxcvbnm", "qwrxzyt")
        if w in ("asdfghjkl", "qwertyuiop", "zxcvbnm", "lkjhgfdsa", "poiuytrewq"):
            return True
        # Vowel ratio check for longer single words
        vowels = set("aeiou")
        v_count = sum(1 for char in w if char in vowels)
        if len(w) >= 7 and (v_count / len(w)) < 0.12:
            return True
        # Repeating 2-char tokens (e.g. "abababababab")
        if len(w) >= 8 and len(set(w[i:i+2] for i in range(0, len(w)-1, 2))) <= 1:
            return True

    # Pure numeric strings without query context (e.g. "1234567890123")
    if cleaned.isdigit() and len(cleaned) >= 6:
        return True

    return False


def check_prompt_safety(question: str) -> dict[str, Any]:
    """Inspects a user query against safety policies and guardrails.

    Returns:
        dict: {
            "blocked": bool,
            "reason": str | None,
            "response": str | None
        }
    """
    if not question or not isinstance(question, str):
        return {"blocked": False, "reason": None, "response": None}

    norm = _normalize_text(question)

    # 1. Bad / Abusive Words (Polite Warning)
    for pattern in _ABUSIVE_PATTERNS:
        if pattern.search(norm) or pattern.search(question):
            return {
                "blocked": True,
                "reason": "abusive",
                "response": RESPONSE_ABUSIVE,
            }

    # 2. Sexual / Inappropriate Content (Refuse & Redirect)
    for pattern in _SEXUAL_PATTERNS:
        if pattern.search(norm) or pattern.search(question):
            return {
                "blocked": True,
                "reason": "sexual",
                "response": RESPONSE_SEXUAL,
            }

    # 3. Threats / Violent Content (Refuse)
    for pattern in _THREAT_PATTERNS:
        if pattern.search(norm) or pattern.search(question):
            return {
                "blocked": True,
                "reason": "threat",
                "response": RESPONSE_THREAT,
            }

    # 4. System Prompt / Secrets Leakage (Block)
    for pattern in _SYSTEM_PROMPT_PATTERNS:
        if pattern.search(norm) or pattern.search(question):
            return {
                "blocked": True,
                "reason": "system_prompt_leak",
                "response": RESPONSE_SYSTEM_PROMPT,
            }

    # 5. Prompt Injection (Block)
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(norm) or pattern.search(question):
            return {
                "blocked": True,
                "reason": "prompt_injection",
                "response": RESPONSE_INJECTION_JAILBREAK,
            }

    # 6. Jailbreak Attempt (Block)
    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(norm) or pattern.search(question):
            return {
                "blocked": True,
                "reason": "jailbreak",
                "response": RESPONSE_INJECTION_JAILBREAK,
            }

    # 7. SQL Injection (Block)
    for pattern in _SQL_INJECTION_PATTERNS:
        if pattern.search(norm) or pattern.search(question):
            return {
                "blocked": True,
                "reason": "sql_injection",
                "response": RESPONSE_SQL_INJECTION,
            }

    # 8. Gibberish / Random Text (Clarify)
    if _is_gibberish(question):
        return {
            "blocked": True,
            "reason": "gibberish",
            "response": RESPONSE_GIBBERISH,
        }

    # 9. Off-Topic Question (Redirect to SXCA Admissions)
    # Check if query matches an off-topic domain AND has no SXCA domain keywords
    if not _SXCA_DOMAIN_KEYWORDS.search(question) and not _SXCA_DOMAIN_KEYWORDS.search(norm):
        for pattern in _OFF_TOPIC_PATTERNS:
            if pattern.search(norm) or pattern.search(question):
                return {
                    "blocked": True,
                    "reason": "off_topic",
                    "response": RESPONSE_OFF_TOPIC,
                }

    # 10. Normal Admission Question (Pass through)
    return {"blocked": False, "reason": None, "response": None}


def is_prompt_injection(question: str) -> bool:
    """Returns True if the text contains a prompt injection or security attack."""
    return check_prompt_safety(question)["blocked"]


def inspect_query(question: str) -> dict[str, Any]:
    """Alias for check_prompt_safety."""
    return check_prompt_safety(question)
