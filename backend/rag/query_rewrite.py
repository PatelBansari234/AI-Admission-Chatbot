"""Understands a student's question before retrieval sees it: resolves a
follow-up like "what about the fee?" into a standalone search query, tracks
which course/qualification the conversation has established, and flags
whether the question should also be checked against the structured
course_eligibility table (see eligibility.py) - a fee/eligibility question
about a specific course is answered exactly from that table, sidestepping
the chunk-boundary ambiguity RAG retrieval has on wide fee tables (see the
README's fee-table limitation).

Runs on every question, not just follow-ups - "Can a BCA graduate apply for
MSc AI?" needs course/qualification extraction and intent classification
even as someone's very first message, so this can't be gated on
conversation history the way pure follow-up resolution could be.

Fails open throughout: if the model's output isn't parseable, the original
question is used unchanged and needs_structured_lookup defaults to False -
a missed structured lookup just falls back to the RAG path that existed
before this module did, never a broken request.
"""

import json
import re

from backend.llm import gemini, qwen

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = """You read a student's latest question, in the context of the
recent conversation, and extract structured information about it. Reply
with ONLY a JSON object, no other text:

{"query": "...", "course": "... or null", "qualification": "... or null", "needs_structured_lookup": true or false}

query:
- A compact, keyword-style search query for the question - "MSc Artificial
  Intelligence fee", not "What is the fee for MSc Artificial Intelligence?".
  The extra question words add nothing a search index can use.
- If the latest question uses a pronoun or leaves something implicit ("it",
  "that", "the fee", "what about..."), name the actual thing explicitly,
  using the most recent course/topic named in the conversation - even if it
  was named by the assistant, not the student. If the question is already
  standalone, just turn it into search keywords.

course / qualification:
- Report "course" whenever a specific course has been named anywhere in the
  conversation so far (student or assistant), not only if the latest
  question names it again. Always report ONE clean, full form ("MSc
  Artificial Intelligence") - ALWAYS expand a known abbreviation ("MSc AI")
  to that full name, even on a fresh question with no prior conversation at
  all - this does not depend on the full name having been said anywhere
  first, it's a fixed, known abbreviation. Never write both forms together
  in parentheses. This includes casual/colloquial references too, not just
  the standard abbreviation - "AI master's", "the AI PG course", "masters
  in artificial intelligence" all mean "MSc Artificial Intelligence" and
  must be reported as that exact full name, not copied as the casual
  phrase - a casual phrase OR an unexpanded abbreviation copied verbatim
  won't match anything in the admission records lookup (confirmed: "MSc
  AI" alone does not match the stored name "MSc-Artificial Intelligence" -
  neither contains the other as a substring).
- Report "qualification" whenever the student has stated their own prior
  qualification/stream (e.g. "12th Science", "BCA", "completed Commerce") -
  what THEY have, not what a course requires.
- The college's admission records only classify students by broad stream
  (Science / Commerce / Arts), not by individual subject. If the student
  names specific 12th-standard subjects instead of a stream (Maths,
  Biology, Physics, Chemistry -> Science; Accountancy, Business Studies,
  Economics -> Commerce; History, Political Science, Psychology -> Arts),
  report "qualification" as the underlying stream, not the raw subject
  list - "I have Maths but not Biology" still means qualification="Science"
  (Maths and Biology are both Science-stream subjects), because that's the
  only granularity admission records actually match against. This is about
  finding the right stream to search for, not about deciding whether the
  student is eligible - that judgment call belongs to the answer step, not
  here.
- Never invent either - use null, not a guess, if it was never actually
  stated.

needs_structured_lookup:
- true if the question is about whether someone qualifies for a course,
  which courses fit a given qualification/stream, or a specific course's
  eligibility/admission requirement/fee - anything a college's admission
  eligibility+fee table would answer.
- false for everything else (course content/curriculum, faculty, general
  college info, documents required, dates, etc.) - including a bare
  greeting or a question with no course/qualification/admission angle at
  all.
- true regardless of phrasing or language - "Can a BCA graduate apply for
  MSc AI?", "I completed BCA, am I eligible for AI master's?", "BCA ke baad
  MSc AI kar sakte hai?" must all get true, because they ask the same thing.

Examples:

Recent conversation: (none yet)
Known context so far: course=null, qualification=null
Latest question: Can a BCA graduate apply for MSc AI?
Output:
{"query": "MSc Artificial Intelligence eligibility BCA", "course": "MSc Artificial Intelligence", "qualification": "BCA", "needs_structured_lookup": true}

Recent conversation: (none yet)
Known context so far: course=null, qualification=null
Latest question: Does St. Xavier's College Mumbai offer MSc AI?
Output:
{"query": "MSc Artificial Intelligence Mumbai", "course": "MSc Artificial Intelligence", "qualification": null, "needs_structured_lookup": true}
(course MUST still be expanded to "MSc Artificial Intelligence" here even
though the question is mainly about a location, not eligibility/fee - a
real, confirmed failure left course as the bare abbreviation "MSc AI" here,
which matches nothing in the admission records lookup, so the question
fell through to general search instead and produced an answer not actually
grounded in what was retrieved)

Recent conversation:
Student: What about MSc AI?
Assistant: MSc AI is a postgraduate program offered by the Department of Data Science.
Known context so far: course=null, qualification=null
Latest question: What is the fee?
Output:
{"query": "MSc Artificial Intelligence fee", "course": "MSc Artificial Intelligence", "qualification": null, "needs_structured_lookup": true}
(wrong would be: leaving "the fee" unresolved, dropping the established course, or setting needs_structured_lookup false for a fee question)

Recent conversation: (none yet)
Known context so far: course=null, qualification=null
Latest question: Who teaches in the Data Science department?
Output:
{"query": "Data Science department faculty", "course": null, "qualification": null, "needs_structured_lookup": false}
(false because this is a faculty question, not an eligibility/fee one, even though it names a department)

Recent conversation: (none yet)
Known context so far: course=null, qualification=null
Latest question: I completed BCA. Am I eligible for AI master's?
Output:
{"query": "MSc Artificial Intelligence eligibility BCA", "course": "MSc Artificial Intelligence", "qualification": "BCA", "needs_structured_lookup": true}
(wrong would be: {"course": "AI master's", ...} - a casual phrase copied
as-is instead of resolved to the program's actual full name)

Recent conversation: (none yet)
Known context so far: course=null, qualification=null
Latest question: I have Maths but not Biology, which courses can I apply for?
Output:
{"query": "courses eligible Science stream", "course": null, "qualification": "Science", "needs_structured_lookup": true}
(true because "which courses can I apply for" is exactly the "which courses
fit a given qualification" case; qualification is "Science" - the stream
that Maths/Biology belong to - not "Maths", since that's the granularity
admission records actually use; wrong would be needs_structured_lookup:
false just because the phrasing doesn't look like a typical eligibility
question, or qualification: "Maths, not Biology" which won't match
anything)

Recent conversation: (none yet)
Known context so far: course=null, qualification=null
Latest question: What is the admission procedure?
Output:
{"query": "admission procedure", "course": null, "qualification": null, "needs_structured_lookup": false}
(course MUST stay null - the question names no course at all, so there is
nothing to report; a real, confirmed failure here was reporting
course: "MSc Artificial Intelligence" for this exact question with no
course ever mentioned, which then biased the search query and narrowed a
general "how do I apply" question into an answer about one specific
program the student never asked about)"""


_SPELLING_CORRECTIONS = [
    (r"\bcource\b|\bcousre\b|\bcourses\b", "course"),
    (r"\bphyisic\b|\bphyisics\b|\bphyiscs\b|\bphyisc\b|\bphyics\b|\bphysic\b|\bphysix\b|\bphy\b|\bphisics\b|\bphisic\b|\bphys\b|ફિઝિક્સ\S*|ભૌતિકશાસ્ત્ર\S*", "physics"),
    (r"\bchem\b|\bchemi\b|\bchemestry\b|\bchmistry\b|\bchemis\b|\bchemistr\b|કેમિસ્ટ્રી\S*|રસાયણશાસ્ત્ર\S*", "chemistry"),
    (r"\bmath\b|\bmaths\b|\bmathematic\b|\bmathemetics\b|મેથ્સ\S*|ગણિત\S*", "mathematics"),
    (r"\bstat\b|\bstats\b|\bstatastic\b|\bstatastics\b", "statistics"),
    (r"\bbt\b|\bbiotech\b", "biotechnology"),
    (r"\bbc\b|\bbiochem\b", "biochemistry"),
    (r"\bmicro\b|\bmicrobio\b", "microbiology"),
    (r"\bpsych\b|\bpsycho\b|\bpsychlogy\b|\bpsycology\b", "psychology"),
    (r"\bsocio\b|\bsociolgy\b", "sociology"),
    (r"\bsans\b|\bsanskrt\b", "sanskrit"),
    (r"\bguj\b|\bgujrati\b", "gujarati"),
    (r"\badmision\b|\badmisn\b|\badmsn\b|\badmisson\b|\badmissions\b|એડમિશન\S*|પ્રવેશ\S*", "admission"),
    (r"\beligiblity\b|\belgibility\b|\beligble\b|\belgible\b|લાયકાત\S*", "eligibility"),
    (r"\baply\b|\baplying\b|\baplication\b|\bapplictn\b|અરજી\S*", "apply"),
    (r"\bdoc\b|\bdocs\b|\bdocumnt\b|\bdocumnts\b|\bdocumen\b|દસ્તાવેજ\S*|કાગળો\S*", "documents"),
    (r"\breq\b|\breqd\b|\brequirment\b|\brequirments\b", "required"),
    (r"\bsci\b|\bscince\b|\bscie\b", "science"),
    (r"\bcomm\b|\bcommrce\b|\bcomerce\b", "commerce"),
    (r"\bfe\b|\bfees\b|ફી\S*", "fee"),
    (r"મારે\S*|લેવું\S*|લેવુ\S*|કરવું\S*", "i want"),
]


def normalize_student_text(text: str) -> str:
    """Normalize spelling mistakes, typos, slang, and short forms."""
    t = text.strip()
    for pattern, replacement in _SPELLING_CORRECTIONS:
        t = re.sub(pattern, replacement, t, flags=re.I)
    return t


_COURSE_PATTERNS = [
    (r"\b(m\.?sc\.?\s*(?:in\s*)?artificial\s*intelligence|msc\s*ai|ai\s*master'?s?)\b", "MSc-Artificial Intelligence"),
    (r"\b(m\.?sc\.?\s*(?:in\s*)?big\s*data\s*analytics|msc\s*bda|big\s*data\s*analytics)\b", "MSc-Big Data Analytics"),
    (r"\b(m\.?sc\.?\s*(?:in\s*)?biochemistry)\b", "MSc-Biochemistry"),
    (r"\b(m\.?sc\.?\s*(?:in\s*)?biotechnology)\b", "MSc-Biotechnology"),
    (r"\b(m\.?sc\.?\s*(?:in\s*)?botany)\b", "MSc-Botany"),
    (r"\b(m\.?sc\.?\s*(?:in\s*)?chemistry\s*\(?organic\)?|msc\s*chemistry\s*organic)\b", "MSc-Chemistry (Organic)"),
    (r"\b(m\.?sc\.?\s*(?:in\s*)?chemistry\s*\(?analytical\)?|msc\s*chemistry\s*analytical)\b", "MSc-Chemistry (Analytical)"),
    (r"\b(m\.?sc\.?\s*(?:in\s*)?chemistry)\b", "MSc-Chemistry (Organic)"),
    (r"\b(m\.?sc\.?\s*(?:in\s*)?mathematics|msc\s*maths?)\b", "MSc-Mathematics"),
    (r"\b(m\.?sc\.?\s*(?:in\s*)?microbiology)\b", "MSc-Microbiology"),
    (r"\b(m\.?sc\.?\s*(?:in\s*)?physics)\b", "MSc-Physics"),
    (r"\b(m\.?a\.?\s*(?:in\s*)?clinical\s*psychology|ma\s*psychology)\b", "MA-Clinical Psychology"),
    (r"\b(m\.?a\.?\s*(?:in\s*)?english|ma\s*english)\b", "MA-English"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?artificial\s*intelligence(?:\s*&\s*machine\s*learning)?|bsc\s*aiml|bsc\s*ai)\b", "BSc-Artificial Intelligence & Machine Learning"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?information\s*technology|bsc\s*it|bsc\s*cs|bsc\s*comp.*sci.*)\b", "BSc-Information Technology"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?biotechnology|bsc\s*bt)\b", "BSc-Biotechnology"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?biochemistry\s*with\s*voc\.?\s*bt|bsc\s*biochem.*voc)\b", "BSc-Biochem. with Voc. BT"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?biochemistry)\b", "BSc-Biochemistry"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?botany)\b", "BSc-Botany"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?chem.*with\s*voc\.?\s*ic|bsc\s*chem.*voc)\b", "BSc-Chem. with Voc. IC"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?chemistry)\b", "BSc-Chemistry"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?electronics)\b", "BSc-Electronics"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?mathematics|bsc\s*maths?)\b", "BSc-Mathematics"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?physics|bsc\s*phy\b)\b", "BSc-Physics"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?statistics|bsc\s*stats?)\b", "BSc-Statistics"),
    (r"\b(b\.?s\.?c\.?\s*(?:in\s*)?zoology)\b", "BSc-Zoology"),
    (r"\b(b\.?c\.?a\.?|bachelor of computer applications?)\b", "BCA"),
    (r"\b(b\.?s\.?-?business\s*administration|bba|business\s*administration)\b", "BS-Business Administration"),
    (r"\b(b\.?com\.?-?bps|bcom\s*bps)\b", "BCom-BPS"),
    (r"\b(b\.?com\.?|bachelor of commerce)\b", "BCom (General)"),
    (r"\b(b\.?p\.?a\.?|theatre\s*&\s*production)\b", "BPA-Theatre & Production Management"),
    (r"\b(b\.?a\.?\s*(?:in\s*)?economics|ba\s*economics)\b", "BA-Economics"),
    (r"\b(b\.?a\.?\s*(?:in\s*)?english|ba\s*english)\b", "BA-English"),
    (r"\b(b\.?a\.?\s*(?:in\s*)?gujarati|ba\s*gujarati)\b", "BA-Gujarati"),
    (r"\b(b\.?a\.?\s*(?:in\s*)?psychology|ba\s*psychology)\b", "BA-Psychology"),
    (r"\b(b\.?a\.?\s*(?:in\s*)?sanskrit|ba\s*sanskrit)\b", "BA-Sanskrit"),
    (r"\b(b\.?a\.?\s*(?:in\s*)?sociology|ba\s*sociology)\b", "BA-Sociology & Development"),
    # General subjects (when level is not specified)
    (r"\bphysics\b", "BSc-Physics"),
    (r"\bchemistry\b", "BSc-Chemistry"),
    (r"\bmathematics\b", "BSc-Mathematics"),
    (r"\bcomputer science\b", "BSc-Information Technology"),
]

_STREAM_PATTERNS = [
    (r"\b(12th|hsc|class\s*12|plus\s*two|pass\s*12th?)\s*(?:in\s*)?science\b|\bscience\s*stream\b", "Science"),
    (r"\b(12th|hsc|class\s*12|plus\s*two|pass\s*12th?)\s*(?:in\s*)?commerce\b|\bcommerce\s*stream\b", "Commerce"),
    (r"\b(12th|hsc|class\s*12|plus\s*two|pass\s*12th?)\s*(?:in\s*)?arts\b|\barts\s*stream\b", "Arts"),
]


def _fast_understand(question: str) -> dict | None:
    """Fast-path for questions: normalizes spelling, extracts course, stream, and intent instantly."""
    norm_q = normalize_student_text(question)

    course = None
    for pattern, name in _COURSE_PATTERNS:
        if re.search(pattern, norm_q, re.I):
            course = name
            break

    qualification = None
    for pattern, stream in _STREAM_PATTERNS:
        if re.search(pattern, norm_q, re.I):
            qualification = stream
            break

    # 1. Source queries
    is_source_intent = bool(re.search(
        r"^(where\s+(?:did|do)\s+you\s+find\s+(?:this|that|information|it|info)|give\s+me\s+(?:the\s+)?(?:source|link|url|reference)|where\s+is\s+this\s+(?:information|from|data)|what\s+is\s+the\s+source|source\??|link\??|give\s*link|show\s*source|where\s+did\s+you\s+find\s+information\s+about\s+.*)\b",
        norm_q,
        re.I,
    ))
    if is_source_intent:
        return {
            "query": norm_q,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "source_query",
        }

    # 2. Application link / Portal link queries (e.g. "Can you give me a link to apply MSc AI?")
    is_apply_link = bool(re.search(
        r"\b((?:give\s*(?:me)?|send|share|what\s*is|where\s*is|provide|can\s*you\s*(?:give|send|share|provide))\s*(?:me\s*)?(?:the\s*|a\s*)?(?:link|url|portal|website)\s*(?:to|for)\s*(?:apply|application|admission|register)|(?:link|url|portal)\s*(?:to\s*apply|for\s*application|for\s*admission)|apply\s*(?:link|portal|url)|admission\s*portal\s*link|how\s*to\s*get\s*(?:the\s*)?application\s*link)\b|"
        r"\b(?:link\s*to\s*apply|application\s*link|portal\s*link)\b",
        norm_q,
        re.I,
    ))
    if is_apply_link:
        return {
            "query": norm_q,
            "course": course,
            "qualification": qualification,
            "needs_structured_lookup": bool(course),
            "intent": "application_link",
        }

    # 3. Events / Fests / Departmental Activities
    is_events_intent = bool(re.search(
        r"\b(events?|upcoming\s*events?|what\s*events?|college\s*events?|departmental\s*activities|fests?|culfest|sportsfest|seminars?|workshops?|webinars?|conferences?|celebrations?|guest\s*lectures?|activities\s*(?:happening|at\s*the\s*college|in\s*college)?|functions?)\b",
        norm_q,
        re.I,
    ))

    # 4. Curriculum / Subject / Syllabus / Credits intent
    is_curriculum_intent = bool(re.search(
        r"\b(syllabus|curriculum|subject\s*details|subject\s*outline|paper\s*code|course\s*outcome|course\s*outcomes|course\s*structure|course\s*content|credits?|credit\s*hours|learning\s*hours|topics?\s*(?:covered|in)|modules?|what\s+subjects?\s+(?:are|in)|list\s+of\s+subjects?|semester\s+[1-8]\s+subjects?|practical|theory\s+and\s+lab)\b|"
        r"\b(?:details?\s*(?:about|of|regarding)|tell\s*me\s*about|what\s*is|explain|info\s*about|give\s*details?\s*of|what\s*are\s*the\s*details?\s*(?:of|about)?)\s+(?:the\s+)?(python|python\s*programming|machine\s*learning|nlp|natural\s*language\s*processing|computer\s*vision|data\s*science|deep\s*learning|artificial\s*intelligence|big\s*data|data\s*analytics|dbms|database\s*management|java|c\+\+|data\s*structures|algorithms|operating\s*systems|cyber\s*security|cloud\s*computing)\b|"
        r"\b(python\s*programming|python|machine\s*learning|nlp|computer\s*vision|data\s*structures)\s+(?:details|syllabus|curriculum|credits|course|subject|paper)\b|"
        r"\b(python\s*programming|python|machine\s*learning|nlp|computer\s*vision|data\s*structures)\b",
        norm_q,
        re.I,
    ))

    # 5. Calendar / Holiday list intent
    is_calendar_holiday = bool(re.search(
        r"\b(holiday\s*list|holidays?|vacation|diwali\s*vacation|christmas\s*holidays?|academic\s*calendar|when\s*(?:does|do)\s*(?:the\s*)?college\s*(?:reopen|open|start)|when\s*(?:do|does)\s*classes\s*start|term\s*dates|exam\s*schedule|exam\s*dates|semester\s*commencement|working\s*days|janmashtami|gandhi\s*jayanti|independence\s*day|republic\s*day|good\s*friday|eid|muharram|raksha\s*bandhan|holi|mahashivratri)\b",
        norm_q,
        re.I,
    ))

    # 6. Faculty / Person / Professor intent
    is_faculty_intent = bool(re.search(
        r"\b(who\s+is\s+[A-Za-z]+|tell\s+me\s+about\s+(?:dr\.?|prof\.?|mr\.?|ms\.?|[A-Z][a-z]+)|faculty\s*(?:members?|details?|list)?|professors?|teachers?|head\s+of\s+(?:the\s+)?department|hod\s+of|coordinator\s+of|principal|vice\s*principal|who\s+teaches\b)\b",
        norm_q,
        re.I,
    ))

    # 7. Notices / Announcements
    is_notices_intent = bool(re.search(r"\b(notices?|circulars?|announcements?|exam\s*notice|supplementary\s*exam\s*form)\b", norm_q, re.I))

    # 8. Contact / Department info
    is_contact_intent = bool(re.search(r"\b(contact\s*(?:info|details|number)?|email\s*address|phone\s*number|where\s+is\s+(?:the\s+)?college|location\s+of\s+college|campus\s*address|address\s+of)\b", norm_q, re.I))

    # 9. Application / how to apply intent
    is_how_to_apply = bool(re.search(
        r"\b(how\s*(?:can|do)\s*i\s*apply|how\s*to\s*apply|how\s*to\s*get\s*admission|how\s*to\s*take\s*admission|how\s*to\s*register|application\s*(?:process|steps|procedure|link|form|portal)|where\s*(?:to|can\s*i)\s*apply|admission\s*(?:process|procedure|steps))\b",
        norm_q,
        re.I,
    ))
    is_fee_intent = bool(re.search(r"\b(fee|fees|cost|how\s*much\s*(?:is|are|does|to)?|charges|tuition)\b", norm_q, re.I))
    is_doc_intent = bool(re.search(r"\b(documents?|doc|docs|certificate|marksheet|required\s*documents?|which\s*docs)\b", norm_q, re.I))
    is_elig_intent = False
    if not is_how_to_apply and not is_curriculum_intent and not is_events_intent:
        is_elig_intent = bool(re.search(r"\b(eligibility|eligible|criteria|qualify|requirements|which\s*course|which\s*cource|what\s*can\s*i\s*do|can\s*i\s*(?:take|apply|opt|join)|can\s*i\s*do)\b", norm_q, re.I))

    is_broad_counselling = False
    if not (is_curriculum_intent or is_events_intent or is_calendar_holiday or is_faculty_intent or is_notices_intent or is_contact_intent or is_how_to_apply or is_fee_intent or is_doc_intent or is_elig_intent):
        is_broad_counselling = bool(re.search(
            r"\b(i\s*want\b.*(?:admission|apply|take\s*admission|join|enroll|do|study)|admission\b.*(?:in|for|take|want|levu|chahiye|લેવું|લેવુ))\b",
            norm_q,
            re.I,
        ))

    # Also handle short admission intents like "physics admission", "bca admission", "i want bca", "i want physics"
    if not is_broad_counselling and course and not (is_curriculum_intent or is_events_intent or is_calendar_holiday or is_faculty_intent or is_notices_intent or is_contact_intent or is_fee_intent or is_how_to_apply or is_doc_intent or is_elig_intent):
        short_course = course.lower().split("-")[-1]
        if re.search(r"^\s*(?:i\s*want\s*)?" + re.escape(short_course) + r"\s*(?:admission|apply)?\s*$", norm_q, re.I) or \
           re.search(r"^\s*admission\s*(?:in\s*|for\s*)?" + re.escape(short_course) + r"\s*$", norm_q, re.I) or \
           re.search(r"^\s*i\s*want\s+" + re.escape(short_course) + r"\s*$", norm_q, re.I):
            is_broad_counselling = True

    # 1. Routing Events
    if is_events_intent and not is_broad_counselling:
        if course:
            query = f"{course} events activities expert talk workshops seminars guest lectures handbook"
        else:
            query = "college events departmental activities celebrations culfest sportsfest seminars workshops guest lectures academic calendar 2025 2026 2027"
        return {
            "query": query,
            "course": course,
            "qualification": qualification,
            "needs_structured_lookup": False,
            "intent": "events",
        }

    # 2. Routing Curriculum
    if is_curriculum_intent and not is_broad_counselling:
        if re.search(r"\b(python|python\s*programming)\b", norm_q, re.I):
            query = "Python Programming syllabus curriculum credits course code PAI-1803 PBD-1806 learning hours outcomes"
        elif course:
            query = f"{course} syllabus curriculum course structure papers credits"
        else:
            clean_sub = re.sub(r"\b(i\s*want|the|details?|about|of|regarding|tell\s*me|what\s*is|what\s*are|please|give\s*me)\b", "", norm_q, flags=re.I).strip()
            query = f"{clean_sub} syllabus curriculum course details credits"
        return {
            "query": query,
            "course": course,
            "qualification": qualification,
            "needs_structured_lookup": False,
            "intent": "curriculum",
        }

    # 3. Routing Calendar / Holiday
    if is_calendar_holiday:
        hol_m = re.search(r"\b(janmashtami|diwali|christmas|gandhi\s*jayanti|independence\s*day|republic\s*day|good\s*friday|eid|muharram|raksha\s*bandhan|holi|mahashivratri)\b", norm_q, re.I)
        if hol_m:
            specific_hol = hol_m.group(1)
            query = f"{specific_hol} holiday academic calendar handbook dates schedule 2024 2025 2026 2027"
        else:
            query = "academic calendar holiday list schedule vacation dates reopening 2024 2025 2026"
        return {
            "query": query,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "calendar_holiday",
        }

    # 4. Routing Faculty / Person
    if is_faculty_intent:
        ent_m = re.search(r"\b(?:who\s+is|about|tell\s+me\s+about|dr\.?|prof\.?|mr\.?|ms\.?)\s+([A-Za-z\s]+?)(?:\?|\.|$)", norm_q, re.I)
        if ent_m and len(ent_m.group(1).strip()) >= 3:
            ent = ent_m.group(1).strip()
            query = f"{ent} faculty Assistant Professor Coordinator Department of Data Sciences"
        else:
            query = f"{norm_q} faculty members professors teachers department head"
        return {
            "query": query,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "faculty",
        }

    # 5. Routing Notices
    if is_notices_intent:
        return {
            "query": "college notices announcements circulars latest updates",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "general_website",
        }

    # 6. Routing Contact / Department
    if is_contact_intent:
        return {
            "query": f"{norm_q} contact email address location phone office",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "general_website",
        }

    intent = "general"
    if is_fee_intent and is_elig_intent:
        intent = "fee_and_eligibility"
    elif is_fee_intent and not is_broad_counselling:
        intent = "fee"
    elif is_how_to_apply:
        intent = "process"
    elif is_elig_intent and not is_broad_counselling:
        intent = "eligibility"
    elif is_doc_intent:
        intent = "documents"
    elif is_broad_counselling:
        intent = "counselling"

    if course:
        needs_lookup = intent in ("counselling", "fee", "eligibility", "fee_and_eligibility", "process", "documents")
        q_type = intent if intent != "general" else "overview"
        query = f"{course} {q_type}" if needs_lookup else norm_q
        return {
            "query": query,
            "course": course,
            "qualification": qualification,
            "needs_structured_lookup": needs_lookup,
            "intent": intent,
        }

    if qualification and not course:
        return {
            "query": f"courses eligible {qualification} stream",
            "course": None,
            "qualification": qualification,
            "needs_structured_lookup": True,
            "intent": "eligibility",
        }

    if is_doc_intent:
        return {
            "query": "documents required for admission",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "documents",
        }

    if is_how_to_apply:
        return {
            "query": "admission procedure and application process",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "process",
        }

    return {
        "query": norm_q,
        "course": None,
        "qualification": None,
        "needs_structured_lookup": False,
        "intent": "general",
    }


def _build_prompt(question: str, recent_turns: list[dict], context: dict | None) -> list[dict]:
    convo_lines = []
    for turn in recent_turns[-6:]:
        role = "Student" if turn.get("role") == "user" else "Assistant"
        convo_lines.append(f"{role}: {turn.get('content', '').strip()}")
    convo = "\n".join(convo_lines) or "(none yet)"

    known_course = (context or {}).get("course") or "null"
    known_qual = (context or {}).get("qualification") or "null"

    user_msg = (
        f"Recent conversation:\n{convo}\n\n"
        f"Known context so far: course={known_course}, qualification={known_qual}\n\n"
        f"Latest question: {question}"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def _fallback(question: str) -> dict:
    return {"query": question, "course": None, "qualification": None, "needs_structured_lookup": False, "intent": "general"}


def understand(question: str, recent_turns: list[dict], context: dict | None) -> dict:
    """Returns {"query", "course", "qualification", "needs_structured_lookup", "intent"}.
    "query" always falls back to the original question on any failure."""
    fast = _fast_understand(question)
    # Non-admission and dedicated intents use deterministic fast-path
    if fast is not None and fast.get("intent") in ("source_query", "application_link", "events", "curriculum", "calendar_holiday", "faculty", "general_website"):
        return fast

    # Standalone queries that already resolved a course directly
    if fast is not None and fast.get("course"):
        return fast

    norm_q = normalize_student_text(question)

    # Multi-turn context resolution: if a course is already known from context or previous turns,
    # resolve pronoun/follow-up questions (e.g. "what is its fee?", "eligibility?", "how to apply?")
    known_course = (context or {}).get("course")
    if not known_course and recent_turns:
        # Check student/user turns first from most recent to oldest
        for t in reversed(recent_turns):
            if t.get("role") == "user":
                content = t.get("content", "")
                fast_t = _fast_understand(content)
                if fast_t and fast_t.get("course"):
                    known_course = fast_t.get("course")
                    break
        # If not found in user turns, inspect recent turns
        if not known_course:
            for t in reversed(recent_turns):
                content = t.get("content", "")
                fast_t = _fast_understand(content)
                if fast_t and fast_t.get("course"):
                    known_course = fast_t.get("course")
                    break

    if known_course:
        if fast is not None and fast.get("intent") in ("source_query", "application_link", "events", "curriculum", "calendar_holiday", "faculty", "general_website"):
            return fast

        is_source_intent = bool(re.search(
            r"^(where\s+(?:did|do)\s+you\s+find|give\s+me\s+(?:the\s+)?(?:source|link|url|reference)|what\s+is\s+the\s+source|source\??|link\??|give\s*link|show\s*source)\b",
            norm_q,
            re.I,
        ))
        if is_source_intent:
            return {"query": norm_q, "course": None, "qualification": None, "needs_structured_lookup": False, "intent": "source_query"}

        is_apply_link = bool(re.search(
            r"\b((?:give\s*(?:me)?|send|share|what\s*is|where\s*is|provide|can\s*you\s*(?:give|send|share|provide))\s*(?:me\s*)?(?:the\s*|a\s*)?(?:link|url|portal|website)\s*(?:to|for)\s*(?:apply|application|admission|register)|(?:link|url|portal)\s*(?:to\s*apply|for\s*application|for\s*admission)|apply\s*(?:link|portal|url)|admission\s*portal\s*link|how\s*to\s*get\s*(?:the\s*)?application\s*link)\b|"
            r"\b(?:link\s*to\s*apply|application\s*link|portal\s*link)\b",
            norm_q,
            re.I,
        ))
        if is_apply_link:
            return {
                "query": norm_q,
                "course": known_course,
                "qualification": (context or {}).get("qualification"),
                "needs_structured_lookup": True,
                "intent": "application_link",
            }

        is_how_to_apply = bool(re.search(
            r"\b(how\s*(?:can|do)\s*i\s*apply|how\s*to\s*apply|how\s*to\s*get\s*admission|how\s*to\s*take\s*admission|how\s*to\s*register|application\s*(?:process|steps|procedure|link|form|portal)|where\s*(?:to|can\s*i)\s*apply|admission\s*(?:process|procedure|steps))\b",
            norm_q,
            re.I,
        ))
        is_fee_intent = bool(re.search(r"\b(fee|fees|cost|how\s*much\s*(?:is|are|does|to)?|charges|tuition)\b", norm_q, re.I))
        is_doc_intent = bool(re.search(r"\b(documents?|doc|docs|certificate|marksheet|required\s*documents?|which\s*docs)\b", norm_q, re.I))
        is_broad_counselling = bool(re.search(
            r"\b(i\s*want\b.*(?:admission|apply|take\s*admission|join|enroll|do|study)|admission\b.*(?:in|for|take|want|levu|chahiye|લેવું|લેવુ)|(?:admission|apply)\s*(?:in|for)?\b)\b",
            norm_q,
            re.I,
        ))

        is_elig_intent = False
        if not is_how_to_apply:
            is_elig_intent = bool(re.search(r"\b(eligibility|eligible|criteria|qualify|requirements|can\s*i\s*(?:take|apply|opt|join))\b", norm_q, re.I))

        has_followup_marker = bool(re.search(r"\b(it|its|this|that|for\s*it|of\s*it)\b", norm_q, re.I)) or len(norm_q.split()) <= 4

        followup_intent = None
        if is_fee_intent and is_elig_intent:
            followup_intent = "fee_and_eligibility"
        elif is_fee_intent:
            followup_intent = "fee"
        elif is_how_to_apply:
            followup_intent = "process"
        elif is_elig_intent:
            followup_intent = "eligibility"
        elif is_doc_intent:
            followup_intent = "documents"
        elif is_broad_counselling:
            followup_intent = "counselling"
        elif has_followup_marker:
            followup_intent = "general"

        if followup_intent:
            needs_lookup = followup_intent in ("counselling", "fee", "eligibility", "fee_and_eligibility", "process", "documents")
            return {
                "query": f"{known_course} {followup_intent}",
                "course": known_course,
                "qualification": (context or {}).get("qualification"),
                "needs_structured_lookup": needs_lookup,
                "intent": followup_intent,
            }

    # If no recent turns and no known course, return fast result if available
    if fast is not None and not recent_turns:
        return fast

    # Optional Gemini multilingual understanding (English, Hindi, Gujarati, Hinglish)
    gemini_res = gemini.understand_multilingual(question, recent_turns, context)
    if gemini_res is not None:
        language = gemini_res.get("language")
        intent = gemini_res.get("intent") or "general"
        course = gemini_res.get("course") or known_course
        confidence = gemini_res.get("confidence")
        needs_lookup = bool(course or intent in ("fee", "eligibility", "counselling", "application", "process", "documents"))
        return {
            "query": question,
            "course": course,
            "qualification": gemini_res.get("qualification") or (context or {}).get("qualification"),
            "needs_structured_lookup": needs_lookup,
            "intent": intent,
            "language": language,
            "confidence": confidence,
        }

    messages = _build_prompt(norm_q, recent_turns, context)
    try:
        raw = qwen.generate(messages, max_new_tokens=35)
    except Exception:
        return _fallback(norm_q)

    match = _JSON_RE.search(raw)
    if not match:
        return _fallback(norm_q)

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _fallback(norm_q)

    query = parsed.get("query")
    if not isinstance(query, str) or not query.strip():
        query = norm_q

    def _clean(value):
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if value and value.lower() != "null" else None

    course = _clean(parsed.get("course")) or known_course
    is_app_intent = bool(re.search(r"\b(i\s*want\b|apply|admission|admission\s*in|join|enroll|take\s*admission)\b", norm_q, re.I))
    intent = "counselling" if is_app_intent and course else "general"

    return {
        "query": query.strip(),
        "course": course,
        "qualification": _clean(parsed.get("qualification")),
        "needs_structured_lookup": bool(parsed.get("needs_structured_lookup")),
        "intent": intent,
    }
