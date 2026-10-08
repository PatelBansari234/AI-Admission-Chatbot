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
from backend.rag import counsellor

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
    (r"\bcouce\b|\bcousre\b|\bcources?\b", "course"),
    (r"\bcomm\b|\bcommrce\b|\bcomerce\b|\bcommiorce\b", "commerce"),
    (r"\bfeculty\b|\bfaculties\b|\bfaculti\b", "faculty"),
    (r"\bnisar\s*vyash\b|\bnisarg\s*vyash\b|\bnisar\s*vyas\b", "nisarg vyas"),
    (r"\bfr\.david\b|\bwho\s+fr\s+david\b", "fr david"),
    (r"\bdr\.david\b", "dr david"),
    (r"\bdr\.sebastian\b", "dr sebastian"),
    (r"\bpresedent\b", "president"),
    (r"\bkerna\b|\bkarva\b|\bkarvu\b", "karna"),
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
    (r"\bfe\b|\bfees\b|\bfeee\b|\bfess\b|\bfeees\b|ફી\S*", "fee"),
    (r"\bb\.?d\.?a\.?\b", "bda"),
    (r"\bsubjet\b|\bsubjets\b|\bsubjectss\b", "subjects"),
    (r"\bsylabus\b|\bsyllbus\b|\bsyllabous\b", "syllabus"),
    (r"\bdetals\b|\bdetailes\b|\bdetials\b", "details"),
    (r"\bhostal\b|\bhostle\b", "hostel"),
    (r"\bnacc\b|\bnac\b", "naac"),
    (r"\bdewail\b|\bdewali\b", "diwali"),
    (r"\bcai\s*2\b|\bcai-2\b|\bcai2\b", "CAI-2"),
    (r"\bca\s*ii\b|\bca-ii\b", "CA-II"),
    (r"\bcia\s*2\b|\bcia-2\b|\bcia2\b|\bcia\s*ii\b|\bcia-ii\b", "CIA-2"),
    (r"\bca\s*2\b|\bca-2\b|\bca2\b", "CA-2"),
    (r"\braja\s*chhe\b|\braja\s*che\b|\brajao\b|\braja\b|રજા\S*", "holiday"),
    (r"\bchutti\b|\bchhutti\b|છુટ્ટી\S*|छुट्टी\S*", "holiday"),
    (r"\blast\s*yeas\b|\blast\s*yr\b", "last year"),
    (r"\bmne\b", "mane"),
    (r"\bclg\b", "college"),
    (r"\bprof\b|\bproff\b", "professor"),
    (r"મારે\S*|લેવું\S*|લેવુ\S*|કરવું\S*", "i want"),
    (r"\bcomplite\b|\bcomplited\b|\bcomplit\b|\bcompletd\b", "completed"),
    (r"\bfree\s+(?:of\s+)?hostel\b|\bhostel\s+free\b", "hostel fee"),
    (r"\b12\s+in\s+arts\b", "12th in arts"),
    (r"\b12\s+in\s+commerce\b", "12th in commerce"),
    (r"\b12\s+in\s+science\b", "12th in science"),
]


def normalize_student_text(text: str) -> str:
    """Normalize spelling mistakes, typos, slang, and short forms."""
    t = text.strip()
    has_pravida = bool(re.search(r"\bpravida\b", t, re.I))
    for pattern, replacement in _SPELLING_CORRECTIONS:
        if has_pravida and replacement == "holiday":
            continue
        t = re.sub(pattern, replacement, t, flags=re.I)
    return t


_COURSE_PATTERNS = [
    (r"\b(m\.?sc\.?\s*(?:in\s*)?artificial\s*intelligence|msc\s*ai|ai\s*master'?s?)\b", "MSc-Artificial Intelligence"),
    (r"\b(m\.?sc\.?\s*(?:in\s*)?big\s*data(?:\s*analytics)?|msc\s*bda|big\s*data\s*analytics|big\s*data|bda)\b", "MSc-Big Data Analytics"),
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
    (r"\b(b\.?a\.?\s*(?:in\s*)?economics|ba\s*economics|b\.?\s*eco\.?|b\s*eco\b|ba\s*eco\.?|b\.?\s*a\.?\s*eco\.?|bachelor\s*(?:of|in)?\s*economics|economics\s*graduate)\b", "BA-Economics"),
    (r"\beconomics\b", "BA-Economics"),
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
    (r"\b(12th?|hsc|class\s*12|plus\s*two|pass\s*12th?|12)\s*(?:in\s*)?science\b|\bscience\s*stream\b", "Science"),
    (r"\b(12th?|hsc|class\s*12|plus\s*two|pass\s*12th?|12)\s*(?:in\s*)?commerce\b|\bcommerce\s*stream\b", "Commerce"),
    (r"\b(12th?|hsc|class\s*12|plus\s*two|pass\s*12th?|12)\s*(?:in\s*)?arts\b|\barts\s*stream\b", "Arts"),
]


def _fast_understand(question: str) -> dict | None:
    """Fast-path for questions: normalizes spelling, extracts course, stream, and intent instantly."""
    norm_q = normalize_student_text(question)

    course = None
    matched_c, _ = counsellor.match_course(norm_q)
    if not matched_c:
        matched_c, _ = counsellor.match_course(question)
    if matched_c:
        course = matched_c
    else:
        for pattern, name in _COURSE_PATTERNS:
            if re.search(pattern, norm_q, re.I):
                course = name
                break

    qualification = None
    for pattern, stream in _STREAM_PATTERNS:
        if re.search(pattern, norm_q, re.I):
            qualification = stream
            break

    # 1. Application link / Portal link queries (e.g. "Can you give me a link to apply MSc AI?", "give me link to apply")
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

    # 1b. Source queries
    is_source_intent = bool(re.search(
        r"^(where\s+(?:did|do)\s+you\s+find\s+(?:this|that|information|it|info)|give\s+me\s+(?:the\s+)?(?:source|reference)|where\s+is\s+this\s+(?:information|from|data)|what\s+is\s+the\s+source|source\??|show\s*source|where\s+did\s+you\s+find\s+information\s+about\s+.*)\b|"
        r"^(give\s*link|link\??)$",
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

    # 1c. Other college queries (strict single-college boundary)
    is_other_college = bool(re.search(
        r"\b(iit\b|iim\b|bits\s*pilani|nirma\b|ld\s*college|bvm\b|gujarat\s*university|delhi\s*university|\bdu\b|mumbai\s*university|st\.?\s*xavier'?s?\s*(?:mumbai|calcutta|kolkata|jaipur|ranchi)|sal\s*college|silver\s*oak|parul\b|marwadi\b|pdeu\b|da-?iict\b|nift\b|gnlu\b|other\s*colleges?)\b",
        norm_q,
        re.I,
    )) and not bool(re.search(r"\b(sxca|st\.?\s*xavier'?s?\s*(?:college)?\s*(?:autonomous)?\s*(?:ahmedabad)?)\b", norm_q, re.I))
    if is_other_college:
        return {
            "query": norm_q,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "other_college",
        }

    # 2b. Rankings & Accreditation (NAAC, NIRF, GSIRF)
    is_ranking_intent = bool(re.search(
        r"\b(naac|nacc|nac\b|nirf|gsirf|ranking|rankings|accreditation|grade|cycle-?iv)\b",
        norm_q,
        re.I,
    )) and not bool(re.search(r"\b(merit\s*list|cut\s*off|cutoff)\b", norm_q, re.I))
    if is_ranking_intent:
        return {
            "query": "NAAC accreditation NIRF ranking GSIRF rating St Xavier's College",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "ranking",
        }

    # 2c. Merit List & Cutoffs (e.g. "what are the last yeas ranking or merit list for the bca")
    is_merit_cutoff_intent = bool(re.search(
        r"\b(merit\s*list|cut\s*off|cutoff|cutoffs|last\s*year\s*(?:ranking|merit|cutoff|marks)|previous\s*year\s*(?:ranking|merit|cutoff|marks)|closing\s*merit)\b",
        norm_q,
        re.I,
    ))
    if is_merit_cutoff_intent:
        return {
            "query": f"{course or 'BCA'} merit list cutoff admission marks",
            "course": course,
            "qualification": qualification,
            "needs_structured_lookup": bool(course),
            "intent": "merit_cutoff",
        }

    # 2d. Percentage eligibility (e.g. "courses available for 65%", "can I get admission with 65%?")
    pct_m = re.search(r"\b(\d{1,2}(?:\.\d+)?)\s*(?:%|percent\b)", norm_q, re.I)
    is_percentage_intent = bool(pct_m) and bool(re.search(r"\b(course|courses|cource|available|eligible|eligibility|admission|can\s*i\s*(?:get|take|apply|do)|male|malshe|મળે|મળશે|મિલેગા)\b", norm_q, re.I))
    if is_percentage_intent:
        pct_val = pct_m.group(1)
        return {
            "query": f"courses available for {pct_val} percent 12th pass",
            "course": course,
            "qualification": qualification,
            "needs_structured_lookup": False,
            "intent": "percentage_eligibility",
            "percentage": pct_val,
        }

    # 3. Events / Fests / Departmental Activities
    is_events_intent = bool(re.search(
        r"\b(events?|upcoming\s*events?|what\s*events?|college\s*events?|departmental\s*activities|fests?|culfest|sportsfest|seminars?|workshops?|webinars?|conferences?|celebrations?|guest\s*lectures?|activities\s*(?:happening|at\s*the\s*college|in\s*college)?|functions?)\b",
        norm_q,
        re.I,
    ))

    # 4. Curriculum / Subject / Syllabus / Credits intent
    is_curriculum_intent = bool(re.search(
        r"\b(subjects?|subject\??|syllabus|curriculum|course\s*structure|course\s*curriculum|"
        r"subject\s*details|subject\s*outline|paper\s*code|course\s*outcome|course\s*outcomes|"
        r"course\s*content|credits?|credit\s*hours|learning\s*hours|topics?\s*(?:covered|in)|modules?|"
        r"what\s+subjects?\s+(?:are|in)|list\s+of\s+subjects?|semester\s+[1-8]\s+subjects?|sem\s+[1-8]\s+subjects?|"
        r"sem\s+[1-8]|semester\s+[1-8]|practical|theory\s+and\s+lab|"
        r"what\s*(?:will\s*i|do\s*we)\s*study|what\s*are\s*the\s*subjects?|"
        r"what\s*subjects?\s*(?:are\s*there|do\s*you\s*teach)|papers?|course\s*papers?|papers?\s*in\b|"
        r"subjects?\s*kya\s*hai|kya\s*subjects?\s*hai|su\s*subjects?\s*che|na\s*subjects?|"
        r"ma\s*kya\s*subjects?\s*che|syllabus\s*kya\s*hai)\b|"
        r"\b(?:details?\s*(?:about|of|regarding)|tell\s*me\s*about|what\s*is|explain|info\s*about|give\s*details?\s*of|what\s*are\s*the\s*details?\s*(?:of|about)?)\s+(?:the\s+)?(python|python\s*programming|machine\s*learning|nlp|natural\s*language\s*processing|computer\s*vision|data\s*science|deep\s*learning|artificial\s*intelligence|big\s*data|data\s*analytics|dbms|database\s*management|java|c\+\+|data\s*structures|algorithms|operating\s*systems|cyber\s*security|cloud\s*computing)\b|"
        r"\b(python\s*programming|python|machine\s*learning|nlp|computer\s*vision|data\s*structures)\s+(?:details|syllabus|curriculum|credits|course|subject|paper)\b|"
        r"\b(python\s*programming|python|machine\s*learning|nlp|computer\s*vision|data\s*structures)\b",
        norm_q,
        re.I,
    ))

    # 4b. Exam Date intent (High Priority - Overrides active course)
    is_exam_date = bool(re.search(
        r"\b(cai-?2|cai\s*2|ca-?ii|ca\s*ii|ca-?2|ca\s*2|cia-?2|cia\s*2|cia2|"
        r"cai-?1|cai\s*1|ca-?i|ca\s*i|cia-?1|cia\s*1|cia1|"
        r"continuous\s*(?:internal\s*)?assessment(?:\s*[12i]+)?|"
        r"internal\s*exams?|midterm|mid\s*term|semester\s*exams?|final\s*exams?|ese(?:\s*exams?)?|"
        r"exam\s*dates?|examination\s*dates?|pariksha|pareeksha)\b",
        norm_q,
        re.I,
    ))

    # 5. Calendar / Holiday list intent
    is_calendar_holiday = bool(re.search(
        r"\b(holiday\s*list|holidays?|vacation|vacations?|diwali\s*vacation|dewail\s*vacation|dewali\s*vacation|diwali\s*holiday|christmas\s*(?:break|holidays?)|summer\s*(?:break|vacation)|academic\s*calendar|when\s*(?:does|do)\s*(?:the\s*)?college\s*(?:reopen|open|start)|when\s*(?:do|does)\s*classes\s*(?:start|begin)|term\s*dates?|semester\s*(?:commencement|start|end)|working\s*days|janmashtami|gandhi\s*jayanti|independence\s*day|republic\s*day|good\s*friday|eid|muharram|raksha\s*bandhan|holi|mahashivratri|rajao|raja\s*che|raja\s*chhe|chutti|chhutti|raja)\b|રજા\S*|છુટ્ટી\S*|छुट्टी\S*",
        norm_q,
        re.I,
    )) and not re.search(r"\bpravida\b", norm_q, re.I)

    # 6. Faculty / Person / Professor intent
    is_faculty_intent = bool(re.search(
        r"\b(who\s+is\s+[A-Za-z]+|tell\s+me\s+about\s+(?:dr\.?|prof\.?|mr\.?|ms\.?|fr\.?|[A-Z][a-z]+)|faculty\s*(?:members?|details?|list)?|professors?|teachers?|head\s+of\s+(?:the\s+)?department|hod\s+of|coordinator\s+of|principal|vice\s*principal|who\s+teaches\b|pravida\s*raja|pravida|nisarg\s*vyas|nisar\s*vyash|nisarg|vyas|vyash|nithya\s*nadar|nithya|david\s*roy|fr\s*david|david|rector|director|president|chairman)\b",
        norm_q,
        re.I,
    )) or "fr david" in norm_q or "fr.david" in norm_q or "dr david" in norm_q or "dr.david" in norm_q or "nisarg vyas" in norm_q or "nisar vyash" in norm_q

    # 6b. Today's date intent
    is_today_date = bool(re.search(
        r"\b(today'?s?\s*date|date\s*today|what\s*(?:is\s*)?(?:the\s*)?date(?:\s*today)?|current\s*date|what\s*day\s*is\s*(?:it\s*)?today|aaj\s*ni\s*tari[kh]|aaj\s*ki\s*tari[kh]|aaje\s*kai\s*tari[kh]|aaj\s*konsi\s*tarikh)\b",
        norm_q,
        re.I,
    ))

    # 6c. Swimming pool & sports facility intent
    is_swimming_pool = bool(re.search(
        r"\b(swimming\s*pool|swimming|pool\s*in\s*(?:the\s*)?(?:clg|college)|pool)\b",
        norm_q,
        re.I,
    ))

    # 6c-2. Hostel & Accommodation intent
    is_hostel = bool(re.search(
        r"\b(hostel|hostels|hostal|hostle|boys\s*hostel|girls\s*hostel|accommodation|stay|mess\b)\b",
        norm_q,
        re.I,
    ))

    # 6d. All UG / All PG courses listing intent
    is_all_ug_courses = bool(re.search(
        r"\b((?:list\s*of\s*(?:all\s*)?|all\s*)?ug\s*(?:courses?|cource|programs?|options?)|undergraduate\s*(?:courses?|cource|programs?)|bachelor\s*(?:courses?|degrees?)|what\s*are\s*the\s*ug\s*courses?|show\s*(?:me\s*)?(?:all\s*)?ug\s*courses?|which\s*ug\s*courses?|list\s+of\s+courses?)\b",
        norm_q,
        re.I,
    )) and not re.search(r"\b(pg|postgraduate|master)\b", norm_q, re.I) and not course

    is_all_pg_courses = bool(re.search(
        r"\b((?:list\s*of\s*(?:all\s*)?|all\s*)?pg\s*(?:courses?|cource|programs?|options?)|postgraduate\s*(?:courses?|cource|programs?)|master'?s?\s*(?:courses?|degrees?)|what\s*are\s*the\s*pg\s*courses?|show\s*(?:me\s*)?(?:all\s*)?pg\s*courses?|which\s*pg\s*courses?)\b",
        norm_q,
        re.I,
    )) and not course

    # 7. Notices / Announcements
    is_notices_intent = bool(re.search(r"\b(notices?|circulars?|announcements?|exam\s*notice|supplementary\s*exam\s*form)\b", norm_q, re.I))

    # 8. Contact / Department info
    is_contact_intent = bool(re.search(r"\b(contact\s*(?:info|details|number)?|email\s*address|phone\s*number|where\s+is\s+(?:the\s+)?college|location\s+of\s+college|campus\s*address|address\s+of)\b", norm_q, re.I))

    # 9. Application / how to apply intent
    is_how_to_apply = bool(re.search(
        r"\b(how\s*(?:can|do)\s*i\s*apply|how\s*to\s*apply|how\s*to\s*get\s*admission|how\s*to\s*take\s*admission|how\s*to\s*register|application\s*(?:process|steps|procedure|link|form|portal)|where\s*(?:to|can\s*i)\s*apply|admission\s*(?:process|procedure|steps)|process\s*(?:su\s*che|kya\s*hai|shun\s*che)|procedure\s*(?:su\s*che|kya\s*hai))\b",
        norm_q,
        re.I,
    ))
    is_fee_intent = bool(re.search(r"\b(fee|fees|cost|how\s*much\s*(?:is|are|does|to)?|charges|tuition)\b", norm_q, re.I))
    is_doc_intent = bool(re.search(r"\b(documents?|doc|docs|certificate|marksheet|required\s*documents?|which\s*docs)\b", norm_q, re.I))
    is_elig_intent = False
    if not is_how_to_apply and not is_curriculum_intent and not is_events_intent and not is_all_ug_courses and not is_all_pg_courses:
        is_elig_intent = bool(re.search(r"\b(eligibility|eligible|criteria|qualify|requirements|which\s*course|which\s*cource|what\s*can\s*i\s*do|can\s*i\s*(?:take|apply|opt|join)|can\s*i\s*do)\b", norm_q, re.I))

    is_broad_counselling = False
    if not (is_curriculum_intent or is_events_intent or is_calendar_holiday or is_faculty_intent or is_notices_intent or is_contact_intent or is_how_to_apply or is_fee_intent or is_doc_intent or is_elig_intent or is_today_date or is_swimming_pool or is_all_ug_courses or is_all_pg_courses):
        is_broad_counselling = bool(re.search(
            r"\b(i\s*want\b.*(?:admission|apply|take\s*admission|join|enroll|do|study)|admission\b.*(?:in|for|take|want|levu|chahiye|લેવું|લેવુ))\b",
            norm_q,
            re.I,
        ))

    # 0. Greeting & Pleasantries intent (instant <5ms response)
    is_greeting = bool(re.search(
        r"^(?:hi+|hello+|hey+|heya|greetings|good\s*(?:morning|afternoon|evening)|kem\s*cho|namaste|namaskar|pranam|kya\s*haal\s*(?:hai)?|halo|who\s*are\s*you|what\s*can\s*you\s*do|help\s*me|hi\s*there)\b",
        norm_q,
        re.I,
    )) and not course and not qualification and not is_how_to_apply and not is_fee_intent and not is_doc_intent and not is_elig_intent and not is_broad_counselling and not is_curriculum_intent and not is_events_intent and not is_calendar_holiday and not is_faculty_intent and not is_today_date and not is_swimming_pool and not is_all_ug_courses and not is_all_pg_courses
    if is_greeting:
        return {
            "query": norm_q,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "greeting",
        }

    # 0b. Chitchat, Gratitude & Goodbyes (instant <5ms response)
    is_chitchat = bool(re.search(
        r"^(?:thanks|thank\s*you|thanku|thx|dhanyawad|aabhar|shukriya|ok\s*thanks|okay\s*thanks|bye|goodbye|see\s*you|cya|take\s*care|ok|okay|cool|great\s*thanks)\b",
        norm_q,
        re.I,
    )) and not course and not qualification and not is_how_to_apply and not is_fee_intent and not is_doc_intent and not is_elig_intent and not is_broad_counselling and not is_today_date and not is_swimming_pool and not is_all_ug_courses and not is_all_pg_courses
    if is_chitchat:
        return {
            "query": norm_q,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "chitchat",
        }

    # 0c. Today's date intent
    if is_today_date:
        return {
            "query": norm_q,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "today_date",
        }

    # 0d. Swimming pool & sports facilities intent
    if is_swimming_pool:
        return {
            "query": "sports facilities swimming pool athletics track football cricket",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "swimming_pool",
        }

    # 0d-2. Hostel & Accommodation intent
    if is_hostel:
        is_fee = bool(re.search(r"\b(fee|fees|rent|cost|charges?|price|how\s+much)\b", question.lower()))
        return {
            "query": "hostel fee fees structure accommodation" if is_fee else "hostel accommodation campus",
            "course": None,
            "qualification": qualification,
            "needs_structured_lookup": False,
            "intent": "hostel",
        }


    # 0e. All UG Courses list intent
    if is_all_ug_courses:
        return {
            "query": "undergraduate programs courses BSc BCA BCom BA list eligibility fee",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "all_ug_courses",
        }

    # 0f. All PG Courses list intent
    if is_all_pg_courses:
        return {
            "query": "postgraduate programs courses MSc MA list eligibility fee",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "all_pg_courses",
        }

    # 1. Routing Faculty / Person (checked before calendar to avoid name collisions)
    if is_faculty_intent:
        if re.search(r"\b(pravida\s*raja|pravida)\b", norm_q, re.I):
            return {
                "query": "Dr Pravida Raja Head Department of Data Science faculty SXCA",
                "course": None,
                "qualification": None,
                "needs_structured_lookup": False,
                "intent": "faculty_pravida_raja",
            }
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

    # 2. Routing Events
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

    # 3. Routing Curriculum
    if is_curriculum_intent and not is_broad_counselling:
        sem_m = re.search(r"\b(?:sem(?:ester)?\s*([1-8])|([1-8])(?:st|nd|rd|th)?\s*sem(?:ester)?)\b", norm_q, re.I)
        sem_str = f" semester {sem_m.group(1) or sem_m.group(2)}" if sem_m else ""
        if re.search(r"\b(python|python\s*programming)\b", norm_q, re.I):
            query = "Python Programming syllabus curriculum credits course code PAI-1803 PBD-1806 learning hours outcomes"
        elif course:
            query = f"{course}{sem_str} syllabus curriculum course structure papers credits subjects"
        else:
            clean_sub = re.sub(r"\b(i\s*want|the|details?|about|of|regarding|tell\s*me|what\s*is|what\s*are|please|give\s*me)\b", "", norm_q, flags=re.I).strip()
            query = f"{clean_sub}{sem_str} syllabus curriculum course structure subjects credits"
        return {
            "query": query,
            "course": course,
            "qualification": qualification,
            "needs_structured_lookup": False,
            "intent": "curriculum",
        }

    # 4a. Routing Exam Dates (High Priority - NEVER inherit active course)
    if is_exam_date:
        if re.search(r"\b(cai-?2|cai\s*2|ca-?ii|ca\s*ii|ca-?2|ca\s*2|cia-?2|cia\s*2|cia2|continuous\s*(?:internal\s*)?assessment\s*(?:2|ii))\b", norm_q, re.I):
            q_norm = "CA-II CIA-2 continuous internal assessment examination date academic calendar St Xavier's College Ahmedabad"
            event_type = "cai2"
        elif re.search(r"\b(cai-?1|cai\s*1|ca-?i|ca\s*i|cia-?1|cia\s*1|cia1|continuous\s*(?:internal\s*)?assessment\s*(?:1|i))\b", norm_q, re.I):
            q_norm = "CA-I CIA-1 continuous internal assessment examination date academic calendar St Xavier's College Ahmedabad"
            event_type = "cia1"
        elif re.search(r"\b(ese|end\s*semester|final\s*exam)\b", norm_q, re.I):
            q_norm = "End Semester Examination ESE theory practical date academic calendar St Xavier's College Ahmedabad"
            event_type = "ese"
        else:
            q_norm = "examination dates CIA-1 CIA-2 ESE academic calendar St Xavier's College Ahmedabad"
            event_type = "general_exam"

        return {
            "query": q_norm,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "exam_date",
            "event_type": event_type,
            "original_question": question,
        }

    # 4b. Routing Calendar / Holiday
    if is_calendar_holiday:
        date_m = re.search(r"\b(\d{1,2}(?:st|nd|rd|th)?\s*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b", norm_q, re.I)
        hol_m = re.search(r"\b(janmashtami|diwali|dewail|dewali|christmas|gandhi\s*jayanti|independence\s*day|republic\s*day|good\s*friday|eid|muharram|raksha\s*bandhan|holi|mahashivratri)\b", norm_q, re.I)
        is_diwali = bool(re.search(r"\b(diwali|dewail|dewali)\b", norm_q, re.I))
        is_cal = bool(re.search(r"\b(academic\s*calendar|when\s*(?:does|do)\s*(?:the\s*)?college\s*(?:reopen|open|start)|when\s*(?:do|does)\s*classes\s*(?:start|begin)|term\s*dates?|semester\s*(?:commencement|start|end))\b", norm_q, re.I))
        
        if date_m:
            query = f"{date_m.group(1)} academic calendar handbook holiday schedule working day"
            intent = "calendar_holiday"
        elif is_diwali:
            query = "Diwali vacation academic calendar dates schedule St Xavier's College Ahmedabad"
            intent = "calendar_holiday"
        elif is_cal:
            query = "academic calendar schedule dates semester commencement reopening St Xavier's College Ahmedabad"
            intent = "academic_calendar"
        elif hol_m:
            specific_hol = hol_m.group(1)
            query = f"{specific_hol} holiday academic calendar handbook dates schedule St Xavier's College Ahmedabad"
            intent = "calendar_holiday"
        else:
            query = "academic calendar holiday list schedule vacation dates reopening St Xavier's College Ahmedabad"
            intent = "academic_calendar"

        return {
            "query": query,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": intent,
            "specific_date": date_m.group(1) if date_m else None,
            "holiday_name": hol_m.group(1) if hol_m else ("diwali" if is_diwali else None),
            "original_question": question,
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


def understand(question: str, recent_turns: list[dict] | None = None, context: dict | None = None) -> dict:
    """Returns {"query", "course", "qualification", "needs_structured_lookup", "intent", "profile"}."""
    current_profile = (context or {}).get("profile") if context else None
    can_inherit = counsellor.should_inherit_course_context(question, current_profile or {}, recent_turns)
    if current_profile is None and context and context.get("course") and can_inherit:
        current_profile = {"active_course": context.get("course")}
    elif current_profile and not current_profile.get("active_course") and context and context.get("course") and can_inherit:
        current_profile["active_course"] = context.get("course")
    profile = counsellor.extract_and_update_profile(current_profile, question, recent_turns)
    active_course = profile.get("active_course")
    c_intent = counsellor.classify_intent(question, profile, recent_turns)

    # 1. Top priority human-like counsellor intents
    if c_intent == "FEEDBACK_REPAIR":
        return {
            "query": question,
            "course": active_course,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "feedback_repair",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "COMMERCE_GRADUATE_COURSES":
        return {
            "query": "postgraduate courses for commerce bcom graduates St Xavier's College Ahmedabad",
            "course": None,
            "qualification": "Commerce",
            "needs_structured_lookup": False,
            "intent": "commerce_graduate_courses",
            "profile": profile,
            "original_question": question,
        }

    if c_intent in ("ELIGIBLE_PG_COURSES", "BA_ECONOMICS_GRADUATE_COURSES"):
        return {
            "query": "postgraduate master courses for BA Economics graduates M.Sc Big Data Analytics M.Sc AI MA English St Xavier's College Ahmedabad",
            "course": None,
            "qualification": "BA-Economics",
            "needs_structured_lookup": False,
            "intent": "eligible_pg_courses",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "COMMERCE_STREAM_INQUIRY":
        return {
            "query": "commerce stream undergraduate and postgraduate courses St Xavier's College Ahmedabad",
            "course": None,
            "qualification": "Commerce",
            "needs_structured_lookup": False,
            "intent": "commerce_stream_inquiry",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "BCA_GRADUATE_COURSES":
        return {
            "query": "postgraduate courses for BCA graduates M.Sc AI Big Data Analytics St Xavier's College Ahmedabad",
            "course": None,
            "qualification": "BCA",
            "needs_structured_lookup": False,
            "intent": "bca_graduate_courses",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "NEGATION":
        _, neg_topic = counsellor.detect_negation(question, profile)
        return {
            "query": question,
            "course": None if (neg_topic and neg_topic.startswith("course:")) else active_course,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "negation",
            "negated_topic": neg_topic,
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "REJECT_CLARIFICATION":
        return {
            "query": question,
            "course": active_course,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "reject_clarification",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "REDIRECT":
        return {
            "query": question,
            "course": active_course,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "redirect",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "CORRECTION":
        return {
            "query": question,
            "course": active_course,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "correction",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "AI_AMBIGUITY":
        profile["pending_clarification"] = {"type": "ai"}
        return {
            "query": question,
            "course": None,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "ai_ambiguity",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "ZERO_COST_CHECK":
        return {
            "query": question,
            "course": active_course,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": bool(active_course),
            "intent": "zero_cost",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "FEE_CONFIRMATION":
        profile["pending_clarification"] = {"type": "fee", "course": active_course}
        return {
            "query": question,
            "course": active_course,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "fee_confirmation",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "NO_COURSE_FEE":
        return {
            "query": question,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "no_course_followup",
            "raw_intent": "FEES",
            "profile": profile,
            "original_question": question,
        }

    # 2. Check for specialized fast intents from _fast_understand (dates, swimming pool, rankings, holidays, pravida raja)
    fast = _fast_understand(question)
    if fast is not None and fast.get("intent") in (
        "today_date", "swimming_pool", "faculty_pravida_raja", "ranking",
        "merit_cutoff", "percentage_eligibility", "calendar_holiday",
        "exam_date", "academic_calendar", "other_college", "source_query",
        "all_ug_courses", "all_pg_courses", "application_link", "hostel"
    ):
        fast["profile"] = profile
        if fast.get("intent") in ("merit_cutoff", "percentage_eligibility", "application_link"):
            fast["course"] = active_course or fast.get("course")
        else:
            fast["course"] = None
        return fast

    # 3. Dedicated counsellor intents
    if c_intent == "SUBJECTS" or (fast and fast.get("intent") == "curriculum"):
        course_to_use = active_course or (fast.get("course") if fast else None)
        from backend.rag.curriculum_retriever import extract_semester_filter
        sem_num = extract_semester_filter(question)
        if course_to_use:
            if sem_num:
                query = f"{course_to_use} semester {sem_num} subjects papers course code credits"
            else:
                query = f"{course_to_use} subjects curriculum course structure papers credits semester"
            return {
                "query": query,
                "course": course_to_use,
                "qualification": profile.get("stream"),
                "needs_structured_lookup": False,
                "intent": "curriculum",
                "semester": sem_num,
                "profile": profile,
                "original_question": question,
            }
        else:
            return {
                "query": question,
                "course": None,
                "qualification": None,
                "needs_structured_lookup": False,
                "intent": "no_course_followup",
                "raw_intent": "SUBJECTS",
                "profile": profile,
            }

    if c_intent == "EXAM_DATE":
        fast_ex = _fast_understand(question)
        if fast_ex and fast_ex.get("intent") == "exam_date":
            fast_ex["profile"] = profile
            return fast_ex
        return {
            "query": "CA-II CIA-2 continuous internal assessment examination date academic calendar St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "exam_date",
            "event_type": "cai2",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "HOLIDAY":
        return {
            "query": "Diwali vacation academic calendar dates schedule St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "calendar_holiday",
            "holiday_name": "diwali",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "ACADEMIC_CALENDAR":
        return {
            "query": "academic calendar schedule dates semester commencement reopening St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "academic_calendar",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "GREETING":
        return {
            "query": question,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "greeting",
            "profile": profile,
        }

    if c_intent == "COMPLAINT":
        return {
            "query": question,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "complaint",
            "profile": profile,
        }

    if c_intent == "CHITCHAT":
        return {
            "query": question,
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "chitchat",
            "profile": profile,
        }

    if c_intent == "LANGUAGE_PREFERENCE":
        return {
            "query": question,
            "course": None,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "language_preference",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "FACULTY":
        rewritten_q = question
        active_p = profile.get("active_person")
        if isinstance(active_p, dict) and active_p.get("status") == "RESOLVED":
            p_name = active_p.get("name") or active_p.get("full_name") or ""
            if p_name:
                rewritten_q = re.sub(r"\b(he|she|they|that\s+person|this\s+person)\b", p_name, question, flags=re.I)
                rewritten_q = re.sub(r"\b(his|her|their)\b", f"{p_name}'s", rewritten_q, flags=re.I)
                rewritten_q = re.sub(r"\b(him|them)\b", p_name, rewritten_q, flags=re.I)
        return {
            "query": f"{rewritten_q} faculty professors leadership director principal St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "faculty",
            "profile": profile,
            "original_question": question,
            "rewritten_question": rewritten_q,
        }

    if c_intent == "CONTACT":
        return {
            "query": "contact admission office phone helpline email address St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "contact",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "OFFICE_TIMING":
        return {
            "query": "office timing hours working days St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "office_timing",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "HOSTEL":
        is_fee = bool(re.search(r"\b(fee|fees|rent|cost|charges?|price|how\s+much)\b", question.lower()))
        return {
            "query": "hostel fee fees structure accommodation" if is_fee else "hostel accommodation campus",
            "course": None,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "hostel",
            "profile": profile,
            "original_question": question,
        }


    if c_intent == "DEPARTMENTS":
        return {
            "query": "academic departments Arts Science Commerce St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "departments",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "CLUBS":
        return {
            "query": "student clubs committees associations activities St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "clubs",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "SPORTS":
        return {
            "query": "sports facilities gymnasium grounds physical education St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "sports",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "NCC_NSS":
        return {
            "query": "NCC Army Navy NSS National Service Scheme St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "ncc_nss",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "EVENTS":
        return {
            "query": "annual events Culfest cultural fest sports day St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "events",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "PLACEMENT":
        return {
            "query": "placement cell career recruiters packages St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "placement",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "SCHOLARSHIPS":
        return {
            "query": "scholarships financial aid MYSY Digital Gujarat SAF St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "scholarships",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "LIBRARY":
        return {
            "query": "central library books reading room timing St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "library",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "SCIENCE_COURSES":
        return {
            "query": "science stream courses undergraduate BSc BCA IT AI ML St Xavier's College Ahmedabad",
            "course": None,
            "qualification": "Science",
            "needs_structured_lookup": False,
            "intent": "science_courses",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "GENERAL_PG_ADMISSION":
        return {
            "query": "postgraduate master admission process application eligibility St Xavier's College Ahmedabad",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "general_pg_admission",
            "profile": profile,
            "original_question": question,
        }

    if c_intent == "DOCUMENTS":
        return {
            "query": f"{active_course or ''} documents required for admission".strip(),
            "course": active_course,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "documents",
            "profile": profile,
        }

    if c_intent == "ADMISSION_PROCESS":
        return {
            "query": f"{active_course or ''} admission procedure and application process".strip(),
            "course": active_course,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": bool(active_course),
            "intent": "process",
            "profile": profile,
        }

    if c_intent == "ALL_UG_COURSES":
        return {
            "query": "undergraduate programs courses BSc BCA BCom BA list eligibility fee",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "all_ug_courses",
            "profile": profile,
        }

    if c_intent == "ALL_PG_COURSES":
        return {
            "query": "postgraduate programs courses MSc MA list eligibility fee",
            "course": None,
            "qualification": None,
            "needs_structured_lookup": False,
            "intent": "all_pg_courses",
            "profile": profile,
        }

    if c_intent == "COURSE_DISCOVERY":
        return {
            "query": "courses available St Xavier's College",
            "course": None,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "course_discovery",
            "profile": profile,
        }

    if c_intent == "COURSE_INQUIRY" and active_course:
        return {
            "query": f"{active_course} admission overview",
            "course": active_course,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": True,
            "intent": "course_inquiry",
            "profile": profile,
        }

    if c_intent == "FEES":
        if active_course:
            return {
                "query": f"{active_course} fee St. Xavier's College Ahmedabad",
                "course": active_course,
                "qualification": profile.get("stream"),
                "needs_structured_lookup": True,
                "intent": "fee",
                "profile": profile,
            }
        else:
            return {
                "query": question,
                "course": None,
                "qualification": None,
                "needs_structured_lookup": False,
                "intent": "no_course_followup",
                "raw_intent": "FEES",
                "profile": profile,
            }

    if c_intent == "ELIGIBILITY":
        if active_course:
            return {
                "query": f"{active_course} eligibility criteria St. Xavier's College Ahmedabad",
                "course": active_course,
                "qualification": profile.get("stream"),
                "needs_structured_lookup": True,
                "intent": "eligibility",
                "profile": profile,
            }
        else:
            return {
                "query": question,
                "course": None,
                "qualification": None,
                "needs_structured_lookup": False,
                "intent": "no_course_followup",
                "raw_intent": "ELIGIBILITY",
                "profile": profile,
            }

    if c_intent == "COUNSELLING":
        return {
            "query": question,
            "course": active_course,
            "qualification": profile.get("stream"),
            "needs_structured_lookup": False,
            "intent": "counselling",
            "profile": profile,
        }

    # 4. Fallback to general query understanding / LLM for open questions
    norm_q = normalize_student_text(question)
    can_inherit = counsellor.should_inherit_course_context(question, profile, recent_turns)
    if not can_inherit or c_intent in (
        "EXAM_DATE", "ACADEMIC_CALENDAR", "HOLIDAY", "COMPLAINT", "GREETING", "CHITCHAT",
        "OTHER_COLLEGE", "ALL_UG_COURSES", "ALL_PG_COURSES", "COURSE_DISCOVERY",
        "LANGUAGE_PREFERENCE", "FACULTY", "CONTACT", "OFFICE_TIMING", "HOSTEL",
        "DEPARTMENTS", "CLUBS", "SPORTS", "NCC_NSS", "EVENTS", "PLACEMENT",
        "SCHOLARSHIPS", "LIBRARY", "SCIENCE_COURSES", "GENERAL_PG_ADMISSION",
    ):
        use_course = None
        query = norm_q
    else:
        use_course = active_course
        query = f"{active_course} {norm_q}" if active_course else norm_q

    return {
        "query": query,
        "course": use_course,
        "qualification": profile.get("stream"),
        "needs_structured_lookup": bool(use_course),
        "intent": "general",
        "profile": profile,
    }
