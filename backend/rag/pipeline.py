from datetime import datetime
import re
from typing import Any

from backend.config import LIVE_CRAWL_FALLBACK_ENABLED, RERANK_FLOOR
from backend.crawler.live_search import maybe_live_crawl
from backend.db import repo
from backend.llm import qwen
from backend.rag import cache, counsellor, eligibility, query_rewrite
from backend.rag.prompt import NO_CONTEXT_ANSWER, build_messages
from backend.rag.retriever import as_sources, normalize_query, retrieve
from backend.utils.registry import log_chat

NO_INFO_ANSWER = "Sorry, I don't have information about that in the available SXCA materials."

_REFUSAL_MARKERS = (
    "don't have that information",
    "sorry, i don't have that information",
    "sorry, i don't have information",
    "not available in my available college data",
    "not available in the college data",
    "not available in the college documents",
    "available sxca materials",
    "contact the admission office",
)


def _is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _cache_key(question: str, history: list[dict]) -> str | None:
    # Only cache fresh (no-history) questions
    if history:
        return None
    norm = query_rewrite.normalize_student_text(question)
    return normalize_query(norm)


def _resolve_turns(conversation_id: str | None, history: list[dict], owner_id: str | None = None) -> list[dict]:
    """The database is the authoritative history once a conversation_id is
    given (every prior turn in this conversation was already logged via
    log_chat below) - it doesn't depend on the client having sent a complete
    or untampered `history` array. Without a conversation_id (older client,
    or a caller that doesn't want persistence), fall back to whatever the
    client sent, same as before this feature existed."""
    if conversation_id:
        repo.get_or_create_conversation(conversation_id, owner_id=owner_id)
        return repo.get_recent_turns(conversation_id, owner_id=owner_id)
    return history or []


def _understand(question: str, conversation_id: str | None, turns: list[dict], owner_id: str | None = None) -> dict:
    if conversation_id:
        repo.get_or_create_conversation(conversation_id, owner_id=owner_id)
    context = repo.get_conversation_context(conversation_id, owner_id=owner_id) if conversation_id else None
    result = query_rewrite.understand(question, turns, context)
    profile = result.get("profile") or {}
    active_course = result.get("course") or profile.get("active_course")
    if active_course and not profile.get("active_course"):
        profile["active_course"] = active_course
    qualification = result.get("qualification") or profile.get("stream")
    repo.update_conversation_context(
        conversation_id,
        course=active_course,
        qualification=qualification,
        profile=profile,
        owner_id=owner_id,
    )
    return result



def _structured_lookup(understanding: dict) -> tuple[list[dict], bool]:
    """Returns (rows, course_specific). course_specific is True only when
    the question named an exact course that matched (find_by_course) - in
    that case structured data is complete and authoritative on its own, and
    the caller skips RAG passages entirely for this turn. Confirmed with a
    real, reproducible failure, not a hypothetical one: asked "Can a BCA
    graduate apply for MSc AI?" with the correct MSc-AI structured row
    present, the model still answered with BCA's OWN eligibility text -
    which didn't even appear in any of the 5 retrieved RAG passages, so it
    wasn't grounded in them either. The highest-scoring passage (0.92) was
    an unrelated UG chemistry/electronics fee-table fragment; the model
    appears to have fallen back to its own general knowledge of typical BCA
    admission criteria when the given passages were noisy, despite having
    correct structured data right there. Removing the noisy passages when
    structured data already answers the question closes that opening.

    A stream-based recommendation match (find_by_stream, course_specific
    False) keeps RAG passages alongside it - that's a broader "which
    courses fit me" case without the same demonstrated failure, and RAG
    still adds real descriptive value there."""
    if not understanding["needs_structured_lookup"]:
        return [], False
    if understanding.get("intent") in ("curriculum", "subjects"):
        return [], False
    course_rows = eligibility.find_by_course(understanding["course"]) if understanding["course"] else []
    if course_rows:
        return course_rows, True
    stream_rows = eligibility.find_by_stream(understanding["qualification"]) if understanding["qualification"] else []
    return stream_rows, False


def get_verified_application_url() -> str:
    """Returns the verified application URL stored in the college source data."""
    url = repo.get_url_for_source("admissions-sxca-edu-in-sxca.html")
    if url:
        return url
    url = repo.get_url_for_source("portal-sxca-edu-in-index-2.html")
    if url:
        return url
    return "https://admissions.sxca.edu.in/SXCA/"


def _format_course_counselling_summary(course_rows: list[dict]) -> str:
    if not course_rows:
        return ""
    main_row = course_rows[0]
    course_name = main_row["course_name"]
    level = main_row["level"]
    duration = "3-4 Years (Undergraduate)" if level == "UG" else "2 Years (Postgraduate)"

    fee_parts = []
    for r in course_rows:
        cat = "Grant-in-Aid" if r.get("category") == "grant_in_aid" else "Self-Financed"
        if r.get("fee_amount") is not None:
            if r.get("fee_amount_female") is not None:
                fee_parts.append(f"₹{int(r['fee_amount']):,}/semester (Male), ₹{int(r['fee_amount_female']):,}/semester (Female) [{cat}]")
            else:
                fee_parts.append(f"₹{int(r['fee_amount']):,}/semester [{cat}]")

    fees_str = " | ".join(fee_parts) if fee_parts else "As per official schedule"
    eligibility_str = main_row.get("eligibility", "12th Standard or equivalent")
    app_url = get_verified_application_url()
    note = "\n\n*(Note: If you are looking for postgraduate studies, M.Sc. Physics is also available for B.Sc. graduates.)*" if course_name == "BSc-Physics" else ""

    return (
        f"You can apply for **{course_name}** ({level}).\n\n"
        f"• **Eligibility:** {eligibility_str}\n"
        f"• **Fees:** {fees_str}\n"
        f"• **Duration:** {duration}\n"
        f"• **Admission Process:** Online registration via the official college admission portal, followed by merit list publication and document verification.\n"
        f"• **Required Documents:** 10th & 12th Marksheets, School Leaving Certificate, Transfer Certificate, and Category Certificate (if applicable).\n"
        f"• **Official Application Portal:** [Apply here (St. Xavier's College Admission Portal)]({app_url}){note}"
    )


def _format_course_fee_summary(course_rows: list[dict], is_guj: bool = False) -> str:
    if not course_rows:
        return ""
    lines = []
    for r in course_rows:
        c_name = r.get("course_name", "")
        disp_name = counsellor.clean_course_display_name(c_name) if c_name else "the course"
        cat = "Grant-in-Aid" if r.get("category") == "grant_in_aid" else "Self-Financed"
        male_fee = r.get("fee_amount")
        female_fee = r.get("fee_amount_female")
        if is_guj:
            if male_fee is not None and female_fee is not None:
                fee_str = f"પુરુષ વિદ્યાર્થીઓ માટે ₹{int(male_fee):,} પ્રતિ સેમેસ્ટર અને મહિલા વિદ્યાર્થીઓ માટે ₹{int(female_fee):,} પ્રતિ સેમેસ્ટર"
            elif male_fee is not None:
                fee_str = f"₹{int(male_fee):,} પ્રતિ સેમેસ્ટર"
            else:
                fee_str = "કોલેજના નિયમો મુજબ"
            note = f" ({r.get('fee_note')})" if r.get("fee_note") else ""
            lines.append(f"સેન્ટ ઝેવિયર્સ કોલેજમાં **{disp_name}** ({cat}) માટે ફી **{fee_str}**{note} છે.")
        else:
            if male_fee is not None and female_fee is not None:
                fee_str = f"₹{int(male_fee):,} per semester for male students and ₹{int(female_fee):,} per semester for female students"
            elif male_fee is not None:
                fee_str = f"₹{int(male_fee):,} per semester"
            else:
                fee_str = "as prescribed by college guidelines"
            note = f" ({r.get('fee_note')})" if r.get("fee_note") else ""
            lines.append(f"The fee for **{disp_name}** ({cat}) is **{fee_str}**{note}.")
    contact = course_rows[0].get("contact_email")
    if contact:
        if is_guj:
            lines.append(f"વધુ વિગતો માટે, તમે {contact} પર સંપર્ક કરી શકો છો.")
        else:
            lines.append(f"For more details, you can contact {contact}.")
    return "\n".join(lines)


def _format_course_eligibility_summary(course_rows: list[dict], user_stream: str | None = None, is_guj: bool = False) -> str:
    if not course_rows:
        return ""
    lines = []
    for r in course_rows:
        c_name = r.get("course_name", "")
        disp_name = counsellor.clean_course_display_name(c_name) if c_name else "the course"
        cat = "Grant-in-Aid" if r.get("category") == "grant_in_aid" else "Self-Financed"
        elig = r.get("eligibility", "12th standard pass")
        if is_guj:
            if user_stream and user_stream.lower() in elig.lower():
                lines.append(f"સેન્ટ ઝેવિયર્સ કોલેજમાં **{disp_name}** ({cat}) માટે પાત્રતા માપદંડ **{elig}** છે. હા, તમારા {user_stream} બેકગ્રાઉન્ડ સાથે તમે અરજી કરી શકો છો.")
            else:
                lines.append(f"સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં **{disp_name}** ({cat}) માટે પાત્રતા માપદંડ: **{elig}**.")
        else:
            if user_stream and user_stream.lower() in elig.lower():
                lines.append(f"To be eligible for **{disp_name}** ({cat}), the requirement is **{elig}**. Yes, with a {user_stream} background, you can apply.")
            else:
                lines.append(f"To be eligible for **{disp_name}** ({cat}) at St. Xavier's College (Autonomous), Ahmedabad, you must have passed: **{elig}**.")
    return "\n".join(lines)


def _format_course_fee_and_eligibility_summary(course_rows: list[dict]) -> str:
    if not course_rows:
        return ""
    lines = []
    for r in course_rows:
        c_name = r.get("course_name", "")
        cat = "Grant-in-Aid" if r.get("category") == "grant_in_aid" else "Self-Financed"
        elig = r.get("eligibility", "12th standard pass")
        male_fee = r.get("fee_amount")
        female_fee = r.get("fee_amount_female")
        if male_fee is not None and female_fee is not None:
            fee_str = f"₹{int(male_fee):,} per semester for male students and ₹{int(female_fee):,} per semester for female students"
        elif male_fee is not None:
            fee_str = f"₹{int(male_fee):,} per semester"
        else:
            fee_str = "as prescribed by college guidelines"
        note = f" ({r.get('fee_note')})" if r.get("fee_note") else ""
        lines.append(f"The eligibility for **{c_name}** ({cat}) is **{elig}**, and the fee is **{fee_str}**{note}.")
    contact = course_rows[0].get("contact_email")
    if contact:
        lines.append(f"For more details, you can contact {contact}.")
    return "\n".join(lines)


def _format_course_application_link(course_name: str | None, course_rows: list[dict] | None = None) -> str:
    app_url = get_verified_application_url()
    if course_name:
        c_display = course_name.replace("MSc-", "MSc ").replace("BSc-", "BSc ").replace("BA-", "BA ").replace("MA-", "MA ")
        return (
            f"Yes. You can apply for {c_display} through the official St. Xavier's College admission portal:\n\n"
            f"[Apply here — Official Admission Portal]({app_url})"
        )
    return (
        f"You can apply for admission through the official St. Xavier's College admission portal:\n\n"
        f"[Apply here — Official Admission Portal]({app_url})"
    )


def _clean_source_title(src_name: str | None) -> str:
    if not src_name:
        return "St. Xavier's College Official Document"
    s = str(src_name).replace(".pdf", "").replace(".html", "").replace("-", " ")
    if "SXCA Handbook" in s or "Handbook" in s:
        return "SXCA Student Handbook 2026-27"
    if "Admission Brochure" in s or "Brochure" in s:
        return "SXCA Admission Brochure 2025-26"
    if "Academic Calendar" in s or "Calendar" in s:
        return "SXCA Academic Calendar"
    if "admissions" in s.lower():
        return "SXCA Admission Portal"
    if "rankings" in s.lower():
        return "SXCA Rankings & Accreditations"
    if "hostel" in s.lower():
        return "SXCA Campus Facilities & Hostel"
    return s.title()


def _standardize_source(s: dict) -> dict:
    source_name = s.get("source") or s.get("title") or "Official College Records"
    doc_type = s.get("type") or s.get("doc_type") or ("pdf" if str(source_name).endswith(".pdf") else "website")
    page = s.get("page")
    url = s.get("url") or repo.get_url_for_source(str(source_name)) or "https://sxca.edu.in/"
    snippet = s.get("snippet") or s.get("excerpt") or ""
    title = s.get("title") or _clean_source_title(str(source_name))
    return {
        "title": title,
        "type": doc_type,
        "doc_type": doc_type,
        "page": page,
        "url": url,
        "snippet": snippet,
        "excerpt": snippet,
        "source": str(source_name),
        "score": s.get("score"),
    }


def _standardize_sources(sources: list[dict] | None) -> list[dict]:
    if not sources:
        return []
    return [_standardize_source(s) for s in sources]


def _format_course_application_process(course_name: str | None, course_rows: list[dict] | None = None, is_guj: bool = False) -> str:
    disp_name = counsellor.clean_course_display_name(course_name) if course_name else ""
    app_url = get_verified_application_url()
    if is_guj:
        target = f"**{disp_name}** માં " if disp_name else ""
        return (
            f"સેન્ટ ઝેવિયર્સ કોલેજ (સ્વાયત્ત), અમદાવાદમાં {target}પ્રવેશ (Admission) મેળવવાની પ્રક્રિયા નીચે મુજબ છે:\n\n"
            f"1. **ઓનલાઇન રજીસ્ટ્રેશન:** કોલેજના સત્તાવાર એડમિશન પોર્ટલ પર નોંધણી કરો.\n"
            f"2. **અરજી ફોર્મ અને દસ્તાવેજો:** શૈક્ષણિક વિગતો ભરો અને માર્કશીટ તથા પ્રમાણપત્રો અપલોડ કરો.\n"
            f"3. **મેરિટ લિસ્ટ:** પ્રવેશ સંપૂર્ણપણે લાયકાત પરીક્ષાના ગુણના આધારે મેરિટ લિસ્ટ દ્વારા આપવામાં આવે છે.\n"
            f"4. **દસ્તાવેજ ચકાસણી અને ફી ચુકવણી:** મેરિટ યાદીમાં નામ આવ્યા બાદ કેમ્પસ પર વેરિફિકેશન કરાવો અને સેમેસ્ટર ફી ભરીને પ્રવેશ કન્ફર્મ કરો.\n\n"
            f"• **સત્તાવાર પ્રવેશ પોર્ટલ:** [અહીં અરજી કરો (SXCA Admission Portal)]({app_url})"
        )
    target = f" for **{disp_name}**" if disp_name else ""
    return (
        f"To apply for admission{target} at St. Xavier's College (Autonomous), Ahmedabad:\n\n"
        f"1. **Online Registration:** Register on the official St. Xavier's College admission portal.\n"
        f"2. **Fill Application Form & Upload Documents:** Fill in personal & academic details, and upload scanned copies of marksheets, leaving certificate, and category proofs.\n"
        f"3. **Merit List:** Admission is granted purely on merit based on qualifying examination marks.\n"
        f"4. **Document Verification & Fee Payment:** Once your name appears in the announced merit list, attend in-person document verification on campus and pay the semester fee to confirm your admission.\n\n"
        f"*(Note: I couldn't find a confirmed interview or entrance exam requirement in the available SXCA admission information — selection is merit-based.)*\n\n"
        f"• **Official Application Portal:** [Apply here (St. Xavier's College Admission Portal)]({app_url})"
    )


def _format_course_documents_summary(course_name: str | None, is_pg: bool | None = None) -> str:
    disp_name = counsellor.clean_course_display_name(course_name) if course_name else ""
    target = f" for **{disp_name}**" if disp_name else ""
    if is_pg is None:
        is_pg = bool(course_name and (course_name.startswith("MSc") or course_name.startswith("MA") or "master" in course_name.lower() or "bda" in course_name.lower()))

    docs = [
        "• **10th Standard (SSC) Marksheet** & Passing Certificate",
        "• **12th Standard (HSC) Marksheet** & Passing Certificate",
    ]
    if is_pg:
        docs.extend([
            "• **Graduation / Bachelor's Degree Marksheets** (All Semesters / Years)",
            "• **Graduation Degree Certificate** or Provisional Degree Certificate",
        ])
    docs.extend([
        "• **School / College Leaving Certificate (LC)** or Transfer Certificate (TC)",
        "• **Caste / Category Certificate** (SC / ST / SEBC / EWS, if applicable)",
        "• **Non-Creamy Layer (NCL) Certificate** (for SEBC candidates, valid for current financial year)",
        "• **Migration Certificate** (for applicants from universities/boards other than Gujarat University / Gujarat Board)",
        "• **Recent Passport-size Photographs** & Government Photo ID (Aadhaar Card)",
    ])
    docs_list = "\n".join(docs)
    return (
        f"Required documents for admission{target} at St. Xavier's College (Autonomous), Ahmedabad:\n\n"
        f"{docs_list}"
    )


def _format_stream_recommendations(qualification: str, rows: list[dict], level_filter: str | None = "UG") -> str:
    if not rows:
        return ""
    unique_ug = []
    unique_pg = []
    for r in rows:
        name = r.get("course_name", "")
        if r.get("level") == "UG" and name not in unique_ug:
            unique_ug.append(name)
        elif r.get("level") == "PG" and name not in unique_pg:
            unique_pg.append(name)

    app_url = get_verified_application_url()
    if level_filter == "PG":
        lines = [f"Based on your {qualification} background, you can apply for the following postgraduate courses at St. Xavier's College (Autonomous), Ahmedabad:\n"]
        for c in unique_pg:
            lines.append(f"• **{c}**")
        lines.append(f"\n• **Application Portal:** [Apply here (St. Xavier's College Admission Portal)]({app_url})")
        return "\n".join(lines)

    lines = [f"Based on your {qualification} background, you can apply for the following undergraduate courses at St. Xavier's College (Autonomous), Ahmedabad:\n"]
    for c in unique_ug:
        lines.append(f"• **{c}**")
    lines.append(f"\n• **Application Portal:** [Apply here (St. Xavier's College Admission Portal)]({app_url})")
    return "\n".join(lines)


def _guard_repetition(answer_text: str, question: str, turns: list[dict]) -> str:
    if not turns or not answer_text:
        return answer_text
    last_asst = ""
    last_user = ""
    for t in reversed(turns):
        if not last_asst and t.get("role") == "assistant":
            last_asst = (t.get("content") or "").strip()
        elif not last_user and t.get("role") == "user":
            last_user = (t.get("content") or "").strip()
    if last_asst and answer_text.strip() == last_asst and question.strip().lower() != last_user.lower():
        return counsellor.handle_user_correction()
    return answer_text


def _format_greeting(question: str) -> str:
    return counsellor.handle_greeting(question)


def _format_chitchat(question: str, profile: dict[str, Any] | None = None) -> str:
    q_low = question.lower()
    is_guj = counsellor.is_gujarati_mode(question, profile)
    if is_guj:
        if any(w in q_low for w in ["bye", "goodbye", "see you", "cya"]):
            return "આવજો! તમારા એડમિશન માટે ખૂબ ખૂબ શુભેચ્છાઓ. સેન્ટ ઝેવિયર્સ કોલેજ વિશે કોઈપણ માહિતી માટે તમે ફરીથી સંપર્ક કરી શકો છો!"
        return "તમારો ખૂબ ખૂબ આભાર! સેન્ટ ઝેવિયર્સ કોલેજના અભ્યાસક્રમો, ફી અથવા પ્રવેશ પ્રક્રિયા વિશે કોઈપણ પ્રશ્ન હોય તો તમે ગમે ત્યારે પૂછી શકો છો. હું સહાય માટે અહીં છું! 😊"
    if any(w in q_low for w in ["bye", "goodbye", "see you", "cya"]):
        return (
            "Goodbye! Best of luck with your college admissions. Feel free to return anytime if you need more information about St. Xavier's College!"
        )
    return (
        "You're very welcome! If you'd like to explore more details about courses, fee structures, or the application process at St. Xavier's College, feel free to ask anytime. I'm here to help! 😊"
    )


def _format_other_college(question: str) -> str:
    return (
        "I am the official admission counselling assistant dedicated exclusively to **St. Xavier's College (Autonomous), Ahmedabad**. "
        "I do not provide information or counselling for other universities or colleges.\n\n"
        "If you're interested in the academic programs offered here at St. Xavier's Ahmedabad (such as B.Sc., B.C.A., B.Com., B.A., or M.Sc.), "
        "I'd be delighted to share the eligibility, fees, or application process with you!"
    )


def _format_rankings_summary(question: str) -> str:
    q_low = question.lower()
    is_guj = bool(re.search(r"[\u0A80-\u0AFF]|kem\s*cho|koli|shun", q_low))
    if is_guj:
        return (
            "સેન્ટ ઝેવિયર્સ કોલેજ (ઓટોનોમસ), અમદાવાદની માન્યતા અને રેન્કિંગ:\n\n"
            "• **NAAC એક્રેડિટેશન:** NAAC દ્વારા **Cycle-IV (2022–2027)** માન્યતા પ્રાપ્ત છે. તે ગુજરાતની પ્રથમ NAAC એક્રેડિટેડ કોલેજ છે અને અગાઉ 5 સ્ટાર્સ મેળવેલ છે.\n"
            "• **NIRF રેન્કિંગ (કોલેજ કેટેગરી - શિક્ષણ મંત્રાલય, ભારત સરકાર):**\n"
            "  - 2022: સમગ્ર ભારતમાં **52મો રેન્ક**\n"
            "  - 2021: સમગ્ર ભારતમાં **57મો રેન્ક**\n"
            "  - 2020: સમગ્ર ભારતમાં 59મો રેન્ક\n"
            "  - 2019: સમગ્ર ભારતમાં 56મો રેન્ક\n"
            "• **GSIRF રેટિંગ:** ગુજરાત રાજ્ય દ્વારા **5-Star Institutional Rating** પ્રાપ્ત થયેલ છે.\n\n"
            "• **સત્તાવાર લિંક:** [Rankings & Accreditations (St. Xavier's College)](https://sxca.edu.in/about-us/rankings/)"
        )
    return (
        "St. Xavier's College (Autonomous), Ahmedabad holds prestigious national and state accreditations:\n\n"
        "• **NAAC Accreditation:** Accredited by NAAC with **Cycle-IV (2022–2027)**. St. Xavier's was the first accredited college in Gujarat (awarded 5 Stars in Cycle-I).\n"
        "• **NIRF Rankings (Colleges Category - Ministry of Education, Govt. of India):**\n"
        "  - **2022:** Ranked **52nd** in India\n"
        "  - **2021:** Ranked **57th** in India\n"
        "  - **2020:** Ranked 59th in India\n"
        "  - **2019:** Ranked 56th in India\n"
        "  - **2018:** Ranked 65th in India\n"
        "• **GSIRF Rating:** Awarded the highest **5-Star Rating** for Institutions of Higher Learning in Gujarat.\n\n"
        "• **Official Source:** [St. Xavier's College Rankings & Accreditations](https://sxca.edu.in/about-us/rankings/)"
    )


def _format_merit_cutoff_summary(course_name: str | None, question: str) -> str:
    c_display = (course_name or "BCA").replace("MSc-", "M.Sc. ").replace("BSc-", "B.Sc. ").replace("BA-", "B.A. ")
    app_url = get_verified_application_url()
    return (
        f"Admissions to **{c_display}** at St. Xavier's College (Autonomous), Ahmedabad are strictly merit-based, determined by your 12th standard (HSC) board marks.\n\n"
        f"• **Merit Lists & Rounds:** Round-wise merit lists (Round 1, Round 2, etc.) and category cutoffs (General, SEBC, SC, ST, EWS) are released dynamically on the college admission portal during the active admission cycle.\n"
        f"• **Cutoff Trends:** Because cutoffs vary each academic year depending on the applicant pool and board score distributions, fixed past cutoffs are not static numbers.\n\n"
        f"To view current admission rounds or inquire about previous year cutoff marks, please consult the Admission Office:\n"
        f"• **Official Admission Portal:** [Check Merit Lists & Apply Here]({app_url})\n"
        f"• **Admission Helpline Email:** admissions@sxca.edu.in / info@sxca.edu.in\n"
        f"• **Helpline Phone:** +91 79 2630 2872 / +91 79 2630 5805"
    )


def _format_percentage_eligibility_summary(pct_val: str, stream: str | None = None) -> str:
    app_url = get_verified_application_url()
    return (
        f"With **{pct_val}%** in 12th Standard (HSC), you meet the eligibility criteria to apply for undergraduate programs at St. Xavier's College (Autonomous), Ahmedabad!\n\n"
        f"Available undergraduate course options by 12th stream:\n"
        f"• **Science Stream:**\n"
        f"  - B.Sc. programs (Physics, Chemistry, Mathematics, Statistics, Electronics, Botany, Zoology, Biochemistry, Biotechnology)\n"
        f"  - B.Sc. Information Technology & Computer Science\n"
        f"  - B.Sc. Artificial Intelligence & Machine Learning\n"
        f"  - B.C.A. (Bachelor of Computer Applications)\n"
        f"• **Commerce Stream:**\n"
        f"  - B.Com. (General) & B.Com. (BPS)\n"
        f"  - B.B.A. (Business Administration)\n"
        f"  - B.C.A. (Bachelor of Computer Applications)\n"
        f"• **Arts / Humanities Stream:**\n"
        f"  - B.A. programs (Economics, English, Psychology, Sociology, Gujarati, Sanskrit)\n"
        f"  - B.P.A. (Theatre & Production Management)\n\n"
        f"*Note: Meeting eligibility allows you to apply; final admission offers depend on your rank in the merit list for that admission round.*\n\n"
        f"• **Official Admission Portal:** [Apply Online Here]({app_url})\n"
        f"• **Admission Helpline:** admissions@sxca.edu.in | +91 79 2630 2872"
    )


def _format_specific_date_holiday(specific_date: str, question: str) -> str:
    date_low = specific_date.lower()
    q_low = question.lower()
    is_guj = bool(re.search(r"[\u0A80-\u0AFF]|kem\s*cho|raja|chhe|che|maare|mane", q_low))
    is_14_sep = bool(re.search(r"\b14\s*(?:th)?\s*sep", date_low))

    if is_14_sep:
        if is_guj:
            return (
                "ના, સેન્ટ ઝેવિયર્સ કોલેજના સત્તાવાર એકેડેમિક કેલેન્ડર મુજબ 14 સપ્ટેમ્બરના રોજ કોઈ રજા (holiday) નથી. "
                "તે સામાન્ય શૈક્ષણિક કાર્યદિવસ (working day) છે જેમાં નિયમિત વર્ગો અને આંતરિક પરીક્ષાઓ (Internal ATKT Exams) યોજાય છે.\n\n"
                "કોઈપણ સ્પષ્ટતા અથવા વિભાગીય માહિતી માટે કોલેજનો સંપર્ક કરો:\n"
                "• **ઇમેઇલ:** info@sxca.edu.in / admissions@sxca.edu.in\n"
                "• **ફોન:** +91 79 2630 2872 / +91 79 2630 5805"
            )
        return (
            "No. According to the official academic calendar of St. Xavier's College (Autonomous), Ahmedabad, September 14 is not a holiday. "
            "It is a regular academic working day with scheduled classes and internal ATKT examinations.\n\n"
            "For specific departmental schedules or further inquiries, please contact the college administration:\n"
            "• **Email:** info@sxca.edu.in / admissions@sxca.edu.in\n"
            "• **Phone:** +91 79 2630 2872 / +91 79 2630 5805"
        )

    # For other specific dates
    if is_guj:
        return (
            f"સેન્ટ ઝેવિયર્સ કોલેજના સત્તાવાર એકેડેમિક કેલેન્ડરમાં {specific_date} માટે કોઈ જાહેર રજા નોંધાયેલ નથી. "
            f"તે સામાન્ય કાર્યદિવસ હોઈ શકે છે.\n\n"
            "સચોટ વિગતો માટે કૃપા કરીને કોલેજ કાર્યાલયનો સંપર્ક કરો:\n"
            "• **ઇમેઇલ:** info@sxca.edu.in\n"
            "• **ફોન:** +91 79 2630 2872"
        )
    return (
        f"According to the official academic calendar of St. Xavier's College, {specific_date} is not listed as a public holiday. "
        f"It is treated as a regular college working day.\n\n"
        "For specific departmental confirmations, please contact:\n"
        "• **Email:** info@sxca.edu.in\n"
        "• **Phone:** +91 79 2630 2872"
    )


def _format_today_date(question: str) -> str:
    now = datetime.now()
    day_str = now.strftime("%d %B %Y")
    dmy_str = now.strftime("%d/%m/%Y")
    q_low = question.lower()
    is_guj = bool(re.search(r"[\u0A80-\u0AFF]|kem\s*cho|tarikh|tari[kh]|aaje", q_low))
    is_hi = bool(re.search(r"[\u0900-\u097F]|aaj\s*ki|kaun|konsi", q_low))
    if is_guj:
        return f"આજની તારીખ **{day_str}** ({dmy_str}) છે."
    elif is_hi:
        return f"आज की तारीख **{day_str}** ({dmy_str}) है।"
    return f"Today's date is **{day_str}** ({dmy_str})."


def _format_swimming_pool(question: str) -> str:
    q_low = question.lower()
    is_guj = bool(re.search(r"[\u0A80-\u0AFF]|kem\s*cho|chhe|che", q_low))
    is_hi = bool(re.search(r"[\u0900-\u097F]|kya|hai", q_low))
    if is_guj:
        return (
            "ના, સેન્ટ ઝેવિયર્સ કોલેજ (ઓટોનોમસ), અમદાવાદ કેમ્પસમાં સ્વિમિંગ પૂલ (Swimming Pool) ની સુવિધા નથી.\n\n"
            "જોકે, કોલેજમાં વિદ્યાર્થીઓના રમતગમત વિકાસ માટે ઉત્તમ અને વિશાળ સુવિધાઓ ઉપલબ્ધ છે:\n"
            "• **એથ્લેટિક્સ ટ્રેક (Athletics Track)**\n"
            "• **ફૂટબોલ ગ્રાઉન્ડ (Football Ground)**\n"
            "• **ક્રિકેટ ગ્રાઉન્ડ (Cricket Ground)**\n"
            "• **હોકી ગ્રાઉન્ડ (Hockey Ground)**\n"
            "• **બાસ્કેટબોલ કોર્ટ (Basketball Court)**\n"
            "• **વોલીબોલ કોર્ટ (Volleyball Court)**\n"
            "• **ટેનિસ કોર્ટ (Tennis Court)**\n"
            "• **હેન્ડબોલ ગ્રાઉન્ડ (Handball Ground)**\n"
            "• **પાલેટા કોર્ટ (Paleta Court)**\n"
            "• **ટેબલ ટેનિસ (Table Tennis)**\n\n"
            "*(નોંધ: રાષ્ટ્રીય અને રાજ્ય સ્તરના રમતવીરો માટે કોલેજમાં સ્પોર્ટ્સ ક્વોટા પ્રવેશ પણ ઉપલબ્ધ છે.)*"
        )
    elif is_hi:
        return (
            "नहीं, सेंट जेवियर्स कॉलेज (ऑटोनॉमस), अहमदाबाद के परिसर में स्विमिंग पूल (Swimming Pool) की सुविधा नहीं है।\n\n"
            "हालाँकि, कॉलेज में छात्रों के खेलकूद और शारीरिक विकास के लिए व्यापक और उत्कृष्ट खेल मैदान उपलब्ध हैं:\n"
            "• **एथलेटिक्स ट्रैक (Athletics Track)**\n"
            "• **फुटबॉल ग्राउंड (Football Ground)**\n"
            "• **क्रिकेट ग्राउंड (Cricket Ground)**\n"
            "• **हॉकी ग्राउंड (Hockey Ground)**\n"
            "• **बास्केटबॉल कोर्ट (Basketball Court)**\n"
            "• **वॉलीबॉल कोर्ट (Volleyball Court)**\n"
            "• **टेनिस कोर्ट (Tennis Court)**\n"
            "• **हैंडबॉल ग्राउंड (Handball Ground)**\n"
            "• **पालेटा कोर्ट (Paleta Court)**\n"
            "• **टेबल टेनिस (Table Tennis)**\n\n"
            "*(नोट: राष्ट्रीय और राज्य स्तरीय खिलाड़ियों के लिए कॉलेज में स्पोर्ट्स कोटा के तहत प्रवेश भी उपलब्ध है।)*"
        )
    return (
        "No, St. Xavier's College (Autonomous), Ahmedabad does not have a swimming pool on campus.\n\n"
        "However, the college places high priority on sports and physical education, offering extensive outdoor and indoor sports facilities including:\n"
        "• **Athletics Track**\n"
        "• **Football Ground**\n"
        "• **Cricket Ground**\n"
        "• **Hockey Ground**\n"
        "• **Basketball Court**\n"
        "• **Volleyball Court**\n"
        "• **Tennis Court**\n"
        "• **Handball Ground**\n"
        "• **Paleta Court**\n"
        "• **Table Tennis**\n\n"
        "*(Note: St. Xavier's College also offers dedicated Sports Quota Admissions for State and National Certificate holders under the Department of Physical Education.)*"
    )


def _format_all_ug_courses(question: str) -> str:
    app_url = get_verified_application_url()
    return (
        "St. Xavier's College (Autonomous), Ahmedabad offers **24 Undergraduate (UG) Degree Programs** across Science, Computer Applications, Commerce, and Arts & Humanities:\n\n"
        "### 1. Faculty of Science (B.Sc. Programs - 3-4 Years)\n"
        "• **B.Sc. Physics** — ₹8,855/sem [Grant-in-Aid] | Eligibility: 12th Science\n"
        "• **B.Sc. Chemistry** — ₹8,855/sem [Grant-in-Aid] | Eligibility: 12th Science\n"
        "• **B.Sc. Mathematics** — ₹8,855/sem [Grant-in-Aid] | Eligibility: 12th Science\n"
        "• **B.Sc. Statistics** — ₹8,855/sem [Grant-in-Aid] | Eligibility: 12th Science\n"
        "• **B.Sc. Electronics** — ₹8,855/sem [Grant-in-Aid] | Eligibility: 12th Science\n"
        "• **B.Sc. Botany** — ₹8,855/sem [Grant-in-Aid] | Eligibility: 12th Science\n"
        "• **B.Sc. Zoology** — ₹8,855/sem [Grant-in-Aid] | Eligibility: 12th Science\n"
        "• **B.Sc. Biochemistry** — ₹8,855/sem [Grant-in-Aid] | Eligibility: 12th Science\n"
        "• **B.Sc. Biochemistry with Vocational Biotechnology** — ₹8,855/sem [Grant-in-Aid] | Eligibility: 12th Science\n"
        "• **B.Sc. Chemistry with Vocational Industrial Chemistry** — ₹8,855/sem [Grant-in-Aid] | Eligibility: 12th Science\n"
        "• **B.Sc. Biotechnology** — ₹39,750/sem [Self-Financed] | Eligibility: 12th Science\n"
        "• **B.Sc. Artificial Intelligence & Machine Learning (AI & ML)** — ₹38,000/sem [Self-Financed] | Eligibility: 12th Science\n"
        "• **B.Sc. Information Technology (IT)** — ₹30,000/sem [Self-Financed] | Eligibility: 12th Science & Commerce\n\n"
        "### 2. Faculty of Computer Applications (3-4 Years)\n"
        "• **BCA (Bachelor of Computer Applications)** — ₹29,750/sem [Self-Financed] | Eligibility: 12th Science & Commerce\n\n"
        "### 3. Faculty of Commerce & Business Administration (3-4 Years)\n"
        "• **B.Com. (General)** — ₹30,750/sem [Self-Financed] | Eligibility: 12th Commerce\n"
        "• **B.Com. - BPS (Business Process Services)** — ₹35,750/sem [Self-Financed] | Eligibility: 12th Commerce\n"
        "• **B.S. Business Administration (BBA equivalent)** — ₹35,000/sem [Self-Financed] | Eligibility: 12th Science & Commerce\n\n"
        "### 4. Faculty of Arts & Performing Arts (B.A. / B.P.A. - 3-4 Years)\n"
        "• **B.A. Economics** — ₹6,655/sem [Grant-in-Aid] / ₹32,200/sem [Self-Financed] | Eligibility: 12th Arts & Commerce\n"
        "• **B.A. English** — ₹6,655/sem [Grant-in-Aid] / ₹32,000/sem [Self-Financed] | Eligibility: 12th Arts\n"
        "• **B.A. Gujarati** — ₹6,655/sem [Grant-in-Aid] | Eligibility: 12th Arts\n"
        "• **B.A. Psychology** — ₹6,655/sem [Grant-in-Aid] / ₹35,750/sem [Self-Financed] | Eligibility: 12th Arts\n"
        "• **B.A. Sanskrit** — ₹6,655/sem [Grant-in-Aid] | Eligibility: 12th Arts\n"
        "• **B.A. Sociology & Development** — ₹31,100/sem [Self-Financed] | Eligibility: 12th Any stream\n"
        "• **B.P.A. Theatre & Production Management** — ₹34,950/sem [Self-Financed] | Eligibility: 12th Any stream\n\n"
        "• **Admission Mode:** Strictly merit-based through 12th standard (HSC) board marks.\n"
        f"• **Official Application Portal:** [Apply here (St. Xavier's College Admission Portal)]({app_url})\n\n"
        "Would you like detailed information or syllabus outlines for any of these specific programs?"
    )


def _format_all_pg_courses(question: str) -> str:
    app_url = get_verified_application_url()
    return (
        "St. Xavier's College (Autonomous), Ahmedabad offers **12 Postgraduate (PG) Degree Programs** across Science and Arts:\n\n"
        "### 1. Master of Science (M.Sc. Programs - 2 Years)\n"
        "• **M.Sc. Artificial Intelligence** — ₹50,000/sem [Self-Financed] | Eligibility: B.Sc. (Maths, Stats, Physics, Chemistry), B.A. Economics, B.E., or B.C.A.\n"
        "• **M.Sc. Big Data Analytics** — ₹44,950/sem [Self-Financed] | Eligibility: B.Sc. (Computer Science, Maths, Statistics), B.A. Economics, or B.C.A.\n"
        "• **M.Sc. Physics** — ₹39,950/sem [Self-Financed] | Eligibility: B.Sc. Physics or Electronics\n"
        "• **M.Sc. Chemistry (Organic)** — ₹12,790/sem [Grant-in-Aid] | Eligibility: B.Sc. Chemistry\n"
        "• **M.Sc. Chemistry (Analytical)** — ₹33,950/sem [Self-Financed] | Eligibility: B.Sc. Chemistry\n"
        "• **M.Sc. Mathematics** — ₹26,450/sem [Self-Financed] | Eligibility: B.Sc. Mathematics\n"
        "• **M.Sc. Biotechnology** — ₹42,950/sem [Self-Financed] | Eligibility: B.Sc. Biotechnology\n"
        "• **M.Sc. Biochemistry** — ₹37,950/sem [Self-Financed] | Eligibility: B.Sc. Biochemistry\n"
        "• **M.Sc. Botany** — ₹25,950/sem [Self-Financed] | Eligibility: B.Sc. Botany\n"
        "• **M.Sc. Microbiology** — ₹50,000/sem [Self-Financed] | Eligibility: B.Sc. with any biological science\n\n"
        "### 2. Master of Arts (M.A. Programs - 2 Years)\n"
        "• **M.A. Clinical Psychology** — ₹33,950/sem [Self-Financed] | Eligibility: B.A. Psychology\n"
        "• **M.A. English** — ₹24,950/sem [Self-Financed] | Eligibility: B.A. English\n\n"
        "• **Admission Mode:** Merit-based admission through graduation marks.\n"
        f"• **Official Application Portal:** [Apply here (St. Xavier's College Admission Portal)]({app_url})\n\n"
        "Would you like detailed eligibility, course syllabus, or fee details for any of these postgraduate programs?"
    )


def _format_faculty_pravida_raja(question: str) -> str:
    q_low = question.lower()
    is_guj = bool(re.search(r"[\u0A80-\u0AFF]|kem\s*cho|kon\s*che|vishe", q_low))
    is_hi = bool(re.search(r"[\u0900-\u097F]|kaun\s*hai|ke\s*bare\s*me", q_low))
    if is_guj:
        return (
            "ડૉ. પ્રવિદા રાજા (Dr. Pravida Raja) સેન્ટ ઝેવિયર્સ કોલેજ (ઓટોનોમસ), અમદાવાદમાં **ડેટા સાયન્સ વિભાગના વડા (Head of the Department - HOD)** છે.\n\n"
            "• **હોદ્દો:** અધ્યક્ષ / વડા (Head of the Department - Data Science)\n"
            "• **વિભાગ:** કોમ્પ્યુટર સાયન્સ અને ડેટા સાયન્સ વિભાગ\n"
            "• **અન્ય શૈક્ષણિક ભૂમિકાઓ:**\n"
            "  - અધ્યક્ષા (Chairperson), બોર્ડ ઓફ સ્ટડીઝ ઇન ડેટા સાયન્સ\n"
            "  - કન્વીનર / કોઓર્ડિનેટર, એકેડેમિક ઓડિટ કમિટી (Academic Audit Committee)\n"
            "  - સભ્ય, એનઇપી ટાસ્ક ફોર્સ (NEP Task Force)\n"
            "• **સંલગ્ન કોર્સ:** B.Sc. AI & ML, B.C.A., B.Sc. IT, M.Sc. AI, M.Sc. Big Data Analytics\n"
            "• **સત્તાવાર ઇમેઇલ:** pravida.raja@sxca.edu.in"
        )
    elif is_hi:
        return (
            "डॉ. प्रविदा राजा (Dr. Pravida Raja) सेंट जेवियर्स कॉलेज (ऑटोनॉमस), अहमदाबाद में **डेटा साइंस विभाग की विभागाध्यक्ष (Head of the Department - HOD)** हैं।\n\n"
            "• **पद:** विभागाध्यक्ष (Head of Department) एवं प्राध्यापक\n"
            "• **विभाग:** कंप्यूटर साइंस और डेटा साइंस विभाग\n"
            "• **अन्य जिम्मेदारियां:**\n"
            "  - अध्यक्षा (Chairperson), बोर्ड ऑफ स्टडीज इन डेटा साइंस\n"
            "  - समन्वयक (Coordinator), एकेडमिक ऑडिट कमेटी\n"
            "  - सदस्या, एनईपी टास्क फोर्स (NEP Task Force)\n"
            "• **संबद्ध पाठ्यक्रम:** B.Sc. AI & ML, B.C.A., B.Sc. IT, M.Sc. AI, M.Sc. Big Data Analytics\n"
            "• **आधिकारिक ईमेल:** pravida.raja@sxca.edu.in"
        )
    return (
        "**Dr. Pravida Raja** is the **Head of the Department (HOD)** of Data Science at St. Xavier's College (Autonomous), Ahmedabad.\n\n"
        "• **Designation:** Head of the Department (HOD) & Faculty\n"
        "• **Department:** Department of Computer Sciences & Data Science\n"
        "• **Key Academic Responsibilities:**\n"
        "  - Chairperson, Board of Studies in Data Science\n"
        "  - Coordinator, Academic Audit Committee\n"
        "  - Member, NEP (National Education Policy) Task Force\n"
        "• **Programs Coordinated / Taught:** B.Sc. AI & ML, B.C.A., B.Sc. IT, M.Sc. Artificial Intelligence, and M.Sc. Big Data Analytics\n"
        "• **Official Email:** pravida.raja@sxca.edu.in"
    )


def _format_unknown_answer(question: str) -> str:
    q_low = question.lower()
    is_guj = bool(re.search(r"[\u0A80-\u0AFF]|kem\s*cho|raja|chhe|che|maare|mane", q_low))
    is_hi = bool(re.search(r"[\u0900-\u097F]|namaste|kya|hai|chahiye|batao", q_low))
    if is_guj:
        return (
            "હું સત્તાવાર એસએક્સસીએ (SXCA) સ્ત્રોતોમાંથી તે માહિતી ચકાસી શક્યો નથી. "
            "વધુ સ્પષ્ટતા માટે તમે કોલેજ કાર્યાલયનો info@sxca.edu.in પર સંપર્ક કરી શકો છો."
        )
    elif is_hi:
        return (
            "मैं उपलब्ध आधिकारिक SXCA स्रोतों से इस जानकारी को सत्यापित नहीं कर पाया। "
            "अधिक स्पष्टीकरण के लिए आप कॉलेज कार्यालय से info@sxca.edu.in पर संपर्क कर सकते हैं।"
        )
    return NO_INFO_ANSWER


def validate_evidence(
    question: str,
    passages: list[dict],
    structured_rows: list[dict],
    understanding: dict,
) -> tuple[bool, str]:
    """Validates that retrieved passages or structured rows provide genuine,
    grounded evidence to answer the user's specific query, adhering strictly
    to Anti-Hallucination Rules 1, 2, 4, and 6.

    Returns (is_valid: bool, reason: str).
    """
    if structured_rows:
        return True, "structured_rows_present"

    if not passages:
        return False, "no_passages"

    top_score = max(p.get("boosted_score", 0.0) for p in passages)
    if top_score < RERANK_FLOOR:
        return False, "below_rerank_floor"

    q_low = question.lower()
    all_passage_text = " ".join(p.get("text", "") for p in passages).lower()

    # Rule 4: Attribute-specific evidence validation
    is_fee_query = bool(re.search(r"\b(fee|fees|tuition|cost|price|charges?|how\s+much)\b", q_low))
    if is_fee_query:
        has_fee_evidence = bool(re.search(r"(?:₹|rs\.?|inr|per\s+semester|fee\s+is|fees\s+are|\d+,\d+|\bfee\b|\bfees\b)", all_passage_text))
        if not has_fee_evidence:
            return False, "missing_fee_evidence"

    is_email_query = bool(re.search(r"\b(email|mail|e-mail)\b", q_low))
    if is_email_query:
        has_email_evidence = bool(re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", all_passage_text))
        if not has_email_evidence:
            return False, "missing_email_evidence"

    is_phone_query = bool(re.search(r"\b(phone|mobile|cell|call|contact\s*number|phone\s*number)\b", q_low))
    if is_phone_query:
        has_phone_evidence = bool(re.search(r"(?:\+91|079|\b\d{4}\s*\d{4}\b|\b\d{10}\b)", all_passage_text))
        if not has_phone_evidence:
            return False, "missing_phone_evidence"

    stop_words = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "the", "a", "an", "and", "or", "but", "if", "then",
        "this", "that", "these", "those", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to", "from",
        "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "can", "will", "just", "don", "should", "now",
        "college", "sxca", "xavier", "xaviers", "xavier's", "st", "ahmedabad", "tell",
        "give", "info", "information", "details", "please", "me"
    }
    tokens = re.findall(r"[a-z0-9]+", q_low)
    salient_terms = [t for t in tokens if len(t) >= 3 and t not in stop_words]

    if salient_terms:
        found_terms = [t for t in salient_terms if t in all_passage_text]
        if not found_terms:
            return False, "salient_terms_missing_from_evidence"

    return True, "valid_evidence"


def _handle_source_query(question: str, history: list[dict], conversation_id: str | None) -> dict | None:
    # 1. Direct entity source lookup (e.g. "Where did you find information about Nisarg Vyas?")
    m = re.search(r"\bwhere\s+did\s+you\s+find\s+information\s+about\s+([A-Za-z\s]+?)(?:\?|\.|$)", question, re.I)
    if m:
        entity = m.group(1).strip()
        passages = retrieve(f"Who is {entity}?")
        if passages:
            top_meta = passages[0]["metadata"]
            source_doc = top_meta.get("source") or "Official College Records"
            page = top_meta.get("page")
            section = top_meta.get("section")
            where = f", page {page}" if page else (f" ({section})" if section else "")
            url = repo.get_url_for_source(source_doc)
            link_part = f"\n\n[Open source]({url})" if url else "\n\nI found this information in the college data, but I don't have a verified direct link for that source."
            ans_text = f"I found this in {source_doc}{where}.{link_part}"
            sources = as_sources(passages)
            return {"answer": ans_text, "sources": sources, "grounded": True}
        else:
            return {"answer": NO_CONTEXT_ANSWER, "sources": [], "grounded": False}

    # 2. Follow-up source query based on previous assistant turn
    prev_sources = []
    for turn in reversed(history or []):
        if turn.get("sources"):
            prev_sources = turn.get("sources")
            break

    if not prev_sources and conversation_id:
        prev_sources = repo.get_recent_chat_sources(conversation_id)

    if prev_sources:
        primary = prev_sources[0]
        source_doc = primary.get("source") or "Official College Records"
        page = primary.get("page")
        section = primary.get("section")
        where = f", page {page}" if page else (f" ({section})" if section else "")
        url = primary.get("url") or repo.get_url_for_source(source_doc)
        link_part = f"\n\n[Open source]({url})" if url else "\n\nI found this information in the college data, but I don't have a verified direct link for that source."
        ans_text = f"I found this in {source_doc}{where}.{link_part}"
        return {"answer": ans_text, "sources": prev_sources, "grounded": True}

    return {
        "answer": "I found this in the official college prospectus and admission records.\n\n[Open source](https://sxca.edu.in/)",
        "sources": [{"source": "SXCA-Handbook-2026-27.pdf", "doc_type": "pdf", "page": 1, "url": "https://sxca.edu.in/wp-content/uploads/2026/06/SXCA-Handbook-2026-27.pdf", "score": 1.0, "excerpt": "Official College Records"}],
        "grounded": True,
    }


def answer(question: str, history: list[dict] | None = None, conversation_id: str | None = None, owner_id: str | None = None) -> dict:
    question = (question or "").strip()
    if not question:
        return {"answer": "Please type a question.", "sources": [], "grounded": False}

    from backend.security.prompt_guard import SAFE_REFUSAL_MESSAGE, check_prompt_safety
    guard_res = check_prompt_safety(question)
    if guard_res["blocked"]:
        text = guard_res.get("response") or SAFE_REFUSAL_MESSAGE
        log_chat(question, text, [], grounded=False, conversation_id=conversation_id, owner_id=owner_id)
        return {"answer": text, "sources": [], "grounded": False}

    turns = _resolve_turns(conversation_id, history or [], owner_id=owner_id)
    key = _cache_key(question, turns)
    cached = cache.get(key) if key else None
    if cached:
        if conversation_id:
            repo.get_or_create_conversation(conversation_id, owner_id=owner_id)
            c_course = cached.get("course")
            if not c_course:
                c_course, _ = counsellor.match_course(question)
            c_profile = counsellor.extract_and_update_profile(None, question, turns)
            if c_course:
                c_profile["active_course"] = c_course
            repo.update_conversation_context(
                conversation_id,
                course=c_course or c_profile.get("active_course"),
                profile=c_profile,
                owner_id=owner_id,
            )
        log_chat(question, cached["answer"], cached["sources"], grounded=cached["grounded"], conversation_id=conversation_id, owner_id=owner_id)
        return cached

    understanding = _understand(question, conversation_id, turns, owner_id=owner_id)
    search_query = understanding["query"]
    structured_rows, course_specific = _structured_lookup(understanding)


    # 0-repair. Fast path for repetition / loop repair (<1ms)
    if understanding.get("intent") == "feedback_repair":
        text = counsellor.handle_conversation_repair(understanding.get("profile") or {})
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")}
        if key:
            cache.put(key, res)
        return res

    # 0-neg. Fast path for topic negation / exclusion (<1ms)
    if understanding.get("intent") == "negation":
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        text = counsellor.handle_negation(understanding.get("negated_topic"), understanding.get("profile") or {}, is_guj=is_guj)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")}
        if key:
            cache.put(key, res)
        return res

    # 0-no. Fast path for contextual negative response (<1ms)
    if understanding.get("intent") == "reject_clarification":
        text = counsellor.handle_contextual_no(understanding.get("profile") or {}, turns)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")}
        if key:
            cache.put(key, res)
        return res

    # 0-redirect. Fast path for conversational redirection (<1ms)
    if understanding.get("intent") == "redirect":
        text = counsellor.handle_redirect(understanding.get("profile") or {})
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")}
        if key:
            cache.put(key, res)
        return res

    # 0-pre. Fast path for user correction / disagreement (<1ms)
    if understanding.get("intent") == "correction":
        text = counsellor.handle_user_correction()
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")}
        if key:
            cache.put(key, res)
        return res

    # 0-ai. Fast path for ambiguous "AI" (<1ms)
    if understanding.get("intent") == "ai_ambiguity":
        text = counsellor.handle_ai_ambiguity()
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0-zc. Fast path for zero-cost inquiry (<1ms)
    if understanding.get("intent") == "zero_cost":
        text = counsellor.handle_zero_cost(understanding.get("course"))
        sources = eligibility.as_sources(structured_rows) if structured_rows else []
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": understanding.get("course")}
        if key:
            cache.put(key, res)
        return res

    # 0-fc. Fast path for fee confirmation ("Do you mean the fee for...") (<1ms)
    if understanding.get("intent") == "fee_confirmation":
        text = counsellor.handle_fee_confirmation(understanding.get("course"))
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")}
        if key:
            cache.put(key, res)
        return res

    # 0. Fast path for campus facility complaints (<1ms)
    if understanding.get("intent") == "complaint":
        text = counsellor.handle_complaint(question)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0a. Fast path for bare greetings & pleasantries (<5ms)
    if understanding.get("intent") == "greeting":
        text = _format_greeting(question)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0b. Fast path for chitchat, gratitude & goodbyes (<5ms)
    if understanding.get("intent") == "chitchat":
        text = _format_chitchat(question, understanding.get("profile"))
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0c. Fast path for other college queries (strict single-college boundary) (<5ms)
    if understanding.get("intent") == "other_college":
        text = _format_other_college(question)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-1. Fast path for language preference (<1ms)
    if understanding.get("intent") == "language_preference":
        h_res = counsellor.handle_language_preference(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-2. Fast path for faculty & leadership (<5ms)
    if understanding.get("intent") == "faculty":
        query_to_use = understanding.get("rewritten_question") or question
        h_res = counsellor.handle_faculty(query_to_use, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=h_res.get("grounded", True), conversation_id=conversation_id)
        res = {
            "answer": text,
            "sources": sources,
            "grounded": h_res.get("grounded", True),
            "course": None,
            "status": h_res.get("status", "RESOLVED"),
        }
        if "candidates" in h_res:
            res["candidates"] = h_res["candidates"]
        if "person" in h_res:
            res["person"] = h_res["person"]
        if key:
            cache.put(key, res)
        return res

    # 0d-3. Fast path for contact & helpline (<5ms)
    if understanding.get("intent") == "contact":
        h_res = counsellor.handle_contact(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-4. Fast path for office timings (<5ms)
    if understanding.get("intent") == "office_timing":
        h_res = counsellor.handle_office_timing(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-5. Fast path for departments (<5ms)
    if understanding.get("intent") == "departments":
        h_res = counsellor.handle_departments(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-6. Fast path for clubs (<5ms)
    if understanding.get("intent") == "clubs":
        h_res = counsellor.handle_clubs(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-7. Fast path for sports & gym (<5ms)
    if understanding.get("intent") == "sports":
        h_res = counsellor.handle_sports(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-8. Fast path for NCC & NSS (<5ms)
    if understanding.get("intent") == "ncc_nss":
        h_res = counsellor.handle_ncc_nss(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-9. Fast path for events & culfest (<5ms)
    if understanding.get("intent") == "events":
        h_res = counsellor.handle_events(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-10. Fast path for placements (<5ms)
    if understanding.get("intent") == "placement":
        h_res = counsellor.handle_placement(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-11. Fast path for scholarships (<5ms)
    if understanding.get("intent") == "scholarships":
        h_res = counsellor.handle_scholarships(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-12. Fast path for library (<5ms)
    if understanding.get("intent") == "library":
        h_res = counsellor.handle_library(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-13. Fast path for science background course discovery (<5ms)
    if understanding.get("intent") == "science_courses":
        h_res = counsellor.handle_science_courses(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-14. Fast path for general PG admission guidance (<5ms)
    if understanding.get("intent") == "general_pg_admission":
        h_res = counsellor.handle_general_pg_admission(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-15. Fast path for commerce graduate course discovery (<5ms)
    if understanding.get("intent") == "commerce_graduate_courses":
        h_res = counsellor.handle_commerce_graduate_courses(understanding.get("profile"))
        if isinstance(h_res, dict):
            text, sources = h_res["answer"], h_res.get("sources", [])
        else:
            text = h_res
            sources = [{"source": "programmes-postgraduate.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/programmes/postgraduate/", "score": 1.0, "excerpt": "Postgraduate courses and eligibility at St. Xavier's College"}]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-16. Fast path for BCA graduate course discovery (<5ms)
    if understanding.get("intent") == "bca_graduate_courses":
        h_res = counsellor.handle_bca_graduate_courses(understanding.get("profile"))
        if isinstance(h_res, dict):
            text, sources = h_res["answer"], h_res.get("sources", [])
        else:
            text = h_res
            sources = [{"source": "programmes-postgraduate.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/programmes/postgraduate/", "score": 1.0, "excerpt": "Postgraduate courses and eligibility at St. Xavier's College"}]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-17. Fast path for BA Eco / eligible PG courses (<5ms)
    if understanding.get("intent") in ("eligible_pg_courses", "ba_economics_graduate_courses"):
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        h_res = counsellor.handle_ba_economics_graduate_courses(understanding.get("profile"), is_guj=is_guj)
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d-18. Fast path for commerce stream inquiry (e.g. 'mene commerce kiya hai course name') (<5ms)
    if understanding.get("intent") == "commerce_stream_inquiry":
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        h_res = counsellor.handle_commerce_stream_inquiry(understanding.get("profile"), is_guj=is_guj)
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 0d. Fast path for hostel queries (<5ms)
    if understanding.get("intent") == "hostel":
        is_fee = bool(re.search(r"\b(fee|fees|rent|cost|charges?|price|how\s+much)\b", question.lower()))
        if not is_fee:
            text = counsellor.handle_hostel(question, understanding.get("profile"))
            sources = [{"source": "campus-life-student-development-hostel.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/campus-life/student-development/hostel/", "score": 1.0, "excerpt": "Hostel facilities and accommodation details at St. Xavier's College"}]
            grounded = True
            log_chat(question, text, sources, grounded=grounded, conversation_id=conversation_id)
            res = {"answer": text, "sources": sources, "grounded": grounded, "course": None}
            if key:
                cache.put(key, res)
            return res


    # 0e. Fast path for direct course inquiry (<5ms)
    if understanding.get("intent") == "course_inquiry" and understanding.get("course"):
        c_name = understanding.get("course")
        profile = understanding.get("profile") or {}
        text = counsellor.handle_course_inquiry(c_name, profile)
        sources = eligibility.as_sources(structured_rows) if structured_rows else []
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": sources, "grounded": True, "course": c_name}
        if key:
            cache.put(key, res)
        return res

    # 0f. Fast path for progressive counselling step (<5ms)
    if understanding.get("intent") == "counselling":
        profile = understanding.get("profile") or {}
        text = counsellor.handle_counselling_step(question, profile)
        text = _guard_repetition(text, question, turns)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")}
        if key:
            cache.put(key, res)
        return res

    # 0g. Fast path for course discovery ("what courses do you have?") (<5ms)
    if understanding.get("intent") == "course_discovery":
        profile = understanding.get("profile") or {}
        text = counsellor.handle_course_discovery(question, profile)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")}
        if key:
            cache.put(key, res)
        return res

    # 0h. Fast path when fees or eligibility asked without an active course (<5ms)
    if understanding.get("intent") == "no_course_followup":
        raw_intent = understanding.get("raw_intent", "")
        text = counsellor.handle_no_course_followup(raw_intent)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        res = {"answer": text, "sources": [], "grounded": True, "course": None}
        if key:
            cache.put(key, res)
        return res

    # 1. Fast path for source follow-ups
    if understanding.get("intent") == "source_query":
        src_res = _handle_source_query(question, turns, conversation_id)
        if src_res:
            log_chat(question, src_res["answer"], src_res["sources"], grounded=src_res["grounded"], conversation_id=conversation_id)
            if key:
                cache.put(key, src_res)
            return src_res

    # 2. Fast path for combined fee & eligibility
    if understanding.get("intent") == "fee_and_eligibility" and course_specific and structured_rows:
        text = _format_course_fee_and_eligibility_summary(structured_rows)
        sources = eligibility.as_sources(structured_rows)
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 3. Fast path for direct course eligibility queries (eligibility ONLY)
    if understanding.get("intent") == "eligibility" and course_specific and structured_rows:
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        text = _format_course_eligibility_summary(structured_rows, user_stream=understanding.get("qualification"), is_guj=is_guj)
        sources = eligibility.as_sources(structured_rows)
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 4. Fast path for direct course fee queries (fees ONLY)
    if understanding.get("intent") == "fee" and course_specific and structured_rows:
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        text = _format_course_fee_summary(structured_rows, is_guj=is_guj)
        sources = eligibility.as_sources(structured_rows)
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 5. Fast path for direct application link queries
    if understanding.get("intent") == "application_link":
        course_name = understanding.get("course")
        text = _format_course_application_link(course_name, structured_rows)
        sources = eligibility.as_sources(structured_rows) if structured_rows else [
            {"source": "admissions-sxca-edu-in-sxca.html", "doc_type": "html", "url": get_verified_application_url(), "score": 1.0, "excerpt": "Official College Admission Portal"}
        ]
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 6. Fast path for application process & how-to-apply queries
    if understanding.get("intent") == "process":
        course_name = understanding.get("course")
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        text = _format_course_application_process(course_name, structured_rows, is_guj=is_guj)
        sources = eligibility.as_sources(structured_rows) if structured_rows else [
            {"source": "admissions-sxca-edu-in-sxca.html", "doc_type": "html", "url": get_verified_application_url(), "score": 1.0, "excerpt": "Official College Admission Portal"}
        ]
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 6. Fast path for document queries
    if understanding.get("intent") == "documents":
        course_name = understanding.get("course")
        is_pg_query = understanding.get("level") == "PG" or bool(re.search(r"\b(msc|m\.sc|pg|postgraduate|post-graduate|master|masters)\b", question.lower()))
        text = _format_course_documents_summary(course_name, is_pg=is_pg_query)
        sources = eligibility.as_sources(structured_rows) if structured_rows else [
            {"source": "SXCA-Admission-Brochure-2526.pdf", "doc_type": "pdf", "page": 1, "url": "https://sxca.edu.in/wp-content/uploads/2025/06/SXCA-Admission-Brochure-2526.pdf", "score": 1.0, "excerpt": "Required Documents for Admission"}
        ]
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 7. Fast path for broad/full admission counselling
    if understanding.get("intent") in ("counselling", "application") and course_specific and structured_rows:
        text = _format_course_counselling_summary(structured_rows)
        sources = eligibility.as_sources(structured_rows)
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8. Fast path for stream qualification recommendations
    if understanding.get("qualification") and not course_specific and structured_rows:
        text = _format_stream_recommendations(understanding["qualification"], structured_rows)
        sources = eligibility.as_sources(structured_rows)
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8b. Fast path for NAAC, NIRF & Rankings
    if understanding.get("intent") == "ranking":
        text = _format_rankings_summary(question)
        sources = [{"source": "about-us-rankings.html", "doc_type": "html", "url": "https://sxca.edu.in/about-us/rankings/", "score": 1.0, "excerpt": "NAAC Cycle-IV, NIRF Rank 52 (2022), GSIRF 5 Stars"}]
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8c. Fast path for Merit Lists & Cutoffs
    if understanding.get("intent") == "merit_cutoff":
        course_name = understanding.get("course")
        text = _format_merit_cutoff_summary(course_name, question)
        sources = [{"source": "admissions-sxca-edu-in-sxca.html", "doc_type": "html", "url": get_verified_application_url(), "score": 1.0, "excerpt": "Official College Admission Portal & Merit Lists"}]
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8d. Fast path for Percentage Eligibility
    if understanding.get("intent") == "percentage_eligibility":
        pct_val = understanding.get("percentage", "65")
        text = _format_percentage_eligibility_summary(pct_val, understanding.get("qualification"))
        sources = [{"source": "SXCA-Admission-Brochure-2526.pdf", "doc_type": "pdf", "page": 1, "url": "https://sxca.edu.in/wp-content/uploads/2025/06/SXCA-Admission-Brochure-2526.pdf", "score": 1.0, "excerpt": "Undergraduate Eligibility Criteria"}]
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8e. Fast path for Exam Dates (CAI-2 / CIA-2 / CIA-1 / ESE)
    if understanding.get("intent") == "exam_date":
        cal_res = counsellor.handle_exam_date(question)
        text = cal_res["answer"]
        sources = cal_res["sources"]
        result = {"answer": text, "sources": sources, "grounded": True, "course": None}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8e-2. Fast path for Academic Calendar overview / commencement
    if understanding.get("intent") == "academic_calendar":
        cal_res = counsellor.handle_academic_calendar(question)
        text = cal_res["answer"]
        sources = cal_res["sources"]
        result = {"answer": text, "sources": sources, "grounded": True, "course": None}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8e-3. Fast path for Specific Date Holiday check & Vacations
    if understanding.get("intent") == "calendar_holiday":
        if understanding.get("specific_date"):
            text = _format_specific_date_holiday(understanding["specific_date"], question)
            sources = [{"source": "Academic-Calendar-2025-26.pdf", "doc_type": "pdf", "page": 1, "url": "https://sxca.edu.in/wp-content/uploads/2026/03/Academic-Calendar-2025-26.pdf", "score": 1.0, "excerpt": "Academic Calendar & Internal ATKT Examinations Schedule"}]
            result = {"answer": text, "sources": sources, "grounded": True, "course": None}
        else:
            cal_res = counsellor.handle_vacation_holiday(question)
            result = {"answer": cal_res["answer"], "sources": cal_res["sources"], "grounded": True, "course": None}
            text = cal_res["answer"]
            sources = cal_res["sources"]

        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8e-4. Fast path for Curriculum / Subjects / Syllabus / Course Structure
    if understanding.get("intent") in ("curriculum", "subjects"):
        course_name = understanding.get("course")
        curr_res = counsellor.handle_subjects(course_name, question)
        text = curr_res["answer"]
        sources = curr_res["sources"]
        grounded = curr_res.get("grounded", True)
        result = {"answer": text, "sources": sources, "grounded": grounded, "course": course_name}
        log_chat(question, text, sources, grounded=grounded, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8f. Fast path for Today's Date
    if understanding.get("intent") == "today_date":
        text = _format_today_date(question)
        result = {"answer": text, "sources": [], "grounded": True}
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8g. Fast path for Swimming Pool & Sports facilities
    if understanding.get("intent") == "swimming_pool":
        text = _format_swimming_pool(question)
        sources = [{"source": "campus-life-student-development-sports.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/campus-life/student-development/sports/", "score": 1.0, "excerpt": "Sports facilities and ground details at St. Xavier's College"}]
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8h. Fast path for All UG Courses
    if understanding.get("intent") == "all_ug_courses":
        text = _format_all_ug_courses(question)
        sources = [{"source": "SXCA-Admission-Brochure-2526.pdf", "doc_type": "pdf", "page": 1, "url": "https://admissions.sxca.edu.in/SXCA/", "score": 1.0, "excerpt": "Undergraduate Programs and Fee Structure"}]
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8i. Fast path for All PG Courses
    if understanding.get("intent") == "all_pg_courses":
        text = _format_all_pg_courses(question)
        sources = [{"source": "SXCA-Admission-Brochure-2526.pdf", "doc_type": "pdf", "page": 1, "url": "https://admissions.sxca.edu.in/SXCA/", "score": 1.0, "excerpt": "Postgraduate Programs and Fee Structure"}]
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 8j. Fast path for Dr. Pravida Raja
    if understanding.get("intent") == "faculty_pravida_raja":
        text = _format_faculty_pravida_raja(question)
        sources = [{"source": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/", "score": 1.0, "excerpt": "Dr. Pravida Raja, Head of Department of Data Science"}]
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    passages = [] if course_specific else retrieve(search_query)
    is_valid, reason = validate_evidence(question, passages, structured_rows, understanding)
    if not is_valid:
        text = NO_INFO_ANSWER
        sources = []
        grounded = False
    else:
        text = qwen.generate(build_messages(question, passages, turns, structured_rows))
        grounded = not _is_refusal(text)
        if not grounded or _is_refusal(text):
            text = NO_INFO_ANSWER
            sources = []
            grounded = False
        else:
            sources = (as_sources(passages) + eligibility.as_sources(structured_rows))

    result = {"answer": text, "sources": sources, "grounded": grounded, "course": understanding.get("course")}
    log_chat(question, text, sources, grounded=grounded, conversation_id=conversation_id)
    if key:
        cache.put(key, result)
    return result


def _stream_attempt(question: str, passages: list[dict], history: list[dict], structured_rows: list[dict]):
    """Yields ("token", piece) as they arrive, then returns (full_text, grounded)."""
    collected = []
    for piece in qwen.stream(build_messages(question, passages, history, structured_rows)):
        collected.append(piece)
        yield "token", piece
    full = "".join(collected).strip()
    return full, not _is_refusal(full)


def answer_stream(question: str, history: list[dict] | None = None, conversation_id: str | None = None, owner_id: str | None = None):
    """Yield ("token", str) chunks as the answer streams, then ("sources",
    [...]) and ("done", full) once it's complete.

    A cache hit (see backend/rag/cache.py) skips retrieval and generation
    entirely and yields the stored answer as one immediate "token" chunk -
    still fast, just not incrementally typed, since the whole point of a
    cache hit is not spending any time regenerating it.

    On a miss, if the first attempt isn't grounded, a live crawl runs and a
    second attempt streams in its place - the client gets a ("restart",
    None) event telling it to discard whatever was streamed so far before
    the retry's tokens arrive, so a discarded refusal is never left on
    screen next to its own correction.
    """
    question = (question or "").strip()
    if not question:
        yield "done", "Please type a question."
        return

    from backend.security.prompt_guard import SAFE_REFUSAL_MESSAGE, check_prompt_safety
    guard_res = check_prompt_safety(question)
    if guard_res["blocked"]:
        text = guard_res.get("response") or SAFE_REFUSAL_MESSAGE
        yield "token", text
        yield "sources", []
        yield "done", {"answer": text, "sources": [], "grounded": False}
        log_chat(question, text, [], grounded=False, conversation_id=conversation_id, owner_id=owner_id)
        return

    turns = _resolve_turns(conversation_id, history or [], owner_id=owner_id)
    key = _cache_key(question, turns)
    cached = cache.get(key) if key else None
    if cached:
        if conversation_id:
            repo.get_or_create_conversation(conversation_id, owner_id=owner_id)
            c_course = cached.get("course")
            if not c_course:
                c_course, _ = counsellor.match_course(question)
            c_profile = counsellor.extract_and_update_profile(None, question, turns)
            if c_course:
                c_profile["active_course"] = c_course
            repo.update_conversation_context(
                conversation_id,
                course=c_course or c_profile.get("active_course"),
                profile=c_profile,
                owner_id=owner_id,
            )
        log_chat(question, cached["answer"], cached["sources"], grounded=cached["grounded"], conversation_id=conversation_id, owner_id=owner_id)
        if cached.get("course"):
            yield "course", cached["course"]
        yield "token", cached["answer"]
        yield "sources", cached["sources"]
        yield "done", cached["answer"]
        return

    understanding = _understand(question, conversation_id, turns, owner_id=owner_id)
    search_query = understanding["query"]
    structured_rows, course_specific = _structured_lookup(understanding)


    # 0-repair. Fast path for repetition / loop repair (<1ms)
    if understanding.get("intent") == "feedback_repair":
        text = counsellor.handle_conversation_repair(understanding.get("profile") or {})
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")})
        yield "course", understanding.get("course")
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0-neg. Fast path for topic negation / exclusion (<1ms)
    if understanding.get("intent") == "negation":
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        text = counsellor.handle_negation(understanding.get("negated_topic"), understanding.get("profile") or {}, is_guj=is_guj)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")})
        yield "course", understanding.get("course")
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0-no. Fast path for contextual negative response (<1ms)
    if understanding.get("intent") == "reject_clarification":
        text = counsellor.handle_contextual_no(understanding.get("profile") or {}, turns)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")})
        yield "course", understanding.get("course")
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0-redirect. Fast path for conversational redirection (<1ms)
    if understanding.get("intent") == "redirect":
        text = counsellor.handle_redirect(understanding.get("profile") or {})
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")})
        yield "course", understanding.get("course")
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0-pre. Fast path for user correction / disagreement (<1ms)
    if understanding.get("intent") == "correction":
        text = counsellor.handle_user_correction()
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")})
        yield "course", understanding.get("course")
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0-ai. Fast path for ambiguous "AI" (<1ms)
    if understanding.get("intent") == "ai_ambiguity":
        text = counsellor.handle_ai_ambiguity()
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0-zc. Fast path for zero-cost inquiry (<1ms)
    if understanding.get("intent") == "zero_cost":
        text = counsellor.handle_zero_cost(understanding.get("course"))
        sources = eligibility.as_sources(structured_rows) if structured_rows else []
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": understanding.get("course")})
        yield "course", understanding.get("course")
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0-fc. Fast path for fee confirmation ("Do you mean the fee for...") (<1ms)
    if understanding.get("intent") == "fee_confirmation":
        text = counsellor.handle_fee_confirmation(understanding.get("course"))
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")})
        yield "course", understanding.get("course")
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0a. Fast path for bare greetings & pleasantries (<5ms)
    # 0. Fast path for campus facility complaints (<1ms)
    if understanding.get("intent") == "complaint":
        text = counsellor.handle_complaint(question)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0a. Fast path for bare greetings & pleasantries (<5ms)
    if understanding.get("intent") == "greeting":
        text = _format_greeting(question)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0b. Fast path for chitchat, gratitude & goodbyes (<5ms)
    if understanding.get("intent") == "chitchat":
        text = _format_chitchat(question, understanding.get("profile"))
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0c. Fast path for other college queries (strict single-college boundary) (<5ms)
    if understanding.get("intent") == "other_college":
        text = _format_other_college(question)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0d-1. Fast path for language preference (<1ms)
    if understanding.get("intent") == "language_preference":
        h_res = counsellor.handle_language_preference(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-2. Fast path for faculty & leadership (<5ms)
    if understanding.get("intent") == "faculty":
        query_to_use = understanding.get("rewritten_question") or question
        h_res = counsellor.handle_faculty(query_to_use, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=h_res.get("grounded", True), conversation_id=conversation_id)
        if key:
            res_payload = {
                "answer": text,
                "sources": sources,
                "grounded": h_res.get("grounded", True),
                "course": None,
                "status": h_res.get("status", "RESOLVED"),
            }
            if "candidates" in h_res:
                res_payload["candidates"] = h_res["candidates"]
            if "person" in h_res:
                res_payload["person"] = h_res["person"]
            cache.put(key, res_payload)
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-3. Fast path for contact & helpline (<5ms)
    if understanding.get("intent") == "contact":
        h_res = counsellor.handle_contact(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-4. Fast path for office timings (<5ms)
    if understanding.get("intent") == "office_timing":
        h_res = counsellor.handle_office_timing(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-5. Fast path for departments (<5ms)
    if understanding.get("intent") == "departments":
        h_res = counsellor.handle_departments(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-6. Fast path for clubs (<5ms)
    if understanding.get("intent") == "clubs":
        h_res = counsellor.handle_clubs(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-7. Fast path for sports & gym (<5ms)
    if understanding.get("intent") == "sports":
        h_res = counsellor.handle_sports(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-8. Fast path for NCC & NSS (<5ms)
    if understanding.get("intent") == "ncc_nss":
        h_res = counsellor.handle_ncc_nss(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-9. Fast path for events & culfest (<5ms)
    if understanding.get("intent") == "events":
        h_res = counsellor.handle_events(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-10. Fast path for placements (<5ms)
    if understanding.get("intent") == "placement":
        h_res = counsellor.handle_placement(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-11. Fast path for scholarships (<5ms)
    if understanding.get("intent") == "scholarships":
        h_res = counsellor.handle_scholarships(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-12. Fast path for library (<5ms)
    if understanding.get("intent") == "library":
        h_res = counsellor.handle_library(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-13. Fast path for science background course discovery (<5ms)
    if understanding.get("intent") == "science_courses":
        h_res = counsellor.handle_science_courses(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-14. Fast path for general PG admission guidance (<5ms)
    if understanding.get("intent") == "general_pg_admission":
        h_res = counsellor.handle_general_pg_admission(question, understanding.get("profile"))
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-15. Fast path for commerce graduate course discovery (<5ms)
    if understanding.get("intent") == "commerce_graduate_courses":
        h_res = counsellor.handle_commerce_graduate_courses(understanding.get("profile"))
        if isinstance(h_res, dict):
            text, sources = h_res["answer"], h_res.get("sources", [])
        else:
            text = h_res
            sources = [{"source": "programmes-postgraduate.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/programmes/postgraduate/", "score": 1.0, "excerpt": "Postgraduate courses and eligibility at St. Xavier's College"}]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-16. Fast path for BCA graduate course discovery (<5ms)
    if understanding.get("intent") == "bca_graduate_courses":
        h_res = counsellor.handle_bca_graduate_courses(understanding.get("profile"))
        if isinstance(h_res, dict):
            text, sources = h_res["answer"], h_res.get("sources", [])
        else:
            text = h_res
            sources = [{"source": "programmes-postgraduate.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/programmes/postgraduate/", "score": 1.0, "excerpt": "Postgraduate courses and eligibility at St. Xavier's College"}]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-17. Fast path for BA Eco / eligible PG courses (<5ms)
    if understanding.get("intent") in ("eligible_pg_courses", "ba_economics_graduate_courses"):
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        h_res = counsellor.handle_ba_economics_graduate_courses(understanding.get("profile"), is_guj=is_guj)
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d-18. Fast path for commerce stream inquiry (e.g. 'mene commerce kiya hai course name') (<5ms)
    if understanding.get("intent") == "commerce_stream_inquiry":
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        h_res = counsellor.handle_commerce_stream_inquiry(understanding.get("profile"), is_guj=is_guj)
        text, sources = h_res["answer"], h_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0d. Fast path for hostel queries (<5ms)
    if understanding.get("intent") == "hostel":
        is_fee = bool(re.search(r"\b(fee|fees|rent|cost|charges?|price|how\s+much)\b", question.lower()))
        if not is_fee:
            text = counsellor.handle_hostel(question, understanding.get("profile"))
            sources = [{"source": "campus-life-student-development-hostel.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/campus-life/student-development/hostel/", "score": 1.0, "excerpt": "Hostel facilities and accommodation details at St. Xavier's College"}]
            log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
            if key:
                cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": None})
            yield "course", None
            yield "token", text
            yield "sources", sources
            yield "done", text
            return


    # 0e. Fast path for direct course inquiry (<5ms)
    if understanding.get("intent") == "course_inquiry" and understanding.get("course"):
        c_name = understanding.get("course")
        profile = understanding.get("profile") or {}
        text = counsellor.handle_course_inquiry(c_name, profile)
        sources = eligibility.as_sources(structured_rows) if structured_rows else []
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True, "course": c_name})
        yield "course", c_name
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 0f. Fast path for progressive counselling step (<5ms)
    if understanding.get("intent") == "counselling":
        profile = understanding.get("profile") or {}
        text = counsellor.handle_counselling_step(question, profile)
        text = _guard_repetition(text, question, turns)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")})
        yield "course", understanding.get("course")
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0g. Fast path for course discovery ("what courses do you have?") (<5ms)
    if understanding.get("intent") == "course_discovery":
        profile = understanding.get("profile") or {}
        text = counsellor.handle_course_discovery(question, profile)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": understanding.get("course")})
        yield "course", understanding.get("course")
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 0h. Fast path when fees or eligibility asked without an active course (<5ms)
    if understanding.get("intent") == "no_course_followup":
        raw_intent = understanding.get("raw_intent", "")
        text = counsellor.handle_no_course_followup(raw_intent)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True, "course": None})
        yield "course", None
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    yield "course", understanding.get("course")

    # 1. Fast path for source follow-ups
    if understanding.get("intent") == "source_query":
        src_res = _handle_source_query(question, turns, conversation_id)
        if src_res:
            log_chat(question, src_res["answer"], src_res["sources"], grounded=src_res["grounded"], conversation_id=conversation_id)
            if key:
                cache.put(key, src_res)
            yield "token", src_res["answer"]
            yield "sources", src_res["sources"]
            yield "done", src_res["answer"]
            return

    # 2. Fast path for combined fee & eligibility
    if understanding.get("intent") == "fee_and_eligibility" and course_specific and structured_rows:
        text = _format_course_fee_and_eligibility_summary(structured_rows)
        sources = eligibility.as_sources(structured_rows)
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 3. Fast path for direct course eligibility queries (eligibility ONLY)
    if understanding.get("intent") == "eligibility" and course_specific and structured_rows:
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        text = _format_course_eligibility_summary(structured_rows, user_stream=understanding.get("qualification"), is_guj=is_guj)
        sources = eligibility.as_sources(structured_rows)
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 4. Fast path for direct course fee queries (fees ONLY)
    if understanding.get("intent") == "fee" and course_specific and structured_rows:
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        text = _format_course_fee_summary(structured_rows, is_guj=is_guj)
        sources = eligibility.as_sources(structured_rows)
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 5. Fast path for direct application link queries
    if understanding.get("intent") == "application_link":
        course_name = understanding.get("course")
        text = _format_course_application_link(course_name, structured_rows)
        raw_sources = eligibility.as_sources(structured_rows) if structured_rows else [
            {"source": "admissions-sxca-edu-in-sxca.html", "doc_type": "html", "url": get_verified_application_url(), "score": 1.0, "excerpt": "Official College Admission Portal"}
        ]
        sources = _standardize_sources(raw_sources)
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 6. Fast path for application process & how-to-apply queries
    if understanding.get("intent") == "process":
        course_name = understanding.get("course")
        is_guj = counsellor.is_gujarati_mode(question, understanding.get("profile") or {})
        text = _format_course_application_process(course_name, structured_rows, is_guj=is_guj)
        raw_sources = eligibility.as_sources(structured_rows) if structured_rows else [
            {"source": "admissions-sxca-edu-in-sxca.html", "doc_type": "html", "url": get_verified_application_url(), "score": 1.0, "excerpt": "Official College Admission Portal"}
        ]
        sources = _standardize_sources(raw_sources)
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 6. Fast path for document queries
    if understanding.get("intent") == "documents":
        course_name = understanding.get("course")
        text = _format_course_documents_summary(course_name)
        raw_sources = eligibility.as_sources(structured_rows) if structured_rows else [
            {"source": "SXCA-Admission-Brochure-2526.pdf", "doc_type": "pdf", "page": 1, "url": "https://sxca.edu.in/wp-content/uploads/2025/06/SXCA-Admission-Brochure-2526.pdf", "score": 1.0, "excerpt": "Required Documents for Admission"}
        ]
        sources = _standardize_sources(raw_sources)
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 7. Fast path for broad/full admission counselling
    if understanding.get("intent") in ("counselling", "application") and course_specific and structured_rows:
        text = _format_course_counselling_summary(structured_rows)
        sources = eligibility.as_sources(structured_rows)
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 8. Fast path for stream qualification recommendations
    if understanding.get("qualification") and not course_specific and structured_rows:
        text = _format_stream_recommendations(understanding["qualification"], structured_rows)
        sources = eligibility.as_sources(structured_rows)
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 8b. Fast path for NAAC, NIRF & Rankings
    if understanding.get("intent") == "ranking":
        text = _format_rankings_summary(question)
        sources = [{"source": "about-us-rankings.html", "doc_type": "html", "url": "https://sxca.edu.in/about-us/rankings/", "score": 1.0, "excerpt": "NAAC Cycle-IV, NIRF Rank 52 (2022), GSIRF 5 Stars"}]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 8c. Fast path for Merit Lists & Cutoffs
    if understanding.get("intent") == "merit_cutoff":
        course_name = understanding.get("course")
        text = _format_merit_cutoff_summary(course_name, question)
        sources = [{"source": "admissions-sxca-edu-in-sxca.html", "doc_type": "html", "url": get_verified_application_url(), "score": 1.0, "excerpt": "Official College Admission Portal & Merit Lists"}]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 8d. Fast path for Percentage Eligibility
    if understanding.get("intent") == "percentage_eligibility":
        pct_val = understanding.get("percentage", "65")
        text = _format_percentage_eligibility_summary(pct_val, understanding.get("qualification"))
        sources = [{"source": "SXCA-Admission-Brochure-2526.pdf", "doc_type": "pdf", "page": 1, "url": "https://sxca.edu.in/wp-content/uploads/2025/06/SXCA-Admission-Brochure-2526.pdf", "score": 1.0, "excerpt": "Undergraduate Eligibility Criteria"}]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 8e. Fast path for Exam Dates (CAI-2 / CIA-2 / CIA-1 / ESE)
    if understanding.get("intent") == "exam_date":
        cal_res = counsellor.handle_exam_date(question)
        text = cal_res["answer"]
        sources = cal_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 8e-2. Fast path for Academic Calendar overview / commencement
    if understanding.get("intent") == "academic_calendar":
        cal_res = counsellor.handle_academic_calendar(question)
        text = cal_res["answer"]
        sources = cal_res["sources"]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 8e-3. Fast path for Specific Date Holiday check & Vacations
    if understanding.get("intent") == "calendar_holiday":
        if understanding.get("specific_date"):
            text = _format_specific_date_holiday(understanding["specific_date"], question)
            sources = [{"source": "Academic-Calendar-2025-26.pdf", "doc_type": "pdf", "page": 1, "url": "https://sxca.edu.in/wp-content/uploads/2026/03/Academic-Calendar-2025-26.pdf", "score": 1.0, "excerpt": "Academic Calendar & Internal ATKT Examinations Schedule"}]
        else:
            cal_res = counsellor.handle_vacation_holiday(question)
            text = cal_res["answer"]
            sources = cal_res["sources"]

        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 8e-4. Fast path for Curriculum / Subjects / Syllabus / Course Structure
    if understanding.get("intent") in ("curriculum", "subjects"):
        course_name = understanding.get("course")
        curr_res = counsellor.handle_subjects(course_name, question)
        text = curr_res["answer"]
        sources = curr_res["sources"]
        grounded = curr_res.get("grounded", True)
        log_chat(question, text, sources, grounded=grounded, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": grounded, "course": course_name})
        yield "token", text
        if sources:
            yield "sources", sources
        yield "done", text
        return

    # 8f. Fast path for Today's Date
    if understanding.get("intent") == "today_date":
        text = _format_today_date(question)
        log_chat(question, text, [], grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": [], "grounded": True})
        yield "token", text
        yield "sources", []
        yield "done", text
        return

    # 8g. Fast path for Swimming Pool & Sports facilities
    if understanding.get("intent") == "swimming_pool":
        text = _format_swimming_pool(question)
        sources = [{"source": "campus-life-student-development-sports.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/campus-life/student-development/sports/", "score": 1.0, "excerpt": "Sports facilities and ground details at St. Xavier's College"}]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 8h. Fast path for All UG Courses
    if understanding.get("intent") == "all_ug_courses":
        text = _format_all_ug_courses(question)
        sources = [{"source": "SXCA-Admission-Brochure-2526.pdf", "doc_type": "pdf", "page": 1, "url": "https://admissions.sxca.edu.in/SXCA/", "score": 1.0, "excerpt": "Undergraduate Programs and Fee Structure"}]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 8i. Fast path for All PG Courses
    if understanding.get("intent") == "all_pg_courses":
        text = _format_all_pg_courses(question)
        sources = [{"source": "SXCA-Admission-Brochure-2526.pdf", "doc_type": "pdf", "page": 1, "url": "https://admissions.sxca.edu.in/SXCA/", "score": 1.0, "excerpt": "Postgraduate Programs and Fee Structure"}]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    # 8j. Fast path for Dr. Pravida Raja
    if understanding.get("intent") == "faculty_pravida_raja":
        text = _format_faculty_pravida_raja(question)
        sources = [{"source": "academics-departments-computer-sciences-faculty-computer-sciences-data-sciences.html", "doc_type": "html", "page": 1, "url": "https://sxca.edu.in/academics/departments/computer-sciences/faculty-computer-sciences-data-sciences/", "score": 1.0, "excerpt": "Dr. Pravida Raja, Head of Department of Data Science"}]
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": text, "sources": sources, "grounded": True})
        yield "token", text
        yield "sources", sources
        yield "done", text
        return

    passages = [] if course_specific else retrieve(search_query)
    is_valid, reason = validate_evidence(question, passages, structured_rows, understanding)
    if not is_valid:
        full = NO_INFO_ANSWER
        sources = []
        grounded = False
        yield "token", full
        yield "sources", sources
        yield "done", full
        log_chat(question, full, sources, grounded=grounded, conversation_id=conversation_id)
        if key:
            cache.put(key, {"answer": full, "sources": sources, "grounded": grounded})
        return

    full, grounded = yield from _stream_attempt(question, passages, turns, structured_rows)

    if not grounded or _is_refusal(full):
        full = NO_INFO_ANSWER
        sources = []
        grounded = False
        yield "restart", None
        yield "token", full
    else:
        sources = _standardize_sources(as_sources(passages) + eligibility.as_sources(structured_rows))

    log_chat(question, full, sources, grounded=grounded, conversation_id=conversation_id)
    if key:
        cache.put(key, {"answer": full, "sources": sources, "grounded": grounded})
    yield "sources", sources
    yield "done", full
