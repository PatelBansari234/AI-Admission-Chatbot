import re
import typing

from backend.config import LIVE_CRAWL_FALLBACK_ENABLED
from backend.crawler.live_search import maybe_live_crawl
from backend.db import repo
from backend.llm import qwen
from backend.rag import cache, eligibility, query_rewrite
from backend.rag.prompt import NO_CONTEXT_ANSWER, build_messages
from backend.rag.retriever import as_sources, normalize_query, retrieve
from backend.utils.registry import log_chat

_REFUSAL_MARKERS = (
    "don't have that information",
    "sorry, i don't have that information",
    "not available in my available college data",
    "not available in the college data",
    "not available in the college documents",
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


def _resolve_turns(conversation_id: str | None, history: list[dict]) -> list[dict]:
    """The database is the authoritative history once a conversation_id is
    given (every prior turn in this conversation was already logged via
    log_chat below) - it doesn't depend on the client having sent a complete
    or untampered `history` array. Without a conversation_id (older client,
    or a caller that doesn't want persistence), fall back to whatever the
    client sent, same as before this feature existed."""
    if conversation_id:
        repo.get_or_create_conversation(conversation_id)
        return repo.get_recent_turns(conversation_id)
    return history or []


def _understand(question: str, conversation_id: str | None, turns: list[dict]) -> dict:
    """Runs on every question, not just follow-ups - "Can a BCA graduate
    apply for MSc AI?" needs course/qualification extraction and intent
    classification even as someone's first message, unlike pure follow-up
    resolution which only matters once there's a prior turn to resolve
    against (see query_rewrite.py). Any newly-mentioned course/qualification
    gets persisted so later turns in the same conversation already know it
    without re-extracting it every time."""
    context = repo.get_conversation_context(conversation_id) if conversation_id else None
    result = query_rewrite.understand(question, turns, context)
    if conversation_id and (result["course"] or result["qualification"]):
        repo.update_conversation_context(
            conversation_id, course=result["course"], qualification=result["qualification"]
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


def _format_course_fee_summary(course_rows: list[dict]) -> str:
    if not course_rows:
        return ""
    lines = []
    for r in course_rows:
        c_name = r.get("course_name", "")
        cat = "Grant-in-Aid" if r.get("category") == "grant_in_aid" else "Self-Financed"
        male_fee = r.get("fee_amount")
        female_fee = r.get("fee_amount_female")
        if male_fee is not None and female_fee is not None:
            fee_str = f"₹{int(male_fee):,} per semester for male students and ₹{int(female_fee):,} per semester for female students"
        elif male_fee is not None:
            fee_str = f"₹{int(male_fee):,} per semester"
        else:
            fee_str = "as prescribed by college guidelines"
        note = f" ({r.get('fee_note')})" if r.get("fee_note") else ""
        lines.append(f"The fee for **{c_name}** ({cat}) is **{fee_str}**{note}.")
    contact = course_rows[0].get("contact_email")
    if contact:
        lines.append(f"For more details, you can contact {contact}.")
    return "\n".join(lines)


def _format_course_eligibility_summary(course_rows: list[dict], user_stream: str | None = None) -> str:
    if not course_rows:
        return ""
    lines = []
    for r in course_rows:
        c_name = r.get("course_name", "")
        cat = "Grant-in-Aid" if r.get("category") == "grant_in_aid" else "Self-Financed"
        elig = r.get("eligibility", "12th standard pass")
        if user_stream and user_stream.lower() in elig.lower():
            lines.append(f"The eligibility for **{c_name}** ({cat}) is **{elig}**. Yes, with a {user_stream} background, you can apply for {c_name}.")
        else:
            lines.append(f"The eligibility for **{c_name}** ({cat}) is **{elig}**.")
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


def _format_course_application_process(course_name: str | None, course_rows: list[dict]) -> str:
    app_url = get_verified_application_url()
    target = f" for **{course_name}**" if course_name else ""
    return (
        f"To apply for admission{target} at St. Xavier's College (Autonomous), Ahmedabad:\n\n"
        f"1. **Online Registration:** Register on the official college admission portal.\n"
        f"2. **Fill Application Form:** Enter your personal and academic details, and select your course.\n"
        f"3. **Upload Documents:** Upload scanned copies of required documents (10th & 12th marksheets, School Leaving Certificate, and Category Certificate if applicable).\n"
        f"4. **Merit List & Verification:** Follow the merit list announcements on the college portal, complete document verification, and pay the admission fee.\n\n"
        f"• **Official Application Portal:** [Apply here (St. Xavier's College Admission Portal)]({app_url})"
    )


def _format_course_documents_summary(course_name: str | None) -> str:
    target = f" for **{course_name}**" if course_name else ""
    return (
        f"Required documents for admission{target} at St. Xavier's College (Autonomous), Ahmedabad:\n\n"
        f"• **10th Standard (SSC) Marksheet** & Passing Certificate\n"
        f"• **12th Standard (HSC) Marksheet** & Passing Certificate\n"
        f"• **School Leaving Certificate (LC)** / Transfer Certificate (TC)\n"
        f"• **Caste / Category Certificate** (SC / ST / SEBC / EWS, if applicable)\n"
        f"• **Non-Creamy Layer (NCL) Certificate** (for SEBC candidates, valid for current year)\n"
        f"• **Recent Passport-size Photographs**\n"
        f"• **Migration Certificate** (for students from boards other than Gujarat Board)"
    )


def _format_stream_recommendations(qualification: str, rows: list[dict]) -> str:
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
    lines = [f"Based on your {qualification} background, you can apply for the following undergraduate courses at St. Xavier's College (Autonomous), Ahmedabad:\n"]
    for c in unique_ug:
        lines.append(f"• **{c}**")
    if unique_pg:
        lines.append(f"\nPostgraduate / Integrated options include: {', '.join(unique_pg)}.")
    lines.append(f"\n• **Application Portal:** [Apply here (St. Xavier's College Admission Portal)]({app_url})")
    return "\n".join(lines)


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


def answer(question: str, history: list[dict] | None = None, conversation_id: str | None = None) -> dict:
    question = (question or "").strip()
    if not question:
        return {"answer": "Please type a question.", "sources": [], "grounded": False}

    turns = _resolve_turns(conversation_id, history or [])
    key = _cache_key(question, turns)
    cached = cache.get(key) if key else None
    if cached:
        log_chat(question, cached["answer"], cached["sources"], grounded=cached["grounded"], conversation_id=conversation_id)
        return cached

    understanding = _understand(question, conversation_id, turns)
    search_query = understanding["query"]
    structured_rows, course_specific = _structured_lookup(understanding)

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
        text = _format_course_eligibility_summary(structured_rows, user_stream=understanding.get("qualification"))
        sources = eligibility.as_sources(structured_rows)
        result = {"answer": text, "sources": sources, "grounded": True}
        log_chat(question, text, sources, grounded=True, conversation_id=conversation_id)
        if key:
            cache.put(key, result)
        return result

    # 4. Fast path for direct course fee queries (fees ONLY)
    if understanding.get("intent") == "fee" and course_specific and structured_rows:
        text = _format_course_fee_summary(structured_rows)
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
        text = _format_course_application_process(course_name, structured_rows)
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
        text = _format_course_documents_summary(course_name)
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

    passages = [] if course_specific else retrieve(search_query)
    has_evidence = bool(passages) or bool(structured_rows)
    text = qwen.generate(build_messages(question, passages, turns, structured_rows)) if has_evidence else NO_CONTEXT_ANSWER
    grounded = has_evidence and not _is_refusal(text)

    # The static index came up short - either nothing at all, or something
    # that looked plausible but didn't actually answer it (e.g. a bare page
    # heading with no body text). Either way, try a live crawl guided by this
    # question's own words and retry once with whatever it finds - unless
    # the fallback is switched off (e.g. the index is deliberately scoped to
    # specific uploaded documents only), or this was a course-specific
    # structured match, where reintroducing RAG passages is exactly the
    # distraction risk just described above.
    if not grounded and not course_specific and LIVE_CRAWL_FALLBACK_ENABLED and maybe_live_crawl(search_query):
        passages = retrieve(search_query)
        if passages or structured_rows:
            text = qwen.generate(build_messages(question, passages, turns, structured_rows))
            grounded = not _is_refusal(text)

    sources = (as_sources(passages) + eligibility.as_sources(structured_rows)) if grounded else []
    result = {"answer": text, "sources": sources, "grounded": grounded}
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


def answer_stream(question: str, history: list[dict] | None = None, conversation_id: str | None = None):
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

    turns = _resolve_turns(conversation_id, history or [])
    key = _cache_key(question, turns)
    cached = cache.get(key) if key else None
    if cached:
        log_chat(question, cached["answer"], cached["sources"], grounded=cached["grounded"], conversation_id=conversation_id)
        yield "token", cached["answer"]
        yield "sources", cached["sources"]
        yield "done", cached["answer"]
        return

    understanding = _understand(question, conversation_id, turns)
    search_query = understanding["query"]
    structured_rows, course_specific = _structured_lookup(understanding)

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
        text = _format_course_eligibility_summary(structured_rows, user_stream=understanding.get("qualification"))
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
        text = _format_course_fee_summary(structured_rows)
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
        sources = eligibility.as_sources(structured_rows) if structured_rows else [
            {"source": "admissions-sxca-edu-in-sxca.html", "doc_type": "html", "url": get_verified_application_url(), "score": 1.0, "excerpt": "Official College Admission Portal"}
        ]
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
        text = _format_course_application_process(course_name, structured_rows)
        sources = eligibility.as_sources(structured_rows) if structured_rows else [
            {"source": "admissions-sxca-edu-in-sxca.html", "doc_type": "html", "url": get_verified_application_url(), "score": 1.0, "excerpt": "Official College Admission Portal"}
        ]
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
        sources = eligibility.as_sources(structured_rows) if structured_rows else [
            {"source": "SXCA-Admission-Brochure-2526.pdf", "doc_type": "pdf", "page": 1, "url": "https://sxca.edu.in/wp-content/uploads/2025/06/SXCA-Admission-Brochure-2526.pdf", "score": 1.0, "excerpt": "Required Documents for Admission"}
        ]
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

    passages = [] if course_specific else retrieve(search_query)
    if passages or structured_rows:
        full, grounded = yield from _stream_attempt(question, passages, turns, structured_rows)
    else:
        # Nothing retrieved yet, so no tokens have been shown - try the live
        # crawl now rather than displaying a refusal it might not need.
        full, grounded = NO_CONTEXT_ANSWER, False

    if not grounded and not course_specific and LIVE_CRAWL_FALLBACK_ENABLED and maybe_live_crawl(search_query):
        retried = retrieve(search_query)
        if retried or structured_rows:
            yield "restart", None
            full, grounded = yield from _stream_attempt(question, retried, turns, structured_rows)
            passages = retried

    if not passages and not structured_rows:
        yield "token", full  # still nothing after the retry - reveal the refusal now

    sources = (as_sources(passages) + eligibility.as_sources(structured_rows)) if grounded else []
    log_chat(question, full, sources, grounded=grounded, conversation_id=conversation_id)
    if key:
        cache.put(key, {"answer": full, "sources": sources, "grounded": grounded})
    yield "sources", sources
    yield "done", full
