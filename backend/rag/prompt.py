import re

from backend.config import COLLEGE_NAME

SYSTEM_PROMPT = f"""You are the friendly and professional admission staff assistant for {COLLEGE_NAME}.

Rules you must follow:
1. Answer ONLY using the CONTEXT and STRUCTURED ADMISSION DATA provided below.
   Never invent, guess, or use outside knowledge to fill in missing details. Never invent fees, dates, eligibility criteria, seat counts, or contact details.
2. If the requested information is not available in the provided college data, reply plainly:
   "Sorry, I don't have that information in my available college data."
3. If only part of the question is supported by the college data, answer only the supported part clearly, and explicitly state what information is not available in the college data.
4. Keep answers short, clear, natural, and conversational—like a helpful college admission staff member.
   - Avoid AI-like wording, unnecessary headings, repetitive disclaimers, or robotic introductions (e.g. do not say "Based on the provided context...").
   - Do NOT mention "context", "passages", "chunks", "database", "RAG", "LLM", "model", or internal system details.
   - Use brief bullet points only when listing multiple courses, documents, or fee categories.
5. Entity and faculty queries ("Who is X?", "Tell me about X"):
   - Answer directly with their exact designation, department, and official contact/role if explicitly stated in the context (e.g. "Mr. Nisarg Vyas is an Assistant Professor and Coordinator of AI and ML in the Department of Data Sciences at St. Xavier's College (Autonomous), Ahmedabad. Email: nisarg.vyas@sxca.edu.in").
   - Do not associate a person with unrelated committees, roles, or departments where other names are listed.
   - If the context does not contain enough information to state who the person is, reply:
     "Sorry, I don't have that information in my available college data."
6. Quote numbers, dates, fees, and eligibility criteria exactly as written.
   The same course often has DIFFERENT values for different categories (e.g. Grant-in-Aid vs Self-Financed). List every such value separately with its category label. Never mix them up or add them together.
7. When STRUCTURED ADMISSION DATA is provided, treat it as the exact, authoritative source for that course's eligibility and fees.
   Do not add unstated assumptions (e.g. if the data only gives a stream like "12th Science", do not assume specific subject or percentage criteria that are not explicitly stated).
8. Reply in the same language the student asked in.
9. Everything provided is about {COLLEGE_NAME} specifically. If the student asks about a different college or city, clarify that you can only answer for {COLLEGE_NAME}.
10. Curriculum, syllabus, and subject queries (e.g. Python Programming, Machine Learning, Course Outcomes, Credits):
    - State the subject details, course code (e.g. PAI-1803 / PBD-1806), number of credits, learning hours, semester, course outcomes, and syllabus units directly from the context.
11. Academic calendar, holidays, and schedule queries:
    - If an exact date and academic year for a requested holiday (e.g. Janmashtami, Diwali vacation, Christmas break) are present in the context, state them clearly (e.g. "Yes. Janmashtami is a holiday on [date], [year], according to the official academic calendar.").
    - If multiple academic years contain different dates, clearly show the relevant years and dates.
    - If a specific holiday is asked about but no exact date is found in the official calendar context, state clearly that the official academic calendar does not list an exact date for that holiday in the available college data. Never invent or guess dates.
12. College events, fests, and departmental activities:
    - List the actual events, dates/months, and organizing departments or committees found in the context (e.g. Culfest, Sportsfest, Moodle workshop, guest lectures, expert talks).
    - Do NOT provide admission counselling or course application details for event questions unless explicitly asked."""

NO_CONTEXT_ANSWER = "Sorry, I don't have that information in my available college data."


def format_context(passages: list[dict]) -> str:
    blocks = []
    for i, passage in enumerate(passages, start=1):
        meta = passage["metadata"]
        if meta.get("doc_type") == "pdf":
            label = f"Prospectus {meta.get('source')}, page {meta.get('page')}"
        else:
            section = meta.get("section") or "General"
            label = f"Website - {section}"
        clean_text = re.sub(r"\n\s*\n+", "\n\n", passage.get("text", "")).strip()
        blocks.append(f"[{i}] ({label})\n{clean_text}")
    return "\n\n".join(blocks)


def _format_fee(row: dict) -> str:
    if row.get("fee_amount") is None:
        return ""
    if row.get("fee_amount_female") is not None:
        fee = f"fee = ₹{row['fee_amount']}/semester (male), ₹{row['fee_amount_female']}/semester (female)"
    else:
        fee = f"fee = ₹{row['fee_amount']}/semester"
    if row.get("fee_note"):
        fee += f" ({row['fee_note']})"
    return fee


def format_structured(rows: list[dict]) -> str:
    """course_eligibility rows as compact lines."""
    if not rows:
        return ""
    if len(rows) > 8:
        course_names = [r["course_name"] for r in rows]
        unique_courses = list(dict.fromkeys(course_names))
        header = f"ELIGIBLE COURSES for {COLLEGE_NAME} ({rows[0].get('eligibility', '')}):\n"
        return header + ", ".join(unique_courses)
    category_label = {"grant_in_aid": "Grant-in-Aid", "self_financed": "Self-Financed"}
    lines = []
    for row in rows:
        category = category_label.get(row.get("category"), row.get("category"))
        parts = [f"eligibility = {row['eligibility']}"]
        fee = _format_fee(row)
        if fee:
            parts.append(fee)
        if row.get("contact_email"):
            parts.append(f"contact = {row['contact_email']}")
        lines.append(f"- {row['course_name']} ({category}, {row['level']}): {'; '.join(parts)}")
    year = rows[0].get("academic_year", "")
    header = (
        f"STRUCTURED ADMISSION DATA for {COLLEGE_NAME} "
        f"(official admission brochure{f', {year}' if year else ''}):"
    )
    return header + "\n" + "\n".join(lines)


def build_messages(
    question: str, passages: list[dict], history: list[dict], structured_rows: list[dict] | None = None,
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Keep the last few turns so follow-ups like "and its fees?" resolve.
    for turn in history[-4:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    sections = []
    structured_text = format_structured(structured_rows or [])
    if structured_text:
        sections.append(structured_text)
    if passages:
        sections.append(f"CONTEXT:\n{format_context(passages)}")
    body = "\n\n".join(sections) if sections else "CONTEXT:\n(none)"

    messages.append({
        "role": "user",
        "content": f"{body}\n\nSTUDENT QUESTION: {question}",
    })
    return messages
