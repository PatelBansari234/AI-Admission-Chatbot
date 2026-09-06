"""Build a QLoRA training set from logged Q&A.

    python -m backend.finetune.prepare_dataset

Reuses the exact prompt structure the model sees at inference (system prompt
+ retrieved context + question -> answer, from backend/rag/prompt.py), so
fine-tuning reinforces *how* to phrase/extract answers from context - tone and
style - not the facts themselves, which stay with RAG and can change the
moment an admin uploads a new document. See README "Why RAG and not
fine-tuning" for the reasoning.

Context is rebuilt by re-running retrieval now, rather than trusting whatever
was stored with the log row, because the logged "sources" are a trimmed
citation list for the UI, not the full passage text the model actually saw -
and because re-retrieving means a question logged before an index rebuild
still gets today's context, not stale text.

A logged row is used only if the college documents can currently answer it
(retrieval still finds something) - if the index changed since it was
logged, silently pairing an old answer with unrelated new context would
teach the wrong thing, so that row is dropped instead.

Refusal examples (empty/irrelevant context -> the fixed decline message) are
seeded by hand rather than mined from logs: pipeline.py never calls the LLM
at all when retrieval finds literally nothing, so there is no naturally
logged (context, refusal) pair for that path. A few hand-written ones here
cover the case that *does* need the model to decide - a weak, borderline
passage that retrieval hands over but that still doesn't answer the
question.
"""

import json

from backend.config import PROCESSED_DIR
from backend.db import repo
from backend.rag.prompt import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, format_context
from backend.rag.retriever import retrieve

OUTPUT_PATH = PROCESSED_DIR / "finetune_dataset.jsonl"
MIN_ANSWER_LEN = 15

# Hand-seeded borderline-refusal examples: a real passage that LOOKS related
# by keyword overlap but doesn't contain the answer, paired with the correct
# decline. This is exactly the case the retriever's low-confidence fallback
# can hand to the model (see backend/rag/pipeline.py's _is_refusal fix).
REFUSAL_SEEDS = [
    {
        "context": "[1] (Prospectus, page 4)\nAcademic Life\nStudents thrive in a "
        "relaxed and open learning environment, where teaching encourages "
        "discussion and critical thinking. A course credit structure can vary "
        "from 2-4 credits and a student must acquire 22 credits in a semester.",
        "question": "What is the hostel fee for MSc students?",
        "answer": NO_CONTEXT_ANSWER,
    },
    {
        "context": "[1] (College website - Site links)\nOfficial page links: "
        "UG Admissions: https://sxca.edu.in/admissions/ug-admission/\n"
        "PG Admissions: https://sxca.edu.in/admissions/pg-admission/",
        "question": "What is the last date to pay the semester fees?",
        "answer": NO_CONTEXT_ANSWER,
    },
]


def build_examples() -> list[dict]:
    examples = []
    for row in repo.list_chat():
        question = row["question"].strip()
        answer = row["answer"].strip()
        if not question or len(answer) < MIN_ANSWER_LEN:
            continue
        if NO_CONTEXT_ANSWER in answer:
            continue  # covered by the hand-seeded borderline examples instead

        passages = retrieve(question)
        if not passages:
            continue  # index has since changed - drop rather than mismatch

        examples.append({
            "system": SYSTEM_PROMPT,
            "context": format_context(passages),
            "question": question,
            "answer": answer,
        })

    for seed in REFUSAL_SEEDS:
        examples.append({"system": SYSTEM_PROMPT, **seed})

    return examples


def main() -> None:
    examples = build_examples()
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    grounded = sum(1 for e in examples if e["answer"] != NO_CONTEXT_ANSWER)
    print(f"Wrote {len(examples)} examples ({grounded} grounded, "
          f"{len(examples) - grounded} refusal) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
