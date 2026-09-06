"""One-time bootstrap for course_eligibility: every row below was copied
verbatim (not inferred) from the one clean, complete admission-fee table in
data/raw/pdfs/SXCA-Admission-Brochure-2526.pdf ("Government Aided
Programmes" and "Self-Financed Programmes - UG/PG"). That PDF is a
print-shop imposition file whose content repeats 4-8x for print layout
(see the README's table-aware chunking note) - this script reads the table
once, by hand-verified transcription, rather than trusting an automated
pass over the duplicated raw text.

This is a bootstrap for ONE known-good table, not a general extraction
pipeline - Phase 3.2 (repeatable extraction for future document updates)
is separate, deliberately unbuilt work.

    python -m backend.db.seed_eligibility
"""

from backend.db.repo import upsert_course_eligibility

ACADEMIC_YEAR = "2025-26"
SOURCE = "SXCA-Admission-Brochure-2526.pdf"

# (course_name, level, category, eligibility, fee_amount, fee_amount_female, fee_note, contact_email)
ROWS = [
    # Government Aided - Arts (UG), male/female fee split
    ("BA-Economics", "UG", "grant_in_aid", "12th Arts", 6655, 6055, None, "economics@sxca.edu.in"),
    ("BA-English", "UG", "grant_in_aid", "12th Arts", 6655, 6055, None, "english@sxca.edu.in"),
    ("BA-Gujarati", "UG", "grant_in_aid", "12th Arts", 6655, 6055, None, "gujarati@sxca.edu.in"),
    ("BA-Psychology", "UG", "grant_in_aid", "12th Arts", 6655, 6055, None, "psychology@sxca.edu.in"),
    ("BA-Sanskrit", "UG", "grant_in_aid", "12th Arts", 6655, 6055, None, "sanskrit@sxca.edu.in"),
    # Government Aided - Science (UG), male/female fee split
    ("BSc-Biochemistry", "UG", "grant_in_aid", "12th Science", 8855, 8255, None, "biochemistry@sxca.edu.in"),
    ("BSc-Biochem. with Voc. BT", "UG", "grant_in_aid", "12th Science", 8855, 8255,
     "+4,000 (Sem I-IV) vocational add-on", "biochemistry@sxca.edu.in"),
    ("BSc-Botany", "UG", "grant_in_aid", "12th Science", 8855, 8255, None, "botany@sxca.edu.in"),
    ("BSc-Chemistry", "UG", "grant_in_aid", "12th Science", 8855, 8255, None, "chemistry@sxca.edu.in"),
    ("BSc-Chem. with Voc. IC", "UG", "grant_in_aid", "12th Science", 8855, 8255,
     "+3,000 (Sem I-IV) vocational add-on", "chemistry@sxca.edu.in"),
    ("BSc-Electronics", "UG", "grant_in_aid", "12th Science", 8855, 8255, None, "physics-electronics@sxca.edu.in"),
    ("BSc-Mathematics", "UG", "grant_in_aid", "12th Science", 8855, 8255, None, "mathematics@sxca.edu.in"),
    ("BSc-Physics", "UG", "grant_in_aid", "12th Science", 8855, 8255, None, "physics-electronics@sxca.edu.in"),
    ("BSc-Statistics", "UG", "grant_in_aid", "12th Science", 8855, 8255, None, "statistics@sxca.edu.in"),
    ("BSc-Zoology", "UG", "grant_in_aid", "12th Science", 8855, 8255, None, "zoology@sxca.edu.in"),
    # Government Aided - the one PG programme in this table
    ("MSc-Chemistry (Organic)", "PG", "grant_in_aid", "BSc Chemistry", 12790, 10990, None, "chemistry@sxca.edu.in"),
    # Self-Financed - UG, single fee (no gender split)
    ("BA-Economics", "UG", "self_financed", "12th Arts & Commerce", 32200, None, None, "ecohons@sxca.edu.in"),
    ("BA-English", "UG", "self_financed", "12th Arts", 32000, None, None, "english@sxca.edu.in"),
    ("BA-Psychology", "UG", "self_financed", "12th Arts", 35750, None, None, "psychology@sxca.edu.in"),
    ("BA-Sociology & Development", "UG", "self_financed", "12th Any stream", 31100, None, None, "sociology@sxca.edu.in"),
    ("BPA-Theatre & Production Management", "UG", "self_financed", "12th Any stream", 34950, None, None, "communication@sxca.edu.in"),
    ("BCom-BPS", "UG", "self_financed", "12th Commerce", 35750, None, None, "commerce@sxca.edu.in"),
    ("BCom (General)", "UG", "self_financed", "12th Commerce", 30750, None, None, "commerce@sxca.edu.in"),
    ("BCA", "UG", "self_financed", "12th Science & Commerce", 29750, None, None, "bca@sxca.edu.in"),
    ("BS-Business Administration", "UG", "self_financed", "12th Science & Commerce", 35000, None, None, "bba@sxca.edu.in"),
    ("BSc-Artificial Intelligence & Machine Learning", "UG", "self_financed", "12th Science", 38000, None, None, "mscai@sxca.edu.in"),
    ("BSc-Biotechnology", "UG", "self_financed", "12th Science", 39750, None, None, "biochemistry@sxca.edu.in"),
    ("BSc-Information Technology", "UG", "self_financed", "12th Science & Commerce", 30000, None, None, "bca@sxca.edu.in"),
    # Self-Financed - PG, single fee, eligibility is a prior degree, not a 12th-stream
    ("MA-Clinical Psychology", "PG", "self_financed", "BA Psychology", 33950, None, None, "psychology@sxca.edu.in"),
    ("MA-English", "PG", "self_financed", "BA English", 24950, None, None, "english@sxca.edu.in"),
    ("MSc-Artificial Intelligence", "PG", "self_financed", "BSc - Maths, Stats, Phy, Chem, BA Eco, BE & BCA", 50000, None, None, "mscai@sxca.edu.in"),
    ("MSc-Big Data Analytics", "PG", "self_financed", "BSc (CS, Maths, Stats), BA Eco & BCA", 44950, None, None, "bda@sxca.edu.in"),
    ("MSc-Biochemistry", "PG", "self_financed", "BSc Biochemistry", 37950, None, None, "biochemistry@sxca.edu.in"),
    ("MSc-Biotechnology", "PG", "self_financed", "BSc Biotechnology", 42950, None, None, "biochemistry@sxca.edu.in"),
    ("MSc-Botany", "PG", "self_financed", "BSc Botany", 25950, None, None, "botany@sxca.edu.in"),
    ("MSc-Chemistry (Analytical)", "PG", "self_financed", "BSc Chemistry", 33950, None, None, "pganalytical@sxca.edu.in"),
    ("MSc-Mathematics", "PG", "self_financed", "BSc Mathematics", 26450, None, None, "mathematics@sxca.edu.in"),
    ("MSc-Microbiology", "PG", "self_financed", "BSc - with any biological Science", 50000, None, None, "mscmicrobiology@sxca.edu.in"),
    ("MSc-Physics", "PG", "self_financed", "BSc (Physics or Electronics)", 39950, None, None, "mscphysics@sxca.edu.in"),
]

_GLOBAL_FEE_NOTE = "Fee structure is subject to revision (per SXCA-Admission-Brochure-2526.pdf footnote)"


def seed() -> int:
    count = 0
    for course_name, level, category, eligibility, fee, fee_female, note, email in ROWS:
        combined_note = " - ".join(n for n in (note, _GLOBAL_FEE_NOTE) if n)
        upsert_course_eligibility({
            "course_name": course_name,
            "level": level,
            "category": category,
            "eligibility": eligibility,
            "fee_amount": fee,
            "fee_amount_female": fee_female,
            "fee_note": combined_note,
            "contact_email": email,
            "academic_year": ACADEMIC_YEAR,
            "source_document": SOURCE,
        })
        count += 1
    return count


if __name__ == "__main__":
    n = seed()
    print(f"Seeded {n} course_eligibility rows from {SOURCE}")
