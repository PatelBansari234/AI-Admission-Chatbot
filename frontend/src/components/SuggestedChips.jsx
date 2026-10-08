export default function SuggestedChips({ onPick, activeCourse, lastIntent }) {
  let suggestions = [
    "Courses available at SXCA",
    "Admission Process",
    "Eligibility Criteria",
    "Fee Structure",
  ];

  if (lastIntent === "calendar") {
    suggestions = [
      "CIA-1 Exam Schedule",
      "CIA-2 Exam Schedule",
      "Diwali Vacation Dates",
      "Academic Calendar 2026-27",
    ];
  } else if (lastIntent === "subjects" && activeCourse) {
    if (activeCourse.startsWith("MSc") || activeCourse.startsWith("MA")) {
      suggestions = [
        "Semester 1 Subjects",
        "Semester 2 Subjects",
        "Semester 3 Subjects",
        "Semester 4 Subjects",
        "Eligibility",
        "Fees",
      ];
    } else {
      suggestions = [
        "Semester 1 Subjects",
        "Semester 2 Subjects",
        "Semester 3 Subjects",
        "Eligibility",
        "Fees",
      ];
    }
  } else if (activeCourse) {
    if (activeCourse === "MSc-Artificial Intelligence") {
      suggestions = [
        "MSc AI Subjects",
        "MSc AI Eligibility",
        "MSc AI Fees",
        "Admission Process",
      ];
    } else if (activeCourse === "BSc-Artificial Intelligence & Machine Learning") {
      suggestions = [
        "BSc AI & ML Subjects",
        "BSc AI & ML Eligibility",
        "BSc AI & ML Fees",
        "Admission Process",
      ];
    } else if (activeCourse === "BCA") {
      suggestions = [
        "BCA Subjects",
        "BCA Eligibility",
        "BCA Fees",
        "How to apply for BCA?",
      ];
    } else if (activeCourse.startsWith("BCom")) {
      suggestions = [
        "B.Com Subjects",
        "B.Com Eligibility",
        "B.Com Fees",
        "Admission Process",
      ];
    } else {
      const cleanName = activeCourse
        .replace(/^(BSc|MSc|BA|MA|BCom|BS)-/i, "")
        .replace(/\s+/g, " ")
        .trim();
      suggestions = [
        `${cleanName} Subjects`,
        `${cleanName} Eligibility`,
        `${cleanName} Fees`,
        "Admission Process",
      ];
    }
  }

  return (
    <div className="chips-wrapper">
      <div className="chips">
        {suggestions.map((text) => (
          <button key={text} type="button" className="chip" onClick={() => onPick(text)}>
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}
