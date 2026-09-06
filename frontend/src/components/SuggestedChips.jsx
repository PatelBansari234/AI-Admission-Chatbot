const SUGGESTIONS = [
  "What UG courses are offered?",
  "How many credits per semester?",
  "Tell me about the BSc Computer Science programme",
  "What is the admission process?",
];

export default function SuggestedChips({ onPick }) {
  return (
    <div className="chips">
      {SUGGESTIONS.map((text) => (
        <button key={text} type="button" className="chip" onClick={() => onPick(text)}>
          {text}
        </button>
      ))}
    </div>
  );
}
