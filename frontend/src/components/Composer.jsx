import { useState } from "react";

export default function Composer({ onSend, disabled }) {
  const [value, setValue] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    const question = value.trim();
    if (!question) return;
    setValue("");
    onSend(question);
  }

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Ask about courses, eligibility, fees, documents..."
        autoComplete="off"
        required
      />
      <button className="primary" type="submit" disabled={disabled}>
        Send
      </button>
    </form>
  );
}
