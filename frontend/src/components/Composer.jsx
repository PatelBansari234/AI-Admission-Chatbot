import { useEffect, useRef, useState } from "react";

export default function Composer({
  onSend,
  disabled,
  isGenerating,
  onStop,
  editValue,
  onCancelEdit,
}) {
  const [value, setValue] = useState("");
  const [listening, setListening] = useState(false);
  const [lang, setLang] = useState("en-IN");
  const recognitionRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (editValue !== undefined && editValue !== null && editValue !== "") {
      setValue(editValue);
      if (inputRef.current) {
        inputRef.current.focus();
        inputRef.current.select();
      }
    }
  }, [editValue]);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const rec = new SpeechRecognition();
      rec.continuous = false;
      rec.interimResults = true;
      rec.lang = lang;

      rec.onresult = (e) => {
        const transcript = Array.from(e.results)
          .map((r) => r[0].transcript)
          .join("");
        setValue(transcript);
      };

      rec.onerror = (err) => {
        console.warn("Speech recognition error:", err);
        setListening(false);
      };

      rec.onend = () => {
        setListening(false);
      };

      recognitionRef.current = rec;
    }
    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch {}
      }
    };
  }, [lang]);

  function toggleVoice() {
    if (!recognitionRef.current) {
      alert("Voice input is supported in Chrome, Edge, Safari, and other modern browsers.");
      return;
    }
    if (listening) {
      recognitionRef.current.stop();
      setListening(false);
    } else {
      try {
        recognitionRef.current.lang = lang;
        recognitionRef.current.start();
        setListening(true);
      } catch (err) {
        console.warn("Failed to start speech recognition:", err);
        setListening(false);
      }
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (listening && recognitionRef.current) {
      recognitionRef.current.stop();
      setListening(false);
    }
    const question = value.trim();
    if (!question) return;
    if (isGenerating && onStop) {
      onStop();
    }
    setValue("");
    if (onCancelEdit) onCancelEdit();
    onSend(question);
  }

  return (
    <div className="composer-wrapper">
      <form className="composer" onSubmit={handleSubmit}>
        <select
          className="lang-select"
          value={lang}
          onChange={(e) => setLang(e.target.value)}
          title="Voice Language"
        >
          <option value="en-IN">EN</option>
          <option value="hi-IN">HI</option>
          <option value="gu-IN">GU</option>
        </select>
        <input
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={
            isGenerating
              ? "Niaa is typing... You can type your next question..."
              : listening
              ? "Listening... Speak now..."
              : editValue
              ? "Edit your question and press Send..."
              : "Type your message here..."
          }
          autoComplete="off"
          disabled={false}
          required
        />
        {editValue && (
          <button
            type="button"
            className="cancel-edit-btn"
            onClick={() => {
              setValue("");
              if (onCancelEdit) onCancelEdit();
            }}
            title="Cancel editing"
          >
            Cancel
          </button>
        )}
        {isGenerating && (
          <button
            type="button"
            className="stop-btn"
            onClick={onStop}
            title="Stop generation"
          >
            <span className="stop-icon">⏹</span> Stop
          </button>
        )}
        <button
          type="button"
          className={`voice-btn ${listening ? "listening" : ""}`}
          onClick={toggleVoice}
          title={listening ? "Stop listening" : "Speak with Voice"}
        >
          {listening ? "🔴" : "🎤"}
        </button>
        <button
          className="send-btn"
          type="submit"
          disabled={!value.trim()}
          title={isGenerating ? "Send next question" : "Send Message"}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
          </svg>
        </button>
      </form>
    </div>
  );
}
