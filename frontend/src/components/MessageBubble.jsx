import { useEffect, useState } from "react";

export function cleanForSpeech(raw) {
  if (!raw) return "";
  let s = raw;

  // Replace markdown links with readable text
  s = s.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");

  // Replace bare URLs
  s = s.replace(/https?:\/\/[^\s)]+/g, "on the official admission portal");

  // Format currency naturally
  s = s.replace(/₹\s*([0-9,]+)/g, "$1 rupees");

  // Replace /sem with per semester
  s = s.replace(/\/sem(?:ester)?\b/gi, " per semester");

  // Expand abbreviations for natural speech
  s = s.replace(/\bUG\b/g, "undergraduate");
  s = s.replace(/\bPG\b/g, "postgraduate");
  s = s.replace(/\bHOD\b/g, "Head of Department");
  s = s.replace(/\bDr\.\s*/g, "Doctor ");
  s = s.replace(/\bProf\.\s*/g, "Professor ");
  s = s.replace(/\bB\.?Sc\.?\b/gi, "B.Sc.");
  s = s.replace(/\bM\.?Sc\.?\b/gi, "M.Sc.");
  s = s.replace(/\bB\.?A\.?\b/gi, "B.A.");
  s = s.replace(/\bM\.?A\.?\b/gi, "M.A.");
  s = s.replace(/\bB\.?C\.?A\.?\b/gi, "B.C.A.");
  s = s.replace(/\bB\.?Com\.?\b/gi, "B.Com.");

  // Clean technical formatting, bullets, brackets, and pipes
  s = s.replace(/[#*•_`~[\]()|]/g, " ");
  s = s.replace(/[-–—]{2,}/g, ". ");
  s = s.replace(/\s+/g, " ");

  // Ensure clean pauses
  s = s.replace(/\.\s*\./g, ".");
  return s.trim();
}

function findHumanVoice(langCode) {
  if (typeof window === "undefined" || !window.speechSynthesis) return null;
  const voices = window.speechSynthesis.getVoices() || [];
  if (!voices.length) return null;

  const isGuj = langCode === "gu-IN";
  const isHi = langCode === "hi-IN";

  if (isGuj) {
    const guj = voices.find((v) => v.lang.startsWith("gu") && (v.name.includes("Natural") || v.name.includes("Google")));
    if (guj) return guj;
    const anyGuj = voices.find((v) => v.lang.startsWith("gu"));
    if (anyGuj) return anyGuj;
  }

  if (isHi) {
    const hi = voices.find((v) => v.lang.startsWith("hi") && (v.name.includes("Natural") || v.name.includes("Swara") || v.name.includes("Google")));
    if (hi) return hi;
    const anyHi = voices.find((v) => v.lang.startsWith("hi"));
    if (anyHi) return anyHi;
  }

  // Look for natural neural Indian English female voice (matches Niaa)
  const naturalIn = voices.find(
    (v) =>
      (v.name.includes("Neerja") || v.name.includes("Sonia") || v.name.includes("Heera") || v.name.includes("Natural")) &&
      (v.lang.includes("IN") || v.lang.includes("en"))
  );
  if (naturalIn) return naturalIn;

  // Look for Google Female or Natural English voices
  const googleFemale = voices.find(
    (v) =>
      (v.name.includes("Google") ||
        v.name.includes("Natural") ||
        v.name.includes("Female") ||
        v.name.includes("Zira") ||
        v.name.includes("Jenny") ||
        v.name.includes("Aria")) &&
      v.lang.startsWith("en")
  );
  if (googleFemale) return googleFemale;

  const anyEnIn = voices.find((v) => v.lang === "en-IN");
  if (anyEnIn) return anyEnIn;

  const anyEn = voices.find((v) => v.lang.startsWith("en"));
  return anyEn || voices[0];
}

export default function MessageBubble({ role, text, pending, sources, onEdit, onOpenSource }) {
  const [speaking, setSpeaking] = useState(false);
  const classes = ["msg", role, pending ? "pending" : ""].filter(Boolean).join(" ");

  // Ensure browser voices are loaded
  useEffect(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => {
        window.speechSynthesis.getVoices();
      };
    }
  }, []);

  useEffect(() => {
    return () => {
      if (speaking && window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    };
  }, [speaking]);

  function toggleSpeech() {
    if (!window.speechSynthesis) {
      alert("Text-to-speech is not supported in this browser.");
      return;
    }

    if (speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
      return;
    }

    window.speechSynthesis.cancel();

    const spokenText = cleanForSpeech(text);
    if (!spokenText) return;

    const utterance = new SpeechSynthesisUtterance(spokenText);

    // Detect language
    let langCode = "en-IN";
    if (/[\u0A80-\u0AFF]/.test(spokenText)) {
      langCode = "gu-IN";
    } else if (/[\u0900-\u097F]/.test(spokenText)) {
      langCode = "hi-IN";
    }
    utterance.lang = langCode;

    // Select the most natural human voice available
    const voice = findHumanVoice(langCode);
    if (voice) {
      utterance.voice = voice;
    }

    // Warm, conversational human pacing and tone
    utterance.rate = 0.94;
    utterance.pitch = 1.04;

    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);

    setSpeaking(true);
    window.speechSynthesis.speak(utterance);
  }

  function renderFormattedText(content) {
    if (!content) return "";

    const tokens = [];
    const regex = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|(\*\*([^*]+)\*\*)|(https?:\/\/[^\s]+)/g;
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(content)) !== null) {
      if (match.index > lastIndex) {
        tokens.push(content.substring(lastIndex, match.index));
      }
      if (match[1] && match[2]) {
        tokens.push(
          <a
            key={match.index}
            href={match[2]}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--clg-crimson, #8B1E2F)", textDecoration: "underline", fontWeight: 600 }}
          >
            {match[1]}
          </a>
        );
      } else if (match[4]) {
        tokens.push(<strong key={match.index}>{match[4]}</strong>);
      } else if (match[5]) {
        tokens.push(
          <a
            key={match.index}
            href={match[5]}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--clg-crimson, #8B1E2F)", textDecoration: "underline", fontWeight: 600 }}
          >
            {match[5]}
          </a>
        );
      }
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < content.length) {
      tokens.push(content.substring(lastIndex));
    }

    return tokens.length > 0 ? tokens : content;
  }

  return (
    <div className={`msg-row ${role === "user" ? "user-row" : "bot-row"}`}>
      {role === "bot" && (
        <img
          src="/niaa_avatar.jpg"
          alt="Niaa"
          className="bot-avatar"
        />
      )}
      <div className="msg-wrapper">
        <div className={classes} style={{ whiteSpace: "pre-wrap" }}>
          {renderFormattedText(text)}
        </div>

        {/* Source citation cards */}
        {role === "bot" && !pending && sources && sources.length > 0 && (
          <div className="source-cards-wrapper">
            {sources.map((src, idx) => (
              <div key={idx} className="source-badge-card">
                <div className="source-badge-info">
                  <span className="source-badge-icon">📄</span>
                  <span className="source-badge-title">Source: {src.title || src.source || "Official SXCA Document"}</span>
                  {src.page && <span className="source-badge-page">· Page {src.page}</span>}
                </div>
                {src.url ? (
                  <a
                    href={src.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="source-view-action"
                    title="Open official SXCA source document in new tab"
                  >
                    [View source ↗]
                  </a>
                ) : onOpenSource ? (
                  <button
                    type="button"
                    className="source-view-action"
                    onClick={() => onOpenSource(src)}
                    title="View verified source citation details"
                  >
                    [View source ↗]
                  </button>
                ) : null}
              </div>
            ))}
          </div>
        )}

        {role === "user" && onEdit && (
          <button
            type="button"
            className="edit-msg-btn"
            onClick={() => onEdit(text)}
            title="Edit this message"
          >
            ✎ Edit
          </button>
        )}
        {role === "bot" && !pending && (
          <button
            type="button"
            className={`speak-btn ${speaking ? "speaking" : ""}`}
            onClick={toggleSpeech}
            title={speaking ? "Stop reading" : "Listen in natural voice"}
          >
            {speaking ? "⏹️ Stop" : "🔊 Listen"}
          </button>
        )}
      </div>
    </div>
  );
}
