import { Fragment, useEffect, useRef, useState } from "react";
import { streamChat } from "../api.js";
import MessageBubble from "./MessageBubble.jsx";
import SourcesPanel from "./SourcesPanel.jsx";
import SuggestedChips from "./SuggestedChips.jsx";
import Composer from "./Composer.jsx";

const GREETING =
  "Hello! I can answer questions about admissions, courses, eligibility and fees " +
  "using the official college prospectus and website. What would you like to know?";

let nextId = 1;

export default function ChatPage() {
  const [messages, setMessages] = useState([{ id: 0, role: "bot", text: GREETING }]);
  const [sending, setSending] = useState(false);
  const chatRef = useRef(null);
  // Conversation history sent to the backend - kept separate from `messages`
  // state so a stream of token updates doesn't rebuild it on every render.
  const historyRef = useRef([]);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages]);

  function patchMessage(id, patch) {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }

  async function handleSend(question) {
    setMessages((prev) => [...prev, { id: nextId++, role: "user", text: question }]);
    const botId = nextId++;
    setMessages((prev) => [
      ...prev,
      { id: botId, role: "bot", text: "Searching the college documents...", pending: true },
    ]);
    setSending(true);

    let answer = "";
    let sources = [];
    try {
      for await (const evt of streamChat(question, historyRef.current)) {
        if (evt.type === "token") {
          answer += evt.data;
          patchMessage(botId, { text: answer, pending: false });
        } else if (evt.type === "restart") {
          // The first attempt wasn't grounded and a live site check found
          // something better - discard it so it's never shown next to its
          // own correction.
          answer = "";
          patchMessage(botId, { text: "Checking the college website for more...", pending: true });
        } else if (evt.type === "sources") {
          sources = evt.data;
        } else if (evt.type === "error") {
          patchMessage(botId, { text: `Something went wrong: ${evt.data}`, pending: false });
        }
      }
      if (answer) {
        patchMessage(botId, { sources });
        historyRef.current = [
          ...historyRef.current,
          { role: "user", content: question },
          { role: "assistant", content: answer },
        ].slice(-6);
      }
    } catch (err) {
      patchMessage(botId, { text: err.message, pending: false });
    } finally {
      setSending(false);
    }
  }

  return (
    <>
      <div className="card">
        <div id="chat" ref={chatRef}>
          {messages.map((m) => (
            <Fragment key={m.id}>
              <MessageBubble role={m.role} text={m.text} pending={m.pending} />
              {m.role === "bot" && !m.pending && <SourcesPanel sources={m.sources} />}
            </Fragment>
          ))}
        </div>
        <SuggestedChips onPick={handleSend} />
        <Composer onSend={handleSend} disabled={sending} />
      </div>
      <p className="muted disclaimer">
        Answers come only from the college prospectus and website. Always confirm critical
        details with the admission office.
      </p>
    </>
  );
}
