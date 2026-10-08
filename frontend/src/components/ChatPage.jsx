import { Fragment, useEffect, useRef, useState } from "react";
import { streamChat } from "../api.js";
import MessageBubble from "./MessageBubble.jsx";
import SuggestedChips from "./SuggestedChips.jsx";
import Composer from "./Composer.jsx";
import SourceViewerModal from "./SourceViewerModal.jsx";

export function getSingleWelcomeMessage(college) {
  return (
    `🎓 **Welcome to St. Xavier's College (Autonomous), Ahmedabad!**\n\n` +
    `⭐ **NAAC A Grade (Cycle-IV)**  |  📚 **36+ UG & PG Courses**  |  💼 **GSIRF 5-Star Rating**\n\n` +
    `I'm **Niaa**, your Virtual Admission Counsellor.\n\n` +
    `I can help you explore courses, check eligibility requirements, understand fee structures, and guide your admission step-by-step.\n\n` +
    `Type **"HI"** or select a course option below to get started!`
  );
}

let nextId = 10;

export default function ChatPage({ collegeName, restartTrigger }) {
  const cName = collegeName || "St. Xavier's College (Autonomous), Ahmedabad";
  const [messages, setMessages] = useState(() => [
    {
      id: 1,
      role: "bot",
      text: getSingleWelcomeMessage(cName),
    },
  ]);
  const [sending, setSending] = useState(false);
  const [activeCourse, setActiveCourse] = useState(null);
  const [lastIntent, setLastIntent] = useState(null);
  const [editingText, setEditingText] = useState("");
  const [editTargetId, setEditTargetId] = useState(null);
  const [activeSourceModal, setActiveSourceModal] = useState(null);
  const [showScrollBottom, setShowScrollBottom] = useState(false);
  const chatRef = useRef(null);
  const historyRef = useRef([]);
  const abortControllerRef = useRef(null);
  const isNearBottomRef = useRef(true);

  const handleScroll = () => {
    if (chatRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatRef.current;
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
      const isBottom = distanceFromBottom < 50;
      isNearBottomRef.current = isBottom;
      setShowScrollBottom(!isBottom);
    }
  };

  const scrollToBottom = () => {
    if (chatRef.current) {
      chatRef.current.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
      isNearBottomRef.current = true;
      setShowScrollBottom(false);
    }
  };

  // Reset when restartTrigger or collegeName changes
  useEffect(() => {
    if (restartTrigger) {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      isNearBottomRef.current = true;
      setMessages([
        {
          id: 1,
          role: "bot",
          text: getSingleWelcomeMessage(cName),
        },
      ]);
      historyRef.current = [];
      setActiveCourse(null);
      setLastIntent(null);
      setEditingText("");
      setEditTargetId(null);
      setActiveSourceModal(null);
    }
  }, [restartTrigger, cName]);

  // Update initial message if collegeName changes before any user message
  useEffect(() => {
    setMessages((prev) => {
      if (prev.length === 1 && prev[0].id === 1) {
        return [{ id: 1, role: "bot", text: getSingleWelcomeMessage(cName) }];
      }
      return prev;
    });
  }, [cName]);

  useEffect(() => {
    if (chatRef.current && isNearBottomRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  function patchMessage(id, patch) {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }

  function handleStop() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }

  function handleEdit(messageId, userText) {
    setEditingText(userText);
    setEditTargetId(messageId);
  }

  async function handleSend(question) {
    const trimmed = question.trim();
    if (!trimmed) return;

    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch {}
    }

    isNearBottomRef.current = true;

    // Detect calendar/exam or subjects intent to adjust suggested chips
    const isCalendar = /\b(cai|cia|ca[-\s]?[12i]|exam|vacation|holiday|dewai?l|diwali|schedule|dates?|calendar)\b/i.test(trimmed);
    const isSubjects = /\b(subjects?|syllabus|curriculum|sem\s*[1-8]|semester|papers?|what\s*will\s*i\s*study)\b/i.test(trimmed);
    if (isCalendar) {
      setLastIntent("calendar");
    } else if (isSubjects) {
      setLastIntent("subjects");
    }

    setEditingText("");

    // If this send was triggered by editing a previous user message,
    // truncate messages and history from that point forward.
    let baseHistory = historyRef.current;
    if (editTargetId) {
      const targetIdx = messages.findIndex((m) => m.id === editTargetId);
      if (targetIdx !== -1) {
        const truncated = messages.slice(0, targetIdx);
        setMessages(truncated);
        const priorUserCount = truncated.filter((m) => m.role === "user").length;
        baseHistory = baseHistory.slice(0, priorUserCount * 2);
        historyRef.current = baseHistory;
      }
      setEditTargetId(null);
    }

    setMessages((prev) => [...prev, { id: nextId++, role: "user", text: trimmed }]);

    const botId = nextId++;
    setMessages((prev) => [
      ...prev,
      { id: botId, role: "bot", text: "Finding the latest admission details...", pending: true },
    ]);
    setSending(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let answer = "";
    let sources = [];
    try {
      for await (const evt of streamChat(trimmed, baseHistory, controller.signal)) {
        if (evt.type === "token") {
          answer += evt.data;
          patchMessage(botId, { text: answer, pending: false });
        } else if (evt.type === "course" && evt.data) {
          setActiveCourse(evt.data);
          setLastIntent("course");
        } else if (evt.type === "restart") {
          answer = "";
          patchMessage(botId, { text: "Checking latest college information...", pending: true });
        } else if (evt.type === "sources") {
          sources = evt.data;
        } else if (evt.type === "error") {
          patchMessage(botId, { text: `Something went wrong: ${evt.data}`, pending: false });
        }
      }
      if (answer) {
        patchMessage(botId, { sources });
        historyRef.current = [
          ...baseHistory,
          { role: "user", content: trimmed },
          { role: "assistant", content: answer },
        ].slice(-6);
      }
    } catch (err) {
      if (err.name === "AbortError") {
        // Stopped by user: preserve tokens received so far
        patchMessage(botId, { pending: false });
      } else {
        patchMessage(botId, { text: err.message, pending: false });
      }
    } finally {
      if (abortControllerRef.current === controller) {
        setSending(false);
        abortControllerRef.current = null;
      }
    }
  }

  return (
    <div className="card chat-card-container">
      {/* Chat messages feed with doodle wallpaper */}
      <div id="chat" className="doodle-chat-feed" ref={chatRef} onScroll={handleScroll}>
        {messages.map((m) => (
          <Fragment key={m.id}>
            <MessageBubble
              role={m.role}
              text={m.text}
              pending={m.pending}
              sources={m.sources}
              onEdit={m.role === "user" ? (text) => handleEdit(m.id, text) : undefined}
              onOpenSource={setActiveSourceModal}
            />
          </Fragment>
        ))}
      </div>
      {showScrollBottom && (
        <button
          type="button"
          className="jump-bottom-btn"
          onClick={scrollToBottom}
          title="Jump to latest message"
        >
          ↓ Jump to latest
        </button>
      )}

      <SuggestedChips
        onPick={handleSend}
        activeCourse={activeCourse}
        lastIntent={lastIntent}
      />
      <Composer
        onSend={handleSend}
        disabled={false}
        isGenerating={sending}
        onStop={handleStop}
        editValue={editingText}
        onCancelEdit={() => {
          setEditingText("");
          setEditTargetId(null);
        }}
      />

      {activeSourceModal && (
        <SourceViewerModal
          source={activeSourceModal}
          onClose={() => setActiveSourceModal(null)}
        />
      )}
    </div>
  );
}
