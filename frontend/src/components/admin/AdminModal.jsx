import { useEffect, useState } from "react";
import { getToken } from "../../api.js";
import LoginForm from "./LoginForm.jsx";
import DocumentsTab from "./DocumentsTab.jsx";
import CrawlerTab from "./CrawlerTab.jsx";
import LogsTab from "./LogsTab.jsx";
import IntentsTab from "./IntentsTab.jsx";
import AccountTab from "./AccountTab.jsx";

const TABS = [
  { key: "docs", label: "Documents" },
  { key: "crawler", label: "Website crawler" },
  { key: "logs", label: "Questions asked" },
  { key: "intents", label: "Intents" },
  { key: "account", label: "Account" },
];

export default function AdminModal({ onClose }) {
  const [loggedIn, setLoggedIn] = useState(() => !!getToken());
  const [activeTab, setActiveTab] = useState("docs");
  const [note, setNote] = useState(null); // {message, ok}

  function notify(message, ok = true) {
    setNote({ message, ok });
  }

  useEffect(() => {
    if (!note) return;
    const t = setTimeout(() => setNote(null), 5000);
    return () => clearTimeout(t);
  }, [note]);

  useEffect(() => {
    const handler = () => setLoggedIn(false);
    window.addEventListener("admin-unauthorized", handler);
    return () => window.removeEventListener("admin-unauthorized", handler);
  }, []);

  function handleLogout() {
    setLoggedIn(false);
    setActiveTab("docs");
  }

  return (
    <div className="modal-backdrop" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="card">
        <div style={{ display: "flex", alignItems: "center" }}>
          <div style={{ flex: 1 }}>
            <h2>Admin panel</h2>
            <div className="muted">
              {loggedIn
                ? "Manage the documents the chatbot answers from"
                : "Sign in to manage the chatbot's dataset"}
            </div>
          </div>
          <button className="subtle" onClick={onClose}>
            Close
          </button>
        </div>

        {note && <div className={`note ${note.ok ? "ok" : "err"}`}>{note.message}</div>}

        {!loggedIn ? (
          <LoginForm onLoggedIn={() => setLoggedIn(true)} notify={notify} />
        ) : (
          <>
            <div className="tabs">
              {TABS.map((t) => (
                <button
                  key={t.key}
                  className={`tab ${activeTab === t.key ? "active" : ""}`}
                  onClick={() => setActiveTab(t.key)}
                >
                  {t.label}
                </button>
              ))}
            </div>
            {activeTab === "docs" && <DocumentsTab notify={notify} />}
            {activeTab === "crawler" && <CrawlerTab notify={notify} />}
            {activeTab === "logs" && <LogsTab notify={notify} />}
            {activeTab === "intents" && <IntentsTab notify={notify} />}
            {activeTab === "account" && <AccountTab notify={notify} onLogout={handleLogout} />}
          </>
        )}
      </div>
    </div>
  );
}
