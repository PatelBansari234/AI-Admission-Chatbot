import { useEffect, useState } from "react";
import { getHealth, isAdminAuthenticated } from "./api.js";
import ChatPage from "./components/ChatPage.jsx";
import AdminModal from "./components/admin/AdminModal.jsx";

export default function App() {
  const [collegeName, setCollegeName] = useState("St. Xavier's College (Autonomous), Ahmedabad");
  const [adminOpen, setAdminOpen] = useState(false);
  const [restartCount, setRestartCount] = useState(0);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isAdmin, setIsAdmin] = useState(() => isAdminAuthenticated());

  useEffect(() => {
    getHealth()
      .then((d) => {
        if (d.college) setCollegeName(d.college);
      })
      .catch(() => {});
  }, []);

  // Update admin status on unauthorized event, storage change, or keyboard shortcut
  useEffect(() => {
    const handleAuth = () => setIsAdmin(isAdminAuthenticated());
    window.addEventListener("admin-unauthorized", handleAuth);
    window.addEventListener("storage", handleAuth);

    // Administrator shortcut (Ctrl + Shift + A) or URL parameter (?admin or #admin)
    const handleKey = (e) => {
      if (e.ctrlKey && e.shiftKey && (e.key === "A" || e.key === "a")) {
        e.preventDefault();
        setAdminOpen(true);
      }
    };
    window.addEventListener("keydown", handleKey);

    if (window.location.search.includes("admin") || window.location.hash === "#admin") {
      setAdminOpen(true);
    }

    return () => {
      window.removeEventListener("admin-unauthorized", handleAuth);
      window.removeEventListener("storage", handleAuth);
      window.removeEventListener("keydown", handleKey);
    };
  }, []);

  function handleRestart() {
    setRestartCount((c) => c + 1);
  }

  function handleCloseAdmin() {
    setAdminOpen(false);
    setIsAdmin(isAdminAuthenticated());
  }

  return (
    <div className="app-viewport">
      {!isMinimized ? (
        <div className="chat-window-wrapper">
          {/* SXCA Collegiate Header */}
          <header className="niaa-header">
            <div className="niaa-profile">
              <img src="/assets/sxca_logo.png" alt="SXCA Crest" className="sxca-header-crest" />
              <div className="avatar-wrapper">
                <img src="/niaa_avatar.jpg" alt="Niaa" className="niaa-avatar-img" />
                <span className="online-dot" />
              </div>
              <div className="niaa-info">
                <div className="niaa-name-row">
                  <h2 className="niaa-name">Niaa</h2>
                  <span className="niaa-verified-badge" title="Official Virtual Admission Counsellor">✓ Counsellor</span>
                </div>
                <span className="niaa-sub">St. Xavier's College (Autonomous), Ahmedabad</span>
              </div>
            </div>

            <div className="niaa-header-controls">
              <button
                type="button"
                className="header-icon-btn"
                onClick={handleRestart}
                title="Restart conversation"
              >
                ↺
              </button>
              <button
                type="button"
                className="header-icon-btn"
                onClick={() => setIsMinimized(true)}
                title="Minimize chat"
              >
                —
              </button>
            </div>
          </header>

          {/* Main Chat Interface */}
          <main className="chat-body-main">
            <ChatPage collegeName={collegeName} restartTrigger={restartCount} />
          </main>
        </div>
      ) : (
        /* Minimized Floating Launcher */
        <div className="floating-launcher" onClick={() => setIsMinimized(false)}>
          <div className="launcher-avatar-wrap">
            <img src="/niaa_avatar.jpg" alt="Niaa" className="launcher-avatar" />
            <span className="launcher-dot" />
          </div>
          <div className="launcher-text">
            <strong>Chat with Niaa</strong>
            <small>Admission Assistant</small>
          </div>
          <span className="launcher-open-arrow">💬</span>
        </div>
      )}

      {adminOpen && <AdminModal onClose={handleCloseAdmin} />}
    </div>
  );
}
