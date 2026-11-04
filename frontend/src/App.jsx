import { useEffect, useState } from "react";
import { getHealth, isAdminAuthenticated } from "./api.js";
import ChatPage from "./components/ChatPage.jsx";
import AdminPortal from "./admin/AdminPortal.jsx";

export default function App() {
  const [collegeName, setCollegeName] = useState("St. Xavier's College (Autonomous), Ahmedabad");
  const [restartCount, setRestartCount] = useState(0);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isAdminPortal, setIsAdminPortal] = useState(() => {
    return (
      window.location.pathname.startsWith("/admin") ||
      window.location.hash === "#admin" ||
      window.location.hash === "#/admin" ||
      window.location.search.includes("admin")
    );
  });

  useEffect(() => {
    getHealth()
      .then((d) => {
        if (d.college) setCollegeName(d.college);
      })
      .catch(() => {});
  }, []);

  // Listen for route changes (popstate, hashchange) and admin keyboard shortcut
  useEffect(() => {
    const checkRoute = () => {
      const match =
        window.location.pathname.startsWith("/admin") ||
        window.location.hash === "#admin" ||
        window.location.hash === "#/admin" ||
        window.location.search.includes("admin");
      setIsAdminPortal(match);
    };

    window.addEventListener("popstate", checkRoute);
    window.addEventListener("hashchange", checkRoute);

    // Administrator shortcut (Ctrl + Shift + A)
    const handleKey = (e) => {
      if (e.ctrlKey && e.shiftKey && (e.key === "A" || e.key === "a")) {
        e.preventDefault();
        setIsAdminPortal((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKey);

    return () => {
      window.removeEventListener("popstate", checkRoute);
      window.removeEventListener("hashchange", checkRoute);
      window.removeEventListener("keydown", handleKey);
    };
  }, []);

  function handleRestart() {
    setRestartCount((c) => c + 1);
  }

  function handleExitAdmin() {
    window.location.hash = "";
    if (window.location.pathname.startsWith("/admin")) {
      window.history.pushState(null, "", "/");
    }
    setIsAdminPortal(false);
  }

  // Render dedicated Admin Portal if requested
  if (isAdminPortal) {
    return <AdminPortal onExit={handleExitAdmin} />;
  }

  // Existing student chatbot interface (untouched)
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
                onClick={() => setIsAdminPortal(true)}
                title="Admin Portal"
              >
                ⚙
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
    </div>
  );
}
