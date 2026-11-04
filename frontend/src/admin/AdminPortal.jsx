import { useEffect, useState } from "react";
import { authFetch, getToken, isAdminAuthenticated, setToken } from "../api.js";
import AdminLogin from "./AdminLogin.jsx";
import OverviewDashboard from "./OverviewDashboard.jsx";
import PdfKnowledgeManagement from "./PdfKnowledgeManagement.jsx";
import UnansweredQuestionReview from "./UnansweredQuestionReview.jsx";
import SystemHealth from "./SystemHealth.jsx";
import "./admin.css";

const SECTIONS = [
  { id: "overview", label: "Overview Dashboard", icon: "📊" },
  { id: "pdfs", label: "PDF Knowledge Management", icon: "📚" },
  { id: "unanswered", label: "Unanswered-Question Review", icon: "📋" },
  { id: "health", label: "System Health", icon: "⚡" },
];

export default function AdminPortal({ onExit }) {
  const [authed, setAuthed] = useState(() => isAdminAuthenticated());
  const [activeSection, setActiveSection] = useState("overview");
  const [adminUser, setAdminUser] = useState("admin");
  const [notification, setNotification] = useState(null); // { message, ok: bool }

  function notify(message, ok = true) {
    setNotification({ message, ok });
  }

  useEffect(() => {
    if (!notification) return;
    const timer = setTimeout(() => setNotification(null), 5000);
    return () => clearTimeout(timer);
  }, [notification]);

  useEffect(() => {
    if (!authed) return;
    authFetch("/api/admin/me")
      .then((res) => {
        if (res.username) setAdminUser(res.username);
      })
      .catch(() => {
        setAuthed(false);
        setToken("");
      });
  }, [authed]);

  useEffect(() => {
    const handleUnauthorized = () => {
      setAuthed(false);
      setToken("");
    };
    window.addEventListener("admin-unauthorized", handleUnauthorized);
    return () => window.removeEventListener("admin-unauthorized", handleUnauthorized);
  }, []);

  function handleLogout() {
    setToken("");
    setAuthed(false);
    notify("Logged out successfully.");
  }

  if (!authed) {
    return <AdminLogin onLoggedIn={(user) => { setAdminUser(user); setAuthed(true); }} onExit={onExit} />;
  }

  const currentSectionMeta = SECTIONS.find((s) => s.id === activeSection) || SECTIONS[0];

  return (
    <div className="admin-viewport">
      {/* Sidebar Navigation */}
      <aside className="admin-sidebar">
        <div className="admin-sidebar-brand">
          <img src="/assets/sxca_logo.png" alt="SXCA Crest" className="admin-sidebar-crest" />
          <div className="admin-brand-info">
            <h1>SXCA Admin</h1>
            <span>Niaa Admission Assistant</span>
          </div>
        </div>

        <nav className="admin-nav">
          {SECTIONS.map((sec) => (
            <button
              key={sec.id}
              className={`admin-nav-item ${activeSection === sec.id ? "active" : ""}`}
              onClick={() => setActiveSection(sec.id)}
            >
              <div className="admin-nav-label">
                <span>{sec.icon}</span>
                <span>{sec.label}</span>
              </div>
            </button>
          ))}
        </nav>

        <div className="admin-sidebar-footer">
          <div className="admin-user-pill">
            <div className="admin-user-avatar">
              {adminUser.charAt(0).toUpperCase()}
            </div>
            <span>Logged in as <strong>{adminUser}</strong></span>
          </div>

          {onExit && (
            <button
              type="button"
              className="admin-footer-btn"
              onClick={onExit}
              title="Return to public student admission chatbot view"
            >
              ← Back to Student Chatbot
            </button>
          )}

          <button
            type="button"
            className="admin-footer-btn"
            style={{ color: "#fca5a5" }}
            onClick={handleLogout}
          >
            🔒 Sign Out
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="admin-main">
        <header className="admin-topbar">
          <h2 className="admin-topbar-title">
            <span>{currentSectionMeta.icon}</span>
            <span>{currentSectionMeta.label}</span>
          </h2>

          <div className="admin-topbar-actions">
            {onExit && (
              <button
                type="button"
                className="admin-action-btn admin-btn-secondary"
                onClick={onExit}
              >
                💬 Open Chatbot
              </button>
            )}
            <button
              type="button"
              className="admin-action-btn admin-btn-secondary"
              onClick={handleLogout}
            >
              Sign Out
            </button>
          </div>
        </header>

        <div className="admin-content-scroll">
          {/* Notification Alert */}
          {notification && (
            <div className={notification.ok ? "admin-alert-success" : "admin-alert-error"}>
              {notification.message}
            </div>
          )}

          {activeSection === "overview" && (
            <OverviewDashboard onNavigate={(sec) => setActiveSection(sec)} notify={notify} />
          )}

          {activeSection === "pdfs" && (
            <PdfKnowledgeManagement notify={notify} />
          )}

          {activeSection === "unanswered" && (
            <UnansweredQuestionReview notify={notify} />
          )}

          {activeSection === "health" && (
            <SystemHealth notify={notify} />
          )}
        </div>
      </main>
    </div>
  );
}
