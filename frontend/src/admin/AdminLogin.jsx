import { useState } from "react";
import { login, setToken } from "../api.js";

export default function AdminLogin({ onLoggedIn, onExit }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError("Please enter both username and password.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const data = await login(username.trim(), password);
      if (data.token) {
        setToken(data.token);
        if (data.role !== "admin") {
          setError("Your account does not have administrator privileges.");
          setToken("");
          return;
        }
        onLoggedIn(data.username || username.trim());
      } else {
        setError("Login failed: no access token returned.");
      }
    } catch (err) {
      setError(err.message || "Invalid administrator credentials.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="admin-login-wrapper">
      <div className="admin-login-card">
        <div className="admin-login-header">
          <img src="/assets/sxca_logo.png" alt="SXCA Crest" className="admin-login-crest" />
          <h2 className="admin-login-title">Niaa Admin Portal</h2>
          <p className="admin-login-subtitle">
            St. Xavier's College (Autonomous), Ahmedabad
          </p>
        </div>

        <form className="admin-login-body" onSubmit={handleSubmit}>
          {error && <div className="admin-alert-error">{error}</div>}

          <div className="admin-form-group">
            <label className="admin-form-label" htmlFor="admin-username">
              Admin Username
            </label>
            <input
              id="admin-username"
              type="text"
              className="admin-form-input"
              placeholder="e.g. admin"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              required
            />
          </div>

          <div className="admin-form-group">
            <label className="admin-form-label" htmlFor="admin-password">
              Password
            </label>
            <input
              id="admin-password"
              type="password"
              className="admin-form-input"
              placeholder="Enter password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </div>

          <button
            type="submit"
            className="admin-btn-primary"
            disabled={loading}
          >
            {loading ? "Authenticating..." : "Sign In to Admin Panel"}
          </button>

          {onExit && (
            <div className="admin-login-footer">
              <button type="button" className="admin-link-btn" onClick={onExit}>
                ← Back to Student Chatbot
              </button>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
