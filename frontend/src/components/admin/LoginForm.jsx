import { useState } from "react";
import { login, setToken } from "../../api.js";

export default function LoginForm({ onLoggedIn, notify }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleLogin() {
    setBusy(true);
    try {
      const data = await login(username, password);
      setToken(data.token);
      setPassword("");
      onLoggedIn(data.username);
      notify(`Welcome, ${data.username}`);
    } catch (err) {
      notify(err.message, false);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="row">
      <input
        type="text"
        placeholder="Username"
        autoComplete="username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
      />
      <input
        type="password"
        placeholder="Password"
        autoComplete="current-password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleLogin()}
      />
      <button className="primary" disabled={busy} onClick={handleLogin}>
        Log in
      </button>
    </div>
  );
}
