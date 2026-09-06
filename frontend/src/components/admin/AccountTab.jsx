import { useState } from "react";
import { authFetch, setToken } from "../../api.js";

export default function AccountTab({ notify, onLogout }) {
  const [newPassword, setNewPassword] = useState("");

  async function handleUpdate() {
    const body = new FormData();
    body.append("new_password", newPassword);
    try {
      const data = await authFetch("/api/admin/password", { method: "POST", body });
      setNewPassword("");
      notify(data.message);
    } catch (err) {
      notify(err.message, false);
    }
  }

  function handleLogout() {
    setToken("");
    onLogout();
    notify("Logged out");
  }

  return (
    <section>
      <div className="row">
        <input
          type="password"
          placeholder="New password (8+ characters)"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
        />
        <button className="primary" onClick={handleUpdate}>
          Update password
        </button>
        <button className="subtle" onClick={handleLogout}>
          Log out
        </button>
      </div>
    </section>
  );
}
