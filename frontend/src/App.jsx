import { useEffect, useState } from "react";
import { getHealth } from "./api.js";
import ChatPage from "./components/ChatPage.jsx";
import AdminModal from "./components/admin/AdminModal.jsx";

export default function App() {
  const [collegeName, setCollegeName] = useState(
    "St. Xavier's College (Autonomous), Ahmedabad"
  );
  const [adminOpen, setAdminOpen] = useState(false);

  useEffect(() => {
    getHealth()
      .then((d) => d.college && setCollegeName(d.college))
      .catch(() => {});
  }, []);

  return (
    <>
      <header className="app-header">
        <h1>
          Admission Assistant
          <small>{collegeName}</small>
        </h1>
        <button className="ghost" onClick={() => setAdminOpen(true)}>
          Admin
        </button>
      </header>

      <main className="app-main">
        <ChatPage />
      </main>

      {adminOpen && <AdminModal onClose={() => setAdminOpen(false)} />}
    </>
  );
}
