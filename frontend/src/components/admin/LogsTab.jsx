import { useEffect, useState } from "react";
import { authFetch } from "../../api.js";

function escapeExcerpt(text, limit) {
  if (!text) return "";
  return text.length > limit ? `${text.slice(0, limit)}...` : text;
}

export default function LogsTab({ notify }) {
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [unanswered, setUnanswered] = useState(0);

  useEffect(() => {
    authFetch("/api/admin/chat-logs")
      .then((data) => {
        setRows(data.rows);
        setTotal(data.total);
        setUnanswered(data.unanswered || 0);
      })
      .catch((err) => notify(err.message, false));
  }, []);

  return (
    <section>
      <div className="muted">
        {total} questions asked so far - {unanswered} of the latest {rows.length} could not be
        answered from the documents.
      </div>
      <table>
        <thead>
          <tr>
            <th>Question</th>
            <th>Answer</th>
            <th>Answered?</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={3} className="muted">
                No questions yet.
              </td>
            </tr>
          ) : (
            rows.map((r, i) => (
              <tr key={i}>
                <td>{r.question}</td>
                <td>{escapeExcerpt(r.answer, 150)}</td>
                <td>{r.grounded ? "Yes" : <b style={{ color: "#b4531a" }}>No</b>}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </section>
  );
}
