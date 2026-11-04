import { useEffect, useState } from "react";
import { authFetch } from "../api.js";

function formatDate(dateStr) {
  if (!dateStr) return "—";
  try {
    const d = new Date(dateStr);
    return isNaN(d.getTime())
      ? dateStr
      : d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return dateStr;
  }
}

export default function UnansweredQuestionReview({ notify }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterPendingOnly, setFilterPendingOnly] = useState(false);
  const [search, setSearch] = useState("");
  const [expandedRowId, setExpandedRowId] = useState(null);
  const [updatingId, setUpdatingId] = useState(null);

  async function loadQuestions() {
    setLoading(true);
    try {
      const data = await authFetch("/api/admin/unanswered-questions?limit=150");
      setItems(data.unanswered_questions || []);
    } catch (err) {
      if (notify) notify(err.message || "Failed to load unanswered questions", false);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadQuestions();
  }, []);

  async function toggleReview(chatId, currentReviewed) {
    setUpdatingId(chatId);
    try {
      const newStatus = !currentReviewed;
      const res = await authFetch(`/api/admin/unanswered-questions/${chatId}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewed: newStatus, notes: "" }),
      });
      if (notify) notify(res.message);
      setItems((prev) =>
        prev.map((it) =>
          it.id === chatId
            ? {
                ...it,
                reviewed: newStatus,
                reviewed_by: res.review.reviewed_by,
                reviewed_at: res.review.reviewed_at,
              }
            : it
        )
      );
    } catch (err) {
      if (notify) notify(err.message || "Failed to update review status", false);
    } finally {
      setUpdatingId(null);
    }
  }

  const filtered = items.filter((item) => {
    if (filterPendingOnly && item.reviewed) return false;
    if (search.trim()) {
      const q = search.toLowerCase();
      return (
        item.question?.toLowerCase().includes(q) ||
        item.answer?.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const pendingCount = items.filter((it) => !it.reviewed).length;

  return (
    <div>
      <div className="admin-card">
        <div className="admin-card-header">
          <div>
            <h3 className="admin-card-title">
              📋 Unanswered & Fallback Questions ({items.length})
            </h3>
            <span style={{ fontSize: 12, color: "#64748b" }}>
              Student queries where Niaa triggered verified fallbacks or lacked supporting sources
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <label style={{ fontSize: 13, display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={filterPendingOnly}
                onChange={(e) => setFilterPendingOnly(e.target.checked)}
              />
              Show Pending Only ({pendingCount})
            </label>
            <input
              type="text"
              className="admin-form-input"
              style={{ width: 220, padding: "6px 12px", fontSize: 13 }}
              placeholder="Search questions..."
              value={search}
              onChange={(e) => setSearchTerm ? setSearch(e.target.value) : setSearch(e.target.value)}
            />
          </div>
        </div>

        <div className="admin-card-body" style={{ padding: 0 }}>
          <div className="admin-table-wrapper">
            <table className="admin-table">
              <thead>
                <tr>
                  <th style={{ width: 70 }}>ID</th>
                  <th style={{ width: 140 }}>Timestamp</th>
                  <th>Student Question</th>
                  <th>Response & Reason</th>
                  <th style={{ width: 130 }}>Review Status</th>
                  <th style={{ textAlign: "right", width: 150 }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={6} style={{ textAlign: "center", padding: 30, color: "#64748b" }}>
                      Loading unanswered questions...
                    </td>
                  </tr>
                ) : filtered.length === 0 ? (
                  <tr>
                    <td colSpan={6} style={{ textAlign: "center", padding: 30, color: "#64748b" }}>
                      {search ? "No questions match your filter." : "No unanswered questions found."}
                    </td>
                  </tr>
                ) : (
                  filtered.map((item) => {
                    const isExpanded = expandedRowId === item.id;
                    const isUpdating = updatingId === item.id;
                    return (
                      <tr key={item.id}>
                        <td style={{ color: "#64748b", fontSize: 12 }}>#{item.id}</td>
                        <td style={{ fontSize: 12, color: "#64748b", whiteSpace: "nowrap" }}>
                          {formatDate(item.created_at)}
                        </td>
                        <td>
                          <div style={{ fontWeight: 600, color: "#0f172a" }}>
                            {item.question}
                          </div>
                        </td>
                        <td>
                          <div style={{ fontSize: 13, color: "#475569" }}>
                            {isExpanded
                              ? item.answer
                              : item.answer?.length > 120
                              ? item.answer.slice(0, 120) + "..."
                              : item.answer}
                            {item.answer?.length > 120 && (
                              <button
                                className="admin-link-btn"
                                style={{ marginLeft: 6, fontSize: 12 }}
                                onClick={() =>
                                  setExpandedRowId(isExpanded ? null : item.id)
                                }
                              >
                                {isExpanded ? "Collapse" : "Show full"}
                              </button>
                            )}
                          </div>
                          <div style={{ marginTop: 4 }}>
                            <span style={{ fontSize: 11, color: "#b4531a", background: "#fef3c7", padding: "1px 6px", borderRadius: 4 }}>
                              Reason: Official source unverified / fallback triggered
                            </span>
                          </div>
                        </td>
                        <td>
                          {item.reviewed ? (
                            <div>
                              <span className="admin-status-badge admin-status-reviewed">
                                ✓ Reviewed
                              </span>
                              {item.reviewed_by && (
                                <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>
                                  by {item.reviewed_by}
                                </div>
                              )}
                            </div>
                          ) : (
                            <span className="admin-status-badge admin-status-unreviewed">
                              ● Pending Review
                            </span>
                          )}
                        </td>
                        <td style={{ textAlign: "right" }}>
                          <button
                            className={`admin-action-btn ${
                              item.reviewed ? "admin-btn-secondary" : "admin-btn-success"
                            }`}
                            onClick={() => toggleReview(item.id, item.reviewed)}
                            disabled={isUpdating}
                          >
                            {isUpdating
                              ? "Saving..."
                              : item.reviewed
                              ? "Mark Pending"
                              : "Mark Reviewed"}
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
