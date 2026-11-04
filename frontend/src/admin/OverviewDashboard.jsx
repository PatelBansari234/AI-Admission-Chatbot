import { useEffect, useState } from "react";
import { authFetch } from "../api.js";

export default function OverviewDashboard({ onNavigate, notify }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function loadOverview() {
    setLoading(true);
    setError(null);
    try {
      const res = await authFetch("/api/admin/overview");
      setData(res);
    } catch (err) {
      setError(err.message || "Failed to load dashboard metrics");
      if (notify) notify(err.message, false);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadOverview();
  }, []);

  if (loading) {
    return (
      <div style={{ padding: 40, textAlign: "center", color: "#64748b" }}>
        Loading overview metrics...
      </div>
    );
  }

  if (error) {
    return (
      <div className="admin-alert-error">
        <strong>Error loading metrics:</strong> {error}
        <button
          className="admin-action-btn admin-btn-secondary"
          onClick={loadOverview}
          style={{ marginLeft: 12 }}
        >
          Retry
        </button>
      </div>
    );
  }

  const statusClass =
    data?.system_status === "Healthy"
      ? "admin-status-healthy"
      : data?.system_status === "Degraded"
      ? "admin-status-degraded"
      : "admin-status-unavailable";

  return (
    <div>
      <div className="admin-metrics-grid">
        {/* Card 1: Total Recorded Questions */}
        <div className="admin-metric-card accent-navy">
          <div className="admin-metric-header">
            <span className="admin-metric-title">Total Questions</span>
            <span className="admin-metric-icon">💬</span>
          </div>
          <div className="admin-metric-value">
            {data?.total_questions?.toLocaleString() ?? "Not tracked"}
          </div>
          <p className="admin-metric-subtext">
            Recorded student interactions in database
          </p>
        </div>

        {/* Card 2: Unanswered / Review Needed */}
        <div className="admin-metric-card accent-amber">
          <div className="admin-metric-header">
            <span className="admin-metric-title">Unanswered / Gaps</span>
            <span className="admin-metric-icon">⚠️</span>
          </div>
          <div className="admin-metric-value" style={{ color: "#d97706" }}>
            {data?.unanswered_questions?.toLocaleString() ?? "Not tracked"}
          </div>
          <p className="admin-metric-subtext">
            {data?.unreviewed_count !== undefined ? (
              <span>
                <strong>{data.unreviewed_count}</strong> pending admin review
              </span>
            ) : (
              "Low confidence or verified refusals"
            )}
          </p>
        </div>

        {/* Card 3: Indexed PDF & Document Sources */}
        <div className="admin-metric-card accent-crimson">
          <div className="admin-metric-header">
            <span className="admin-metric-title">Indexed PDF Sources</span>
            <span className="admin-metric-icon">📄</span>
          </div>
          <div className="admin-metric-value">
            {data?.pdf_sources_count?.toLocaleString() ?? "Not tracked"}
          </div>
          <p className="admin-metric-subtext">
            {data?.searchable_chunks !== undefined ? (
              <span>
                <strong>{data.searchable_chunks.toLocaleString()}</strong> searchable chunks in vector DB
              </span>
            ) : (
              "Official prospectuses and circulars"
            )}
          </p>
        </div>

        {/* Card 4: System Health Summary */}
        <div className="admin-metric-card accent-green">
          <div className="admin-metric-header">
            <span className="admin-metric-title">System Health</span>
            <span className="admin-metric-icon">⚡</span>
          </div>
          <div className="admin-metric-value" style={{ fontSize: 24, marginTop: 4 }}>
            <span className={`admin-status-badge ${statusClass}`}>
              ● {data?.system_status || "Healthy"}
            </span>
          </div>
          <p className="admin-metric-subtext" style={{ marginTop: 10 }}>
            API, Postgres 17, Vector DB & Qwen status
          </p>
        </div>
      </div>

      {/* Quick Navigation Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        <div className="admin-card">
          <div className="admin-card-header">
            <h3 className="admin-card-title">📚 Knowledge Base Management</h3>
          </div>
          <div className="admin-card-body">
            <p style={{ margin: "0 0 16px", fontSize: 13.5, color: "#475569" }}>
              Upload new official SXCA admission prospectuses or safely remove outdated brochures without disrupting other documents.
            </p>
            <button
              className="admin-action-btn admin-btn-primary"
              style={{ width: "auto" }}
              onClick={() => onNavigate("pdfs")}
            >
              Open PDF Management →
            </button>
          </div>
        </div>

        <div className="admin-card">
          <div className="admin-card-header">
            <h3 className="admin-card-title">🔍 Unanswered Student Queries</h3>
          </div>
          <div className="admin-card-body">
            <p style={{ margin: "0 0 16px", fontSize: 13.5, color: "#475569" }}>
              Inspect questions where Niaa fell back or lacked verified sources. Mark questions as reviewed as new policies are published.
            </p>
            <button
              className="admin-action-btn admin-btn-secondary"
              style={{ width: "auto" }}
              onClick={() => onNavigate("unanswered")}
            >
              Review Unanswered Questions ({data?.unreviewed_count ?? 0}) →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
