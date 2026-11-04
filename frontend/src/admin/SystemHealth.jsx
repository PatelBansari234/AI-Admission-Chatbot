import { useEffect, useState } from "react";
import { authFetch } from "../api.js";

function getStatusBadge(status) {
  if (status === "Healthy") {
    return <span className="admin-status-badge admin-status-healthy">● Healthy</span>;
  }
  if (status === "Degraded") {
    return <span className="admin-status-badge admin-status-degraded">● Degraded</span>;
  }
  return <span className="admin-status-badge admin-status-unavailable">● Unavailable</span>;
}

export default function SystemHealth({ notify }) {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  async function loadHealth(isRefresh = false) {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);

    try {
      const data = await authFetch("/api/admin/system-health");
      setHealth(data);
      if (isRefresh && notify) notify("System health diagnostics updated.");
    } catch (err) {
      if (notify) notify(err.message || "Failed to retrieve health diagnostics", false);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadHealth();
  }, []);

  if (loading) {
    return (
      <div style={{ padding: 40, textAlign: "center", color: "#64748b" }}>
        Running safe health checks across API, Database, Vector Store, and Model...
      </div>
    );
  }

  return (
    <div>
      <div className="admin-card" style={{ marginBottom: 20 }}>
        <div className="admin-card-header">
          <div>
            <h3 className="admin-card-title">System Health Diagnostics</h3>
            <span style={{ fontSize: 12, color: "#64748b" }}>
              Component-level availability and performance probes (credentials omitted)
            </span>
          </div>
          <button
            className="admin-action-btn admin-btn-secondary"
            onClick={() => loadHealth(true)}
            disabled={refreshing}
          >
            {refreshing ? "Checking..." : "🔄 Run Diagnostics Now"}
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 20 }}>
        {/* Component 1: FastAPI Backend */}
        <div className="admin-card">
          <div className="admin-card-header">
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 18 }}>🚀</span>
              <strong>FastAPI Backend</strong>
            </div>
            {getStatusBadge(health?.api?.status)}
          </div>
          <div className="admin-card-body" style={{ fontSize: 13 }}>
            <div style={{ marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Service:</span>
              <strong style={{ color: "#0f172a" }}>{health?.api?.service}</strong>
            </div>
            <div style={{ marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Version:</span>
              <span style={{ color: "#0f172a" }}>v{health?.api?.version}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Deployment:</span>
              <span style={{ color: "#0f172a" }}>Autonomous College Local</span>
            </div>
          </div>
        </div>

        {/* Component 2: PostgreSQL 17 Database */}
        <div className="admin-card">
          <div className="admin-card-header">
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 18 }}>🐘</span>
              <strong>PostgreSQL Database</strong>
            </div>
            {getStatusBadge(health?.database?.status)}
          </div>
          <div className="admin-card-body" style={{ fontSize: 13 }}>
            <div style={{ marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Engine:</span>
              <strong style={{ color: "#0f172a" }}>{health?.database?.engine}</strong>
            </div>
            <div style={{ marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Database:</span>
              <span style={{ color: "#0f172a" }}>{health?.database?.database}</span>
            </div>
            <div style={{ marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Host:Port:</span>
              <span style={{ color: "#0f172a" }}>
                {health?.database?.host}:{health?.database?.port}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Probe Latency:</span>
              <span style={{ color: "#059669", fontWeight: 600 }}>
                {health?.database?.latency_ms !== null ? `${health?.database?.latency_ms} ms` : "—"}
              </span>
            </div>
          </div>
        </div>

        {/* Component 3: ChromaDB Vector Store */}
        <div className="admin-card">
          <div className="admin-card-header">
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 18 }}>📊</span>
              <strong>ChromaDB Vector Store</strong>
            </div>
            {getStatusBadge(health?.vector_db?.status)}
          </div>
          <div className="admin-card-body" style={{ fontSize: 13 }}>
            <div style={{ marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Engine:</span>
              <strong style={{ color: "#0f172a" }}>{health?.vector_db?.engine}</strong>
            </div>
            <div style={{ marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Collection:</span>
              <span style={{ color: "#0f172a" }}>{health?.vector_db?.collection}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Active Search Chunks:</span>
              <strong style={{ color: "#0b2545" }}>
                {health?.vector_db?.chunks?.toLocaleString()}
              </strong>
            </div>
          </div>
        </div>

        {/* Component 4: Local Qwen LLM */}
        <div className="admin-card">
          <div className="admin-card-header">
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 18 }}>🧠</span>
              <strong>Local Qwen Model</strong>
            </div>
            {getStatusBadge(health?.model?.status)}
          </div>
          <div className="admin-card-body" style={{ fontSize: 13 }}>
            <div style={{ marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Model:</span>
              <strong style={{ color: "#0f172a" }}>{health?.model?.model_name}</strong>
            </div>
            <div style={{ marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Compute Device:</span>
              <span style={{ color: "#0f172a" }}>
                {health?.model?.device} (CUDA GPU: {health?.model?.cuda_available ? "Yes" : "No"})
              </span>
            </div>
            <div style={{ marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Memory Cached:</span>
              <span style={{ color: health?.model?.cached_in_memory ? "#059669" : "#d97706", fontWeight: 600 }}>
                {health?.model?.cached_in_memory ? "Yes (Active)" : "On Demand"}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#64748b" }}>Status Note:</span>
              <span style={{ color: "#64748b", fontSize: 12 }}>{health?.model?.note}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
