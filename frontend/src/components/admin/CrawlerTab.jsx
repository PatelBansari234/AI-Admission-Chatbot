import { useEffect, useState } from "react";
import { authFetch } from "../../api.js";

const STRATEGIES = [
  { value: "astar", label: "A* (most admission-relevant pages first)" },
  { value: "bfs", label: "BFS (level by level from the homepage)" },
  { value: "dfs", label: "DFS (follow one branch deep before backtracking)" },
];

export default function CrawlerTab({ notify }) {
  const [startUrl, setStartUrl] = useState("https://sxca.edu.in/");
  const [strategy, setStrategy] = useState("astar");
  const [maxPages, setMaxPages] = useState(30);
  const [maxDepth, setMaxDepth] = useState(3);
  const [running, setRunning] = useState(false);
  const [lastSummary, setLastSummary] = useState(null);
  const [runs, setRuns] = useState([]);
  const [settings, setSettings] = useState(null);

  function loadData() {
    authFetch("/api/admin/crawl-runs")
      .then((data) => setRuns(data.runs || []))
      .catch((err) => notify(err.message, false));

    authFetch("/api/admin/crawler/settings")
      .then((data) => setSettings(data.settings || {}))
      .catch(() => {});
  }

  useEffect(() => {
    loadData();
  }, []);

  async function handleCrawl() {
    setRunning(true);
    try {
      const data = await authFetch("/api/admin/crawl", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_url: startUrl,
          strategy,
          max_pages: Number(maxPages),
          max_depth: Number(maxDepth),
        }),
      });
      notify(data.message);
      setLastSummary(data.summary);
      loadData();
    } catch (err) {
      notify(err.message, false);
    } finally {
      setRunning(false);
    }
  }

  async function handleUpdateSettings(newSettings) {
    try {
      const data = await authFetch("/api/admin/crawler/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newSettings),
      });
      notify(data.message);
      setSettings(data.settings);
    } catch (err) {
      notify(err.message, false);
    }
  }

  async function handleRollback() {
    if (!confirm("Are you sure you want to rollback to the previous dataset backup?")) return;
    try {
      const data = await authFetch("/api/admin/crawler/rollback", { method: "POST" });
      notify(data.message);
      loadData();
    } catch (err) {
      notify(err.message, false);
    }
  }

  return (
    <section>
      <div style={{ background: "#f8fafc", padding: "12px 16px", borderRadius: 8, border: "1px solid #e2e8f0", marginBottom: 16 }}>
        <h4 style={{ margin: "0 0 8px 0", fontSize: "14px", fontWeight: 600 }}>Automatic Website Freshness & Scheduler</h4>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, fontSize: "13px" }}>
          <div>
            <div className="muted">Scheduler Status</div>
            <strong style={{ color: settings?.scheduler_enabled ? "#16a34a" : "#dc2626" }}>
              {settings?.scheduler_enabled ? "Active (Every " + (settings?.interval_hours || 24) + "h)" : "Disabled"}
            </strong>
          </div>
          <div>
            <div className="muted">Last Run</div>
            <strong>{settings?.last_run_at ? settings.last_run_at.replace("T", " ").slice(0, 19) + " UTC" : "Never"}</strong>
          </div>
          <div>
            <div className="muted">Next Scheduled</div>
            <strong>{settings?.next_run_at ? settings.next_run_at.replace("T", " ").slice(0, 19) + " UTC" : "Auto"}</strong>
          </div>
          <div>
            <div className="muted">Freshness Stats</div>
            <span>Changed: <strong>{settings?.pages_changed || 0}</strong> | New: <strong>{settings?.pages_new || 0}</strong></span>
          </div>
        </div>
        <div className="row" style={{ marginTop: 12, alignItems: "center" }}>
          <button
            onClick={() => handleUpdateSettings({ scheduler_enabled: !settings?.scheduler_enabled })}
            style={{ fontSize: "12px", padding: "4px 10px" }}
          >
            {settings?.scheduler_enabled ? "Disable Scheduler" : "Enable Scheduler"}
          </button>
          <label style={{ fontSize: "12px", display: "inline-flex", alignItems: "center", gap: 6 }}>
            Interval:
            <select
              value={settings?.interval_hours || 24}
              onChange={(e) => handleUpdateSettings({ interval_hours: Number(e.target.value) })}
              style={{ padding: "3px 6px", fontSize: "12px" }}
            >
              <option value={12}>Every 12 Hours</option>
              <option value={24}>Every 24 Hours (Daily)</option>
              <option value={48}>Every 48 Hours</option>
              <option value={168}>Every 7 Days (Weekly)</option>
            </select>
          </label>
          <button onClick={handleRollback} style={{ fontSize: "12px", padding: "4px 10px", marginLeft: "auto" }}>
            Rollback to Previous Version
          </button>
        </div>
      </div>

      <div className="muted" style={{ marginBottom: 12 }}>
        Manual crawl on college domain (auto-skips unchanged pages via content hash):
      </div>
      <div className="row">
        <input
          type="text"
          style={{ minWidth: 260 }}
          value={startUrl}
          onChange={(e) => setStartUrl(e.target.value)}
          placeholder="https://sxca.edu.in/"
        />
        <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
          {STRATEGIES.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
        </select>
      </div>
      <div className="row">
        <label className="muted">
          Max pages{" "}
          <input
            type="text"
            inputMode="numeric"
            style={{ width: 60 }}
            value={maxPages}
            onChange={(e) => setMaxPages(e.target.value.replace(/\D/g, ""))}
          />
        </label>
        <label className="muted">
          Max depth{" "}
          <input
            type="text"
            inputMode="numeric"
            style={{ width: 50 }}
            value={maxDepth}
            onChange={(e) => setMaxDepth(e.target.value.replace(/\D/g, ""))}
          />
        </label>
        <button className="primary" disabled={running} onClick={handleCrawl}>
          {running ? "Crawling..." : "Start Crawl Now"}
        </button>
      </div>

      {lastSummary && (
        <div className="muted" style={{ marginBottom: 8 }}>
          Last run: {lastSummary.fetched} fetched (Changed: {lastSummary.pages_changed || 0}, New: {lastSummary.pages_new || 0}, Unchanged: {lastSummary.pages_unchanged || 0}), {lastSummary.errors} errors
        </div>
      )}

      <table>
        <thead>
          <tr>
            <th>When</th>
            <th>Strategy</th>
            <th>Fetched</th>
            <th>Skipped</th>
            <th>Errors</th>
          </tr>
        </thead>
        <tbody>
          {runs.length === 0 ? (
            <tr>
              <td colSpan={5} className="muted">
                No crawl runs yet.
              </td>
            </tr>
          ) : (
            runs.map((r, i) => (
              <tr key={i}>
                <td>{(r.created_at || "").replace("T", " ").replace("+00:00", " UTC")}</td>
                <td>{r.strategy}</td>
                <td>{r.fetched}</td>
                <td>{r.skipped}</td>
                <td>{r.errors}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </section>
  );
}
