import { useEffect, useState } from "react";
import { authFetch } from "../../api.js";

function formatTime(iso) {
  return (iso || "").replace("T", " ").replace("+00:00", " UTC");
}

export default function IntentsTab({ notify }) {
  const [clusters, setClusters] = useState([]);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [loading, setLoading] = useState(false);

  function load(refresh = false) {
    setLoading(true);
    authFetch(`/api/admin/intents${refresh ? "?refresh=true" : ""}`)
      .then((data) => {
        setClusters(data.clusters);
        setTotalQuestions(data.total_questions);
      })
      .catch((err) => notify(err.message, false))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <section>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="muted">
          {clusters.length} intent{clusters.length === 1 ? "" : "s"} across {totalQuestions}{" "}
          logged questions - grouped by meaning, not exact wording, most-asked first.
        </div>
        <button className="subtle" disabled={loading} onClick={() => load(true)}>
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {clusters.length === 0 ? (
        <p className="muted">No questions logged yet.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Intent</th>
              <th>Asked</th>
              <th>Variants</th>
              <th>Answered from docs</th>
              <th>Last asked</th>
            </tr>
          </thead>
          <tbody>
            {clusters.map((c, i) => (
              <tr key={i}>
                <td>
                  {c.representative}
                  {c.distinct_phrasings > 1 && (
                    <details style={{ marginTop: 4 }}>
                      <summary className="muted">
                        {c.distinct_phrasings} phrasings
                      </summary>
                      <ul>
                        {c.samples.map((s, j) => (
                          <li key={j} className="muted">
                            {s.question} {s.count > 1 && <span>&times;{s.count}</span>}
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </td>
                <td>{c.count}</td>
                <td>{c.distinct_phrasings}</td>
                <td>
                  {Math.round(c.grounded_rate * 100)}%
                  {c.grounded_rate < 0.5 && (
                    <span style={{ color: "#b4531a" }}> (mostly unanswered)</span>
                  )}
                </td>
                <td>{formatTime(c.last_asked)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
