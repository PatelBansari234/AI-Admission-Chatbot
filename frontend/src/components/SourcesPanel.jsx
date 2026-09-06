export default function SourcesPanel({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources">
      <details>
        <summary>Sources ({sources.length})</summary>
        <ul>
          {sources.map((s, i) => {
            const where = s.doc_type === "pdf" ? `page ${s.page}` : s.section || "website";
            return (
              <li key={i}>
                {s.url ? (
                  <a
                    href={s.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ fontWeight: 600, color: "#2563eb", textDecoration: "underline" }}
                  >
                    {s.source}
                  </a>
                ) : (
                  <b>{s.source}</b>
                )}{" "}
                &mdash; {where}
                <br />
                <span>{s.excerpt}</span>
              </li>
            );
          })}
        </ul>
      </details>
    </div>
  );
}
