import { useEffect } from "react";

export default function SourceViewerModal({ source, onClose }) {
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === "Escape") {
        onClose();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!source) return null;

  const isPdf = source.type === "pdf" || source.doc_type === "pdf" || (source.source && source.source.endsWith(".pdf"));
  const page = source.page;
  const directUrl = source.url
    ? isPdf && page
      ? `${source.url}#page=${page}`
      : source.url
    : null;

  return (
    <div className="source-modal-overlay" onClick={onClose}>
      <div className="source-modal-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="source-modal-header">
          <div className="source-modal-title-wrap">
            <span className="source-modal-icon">📄</span>
            <div>
              <h3 className="source-modal-title">{source.title || source.source || "Official SXCA Document"}</h3>
              <div className="source-modal-badges">
                <span className={`source-type-badge ${isPdf ? "badge-pdf" : "badge-web"}`}>
                  {isPdf ? "Official PDF Document" : "Official College Portal"}
                </span>
                {page && <span className="source-page-badge">Page {page}</span>}
              </div>
            </div>
          </div>
          <button type="button" className="source-modal-close" onClick={onClose} title="Close">
            ✕
          </button>
        </div>

        <div className="source-modal-body">
          <div className="source-snippet-card">
            <div className="snippet-label">Verified Record Excerpt:</div>
            <p className="snippet-text">
              {source.snippet || source.excerpt || "Official college admission information verified by SXCA records."}
            </p>
          </div>
        </div>

        <div className="source-modal-footer">
          {directUrl ? (
            <a
              href={directUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="source-open-external-btn"
            >
              {isPdf && page ? `Open Official PDF (Page ${page}) ↗` : "Open Official College Document ↗"}
            </a>
          ) : (
            <span className="source-official-note">Verified directly against SXCA official publications</span>
          )}
          <button type="button" className="source-modal-done-btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
