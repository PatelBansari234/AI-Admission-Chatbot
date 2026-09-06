import { useEffect, useRef, useState } from "react";
import { authFetch } from "../../api.js";

export default function DocumentsTab({ notify }) {
  const [documents, setDocuments] = useState([]);
  const [chunkCount, setChunkCount] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const fileInput = useRef(null);

  async function loadDocs() {
    try {
      const data = await authFetch("/api/admin/documents");
      setChunkCount(data.index.chunks);
      setDocuments(data.documents);
    } catch (err) {
      notify(err.message, false);
    }
  }

  useEffect(() => {
    loadDocs();
  }, []);

  async function handleUpload() {
    const file = fileInput.current?.files?.[0];
    if (!file) {
      notify("Choose a file first", false);
      return;
    }
    const body = new FormData();
    body.append("file", file);
    setUploading(true);
    try {
      const data = await authFetch("/api/admin/upload", { method: "POST", body });
      notify(`${data.message} (${data.chunks} chunks)`);
      fileInput.current.value = "";
      loadDocs();
    } catch (err) {
      notify(err.message, false);
    } finally {
      setUploading(false);
    }
  }

  async function handleReindex() {
    if (!confirm("Rebuild the index from all stored files?")) return;
    setReindexing(true);
    try {
      const body = new FormData();
      body.append("reset", "true");
      const data = await authFetch("/api/admin/reindex", { method: "POST", body });
      notify(`${data.message}: ${data.index.chunks} chunks`);
      loadDocs();
    } catch (err) {
      notify(err.message, false);
    } finally {
      setReindexing(false);
    }
  }

  async function handleDelete(source) {
    if (!confirm(`Remove ${source} from the chatbot's dataset?`)) return;
    try {
      const data = await authFetch(`/api/admin/documents/${encodeURIComponent(source)}`, {
        method: "DELETE",
      });
      notify(data.message);
      loadDocs();
    } catch (err) {
      notify(err.message, false);
    }
  }

  return (
    <section>
      <div className="row">
        <input type="file" ref={fileInput} accept=".pdf,.txt,.html,.htm" />
        <button className="primary" disabled={uploading} onClick={handleUpload}>
          {uploading ? "Indexing..." : "Upload & index"}
        </button>
        <button className="subtle" disabled={reindexing} onClick={handleReindex}>
          Rebuild index
        </button>
      </div>
      <div className="muted">
        Uploading a file with an existing name replaces it; the old copy is kept under{" "}
        <code>_versions</code>.
      </div>
      {chunkCount !== null && (
        <div className="muted" style={{ marginTop: 8 }}>
          Index: {chunkCount} searchable chunks
        </div>
      )}
      <table>
        <thead>
          <tr>
            <th>File</th>
            <th>Type</th>
            <th>Chunks</th>
            <th>Last updated</th>
            <th>By</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {documents.length === 0 ? (
            <tr>
              <td colSpan={6} className="muted">
                No documents indexed yet.
              </td>
            </tr>
          ) : (
            documents.map((d) => (
              <tr key={d.source}>
                <td>{d.source}</td>
                <td>{d.doc_type}</td>
                <td>{d.chunks ?? "-"}</td>
                <td>{(d.updated_at || "").replace("T", " ").replace("+00:00", " UTC")}</td>
                <td>{d.uploaded_by || "-"}</td>
                <td>
                  <button className="danger" onClick={() => handleDelete(d.source)}>
                    Delete
                  </button>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </section>
  );
}
