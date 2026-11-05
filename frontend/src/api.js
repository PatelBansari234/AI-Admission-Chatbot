const TOKEN_KEY = "adminToken";

// crypto.randomUUID() needs a secure context (https, or localhost) - fall
// back to a manual RFC4122 v4 generator so this still works if that's ever
// missing, since the backend validates conversation_id as a real UUID.
function generateUuid() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// One id per page load (module scope = evaluated once per page load), so the
// backend can thread conversation memory/query rewriting through a whole
// chat without the frontend persisting anything - a refresh naturally starts
// a new id, matching "keep history until the user refreshes."
export const conversationId = generateUuid();

// Base URL for API requests. When VITE_API_BASE_URL is provided (e.g. in Vercel production),
// requests route to that backend domain. Otherwise, fallback to empty string (relative paths)
// for local development and monolithic builds.
export const API_BASE_URL = (import.meta.env?.VITE_API_BASE_URL || "").replace(/\/+$/, "");

export function apiUrl(path) {
  if (!path) return API_BASE_URL || "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
}

export function getToken() {
  return sessionStorage.getItem(TOKEN_KEY) || "";
}

export function setToken(token) {
  if (token) sessionStorage.setItem(TOKEN_KEY, token);
  else sessionStorage.removeItem(TOKEN_KEY);
}

export function getUserRole() {
  const token = getToken();
  if (!token) return null;
  try {
    const parts = token.split(".");
    if (parts.length < 2) return null;
    const body = parts[0];
    const padded = body + "=".repeat((4 - (body.length % 4)) % 4);
    const jsonStr = atob(padded.replace(/-/g, "+").replace(/_/g, "/"));
    const payload = JSON.parse(jsonStr);
    return payload.role || null;
  } catch {
    return null;
  }
}

export function isAdminAuthenticated() {
  return getUserRole() === "admin";
}

export async function getHealth() {
  const res = await fetch(apiUrl("/api/health"));
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function login(username, password) {
  const res = await fetch(apiUrl("/api/login"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "Login failed");
  return data; // {token, username, role}
}

/** Fetch wrapper for /api/admin/* routes: attaches the bearer token and
 * normalizes error handling. Throws on 401 after clearing the stored token,
 * so the caller can fall back to the login screen. */
export async function authFetch(path, options = {}) {
  const res = await fetch(apiUrl(path), {
    ...options,
    headers: { ...(options.headers || {}), Authorization: `Bearer ${getToken()}` },
  });
  if (res.status === 401 || res.status === 403) {
    if (res.status === 401) {
      setToken("");
      window.dispatchEvent(new Event("admin-unauthorized"));
    }
    throw new Error(res.status === 403 ? "Admin access required" : "Session expired, please log in again");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "Request failed");
  return data;
}

function mergeSignals(sig1, sig2) {
  if (!sig1) return sig2;
  if (!sig2) return sig1;
  const controller = new AbortController();
  const onAbort = () => controller.abort(sig1.reason || sig2.reason);
  if (sig1.aborted) onAbort();
  else if (sig2.aborted) onAbort();
  else {
    sig1.addEventListener("abort", onAbort, { once: true });
    sig2.addEventListener("abort", onAbort, { once: true });
  }
  return controller.signal;
}

/** Streams /api/chat/stream as an async generator of {type, data} events:
 * "sources" (once), "token" (many), "done" (once), "error" (on failure). */
export async function* streamChat(question, history, signal) {
  const timeoutController = new AbortController();
  const timeoutId = setTimeout(() => {
    const err = new Error("The request timed out. Please check your connection and try again.");
    err.name = "TimeoutError";
    timeoutController.abort(err);
  }, 30000);

  const combinedSignal = mergeSignals(signal, timeoutController.signal);

  try {
    const res = await fetch(apiUrl("/api/chat/stream"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history, conversation_id: conversationId }),
      signal: combinedSignal,
    });
    if (!res.ok || !res.body) {
      throw new Error("Could not reach the server. Please make sure the backend is running.");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      clearTimeout(timeoutId);
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        yield JSON.parse(line.slice(5));
      }
    }
  } finally {
    clearTimeout(timeoutId);
  }
}

