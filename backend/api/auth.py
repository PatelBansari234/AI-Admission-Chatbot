"""Admin authentication: bcrypt password hashes + signed, expiring tokens."""

import base64
import hmac
import json
import secrets
import time
from hashlib import sha256

import bcrypt
from fastapi import Header, HTTPException

from backend.config import DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_USER, SECRET_KEY, TOKEN_TTL_SECONDS
from backend.db import repo


# --- password store (Postgres: users table) --------------------------------
def ensure_default_admin() -> None:
    """Seed the first admin account on first boot."""
    if repo.count_users() > 0:
        return
    password_hash = bcrypt.hashpw(DEFAULT_ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode()
    repo.create_user(DEFAULT_ADMIN_USER, password_hash, role="admin")


def verify_password(username: str, password: str) -> bool:
    user = repo.get_user(username)
    if not user:
        # Constant-ish work even for unknown users, so timing doesn't leak names.
        bcrypt.checkpw(b"x", bcrypt.hashpw(b"y", bcrypt.gensalt()))
        return False
    return bcrypt.checkpw(password.encode(), user["password_hash"].encode())


def change_password(username: str, new_password: str) -> None:
    password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    if not repo.update_password(username, password_hash):
        raise HTTPException(status_code=404, detail="User not found")


# --- tokens ----------------------------------------------------------------
def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: bytes) -> str:
    return _b64(hmac.new(SECRET_KEY.encode(), payload, sha256).digest())


def create_token(username: str, role: str = "admin") -> str:
    payload = json.dumps({
        "sub": username,
        "role": role,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
        "jti": secrets.token_hex(8),
    }).encode()
    return f"{_b64(payload)}.{_sign(payload)}"


def verify_token_payload(token: str) -> dict:
    """Return the decoded payload dict, or raise 401."""
    try:
        body, signature = token.split(".", 1)
        payload = _unb64(body)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    if not hmac.compare_digest(signature, _sign(payload)):
        raise HTTPException(status_code=401, detail="Invalid token")

    data = json.loads(payload)
    if data.get("exp", 0) < time.time():
        raise HTTPException(status_code=401, detail="Session expired, please log in again")
    return data


def verify_token(token: str) -> str:
    """Return the username, or raise 401."""
    return verify_token_payload(token)["sub"]


def require_admin(authorization: str = Header(default="")) -> str:
    """FastAPI dependency guarding every /api/admin route. Strictly requires role='admin'."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Admin login required")
    data = verify_token_payload(authorization[7:])
    if data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return data["sub"]


def resolve_owner_id(authorization: str = "", session_token: str = "") -> str:
    """Resolve conversation owner identity from JWT header or guest session token."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        try:
            payload = verify_token_payload(token)
            username = payload.get("sub")
            if username:
                return f"user:{username}"
        except HTTPException:
            pass
    if session_token and session_token.strip():
        st = session_token.strip()
        hashed = sha256(st.encode()).hexdigest()[:16]
        return f"session:{hashed}"
    return "anonymous"


