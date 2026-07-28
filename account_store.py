"""Encrypted account, session, and fitness-vault persistence."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


PASSWORD_ITERATIONS = 600_000
SESSION_DAYS = 30


class AccountExistsError(ValueError):
    pass


class InvalidCredentialsError(ValueError):
    pass


class AIBudgetExceededError(ValueError):
    def __init__(self, usage: dict[str, Any]):
        super().__init__("Today's AI allowance is used. Local tracking still works.")
        self.usage = usage


class SyncConflictError(ValueError):
    def __init__(self, payload: dict[str, Any], version: int, updated_at: str):
        super().__init__("The cloud copy changed on another device.")
        self.payload = payload
        self.version = version
        self.updated_at = updated_at


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _new_recovery_code() -> str:
    return "-".join(secrets.token_hex(3) for _ in range(4))


def _decode_key(raw: str) -> bytes:
    try:
        decoded = base64.urlsafe_b64decode(raw.encode("ascii"))
    except Exception as exc:
        raise ValueError("CALORIE_LENS_MASTER_KEY must be URL-safe base64.") from exc
    if len(decoded) != 32:
        raise ValueError("CALORIE_LENS_MASTER_KEY must decode to exactly 32 bytes.")
    return decoded


class AccountStore:
    def __init__(
        self,
        db_path: str | Path | None = None,
        key_path: str | Path | None = None,
        master_key: bytes | None = None,
    ):
        self.db_path = Path(
            db_path or os.getenv("CALORIE_LENS_DB_PATH", "data/calorie_lens.db")
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path = Path(
            key_path
            or os.getenv(
                "CALORIE_LENS_KEY_PATH",
                str(self.db_path.parent / "calorie_lens.master.key"),
            )
        )
        self.master_key = master_key or self._load_master_key()
        self._initialize()

    def _load_master_key(self) -> bytes:
        configured = os.getenv("CALORIE_LENS_MASTER_KEY")
        if configured:
            return _decode_key(configured)
        if self.key_path.exists():
            return _decode_key(self.key_path.read_text(encoding="utf-8").strip())
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        key = AESGCM.generate_key(bit_length=256)
        self.key_path.write_text(
            base64.urlsafe_b64encode(key).decode("ascii"),
            encoding="utf-8",
        )
        try:
            self.key_path.chmod(0o600)
        except OSError:
            pass
        return key

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    recovery_salt BLOB NOT NULL,
                    recovery_hash BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_user_id
                    ON sessions(user_id);
                CREATE TABLE IF NOT EXISTS vaults (
                    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    encrypted_payload BLOB NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ai_usage (
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    usage_date TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, usage_date, kind)
                );
                CREATE TABLE IF NOT EXISTS ai_cache (
                    cache_key TEXT PRIMARY KEY,
                    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                    encrypted_response BLOB NOT NULL,
                    expires_at TEXT NOT NULL
                );
                """
            )
            cache_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(ai_cache)").fetchall()
            }
            if "user_id" not in cache_columns:
                connection.execute("ALTER TABLE ai_cache ADD COLUMN user_id TEXT")
                connection.execute("DELETE FROM ai_cache WHERE user_id IS NULL")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ai_cache_user_id ON ai_cache(user_id)"
            )

    def _encrypt(self, user_id: str, payload: dict[str, Any]) -> bytes:
        nonce = secrets.token_bytes(12)
        plaintext = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        ciphertext = AESGCM(self.master_key).encrypt(
            nonce, plaintext, user_id.encode("utf-8")
        )
        return nonce + ciphertext

    def _decrypt(self, user_id: str, encrypted: bytes) -> dict[str, Any]:
        plaintext = AESGCM(self.master_key).decrypt(
            encrypted[:12],
            encrypted[12:],
            user_id.encode("utf-8"),
        )
        return json.loads(plaintext.decode("utf-8"))

    @staticmethod
    def _public_user(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "email": row["email"],
            "displayName": row["display_name"],
            "createdAt": row["created_at"],
        }

    def create_user(
        self, email: str, password: str, display_name: str
    ) -> tuple[dict[str, Any], str]:
        normalized = _normalize_email(email)
        password_salt = secrets.token_bytes(16)
        recovery_salt = secrets.token_bytes(16)
        recovery_code = _new_recovery_code()
        now = _iso()
        user_id = str(uuid.uuid4())
        initial_vault: dict[str, Any] = {}
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users (
                        id, email, display_name, password_salt, password_hash,
                        recovery_salt, recovery_hash, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        normalized,
                        display_name.strip(),
                        password_salt,
                        _password_hash(password, password_salt),
                        recovery_salt,
                        _password_hash(recovery_code, recovery_salt),
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO vaults (
                        user_id, encrypted_payload, version, updated_at
                    ) VALUES (?, ?, 0, ?)
                    """,
                    (user_id, self._encrypt(user_id, initial_vault), now),
                )
                row = connection.execute(
                    "SELECT * FROM users WHERE id = ?", (user_id,)
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise AccountExistsError("An account already exists for this email.") from exc
        assert row is not None
        return self._public_user(row), recovery_code

    def authenticate(self, email: str, password: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email = ?", (_normalize_email(email),)
            ).fetchone()
        if row is None or not hmac.compare_digest(
            _password_hash(password, row["password_salt"]), row["password_hash"]
        ):
            raise InvalidCredentialsError("Email or password is incorrect.")
        return self._public_user(row)

    def create_session(self, user_id: str) -> tuple[str, str]:
        token = secrets.token_urlsafe(32)
        created_at = _utc_now()
        expires_at = created_at + timedelta(days=SESSION_DAYS)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (token_hash, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (_token_hash(token), user_id, _iso(created_at), _iso(expires_at)),
            )
        return token, _iso(expires_at)

    def user_for_token(self, token: str) -> dict[str, Any] | None:
        now = _iso()
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            row = connection.execute(
                """
                SELECT users.*
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ? AND sessions.expires_at > ?
                """,
                (_token_hash(token), now),
            ).fetchone()
        return self._public_user(row) if row else None

    def logout(self, token: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),)
            )

    def recover(
        self, email: str, recovery_code: str, new_password: str
    ) -> tuple[str, str]:
        normalized = _normalize_email(email)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email = ?", (normalized,)
            ).fetchone()
            if row is None or not hmac.compare_digest(
                _password_hash(recovery_code, row["recovery_salt"]),
                row["recovery_hash"],
            ):
                raise InvalidCredentialsError("Recovery details are incorrect.")
            password_salt = secrets.token_bytes(16)
            next_recovery_code = _new_recovery_code()
            recovery_salt = secrets.token_bytes(16)
            connection.execute(
                """
                UPDATE users
                SET password_salt = ?, password_hash = ?,
                    recovery_salt = ?, recovery_hash = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    password_salt,
                    _password_hash(new_password, password_salt),
                    recovery_salt,
                    _password_hash(next_recovery_code, recovery_salt),
                    _iso(),
                    row["id"],
                ),
            )
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
        return row["id"], next_recovery_code

    def change_password(
        self, user_id: str, current_password: str, new_password: str
    ) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if row is None or not hmac.compare_digest(
                _password_hash(current_password, row["password_salt"]),
                row["password_hash"],
            ):
                raise InvalidCredentialsError("Current password is incorrect.")
            salt = secrets.token_bytes(16)
            connection.execute(
                """
                UPDATE users
                SET password_salt = ?, password_hash = ?, updated_at = ?
                WHERE id = ?
                """,
                (salt, _password_hash(new_password, salt), _iso(), user_id),
            )
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def delete_user(self, user_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM ai_cache WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def read_vault(self, user_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM vaults WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            raise KeyError("Fitness vault was not found.")
        return {
            "payload": self._decrypt(user_id, row["encrypted_payload"]),
            "version": row["version"],
            "updatedAt": row["updated_at"],
        }

    def save_vault(
        self,
        user_id: str,
        payload: dict[str, Any],
        base_version: int | None = None,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM vaults WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Fitness vault was not found.")
            if base_version is not None and row["version"] != base_version:
                raise SyncConflictError(
                    self._decrypt(user_id, row["encrypted_payload"]),
                    row["version"],
                    row["updated_at"],
                )
            version = row["version"] + 1
            updated_at = _iso()
            connection.execute(
                """
                UPDATE vaults
                SET encrypted_payload = ?, version = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (
                    self._encrypt(user_id, payload),
                    version,
                    updated_at,
                    user_id,
                ),
            )
        return {
            "payload": payload,
            "version": version,
            "updatedAt": updated_at,
        }

    def read_ai_cache(
        self,
        user_id: str,
        cache_key: str,
    ) -> dict[str, Any] | None:
        now = _iso()
        with self._connect() as connection:
            connection.execute("DELETE FROM ai_cache WHERE expires_at <= ?", (now,))
            row = connection.execute(
                """
                SELECT encrypted_response
                FROM ai_cache
                WHERE cache_key = ? AND user_id = ?
                """,
                (cache_key, user_id),
            ).fetchone()
        if row is None:
            return None
        return self._decrypt(f"ai-cache:{cache_key}", row["encrypted_response"])

    def write_ai_cache(
        self,
        user_id: str,
        cache_key: str,
        response: dict[str, Any],
        ttl: timedelta,
    ) -> None:
        expires_at = _iso(_utc_now() + ttl)
        encrypted = self._encrypt(f"ai-cache:{cache_key}", response)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ai_cache (
                    cache_key, user_id, encrypted_response, expires_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    user_id = excluded.user_id,
                    encrypted_response = excluded.encrypted_response,
                    expires_at = excluded.expires_at
                """,
                (cache_key, user_id, encrypted, expires_at),
            )

    def ai_usage_summary(
        self,
        user_id: str,
        daily_limit: int,
        kind_limits: dict[str, int],
    ) -> dict[str, Any]:
        usage_date = _utc_now().date().isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT kind, requests
                FROM ai_usage
                WHERE user_id = ? AND usage_date = ?
                """,
                (user_id, usage_date),
            ).fetchall()
        by_kind = {row["kind"]: row["requests"] for row in rows}
        total = sum(by_kind.values())
        return {
            "date": usage_date,
            "used": total,
            "limit": daily_limit,
            "remaining": max(0, daily_limit - total),
            "kinds": {
                kind: {
                    "used": by_kind.get(kind, 0),
                    "limit": limit,
                    "remaining": max(0, limit - by_kind.get(kind, 0)),
                }
                for kind, limit in kind_limits.items()
            },
        }

    def consume_ai_request(
        self,
        user_id: str,
        kind: str,
        daily_limit: int,
        kind_limits: dict[str, int],
    ) -> dict[str, Any]:
        usage_date = _utc_now().date().isoformat()
        kind_limit = kind_limits[kind]
        with self._connect() as connection:
            # Serialize the read/check/increment so simultaneous requests cannot
            # both observe the same remaining allowance.
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT kind, requests
                FROM ai_usage
                WHERE user_id = ? AND usage_date = ?
                """,
                (user_id, usage_date),
            ).fetchall()
            by_kind = {row["kind"]: row["requests"] for row in rows}
            total = sum(by_kind.values())
            if total >= daily_limit or by_kind.get(kind, 0) >= kind_limit:
                usage = self.ai_usage_summary(user_id, daily_limit, kind_limits)
                raise AIBudgetExceededError(usage)
            connection.execute(
                """
                INSERT INTO ai_usage (user_id, usage_date, kind, requests)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(user_id, usage_date, kind) DO UPDATE SET
                    requests = requests + 1
                """,
                (user_id, usage_date, kind),
            )
        return self.ai_usage_summary(user_id, daily_limit, kind_limits)
