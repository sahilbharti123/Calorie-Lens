import sqlite3
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import account_store
import api
from account_store import (
    AIBudgetExceededError,
    AccountStore,
    InvalidCredentialsError,
    SyncConflictError,
)
from fastapi.testclient import TestClient


class AccountStoreTests(unittest.TestCase):
    def setUp(self):
        self.iterations = account_store.PASSWORD_ITERATIONS
        account_store.PASSWORD_ITERATIONS = 2_000
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = AccountStore(
            db_path=root / "accounts.db",
            key_path=root / "vault.key",
        )

    def tearDown(self):
        account_store.PASSWORD_ITERATIONS = self.iterations
        self.temp.cleanup()

    def test_account_lifecycle_and_encrypted_vault(self):
        user, recovery_code = self.store.create_user(
            "Sahil@Example.com",
            "correct-horse-battery",
            "Sahil",
        )
        self.assertEqual(user["email"], "sahil@example.com")
        self.assertEqual(
            self.store.authenticate("sahil@example.com", "correct-horse-battery")["id"],
            user["id"],
        )

        token, _ = self.store.create_session(user["id"])
        self.assertEqual(self.store.user_for_token(token)["id"], user["id"])

        payload = {
            "coachMemory": {"notes": "private training schedule"},
            "profile": {
                "primaryGoal": "build-muscle",
                "dietStyle": "vegetarian",
                "injuries": ["sensitive left knee"],
            },
            "days": {"2026-07-29": {"waterMl": 1750}},
        }
        saved = self.store.save_vault(user["id"], payload, 0)
        self.assertEqual(saved["version"], 1)
        self.assertEqual(self.store.read_vault(user["id"])["payload"], payload)

        with sqlite3.connect(self.store.db_path) as connection:
            encrypted = connection.execute(
                "SELECT encrypted_payload FROM vaults WHERE user_id = ?",
                (user["id"],),
            ).fetchone()[0]
        self.assertNotIn(b"private training schedule", encrypted)

        with self.assertRaises(SyncConflictError):
            self.store.save_vault(user["id"], {"days": {}}, 0)

        _, next_recovery_code = self.store.recover(
            "sahil@example.com",
            recovery_code,
            "new-correct-password",
        )
        self.assertIsNone(self.store.user_for_token(token))
        with self.assertRaises(InvalidCredentialsError):
            self.store.authenticate("sahil@example.com", "correct-horse-battery")
        self.store.authenticate("sahil@example.com", "new-correct-password")
        with self.assertRaises(InvalidCredentialsError):
            self.store.recover(
                "sahil@example.com",
                recovery_code,
                "another-password",
            )
        self.assertTrue(next_recovery_code)

        self.store.delete_user(user["id"])
        with self.assertRaises(KeyError):
            self.store.read_vault(user["id"])

    def test_ai_cache_is_encrypted_and_daily_limits_are_enforced(self):
        user, _ = self.store.create_user(
            "budget@example.com",
            "correct-horse-battery",
            "Budget",
        )
        cache_key = "known-request"
        response = {"reply": "private coaching answer"}
        self.store.write_ai_cache(
            user["id"],
            cache_key,
            response,
            timedelta(days=1),
        )
        self.assertEqual(
            self.store.read_ai_cache(user["id"], cache_key),
            response,
        )

        with sqlite3.connect(self.store.db_path) as connection:
            encrypted = connection.execute(
                "SELECT encrypted_response FROM ai_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()[0]
        self.assertNotIn(b"private coaching answer", encrypted)

        limits = {"text": 1, "audio": 1, "coach": 1}
        usage = self.store.consume_ai_request(user["id"], "text", 2, limits)
        self.assertEqual(usage["used"], 1)
        with self.assertRaises(AIBudgetExceededError):
            self.store.consume_ai_request(user["id"], "text", 2, limits)

        self.store.delete_user(user["id"])
        self.assertIsNone(self.store.read_ai_cache(user["id"], cache_key))


class AccountApiTests(unittest.TestCase):
    def setUp(self):
        self.iterations = account_store.PASSWORD_ITERATIONS
        account_store.PASSWORD_ITERATIONS = 2_000
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.original_store = api.STORE
        api.STORE = AccountStore(
            db_path=root / "api.db",
            key_path=root / "api.key",
        )
        self.client = TestClient(api.app)

    def tearDown(self):
        api.STORE = self.original_store
        account_store.PASSWORD_ITERATIONS = self.iterations
        self.temp.cleanup()

    def test_signup_login_sync_backup_logout(self):
        response = self.client.post(
            "/v1/auth/signup",
            json={
                "email": "test@example.com",
                "password": "safe-password-123",
                "display_name": "Test User",
            },
        )
        self.assertEqual(response.status_code, 201)
        auth = response.json()
        self.assertTrue(auth["recoveryCode"])
        headers = {"Authorization": f"Bearer {auth['token']}"}

        me = self.client.get("/v1/auth/me", headers=headers)
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["user"]["displayName"], "Test User")

        usage = self.client.get("/v1/ai/usage", headers=headers)
        self.assertEqual(usage.status_code, 200)
        self.assertEqual(usage.json()["used"], 0)
        self.assertEqual(
            self.client.get("/v1/ai/usage").status_code,
            401,
        )

        first = self.client.put(
            "/v1/sync",
            headers=headers,
            json={
                "base_version": 0,
                "payload": {
                    "goals": {"calories": 2100},
                    "profile": {
                        "primaryGoal": "lose-fat",
                        "dietStyle": "home-indian",
                    },
                    "days": {},
                },
            },
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["version"], 1)
        self.assertEqual(
            api._vault_profile({"id": auth["user"]["id"]})["primaryGoal"],
            "lose-fat",
        )

        conflict = self.client.put(
            "/v1/sync",
            headers=headers,
            json={"base_version": 0, "payload": {"days": {}}},
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["detail"]["version"], 1)

        backup = self.client.get("/v1/backup", headers=headers)
        self.assertEqual(backup.status_code, 200)
        self.assertEqual(backup.json()["format"], "calorie-lens-backup-v1")

        logout = self.client.post("/v1/auth/logout", headers=headers)
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(self.client.get("/v1/auth/me", headers=headers).status_code, 401)


if __name__ == "__main__":
    unittest.main()
