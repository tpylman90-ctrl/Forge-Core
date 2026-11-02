import tempfile
import unittest
from pathlib import Path

from forge_core.learning.vault import (
    VaultError, add_principle, delete_principle, empty_memory, load_vault, save_vault,
)


class TestMemory(unittest.TestCase):
    def test_roundtrip_encrypt_and_forget(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vault"
            memory = empty_memory()
            identifier = add_principle(memory, "Opened editor", "Needed proportions", "Check silhouette first", True)
            save_vault(path, "strong-password-example", memory)
            encrypted = path.read_bytes()
            self.assertNotIn(b"Check silhouette first", encrypted)
            self.assertNotIn(b"Opened editor", encrypted)
            recovered = load_vault(path, "strong-password-example")
            self.assertEqual(recovered["principles"][0]["id"], identifier)
            self.assertTrue(delete_principle(recovered, identifier))
            save_vault(path, "strong-password-example", recovered)
            self.assertEqual(load_vault(path, "strong-password-example")["principles"], [])

    def test_wrong_password_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vault"
            save_vault(path, "test-password", empty_memory())
            with self.assertRaises(VaultError):
                load_vault(path, "incorrect-password")
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 0x7F
            path.write_bytes(payload)
            with self.assertRaises(VaultError):
                load_vault(path, "test-password")

    def test_rejects_unconfirmed_memories(self):
        with self.assertRaises(PermissionError):
            add_principle(empty_memory(), "Action", "Reason", "Lesson", False)
        with self.assertRaises(ValueError):
            add_principle(empty_memory(), "Action", "Reason", "  ", True)


if __name__ == "__main__":
    unittest.main()
