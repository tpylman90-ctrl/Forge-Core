"""Password-protected local memory vault prototype (not audited/Android Keystore).

Always encrypt the complete JSON document before writing. Memory never belongs
in the public repo, CI or model training corpora without independent opt-in.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import tempfile
import uuid

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"FORGEVAULT1\n"
SALT_BYTES = 16
NONCE_BYTES = 12
MAX_FILE_BYTES = 16 * 1024 * 1024


class VaultError(Exception):
    """Invalid password, corrupted vault or unsupported version."""


def _key(password: str, salt: bytes) -> bytes:
    if not password:
        raise ValueError("A nonempty vault passphrase is required")
    return Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(password.encode("utf-8"))


def empty_memory() -> dict:
    return {"schema": 1, "principles": []}


def load_vault(path: Path, password: str) -> dict:
    raw = path.read_bytes()
    if len(raw) < len(MAGIC) + SALT_BYTES + NONCE_BYTES + 16 or len(raw) > MAX_FILE_BYTES:
        raise VaultError("Invalid vault length")
    if not raw.startswith(MAGIC):
        raise VaultError("Unsupported vault format")
    start = len(MAGIC)
    salt = raw[start : start + SALT_BYTES]
    nonce = raw[start + SALT_BYTES : start + SALT_BYTES + NONCE_BYTES]
    ciphertext = raw[start + SALT_BYTES + NONCE_BYTES :]
    try:
        plaintext = AESGCM(_key(password, salt)).decrypt(nonce, ciphertext, MAGIC + salt)
        memory = json.loads(plaintext.decode("utf-8"))
    except (InvalidTag, UnicodeDecodeError, ValueError, TypeError) as exc:
        raise VaultError("Wrong passphrase or damaged vault") from exc
    if not isinstance(memory, dict) or memory.get("schema") != 1 or not isinstance(memory.get("principles"), list):
        raise VaultError("Unsupported memory schema")
    return memory


def save_vault(path: Path, password: str, memory: dict) -> None:
    """Atomically replace an encrypted vault, restricted to its local OS user.

    The caller must unlock an existing vault *before* overwriting it. This
    low-level API intentionally does not provide synchronization.
    """
    if memory.get("schema") != 1 or not isinstance(memory.get("principles"), list):
        raise ValueError("Invalid memory schema")
    plaintext = json.dumps(memory, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    salt, nonce = os.urandom(SALT_BYTES), os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(_key(password, salt)).encrypt(nonce, plaintext, MAGIC + salt)
    payload = MAGIC + salt + nonce + ciphertext
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError("Vault size limit exceeded")
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".forge-", dir=path.parent, delete=False) as file:
            temp_name = file.name
            os.chmod(temp_name, 0o600)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def add_principle(memory: dict, observation: str, why: str, principle: str, confirmed: bool) -> str:
    """Store only an explicitly confirmed explanation, never a model guess."""
    if not confirmed:
        raise PermissionError("Only user-confirmed principles may be stored")
    if not all(value.strip() for value in (observation, why, principle)):
        raise ValueError("Observation, reason and principle are required")
    identifier = str(uuid.uuid4())
    memory["principles"].append({
        "id": identifier, "observation": observation.strip(),
        "why": why.strip(), "principle": principle.strip(),
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    })
    return identifier


def delete_principle(memory: dict, identifier: str) -> bool:
    original = len(memory["principles"])
    memory["principles"] = [p for p in memory["principles"] if p["id"] != identifier]
    return len(memory["principles"]) < original
