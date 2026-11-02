"""Forge Core interactive teacher — human-initiated prototype, not Android tracking.

Run: python -m forge_core.cli learn [--vault /path/to/vault]
"""
from __future__ import annotations

import argparse
import getpass
from pathlib import Path

from forge_core.learning.vault import (
    VaultError, add_principle, delete_principle, empty_memory, load_vault, save_vault,
)


def ask(prompt: str) -> str:
    return input(prompt).strip()


def learn(memory: dict) -> bool:
    print("\nForge: Tell me about a decision or action. I'll try to learn why.")
    observation = ask("What happened? ")
    if not observation:
        return False
    print("Forge: Interesting. What led you to make that decision instead of an alternative?")
    reason = ask("Your reasoning: ")
    if not reason:
        print("Forge: No problem; nothing was saved.")
        return False
    print("Forge: I won't assume one decision is a universal rule.")
    principle = ask("What general lesson, if any, should I take from this? ")
    if not principle:
        print("Forge: No general principle saved.")
        return False
    print("\nForge proposes to remember:")
    print(f"  Situation: {observation}\n  Reason: {reason}\n  Principle: {principle}")
    if ask("Confirm this memory? [yes/no]: ").casefold() not in ("yes", "y"):
        print("Forge: Understood; I won't store it.")
        return False
    identifier = add_principle(memory, observation, reason, principle, confirmed=True)
    print(f"Forge: Confirmed. Memory ID: {identifier}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge Core's first interactive learning loop")
    parser.add_argument("command", choices=("learn", "review", "forget", "demo"))
    parser.add_argument("--vault", type=Path, default=Path.home() / ".forge-core" / "memory.vault")
    parser.add_argument("--id", help="Memory ID to delete with 'forget'")
    args = parser.parse_args()

    if args.command == "demo":
        print("Forge Core v0.2: working learning loop. No phone monitoring or autonomous language intelligence yet.")
        print("Example: 'You revised a car body before detailing.'")
        print("Forge asks: 'Why did proportions matter more than detail in this case?'")
        print("User explanation -> proposed principle -> confirmation -> encrypted local memory.")
        return

    new = not args.vault.exists()
    password = getpass.getpass("Create vault passphrase: " if new else "Unlock vault passphrase: ")
    if new:
        verify = getpass.getpass("Repeat passphrase: ")
        if password != verify:
            parser.error("Passphrases do not match")
        memory = empty_memory()
    else:
        try:
            memory = load_vault(args.vault, password)
        except VaultError as exc:
            parser.error(str(exc))

    if args.command == "learn":
        changed = learn(memory)
    elif args.command == "review":
        for record in memory["principles"]:
            print(f"{record['id']}: {record['principle']}\n  Because: {record['why']}")
        print(f"{len(memory['principles'])} confirmed principle(s).")
        changed = False
    else:
        if not args.id:
            parser.error("forget requires --id from the review command")
        changed = delete_principle(memory, args.id)
        print("Memory deleted." if changed else "Memory not found.")

    if changed or new:
        save_vault(args.vault, password, memory)
        print(f"Encrypted vault saved locally at {args.vault}")


if __name__ == "__main__":
    main()
