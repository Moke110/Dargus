"""Uninstall command — ``dargus uninstall`` (T6).

Removes the Dargus program (delegating to the uv tool uninstall path) while
preserving the Dargus home data (config, secrets, D-Base, sessions) and
printing where that data remains. It never deletes user data.
"""

from __future__ import annotations


def run_uninstall() -> int:
    """Uninstall Dargus, preserving home data.

    Returns:
        Exit code (0 for success, 1 when the program removal failed).
    """
    from dargus import api

    print()
    print("  Dargus uninstall")
    print("  ────────────────────────────────")
    print()
    print("  Removing the Dargus program. Your data is preserved.")

    result = api.uninstall()

    data = result.get("data") or {}
    print()
    print("  Program removal:")
    if result.get("uninstalled"):
        print(f"    OK — ran: {result['command']}")
    else:
        error = result.get("error") or "unknown"
        print(f"    {error}")
        print("    The program was left in place.")

    print()
    print("  Your data remains (not deleted):")
    for label, path in data.items():
        print(f"    {label:<9} {path}")
    print()
    print(
        "  To back it up or delete it, remove the Dargus home directory "
        "yourself. Reinstall with: curl -LsSf "
        "https://github.com/Moke110/Dargus/releases/latest/download/install.sh | sh"
    )

    return 0 if result.get("uninstalled") else 1
