"""Setup command — the interactive ``dargus setup`` wizard (T4).

Interactive-only (no ``--home`` flag). Steps, in order:
① confirm the Dargus home (default ``$DARGUS_HOME`` or ``~/.dargus/``,
   editable) → ② write the clean default config → ③ API-key wizard into
   ``{home}/.env`` (skippable) → ④ create the D-Base directory structure and
   migrate legacy per-workspace session archives.

All file operations run through the API seam (``dargus.api``); this module
only renders the interactive prompt around it.
"""

from __future__ import annotations

from pathlib import Path


def run_setup_wizard() -> int:
    """Launch the interactive setup wizard.

    Returns:
        Exit code (0 for success).
    """
    from dargus import api

    print()
    print("  Dargus setup")
    print("  ────────────────────────────────")
    print()

    # Step ① — confirm the Dargus home location.
    default_home = api.dargus_home()
    print("  Dargus keeps its config, secrets, D-Base, and session archive in")
    print("  one per-user home directory.")
    answer = input(f"  Dargus home [{default_home}]: ").strip()
    home = Path(answer).expanduser() if answer else None

    # Step ③ — API-key wizard (skippable).
    print()
    choice = input("  Set your LLM API key now? [y/N]: ").strip().lower()
    api_key: str | None = None
    if choice in {"y", "yes"}:
        key = input("  API key: ").strip()
        if key:
            api_key = key
        else:
            print("  No key entered — skipped.")
    else:
        print("  Skipped. Set a key later with: dargus config")

    # Steps ② and ④ + the key — one API call drives config, secrets, D-Base
    # structure, and legacy migration (the wizard never touches files itself).
    result = api.setup(home=home, api_key=api_key)

    # Summary.
    print()
    print("  Dargus is set up:")
    print(f"    home:      {result['home']}")
    print(f"    config:    {result['config']}")
    print(f"    secrets:   {result['env'] or '(none yet — set with: dargus config)'}")
    print(f"    d-base:    {result['dbase']}")
    print(f"    migrated:  {result['migrated']} legacy session(s)")
    print()
    print("  You're ready. Run `dargus` to start the REPL.")
    return 0
