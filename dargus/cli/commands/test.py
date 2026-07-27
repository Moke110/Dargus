"""Test command — Dargus test menu."""

from __future__ import annotations


def run_test_menu() -> int:
    """Launch the Dargus test menu.

    Returns:
        Exit code (0 for success).
    """
    from dargus import api

    while True:
        print()
        print("  Dargus Test Suite")
        print("  ────────────────────────────────")
        print()

        # Get test modules
        modules = api.list_test_modules()
        total_tests = sum(m["n_tests"] for m in modules)

        print(f"  1. Run All Tests ({total_tests} tests)")
        print("  2. DB Input — write single evidence to test D-Base")
        print("  3. Bulk Input — bulk write evidence instances")
        print("  4. Ingest Test — run ingest workflow on test data")
        for i, mod in enumerate(modules, start=5):
            print(f"  {i}. {mod['name']} ({mod['n_tests']} tests)")
        print(f"  {len(modules) + 5}. Back")
        print()

        max_choice = len(modules) + 5
        choice = input(f"  Select option [1-{max_choice}]: ").strip()

        if choice == "1":
            api.run_tests()
        elif choice == "2":
            _run_test_dbase()
        elif choice == "3":
            _run_test_bulk_input()
        elif choice == "4":
            _run_test_ingest()
        elif choice == str(max_choice) or choice == "":
            return 0
        else:
            # Check if it's a module number
            try:
                idx = int(choice) - 5
                if 0 <= idx < len(modules):
                    api.run_tests(modules[idx]["name"])
                else:
                    print("  Invalid option.")
            except ValueError:
                print("  Invalid option.")


def _run_test_dbase() -> None:
    """Write a single evidence record to test D-Base."""
    from dargus import api

    print()
    print("  DB Input — Write Single Evidence")
    print("  ────────────────────────────────")
    print()

    raw_input = input("  Paste evidence JSON or path to file (.yaml/.json): ").strip()
    if not raw_input:
        print("  Cancelled.")
        return

    # Parse input
    import json
    from pathlib import Path

    if raw_input.startswith("{") or raw_input.startswith("["):
        raw_data = json.loads(raw_input)
    else:
        input_path = Path(raw_input).expanduser()
        if not input_path.exists():
            print(f"  File not found: {input_path}")
            return
        content = input_path.read_text(encoding="utf-8")
        if input_path.suffix in (".yaml", ".yml"):
            import yaml

            raw_data = yaml.safe_load(content) or {}
        else:
            raw_data = json.loads(content)

    # Write evidence
    result = api.test_write_evidence(raw_data)

    print()
    print("  Instance written.")
    print("  ────────────────────────────────")
    print(f"  Evidence ID:     {result['evidence_id']}")
    print(f"  Biological level: {result.get('biological_level', '?')}")
    print(f"  Readout type:     {result.get('readout_type', '?')}")
    print(f"  Status:           {result['status']}")


def _run_test_bulk_input() -> None:
    """Bulk-write evidence .json files to test D-Base."""
    from dargus import api

    print()
    print("  Bulk Input — Bulk Write Evidence")
    print("  ────────────────────────────────")
    print()

    # Get saved directory
    saved_dir = api.get_test_config("bulk_input_dir", "~/dargus-dev/tests/db-input-instances")
    print(f"  Bulk Input directory: {saved_dir}")

    choice = input("  Change directory? Enter new path or ENTER to keep: ").strip()
    if choice:
        api.set_test_config("bulk_input_dir", choice)
        saved_dir = choice

    # Run bulk input
    result = api.test_bulk_input(saved_dir)

    print()
    print("  Bulk input complete.")
    print("  ────────────────────────────────")
    print(f"  Directory:        {result['directory']}")
    print(f"  Files processed:  {result['total']}")
    print(f"  Added:            {result['added']}")
    print(f"  Duplicates:       {result['duplicates']}")
    print(f"  Hard rejects:     {result['hard_rejects']}")
    print(f"  Errors:           {result['errors']}")
    print(f"  Time:             {result['elapsed']:.1f}s")


def _run_test_ingest() -> None:
    """Run ingest workflow on test data directory."""
    from dargus import api

    print()
    print("  Ingest Test — Run Ingest Workflow")
    print("  ────────────────────────────────")
    print()

    # Get saved directory
    saved_dir = api.get_test_config("ingest_dir", "~/dargus-dev/test/ingest/slices")
    print(f"  Ingest data directory: {saved_dir}")

    choice = input("  Change directory? Enter new path or ENTER to keep: ").strip()
    if choice:
        api.set_test_config("ingest_dir", choice)
        saved_dir = choice

    # Run ingest test
    result = api.test_ingest_dir(saved_dir)

    print()
    print("  Ingest complete.")
    print("  ────────────────────────────────")
    print(f"  Directory:        {result['directory']}")
    print(f"  Files processed:  {result['total']}")
    print(f"  Added:            {result['added']}")
    print(f"  Duplicates:       {result['duplicates']}")
    print(f"  Hard rejects:     {result['hard_rejects']}")
    print(f"  Errors:           {result['errors']}")
    print(f"  Time:             {result['elapsed']:.1f}s")

    # Ask about report
    report_choice = input("\n  Generate Ingest-test-report.md? [y/N]: ").strip().lower()
    if report_choice in ("y", "yes"):
        report_path = api.write_ingest_report(result)
        print(f"\n  Report written to {report_path}")
