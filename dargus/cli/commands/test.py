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
        for i, mod in enumerate(modules, start=2):
            print(f"  {i}. {mod['name']} ({mod['n_tests']} tests)")
        print(f"  {len(modules) + 2}. Back")
        print()

        max_choice = len(modules) + 2
        choice = input(f"  Select option [1-{max_choice}]: ").strip()

        if choice == "1":
            api.run_tests()
        elif choice == str(max_choice) or choice == "":
            return 0
        else:
            # Check if it's a module number
            try:
                idx = int(choice) - 2
                if 0 <= idx < len(modules):
                    api.run_tests(modules[idx]["name"])
                else:
                    print("  Invalid option.")
            except ValueError:
                print("  Invalid option.")
