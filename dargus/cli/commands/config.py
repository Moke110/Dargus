"""Config command — Dargus configuration menu."""

from __future__ import annotations


def run_config_menu() -> int:
    """Launch the Dargus configuration menu.

    Returns:
        Exit code (0 for success).
    """

    while True:
        print()
        print("  Dargus Configuration")
        print("  ────────────────────────────────")
        print()
        print("  1. Show current LLM configuration")
        print("  2. Set API key")
        print("  3. Run LLM configuration wizard")
        print("  4. Clear D-Base")
        print("  5. Back")
        print()

        choice = input("  Select option [1-5]: ").strip()

        if choice == "1":
            _show_config()
        elif choice == "2":
            _set_api_key()
        elif choice == "3":
            _run_model_wizard()
        elif choice == "4":
            run_clear_dbase()
        elif choice == "5" or choice == "":
            return 0
        else:
            print("  Invalid option.")


def _show_config() -> None:
    """Show current LLM configuration."""
    from dargus import api

    cfg = api.get_llm_config()

    print()
    print("  LLM Configuration:")
    print(f"    provider:    {cfg.get('provider', 'openai_compatible')}")
    print(f"    model:       {cfg.get('model', 'not set')}")
    base_url = cfg.get("base_url", "")
    if base_url:
        print(f"    base_url:    {base_url}")
    print(f"    temperature: {cfg.get('temperature', 0.0)}")
    print(f"    max_tokens:  {cfg.get('max_tokens', 2048)}")

    if cfg.get("has_api_key"):
        print("    api_key:     [from .env]  ********")
    else:
        print("    api_key:     not set")
        print()
        print("    No API key found. Set one with:")
        print("      dargus config set-api-key <provider> <key>")


def _set_api_key() -> None:
    """Set API key interactively."""
    from dargus import api

    print()
    provider = input("  Provider (e.g. openai, anthropic, deepseek): ").strip()
    if not provider:
        print("  Cancelled.")
        return

    key = input("  API key: ").strip()
    if not key:
        print("  Cancelled.")
        return

    env_path = api.set_api_key(provider, key)
    print(f"  API key for '{provider}' saved to {env_path}")


def _run_model_wizard() -> None:
    """Run the interactive LLM configuration wizard."""
    from dargus import api

    print()
    print("  Configure LLM connection")
    print("  ────────────────────────────────")
    print()

    # Load current config
    cfg = api.get_llm_config()
    cur_base_url = cfg.get("base_url", "")
    cur_model = cfg.get("model", "")
    has_key = cfg.get("has_api_key", False)
    key_display = "********" if has_key else "(not set)"

    # Step 1: Base URL
    prompt = f"  Base URL [{cur_base_url}]: "
    new_base_url = input(prompt).strip()
    if not new_base_url:
        new_base_url = cur_base_url

    # Step 2: Model
    prompt = f"  Model [{cur_model}]: "
    new_model = input(prompt).strip()
    if not new_model:
        new_model = cur_model

    # Step 3: API Key
    prompt = f"  API Key [{key_display}]: "
    new_key = input(prompt).strip()

    # Step 4: Test connection
    print()
    print("  Testing connection...")
    print(f"  POST {new_base_url}/chat/completions")

    result = api.test_llm_connection(new_model, new_base_url, new_key or None)
    if result["ok"]:
        print(f"  Model: {result['model']} │ Connected OK ({result['latency_ms']}ms)")
    else:
        print(f"  Error: Connection failed — {result['error']}")
        _print_troubleshooting(result, new_base_url)

    # Step 5: Confirm save
    print()
    choice = input("  Save configuration? [y/N]: ").strip().lower()
    if choice not in {"y", "yes"}:
        print("  Discarded.")
        return

    # Save configuration
    api.save_llm_config(new_model, new_base_url)
    if new_key:
        api.set_api_key("default", new_key)

    print("  Configuration saved.")


def _print_troubleshooting(result: dict, base_url: str) -> None:
    """Print targeted troubleshooting hints based on the connection error."""
    error = result.get("error", "")

    if "404" in error:
        print()
        print("  Troubleshooting: HTTP 404 — endpoint not found.")
        print("  The client POSTs to:  <base_url>/chat/completions")
        print(f"  Full URL attempted:   {base_url}/chat/completions")
        print("  Make sure base_url points to an OpenAI-compatible API root.")
        print("  Examples:")
        print("    DeepSeek:     https://api.deepseek.com/v1")
        print("    OpenAI:       https://api.openai.com/v1")
        print("    Ollama (loc): http://localhost:11434/v1")
        print("    vLLM  (loc):  http://localhost:8000/v1")
    elif "401" in error or "403" in error:
        print()
        print("  Troubleshooting: Authentication failed.")
        print("  Check that your API key is valid and not expired.")
    elif "Connection" in error or "Name or service not known" in error:
        print()
        print("  Troubleshooting: Cannot reach server.")
        print("  Check that the base_url hostname is correct and reachable.")


def run_clear_dbase() -> int:
    """Clear all records from the global D-Base with confirmation code.

    Returns:
        Exit code (0 for success, 1 for aborted).
    """
    from dargus import api

    # Generate confirmation code via API
    code = api.generate_clear_dbase_code()
    print("  WARNING: This will delete ALL records from the global D-Base.")
    print(f"  Confirmation code: {code}")
    user_input = input("  Enter the code exactly to proceed: ").strip()

    # API verifies the code internally
    success = api.clear_dbase(user_input, code)
    if success:
        print("  Global D-Base cleared.")
        return 0
    else:
        print("  Confirmation code mismatch or operation failed. Aborted.")
        return 1
