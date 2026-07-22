"""Dargus command-line interface."""

from __future__ import annotations

import argparse
import json as _json
import logging
import secrets
import sys

from dargus import Iris
from dargus._env import load_dotenv
from dargus.tui import run_app


def _json_arg(raw: str) -> dict:
    """Parse a CLI JSON argument."""
    return _json.loads(raw)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``dargus-cli`` CLI."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="dargus", description="Dargus efficacy prediction")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="ingest data into the global D-Base")
    train_parser.add_argument("--datadir", required=True)
    train_parser.add_argument("--reset", action="store_true", help="clear D-Base before training")
    train_parser.add_argument("--disease-kb-dir", help="path to disease knowledge base directory")

    predict_parser = subparsers.add_parser("predict", help="predict efficacy for drugs/disease")
    predict_parser.add_argument("--drugs", required=True)
    predict_parser.add_argument("--disease", required=True)
    predict_parser.add_argument("--endpoints", nargs="+")
    predict_parser.add_argument("--max-rounds", type=int, default=5)

    subparsers.add_parser("status", help="show global D-Base status")

    subparsers.add_parser("clear-dbase", help="clear all records from the global D-Base")

    config_parser = subparsers.add_parser("config", help="configure Dargus settings")
    config_subs = config_parser.add_subparsers(dest="config_command")

    set_key_parser = config_subs.add_parser(
        "set-api-key", help="set your LLM API key for any provider"
    )
    set_key_parser.add_argument("provider", help="provider name, e.g. openai, anthropic, deepseek")
    set_key_parser.add_argument("key", help="your API key")

    config_subs.add_parser("show", help="show current LLM configuration")

    subparsers.add_parser("model", help="interactive LLM configuration wizard")

    subparsers.add_parser("test", help="run internal test suite")

    args = parser.parse_args(argv)

    if args.command == "train":
        from dargus.workflows.train import run as run_train

        report = run_train(args.datadir, reset=args.reset, disease_kb_dir=args.disease_kb_dir)
        print(f"Records added: {report.n_records}")
        print(f"Duplicates skipped: {report.n_skipped}")
        print(f"Global D-Base size: {report.dbase_size}")
        return 0

    if args.command == "predict":
        from dargus.workflows.predict import run as run_predict

        drug_ids = [d.strip() for d in args.drugs.split(",") if d.strip()]
        result = run_predict(
            drug_ids=drug_ids,
            disease_id=args.disease,
            endpoints=args.endpoints,
            max_rounds=args.max_rounds,
        )
        for drug, disease_eps in result.items():
            print(f"{drug}:")
            for disease, endpoints_dict in disease_eps.items():
                for endpoint, pred in endpoints_dict.items():
                    print(
                        f"  {disease}/{endpoint}: "
                        f"[{pred['efficacy_low']:.3f}, {pred['efficacy_up']:.3f}]"
                    )
        return 0

    if args.command == "status":
        iris = Iris()
        status = iris.status()
        print(status)
        return 0

    if args.command == "clear-dbase":
        return _clear_dbase()

    if args.command == "config":
        if args.config_command == "set-api-key":
            from dargus._env import write_dotenv

            env_path = write_dotenv("DARGUS_LLM_API_KEY", args.key)
            print(f"API key for '{args.provider}' saved to {env_path}")
            print("Run 'dargus' to start the REPL.")
            return 0

        elif args.config_command == "show":
            import os
            from pathlib import Path

            import yaml

            config_path = Path(__file__).resolve().parent / "config" / "dargus_config.yaml"
            with config_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            llm_cfg = cfg.get("llm", {})

            print("LLM Configuration:")
            print(f"  provider:    {llm_cfg.get('provider', 'openai_compatible')}")
            print(f"  model:       {llm_cfg.get('model', 'not set')}")
            base_url = llm_cfg.get("base_url", "")
            if base_url:
                print(f"  base_url:    {base_url}")
            print(f"  temperature: {llm_cfg.get('temperature', 0.0)}")
            print(f"  max_tokens:  {llm_cfg.get('max_tokens', 2048)}")

            api_key = os.environ.get("DARGUS_LLM_API_KEY")
            if api_key:
                print("  api_key:     [from .env]  ********")
            else:
                print("  api_key:     not set")
                print()
                print("  No API key found. Set one with:")
                print("    dargus config set-api-key <provider> <key>")
            return 0

        else:
            config_parser.print_help()
            return 1

    if args.command == "model":
        return _run_model_wizard()

    if args.command == "test":
        return _run_test_suite()

    try:
        run_app()
    except ImportError as exc:
        print(f"Error: Cannot launch REPL — missing dependency: {exc}", file=sys.stderr)
        print("Run: pip install -e .[dev]", file=sys.stderr)
        return 1
    return 0


def _clear_dbase() -> int:
    """Clear all records from the global D-Base with confirmation code."""
    from dargus.dbase import DBase
    from dargus.dbase.manager import DBaseManager

    code = secrets.token_hex(5)
    print("WARNING: This will delete ALL records from the global D-Base.")
    print(f"Confirmation code: {code}")
    user_input = input("Enter the code exactly to proceed: ").strip()
    if user_input != code:
        print("Confirmation code mismatch. Aborted.")
        return 1

    dbase = DBase.global_instance()
    manager = DBaseManager(dbase)
    manager.reset()
    print("Global D-Base cleared.")
    return 0


def _run_test_suite() -> int:
    """Run the internal Dargus test suite (arrow-key menu)."""
    from pathlib import Path

    test_dir = Path(__file__).resolve().parent / "tests"
    modules = sorted(
        p.name for p in test_dir.iterdir() if p.is_dir() and (p / "__init__.py").exists()
    )

    options = [
        f"Run All Tests ({_count_tests(test_dir)} tests)",
        "DB Input — write single evidence to test D-Base",
        "Bulk Input — bulk write evidence instances from db-input-instances/",
    ]
    for mod in modules:
        options.append(f"{mod} ({_count_tests(test_dir / mod)} tests)")
    options.append("Exit")

    while True:
        idx = _arrow_menu(options)
        if idx == len(options) - 1:  # Exit
            print("  Exiting test suite.")
            return 0
        if idx == 0:  # Run All
            import pytest

            pytest.main(["-q", str(test_dir)])
        elif idx == 1:  # DB Input
            _run_test_dbase()
        elif idx == 2:  # Bulk Input
            _run_test_bulk_input()
        else:  # specific module
            import pytest

            mod = modules[idx - 3]
            pytest.main(["-q", str(test_dir / mod)])
        print()
        input("Press ENTER to return to menu...")
    return 0


def _count_tests(path) -> int:
    """Count test files in a directory (path is a Path object)."""
    return len(list(path.glob("test_*.py")))


def _run_test_dbase() -> int:
    """Write a single evidence record to a test D-Base (DARGUS_HOME/dbase-test)."""
    import json
    import os
    from pathlib import Path

    import yaml

    from dargus.dbase import DBase
    from dargus.dbase.manager import DBaseManager
    from dargus.dbase.paths import default_dargus_home

    test_root = default_dargus_home()
    test_dbase_dir = test_root / "dbase-test"
    data_dir = test_dbase_dir / "data"

    # Step 1: check/create test D-Base
    if data_dir.exists():
        records = []
        for shard in sorted(data_dir.glob("shard-*.jsonl")):
            with shard.open("r", encoding="utf-8") as fh:
                records.extend(json.loads(line) for line in fh if line.strip())
        print(f"Test D-Base found ({len(records)} records).")
        choice = input("Clear existing data? [y/N]: ").strip().lower()
        if choice in ("y", "yes"):
            import shutil

            shutil.rmtree(test_dbase_dir)
            test_dbase_dir.mkdir(parents=True)
            (test_dbase_dir / "data").mkdir()
            (test_dbase_dir / "views").mkdir()
            print("Cleared.")
    else:
        test_dbase_dir.mkdir(parents=True)
        data_dir.mkdir()
        (test_dbase_dir / "views").mkdir()
        print(f"Test D-Base created at {test_dbase_dir}")

    # Step 2: switch to test database
    old_working = os.environ.get("WORKING_DBASE")
    os.environ["WORKING_DBASE"] = "dbase-test"

    try:
        # Step 3: get input
        raw_input = input("Paste evidence JSON or path to file (.yaml/.json): ").strip()

        # Step 4: parse input
        if raw_input.startswith("{") or raw_input.startswith("["):
            raw_data = json.loads(raw_input)
        else:
            input_path = Path(raw_input).expanduser()
            if not input_path.exists():
                print(f"File not found: {input_path}")
                return 1
            content = input_path.read_text(encoding="utf-8")
            if input_path.suffix in (".yaml", ".yml"):
                raw_data = yaml.safe_load(content) or {}
            else:
                raw_data = json.loads(content)

        # Step 5: build + write
        dbase = DBase.global_instance()
        manager = DBaseManager(dbase)
        evidence = manager.build_evidence(
            raw_data,
            source_metadata={"type": "file_path", "id": "test-dbase:cli"},
        )

        wrote = manager.write_record(evidence)
        status = "added" if wrote is True else "duplicate — skipped"

        # Step 6: report
        print()
        print("  Instance written.")
        print("  ────────────────────────────────")
        print(f"  Evidence ID:     {evidence['evidence_id']}")
        print(f"  Biological level: {evidence.get('biological_level', '?')}")
        print(f"  Readout type:     {evidence.get('readout_type', '?')}")
        print(f"  Status:           {status}")

    finally:
        # Step 7: restore default D-Base
        if old_working is not None:
            os.environ["WORKING_DBASE"] = old_working
        else:
            os.environ.pop("WORKING_DBASE", None)
        print(f"  Working D-Base restored to default ({default_dargus_home() / 'dbase'}).")

    return 0


def _run_test_bulk_input() -> int:
    """Bulk-write evidence .json files from a configurable directory to the test D-Base."""
    import json
    import os
    import shutil
    import time
    from pathlib import Path

    import yaml

    from dargus.dbase import DBase
    from dargus.dbase.manager import DBaseManager, DuplicateReviewRequest
    from dargus.dbase.paths import default_dargus_home

    config_path = Path(__file__).resolve().parent / "config" / "dargus_config.yaml"

    # Load current config
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # Resolve saved bulk_input_dir with ~ expansion
    saved_dir = cfg.get("test", {}).get("bulk_input_dir", "")
    default_dir = "~/dargus-dev/tests/db-input-instances"
    current_dir = Path(saved_dir).expanduser() if saved_dir else Path(default_dir).expanduser()

    # Show current dir and ask for change
    print(f"\n  Bulk Input directory: {current_dir}")
    choice = input("  Change directory? Enter new path or ENTER to keep: ").strip()
    if choice:
        new_dir = Path(choice).expanduser()
        if not new_dir.is_dir():
            print(f"  Directory not found: {new_dir}")
            return 1
        current_dir = new_dir
        # Persist to config
        cfg.setdefault("test", {})["bulk_input_dir"] = str(new_dir)
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print("  Saved to config.")

    if not current_dir.is_dir():
        print(f"  Instance directory not found: {current_dir}")
        print("  Run Biomni prompt_2 first to generate evidence .json files.")
        return 1

    json_files = sorted(current_dir.glob("*.json"))
    if not json_files:
        print(f"  No .json files found in {current_dir}")
        return 1

    # Step 1: check/create test D-Base
    test_root = default_dargus_home()
    test_dbase_dir = test_root / "dbase-test"
    data_dir = test_dbase_dir / "data"

    if data_dir.exists():
        records = []
        for shard in sorted(data_dir.glob("shard-*.jsonl")):
            with shard.open("r", encoding="utf-8") as fh:
                records.extend(json.loads(line) for line in fh if line.strip())
        print(f"  Test D-Base found ({len(records)} records).")
        choice = input("  Clear existing data? [y/N]: ").strip().lower()
        if choice in ("y", "yes"):
            shutil.rmtree(test_dbase_dir)
            test_dbase_dir.mkdir(parents=True)
            (test_dbase_dir / "data").mkdir()
            (test_dbase_dir / "views").mkdir()
            print("  Cleared.")
    else:
        test_dbase_dir.mkdir(parents=True)
        data_dir.mkdir()
        (test_dbase_dir / "views").mkdir()
        print(f"  Test D-Base created at {test_dbase_dir}")

    # Step 2: switch to test database
    old_working = os.environ.get("WORKING_DBASE")
    os.environ["WORKING_DBASE"] = "dbase-test"

    dbase = DBase.global_instance()
    manager = DBaseManager(dbase)

    added = 0
    duplicates = 0
    hard_rejects = 0
    errors = 0
    error_details: list[str] = []
    total = len(json_files)
    t0 = time.perf_counter()
    bar_width = 30

    try:
        print()

        for i, jf in enumerate(json_files):
            # Parse
            try:
                raw_data = json.loads(jf.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                hard_rejects += 1
                error_details.append(f"{jf.name}: JSON parse error — {exc}")
                _draw_progress(i + 1, total, bar_width, added, duplicates, hard_rejects, errors, t0)
                continue

            # Build
            try:
                evidence = manager.build_evidence(
                    raw_data,
                    source_metadata={"type": "file_path", "id": f"test-dbase:bulk:{jf.name}"},
                )
            except ValueError as exc:
                hard_rejects += 1
                raw_msg = str(exc)
                # Trim validation error list for display
                msg = raw_msg[:120] + ("..." if len(raw_msg) > 120 else "")
                error_details.append(f"{jf.name}: {msg}")
                _draw_progress(i + 1, total, bar_width, added, duplicates, hard_rejects, errors, t0)
                continue
            except Exception as exc:
                errors += 1
                error_details.append(f"{jf.name}: build_evidence failed — {exc}")
                _draw_progress(i + 1, total, bar_width, added, duplicates, hard_rejects, errors, t0)
                continue

            # Write
            try:
                result = manager.write_record(evidence)
            except ValueError as exc:
                hard_rejects += 1
                raw_msg = str(exc)
                msg = raw_msg[:120] + ("..." if len(raw_msg) > 120 else "")
                error_details.append(f"{jf.name}: {msg}")
                _draw_progress(i + 1, total, bar_width, added, duplicates, hard_rejects, errors, t0)
                continue
            except Exception as exc:
                errors += 1
                error_details.append(f"{jf.name}: write_record failed — {exc}")
                _draw_progress(i + 1, total, bar_width, added, duplicates, hard_rejects, errors, t0)
                continue

            if result is True:
                added += 1
            elif isinstance(result, DuplicateReviewRequest):
                duplicates += 1
            else:
                duplicates += 1

            _draw_progress(i + 1, total, bar_width, added, duplicates, hard_rejects, errors, t0)

        # final newline after progress bar
        print()

    finally:
        # Step 7: restore default D-Base
        if old_working is not None:
            os.environ["WORKING_DBASE"] = old_working
        else:
            os.environ.pop("WORKING_DBASE", None)

    elapsed = time.perf_counter() - t0
    rate = total / elapsed if elapsed > 0 else 0

    # Rebuild view for queryability (best-effort; parquet engine may be absent)
    try:
        dbase.rebuild_view()
    except Exception:
        pass

    # Step 6: report
    print()
    print("  Bulk input complete.")
    print("  ────────────────────────────────")
    print(f"  Directory:        {current_dir}")
    print(f"  Files processed:  {total}")
    print(f"  Added:            {added}")
    print(f"  Duplicates:       {duplicates}")
    print(f"  Hard rejects:     {hard_rejects}")
    print(f"  Errors:           {errors}")
    print(f"  Time:             {elapsed:.1f}s ({rate:.0f} files/s)")
    if error_details:
        print("\n  Details (first 10):")
        for detail in error_details[:10]:
            print(f"    - {detail}")
        if len(error_details) > 10:
            print(f"    ... and {len(error_details) - 10} more")
    print(f"\n  Working D-Base restored to default ({default_dargus_home() / 'dbase'}).")

    return 0


def _draw_progress(
    done: int,
    total: int,
    bar_width: int,
    added: int,
    dup: int,
    rejects: int,
    errs: int,
    t0: float,
) -> None:
    """Draw a single-line progress bar, overwriting in-place.

    Never emits a newline. Pads to COLUMNS (or 80) with spaces so a shorter
    successor always fully overwrites a longer predecessor, even after terminal
    resize.
    """
    import shutil
    import sys
    import time

    frac = done / total if total > 0 else 0
    filled = int(bar_width * frac)
    bar = "█" * filled + "░" * (bar_width - filled)
    elapsed = time.perf_counter() - t0
    rate = done / elapsed if elapsed > 0 else 0
    remain = (total - done) / rate if rate > 0 else 0

    col = shutil.get_terminal_size((80, 24)).columns
    line = (
        f"  [{bar}] {done}/{total}  "
        f"added={added}  dup={dup}  reject={rejects}  err={errs}  "
        f"{rate:.0f} f/s  ETA {remain:.0f}s"
    )
    line = line[: col - 1]  # never wrap
    sys.stdout.write("\r" + line.ljust(col - 1, " ") + "\r")
    sys.stdout.flush()


def _arrow_menu(options: list[str]) -> int:
    import sys
    import termios
    import tty

    idx = 0
    n = len(options)
    # total lines drawn: blank line + options + blank + help
    total_lines = n + 3
    _first = True

    def _draw():
        nonlocal _first
        out = sys.stdout
        if not _first:
            out.write(f"\r\033[{total_lines}A")  # return to col 0, move up
        _first = False
        out.write("\r\033[J")  # CR + clear to end of screen
        out.write("\r\n")
        for i, opt in enumerate(options):
            prefix = "  > " if i == idx else "    "
            out.write(f"\r{prefix}{opt}\r\n")
        out.write("\r\n")
        out.write("\rUse ↑/↓ to navigate, ENTER to select\r\n")
        out.flush()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        _draw()
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                break
            if ch == "\x1b":
                nxt = sys.stdin.read(2)
                if nxt == "[A":
                    idx = (idx - 1) % n
                elif nxt == "[B":
                    idx = (idx + 1) % n
                _draw()
            elif ch == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    print()
    return idx


def _run_model_wizard() -> int:
    """Interactive LLM configuration wizard for CLI and REPL."""
    import os
    from pathlib import Path

    import yaml

    from dargus._env import write_dotenv
    from dargus.llm import DargusLLM, check_llm_connection

    print()
    print("  Configure LLM connection")
    print("  ────────────────────────────────")
    print()

    choice = _arrow_menu(["Enter new configuration", "Skip (keep current settings)"])
    if choice == 1:
        print("  Keeping current configuration.")
        return 0

    # Load current config
    config_path = Path(__file__).resolve().parent / "config" / "dargus_config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    llm_cfg = cfg.get("llm", {})

    cur_base_url = _resolve_config_value(llm_cfg.get("base_url", ""))
    cur_model = llm_cfg.get("model", "")
    cur_key = os.environ.get("DARGUS_LLM_API_KEY", "")

    key_display = "********" if cur_key else "(not set)"

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
    if not new_key:
        new_key = cur_key

    # Step 4: Test connection
    print()
    print("  Testing connection...")
    print(f"  POST {new_base_url}/chat/completions")
    llm = DargusLLM(model=new_model, base_url=new_base_url, api_key=new_key or None)
    result = check_llm_connection(llm)
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
        return 0

    # Write model and base_url to YAML
    cfg.setdefault("llm", {})
    cfg["llm"]["model"] = new_model
    cfg["llm"]["base_url"] = new_base_url
    cfg["llm"]["provider"] = "openai_compatible"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # Write API key to .env
    if new_key:
        write_dotenv("DARGUS_LLM_API_KEY", new_key)
        os.environ["DARGUS_LLM_API_KEY"] = new_key

    print("  Configuration saved.")
    return 0


def _print_troubleshooting(result: dict, base_url: str) -> None:
    """Print targeted troubleshooting hints based on the connection error."""
    error = result.get("error", "")

    if "404" in error:
        print()
        print("  Troubleshooting: HTTP 404 — endpoint not found.")
        print("  DargusLLM POSTs to:  <base_url>/chat/completions")
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


def _resolve_config_value(value: str) -> str:
    """Resolve $ENV_VAR references in config values."""
    import os

    if isinstance(value, str) and value.startswith("$"):
        return os.environ.get(value[1:], value)
    return value


def _cli_confirm(plan: dict) -> bool:
    """Default console confirmation callback."""
    if plan.get("kind") == "training_pre_confirmation":
        datadir = plan.get("datadir")
        if datadir is None:
            response = input("Add new data for supplemental training before inference? [y/N] ")
        else:
            response = input(f"Train on {datadir} before inference? [y/N] ")
        return response.strip().lower() in {"y", "yes"}
    response = input(
        f"Confirm prediction plan for {plan.get('disease_id')} with agents "
        f"{plan.get('agents')}? [y/N] "
    )
    return response.strip().lower() in {"y", "yes"}


if __name__ == "__main__":
    sys.exit(main())
