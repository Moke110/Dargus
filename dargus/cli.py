"""Dargus command-line interface."""

from __future__ import annotations

import argparse
import json as _json
import logging
import sys
import warnings

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

    infer_parser = subparsers.add_parser(
        "infer", help="predict efficacy for drugs/disease (deprecated, use predict)"
    )
    infer_parser.add_argument("--drugs", required=True)
    infer_parser.add_argument("--disease", required=True)
    infer_parser.add_argument("--endpoints", nargs="+")
    infer_parser.add_argument("--datadir")

    predict_parser = subparsers.add_parser("predict", help="predict efficacy for drugs/disease")
    predict_parser.add_argument("--drugs", required=True)
    predict_parser.add_argument("--disease", required=True)
    predict_parser.add_argument("--endpoints", nargs="+")
    predict_parser.add_argument("--max-rounds", type=int, default=5)

    # bench group with subcommands
    bench_parser = subparsers.add_parser("bench", help="benchmark and test workflows")
    bench_subs = bench_parser.add_subparsers(dest="bench_command")

    fs_parser = bench_subs.add_parser("full-stack", help="run full benchmark pipeline")
    fs_parser.add_argument(
        "--strip",
        required=True,
        type=_json_arg,
        help='JSON filter for global D-Base records, e.g. \'{"source.type":"benchmark"}\'',
    )
    fs_parser.add_argument("--split", type=_json_arg, help="JSON split config")
    fs_parser.add_argument("--output-dir", help="output directory for reports")

    di_parser = bench_subs.add_parser("data-ingest", help="test ingestion pipeline")
    di_parser.add_argument("--fixture-dir", required=True, help="path to fixture directory")
    di_parser.add_argument(
        "--expected-min", type=int, default=1, help="minimum expected records (default: 1)"
    )
    di_parser.add_argument("--output-dir", help="output directory for report")

    subparsers.add_parser("status", help="show global D-Base status")

    subparsers.add_parser("clear", help="clear all records from the global D-Base")

    subparsers.add_parser("serve-mcp", help="start Dargus MCP server (stdio transport)")

    ingest_report_parser = subparsers.add_parser(
        "ingest-report", help="generate ingestion report without writing"
    )
    ingest_report_parser.add_argument("--datadir", required=True)
    ingest_report_parser.add_argument("--disease-kb-dir")

    config_parser = subparsers.add_parser("config", help="configure Dargus settings")
    config_subs = config_parser.add_subparsers(dest="config_command")

    set_key_parser = config_subs.add_parser(
        "set-api-key", help="set your LLM API key for any provider"
    )
    set_key_parser.add_argument("provider", help="provider name, e.g. openai, anthropic, deepseek")
    set_key_parser.add_argument("key", help="your API key")

    config_subs.add_parser("show", help="show current LLM configuration")

    subparsers.add_parser("model", help="interactive LLM configuration wizard")

    args = parser.parse_args(argv)

    if args.command == "train":
        from dargus.workflows.train import run as run_train

        report = run_train(args.datadir, reset=args.reset, disease_kb_dir=args.disease_kb_dir)
        print(f"Records added: {report.n_records}")
        print(f"Duplicates skipped: {report.n_skipped}")
        print(f"Global D-Base size: {report.dbase_size}")
        return 0

    if args.command == "infer":
        warnings.warn(
            "dargus infer is deprecated, use dargus predict instead",
            DeprecationWarning,
        )
        from dargus.workflows.predict import run as run_predict

        drug_ids = [d.strip() for d in args.drugs.split(",") if d.strip()]
        result = run_predict(
            drug_ids=drug_ids,
            disease_id=args.disease,
            endpoints=args.endpoints,
        )
        for drug, endpoints in result.items():
            print(f"{drug}:")
            for endpoint, pred in endpoints.items():
                print(f"  {endpoint}: [{pred['efficacy_low']:.3f}, {pred['efficacy_up']:.3f}]")
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

    if args.command == "bench":
        if args.bench_command == "full-stack":
            from dargus.workflows.bench_full_stack import run as run_full_stack

            result = run_full_stack(
                strip=args.strip,
                split=args.split,
                output_dir=args.output_dir,
            )
            print(f"Benchmark complete: {result['metrics']}")
            return 0

        elif args.bench_command == "data-ingest":
            from dargus.workflows.bench_data_ingest import run as run_data_ingest

            result = run_data_ingest(
                fixture_dir=args.fixture_dir,
                expected_min=args.expected_min,
                output_dir=args.output_dir,
            )
            print(
                f"Ingest test complete: {result['total_records']} records from "
                f"{len(result['record_counts_by_source'])} sources"
            )
            if result["warnings"]:
                for w in result["warnings"]:
                    print(f"  WARNING: {w}", file=sys.stderr)
            return 0

        else:
            bench_parser.print_help()
            return 1

    if args.command == "status":
        iris = Iris()
        status = iris.status()
        print(status)
        return 0

    if args.command == "clear":
        from dargus.dbase import DBase
        from dargus.dbase.manager import DBaseManager

        dbase = DBase.global_instance()
        manager = DBaseManager(dbase)
        manager.reset()
        print("Global D-Base cleared.")
        return 0

    if args.command == "serve-mcp":
        print("Starting Dargus MCP server on stdio...", file=sys.stderr)
        from dargus.adapters.mcp.server import main as mcp_main

        mcp_main()
        return 0

    if args.command == "ingest-report":
        from dargus.workflows.train import ingest_report

        summary = ingest_report(args.datadir, disease_kb_dir=args.disease_kb_dir)
        print(summary)
        return 0

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

    try:
        run_app()
    except ImportError as exc:
        print(f"Error: Cannot launch REPL — missing dependency: {exc}", file=sys.stderr)
        print("Run: pip install -e .[dev]", file=sys.stderr)
        return 1
    return 0


def _run_model_wizard() -> int:
    """Interactive LLM configuration wizard for CLI and REPL."""
    import os
    from pathlib import Path

    import yaml

    from dargus._env import write_dotenv
    from dargus.llm import DargusLLM, check_llm_connection

    # Load current config
    config_path = Path(__file__).resolve().parent / "config" / "dargus_config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    llm_cfg = cfg.get("llm", {})

    cur_base_url = _resolve_config_value(llm_cfg.get("base_url", ""))
    cur_model = llm_cfg.get("model", "")
    cur_key = os.environ.get("DARGUS_LLM_API_KEY", "")

    key_display = "********" if cur_key else "(not set)"

    print()
    print("  Configure LLM connection")
    print("  ────────────────────────────────")
    print()

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
