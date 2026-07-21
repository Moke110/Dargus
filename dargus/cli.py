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
            print("Run 'dargus' to start the TUI.")
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
            print(f"  provider:    {llm_cfg.get('provider', 'litellm')}")
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

    try:
        run_app()
    except ImportError as exc:
        print(f"Error: Cannot launch REPL — missing dependency: {exc}", file=sys.stderr)
        print("Run: pip install -e .[dev]", file=sys.stderr)
        return 1
    return 0


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
