"""Dargus command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

from dargus import Iris
from dargus._env import load_dotenv


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``dargus`` CLI."""
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

    benchmark_parser = subparsers.add_parser("benchmark", help="run a benchmark configuration")
    benchmark_parser.add_argument("--config", required=True)

    subparsers.add_parser("status", help="show global D-Base status")

    subparsers.add_parser("clear", help="clear all records from the global D-Base")

    subparsers.add_parser("serve-mcp", help="start Dargus MCP server (stdio transport)")

    ingest_report_parser = subparsers.add_parser(
        "ingest-report", help="generate ingestion report without writing"
    )
    ingest_report_parser.add_argument("--datadir", required=True)
    ingest_report_parser.add_argument("--disease-kb-dir")

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

    if args.command == "benchmark":
        from dargus.workflows.benchmark import run as run_benchmark

        report = run_benchmark(args.config)
        print(f"Benchmark complete: {report['metrics']}")
        return 0

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

    parser.print_help()
    return 1


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
