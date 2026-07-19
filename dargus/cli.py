"""Dargus command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys

from dargus import Iris


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``dargus`` CLI."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="dargus", description="Dargus efficacy prediction")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="ingest data into the global D-Base")
    train_parser.add_argument("--datadir", required=True)

    infer_parser = subparsers.add_parser("infer", help="predict efficacy for drugs/disease")
    infer_parser.add_argument("--drugs", required=True)
    infer_parser.add_argument("--disease", required=True)
    infer_parser.add_argument("--endpoints", nargs="+")
    infer_parser.add_argument("--datadir")

    benchmark_parser = subparsers.add_parser("benchmark", help="run a benchmark configuration")
    benchmark_parser.add_argument("--config", required=True)

    subparsers.add_parser("status", help="show global D-Base status")

    args = parser.parse_args(argv)

    if args.command == "train":
        from dargus.workflows.train import run as run_train

        report = run_train(args.datadir)
        print(f"Records added: {report.n_records}")
        print(f"Duplicates skipped: {report.n_skipped}")
        print(f"Global D-Base size: {report.dbase_size}")
        return 0

    if args.command == "infer":
        from dargus.workflows.infer import run as run_infer

        drug_ids = [d.strip() for d in args.drugs.split(",") if d.strip()]
        result = run_infer(
            drug_ids=drug_ids,
            disease_id=args.disease,
            endpoints=args.endpoints,
            datadir=args.datadir,
            confirm_callback=_cli_confirm,
        )
        if isinstance(result, dict) and result.get("aborted"):
            print("Inference aborted.")
            return 1
        for drug, endpoints in result.items():
            print(f"{drug}:")
            for endpoint, pred in endpoints.items():
                print(f"  {endpoint}: [{pred['efficacy_low']:.3f}, {pred['efficacy_up']:.3f}]")
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
