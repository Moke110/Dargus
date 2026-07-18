"""Dargus command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dargus import DirectorAgent


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `dargus` CLI."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="dargus", description="Dargus drug research assistant")
    subparsers = parser.add_subparsers(dest="command")

    scan_parser = subparsers.add_parser("scan", help="target-disease efficacy scan")
    scan_parser.add_argument("target")
    scan_parser.add_argument("disease")
    scan_parser.add_argument("--endpoints", nargs="+")
    scan_parser.add_argument("--drugs", nargs="+")

    scan_v4_parser = subparsers.add_parser("scan-v4", help="v4.0 D-Base + Iris efficacy scan")
    scan_v4_parser.add_argument("--drugs", required=True)
    scan_v4_parser.add_argument("--disease", required=True)
    scan_v4_parser.add_argument("--datadir")
    scan_v4_parser.add_argument("--projects-root", default="projects")

    status_parser = subparsers.add_parser("status", help="check project status")
    status_parser.add_argument("project_id")

    args = parser.parse_args(argv)
    if args.command == "scan":
        from dargus.workflows.target_efficacy_scan import run

        result = run(
            target=args.target,
            disease=args.disease,
            clinical_endpoints=args.endpoints,
            drug_list=args.drugs,
        )
        print(f"Project: {result['project_id']}")
        print(f"Directory: {result['project_dir']}")
        print("Predictions:")
        for endpoint, drugs in result["diris"]["result"]["predictions"].items():
            print(f"  {endpoint}:")
            for drug, pred in drugs.items():
                print(
                    f"    {drug}: {pred['normalized_effect_size']} "
                    f"[{pred['ci_95_lower']}, {pred['ci_95_upper']}]"
                )
        return 0
    if args.command == "scan-v4":
        from dargus.workflows.target_efficacy_scan import run_v4

        drugs_path = Path(args.drugs)
        if drugs_path.exists():
            drug_ids = drugs_path.read_text(encoding="utf-8").strip().split(",")
        else:
            drug_ids = [d.strip() for d in args.drugs.split(",") if d.strip()]

        result = run_v4(
            drugs=drug_ids,
            disease=args.disease,
            datadir=args.datadir,
            projects_root=args.projects_root,
        )
        print(f"Project: {result['project_id']}")
        print("Predictions:")
        for drug, endpoints in result["predictions"].items():
            print(f"  {drug}:")
            for endpoint, pred in endpoints.items():
                print(
                    f"    {endpoint}: {pred['normalized_effect_size']} "
                    f"[{pred['ci95_lower']}, {pred['ci95_upper']}]"
                )
        return 0
    if args.command == "status":
        director = DirectorAgent()
        status = director.status(args.project_id)
        print(status)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
