"""Ingest subcommand handler — parses CLI args, builds task_spec, calls API."""

from __future__ import annotations

from argparse import Namespace


def handle_ingest(args: Namespace) -> int:
    """Execute the ``dargus ingest`` (or ``dargus train``) subcommand.

    Delegates to :func:`dargus.api.ingest`.
    """
    from dargus.api import ingest

    report = ingest(args.datadir, reset=args.reset, disease_kb_dir=args.disease_kb_dir)
    print(f"Records added: {report.n_records}")
    print(f"Duplicates skipped: {report.n_skipped}")
    print(f"Global D-Base size: {report.dbase_size}")
    return 0
