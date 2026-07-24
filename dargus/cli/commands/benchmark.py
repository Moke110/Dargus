"""Benchmark subcommand handler — parses CLI args, builds task_spec, calls API."""

from __future__ import annotations

from argparse import Namespace


def handle_benchmark(args: Namespace) -> int:
    """Execute the ``dargus benchmark`` subcommand.

    Delegates to :func:`dargus.api.benchmark`.
    """
    import json as _json

    from dargus.api import benchmark

    strip = _json.loads(args.strip) if args.strip else {}
    split = _json.loads(args.split) if args.split else None
    result = benchmark(strip=strip, split=split, output_dir=args.output_dir)

    metrics = result.get("metrics", {})
    print("Benchmark Results:")

    def _fmt(key: str) -> str:
        val = metrics.get(key)
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val or "N/A")

    print(f"  Accuracy:  {_fmt('accuracy')}")
    print(f"  Precision: {_fmt('precision')}")
    print(f"  Recall:    {_fmt('recall')}")
    print(f"  F1:        {_fmt('f1')}")
    print(f"  N test:    {result.get('n_test', 'N/A')}")
    return 0
