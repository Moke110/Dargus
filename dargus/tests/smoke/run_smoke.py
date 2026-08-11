"""Smoke-suite driver — run every smoke script as a subprocess and aggregate.

Runs each ``smoke_*.py`` under ``sys.executable``, captures stdout/stderr into
``out/<timestamp>/<name>.log``, folds the exit codes and verdict lines into
``results.json`` + ``summary.txt``, prints a live ``PASS/FAIL/SKIP`` table,
and exits non-zero iff any script FAILed.

Usage:
    python run_smoke.py                 # run the whole suite
    python run_smoke.py dbase agents    # run a subset by name (no smoke_ prefix)
    python run_smoke.py --out /tmp/x    # custom output dir

Verdicts are read from each script's own output: the trailing
``PASS`` / ``FAIL`` / ``SKIP`` line wins; a script that crashes (non-zero exit
with no verdict) is reported as ``FAIL``. On any FAIL the driver exits 1.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "out"
DEFAULT_SCRIPTS = sorted(p.name for p in HERE.glob("smoke_*.py"))


def _verdict_from(output: str) -> str:
    """The verdict is the trailing PASS/FAIL/SKIP line in the script's output.

    A verdict may carry a suffix on the same line (e.g. ``PASS: OK``); only
    the leading word is the contract.
    """
    for line in reversed(output.splitlines()):
        line = line.strip()
        for verdict in ("PASS", "FAIL", "SKIP"):
            if line == verdict or line.startswith(verdict + " ") or line.startswith(verdict + ":"):
                return verdict
    return "FAIL"  # no verdict line — crashed or empty


def _run_one(name: str, script: Path, out_dir: Path, log_path: Path) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = proc.stdout + proc.stderr
        verdict = _verdict_from(output)
    except subprocess.TimeoutExpired:
        output = "TIMEOUT after 300s"
        verdict = "FAIL"
        proc = None

    log_path.write_text(output, encoding="utf-8")

    if verdict == "PASS" and proc is not None and proc.returncode != 0:
        verdict = "FAIL"  # contradictory: said PASS but exited non-zero

    return {
        "name": name,
        "script": str(script),
        "verdict": verdict,
        "exit_code": proc.returncode if proc is not None else "timeout",
        "log": str(log_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Dargus smoke suite.")
    parser.add_argument(
        "names",
        nargs="*",
        help="subset of scripts to run (e.g. 'dbase agents'); default: all",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"output directory; default {OUT_ROOT}/<timestamp>",
    )
    args = parser.parse_args(argv)

    scripts: list[tuple[str, Path]] = []
    for name in args.names or DEFAULT_SCRIPTS:
        # Accept both "dbase" and "smoke_dbase" for a subset selection.
        if not name.startswith("smoke_"):
            name = f"smoke_{name}"
        script = HERE / name if name.endswith(".py") else HERE / f"{name}.py"
        if not script.exists():
            print(f"error: no smoke script '{name}' (wanted {script})", file=sys.stderr)
            return 2
        scripts.append((script.stem, script))

    if not scripts:
        print("error: no smoke scripts to run", file=sys.stderr)
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for name, script in scripts:
        print(f"  running {name} ...", end="", flush=True)
        log_path = out_dir / f"{name}.log"
        result = _run_one(name, script, out_dir, log_path)
        results.append(result)
        print(f" {result['verdict']}")

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        counts[r["verdict"]] += 1

    # results.json
    (out_dir / "results.json").write_text(
        json.dumps({"timestamp": ts, "results": results}, indent=2), encoding="utf-8"
    )

    # summary.txt
    lines = [
        f"Dargus smoke suite — {ts}",
        f"PASS={counts['PASS']} FAIL={counts['FAIL']} SKIP={counts['SKIP']} total={len(results)}",
        "",
    ]
    for r in results:
        lines.append(f"{r['verdict']:5}  {r['name']}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table
    print(f"\n  {'verdict':7} name")
    print(f"  {'───────':7} ────")
    for r in results:
        print(f"  {r['verdict']:7} {r['name']}")
    print(f"\n  outputs: {out_dir}")

    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
