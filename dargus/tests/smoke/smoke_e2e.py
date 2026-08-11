"""Standalone smoke: E2E — real CLI → init → bootstrap → runtime → Iris → live LLM.

Runs the real ``python -m dargus iris "<minimal query>"`` subprocess against the
real environment (real config, real D-Base, real API key if present). Asserts
exit 0 and a non-empty ``[Iris]`` reply.

- **Without an API key:** prints ``SKIP`` and exits 0 ("I'm offline" is not a
  broken build).
- **With a key but a real model error:** prints ``FAIL`` (a wiring break).
- **With a key and success:** prints ``PASS``.

Contract: prints a ``PASS`` / ``FAIL`` / ``SKIP`` verdict line and a matching
exit code (0 pass/skip, non-zero fail). Run directly:  python smoke_e2e.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def _env() -> dict[str, str]:
    """The smoke environment, with the key stripped so we can detect it."""
    env = dict(os.environ)
    env.pop("DARGUS_LLM_API_KEY", None)
    env.pop("DARGUS_CONFIG", None)  # E2E runs against the real config
    return env


def main() -> int:
    if not os.environ.get("DARGUS_LLM_API_KEY"):
        print("SKIP")
        return 0

    query = "Reply with exactly: OK"
    proc = subprocess.run(
        [sys.executable, "-m", "dargus", "iris", query],
        capture_output=True,
        text=True,
        timeout=120,
        env=_env(),
    )

    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(f"FAIL: `python -m dargus iris` exited {proc.returncode}")
        print(output)
        return 1

    # A non-empty [Iris] reply proves the whole pipeline returned a live answer.
    prefix = "[Iris] "
    idx = output.rfind(prefix)
    if idx < 0:
        print("FAIL: no `[Iris]` reply line in CLI output")
        print(output)
        return 1

    reply = output[idx + len(prefix) :].strip()
    if not reply:
        print("FAIL: `[Iris]` reply was empty")
        print(output)
        return 1

    print(f"PASS: {reply}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — smoke scripts report any failure as FAIL
        print(f"FAIL: {exc.__class__.__name__}: {exc}")
        sys.exit(1)
