# Dargus Smoke Suite

Standalone smoke tests that prove the **real, running system** still boots and
works end to end — real imports under the real interpreter, real config
parsing, real persistence, the real CLI. They are deliberately separate from
the offline pytest unit suite (which stubs the LLM and embedding model and
never touches a real key, network, or D-Base).

## What's covered

| Script | Proves | Offline? |
|---|---|---|
| `smoke_dbase.py` | a real Evidence Record validates, writes through the real store, reads back, re-validates | yes |
| `smoke_sessions.py` | create → persist → reload → reopen round-trip, archive is append-only | yes |
| `smoke_tools.py` | ToolRegistry has the file Tools; WorkspaceGuard rejects writes outside the root | yes |
| `smoke_runtime.py` | `bootstrap()` parses a temp config and returns a healthy runtime with models wired | yes |
| `smoke_models.py` | temp-config parsing + deterministic offline embedding round-trip | yes |
| `smoke_agents.py` | a real Iris turn settles through the harness (converged AgentReport) | yes |
| `smoke_api.py` | `new_session` / `resume_session` / `end_session` through the public facade | yes |
| `smoke_e2e.py` | real `python -m dargus iris "<query>"` → CLI → bootstrap → Iris → live LLM reply | **no** — real LLM |

- **Module smokes** run offline in an isolated temp workspace; they never
  write into your real `.dargus/` and never need a key or network.
- **The E2E smoke** hits the real LLM. Without `DARGUS_LLM_API_KEY` it prints
  `SKIP` (exit 0); with a key but a real model error it FAILs; with success it
  prints the `[Iris]` reply and PASSes.

## How to run

```bash
# Whole suite (needs no key; the E2E will SKIP without one)
python dargus/tests/smoke/run_smoke.py

# Subset by name — "dbase" or "smoke_dbase" both work
python dargus/tests/smoke/run_smoke.py dbase agents

# Custom output dir
python dargus/tests/smoke/run_smoke.py --out /tmp/my-smoke

# Any single script on its own
python dargus/tests/smoke/smoke_dbase.py
```

The driver runs every script as a subprocess under `sys.executable`, so a
crash in one script can't corrupt another. It prints a live
`PASS/FAIL/SKIP` table, writes `results.json` + `summary.txt`, and exits
non-zero iff any script FAILed — so it can gate a script or CI step.

## Outputs

Each run lands under `dargus/tests/smoke/out/<timestamp>/`:

- `<name>.log` — one per script (stdout + stderr)
- `results.json` — machine-readable verdicts
- `summary.txt` — the one-line tally plus per-script verdicts

The whole `out/` directory is gitignored — rolling logs are never committed.

## Contract

Every `smoke_*.py` ends with a verdict line — `PASS`, `FAIL`, or `SKIP` —
and a matching exit code (0 pass/skip, non-zero fail). The driver and any
caller consume only this contract.
