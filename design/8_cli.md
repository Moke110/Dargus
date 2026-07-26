# Dargus CLI Design

> The Dargus CLI is the researcher’s primary interface to the system. It provides both a conversational REPL and direct subcommands, with native terminal experience and clear feedback about configuration and state.

## Two interaction modes

1. **REPL.** Running `dargus` with no subcommand launches a Rich-based REPL. The user types natural-language requests; Iris parses intent and responds.
2. **Subcommands.** `dargus predict`, `dargus ingest`, `dargus benchmark`, etc., are scriptable and suitable for CI/automation.

Both modes use the same underlying runtime, Agents, and workflows.

## REPL experience

- **No alternate screen buffer.** All output stays in the terminal scrollback.
- **Logo and greeting.** A boxed ASCII wordmark appears when the terminal is wide enough; a text fallback is shown on narrow terminals.
- **Prompt loop.** `> ` waits for input. Special commands start with `/`.
- **Built-in slash commands:** `/help`, `/quit`, `/config`, `/test`.
- **Configuration hint.** If no LLM API key is configured, the REPL shows setup instructions instead of failing silently.

## Subcommands

| Command | Purpose |
|---|---|
| `dargus` | Launch REPL |
| `dargus predict --drugs ... --disease ... [--endpoints ...]` | Run Predict workflow |
| `dargus ingest --datadir ... [--disease-kb-dir ...] [--reset]` | Run Ingest workflow |
| `dargus benchmark --config ...` | Run Benchmark workflow |
| `dargus status` | Show D-Base status |
| `dargus config` | Configure the reasoning model and other Dargus settings (detailed subcommands to be defined) |
| `dargus config set-api-key <provider> <key>` | Save API key to `.env` |
| `dargus config show` | Display current configuration (keys masked) |
| `dargus test` | Run internal test suite menu |

## Configuration flow

1. User obtains an API key from their LLM provider.
2. `dargus config set-api-key <provider> <key>` writes the key to `.env` with restricted permissions.
3. `dargus config show` displays the active model and other settings without exposing the key.
4. `dargus config` (CLI) and `/config` (REPL) are the single entry point for configuring the reasoning model and other Dargus settings. Detailed configuration capabilities are defined during implementation; this doc does not enumerate them further.

## Deployment wrapper

The `dargus` shell wrapper activates the configured conda environment automatically. If the user is already in the target environment, it runs directly; otherwise it uses `conda run`. The environment name is configurable via `dargus_config.yaml` or the `DARGUS_CONDA_ENV` environment variable.

## Error handling

- Missing API key → clear setup instructions.
- Missing optional dependency → fallback message and exit code.
- Invalid arguments → argparse help.

## v1.0.0 scope

- Rich REPL with natural-language routing.
- Subcommands for predict, ingest, benchmark, status, config, and test.
- API key management and masked configuration display.
- `dargus config` (CLI) and `/config` (REPL) as the single configuration entry point.
- Conda-aware shell wrapper.
