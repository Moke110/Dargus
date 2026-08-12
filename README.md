# Dargus

Clinical efficacy prediction system for drug-development researchers.

## Installation

Install Dargus with one terminal command (Linux + macOS). The installer
bootstraps [uv](https://docs.astral.sh/uv/), provisions a uv-managed Python
when needed, and installs the `dargus` command in an isolated tool
environment — it never touches an existing conda/system Python:

```bash
curl -LsSf https://github.com/Moke110/Dargus/releases/latest/download/install.sh | sh
```

> On Linux servers this is the same path used in WSL.

### From a source checkout (developers)

```bash
pip install -e ".[dev]"
```

## First run

Once installed, initialise Dargus on your machine:

```bash
dargus setup
```

The interactive wizard confirms your **Dargus home** (default `~/.dargus/`,
or `$DARGUS_HOME` if you set it) and sets up everything under it:

- a clean default config (`dargus_config.yaml`)
- your LLM API key in a restricted-permission `{home}/.env`
- the D-Base directory structure (`{home}/dbase`)
- migration of any legacy per-workspace session archives into `{home}/sessions`

Until setup has run, one-shot commands refuse with "run `dargus setup` first"
and the bare REPL shows a setup banner.

## Usage

Dargus provides a unified CLI.

### One-shot commands

```bash
# Send a query to Iris
dargus iris "how does aspirin work?"

# Configuration menu
dargus config

# Test menu
dargus test

# Run the interactive setup wizard
dargus setup
```

### Interactive REPL

```bash
# Launch interactive REPL
dargus
```

In the REPL, you can:

- Type natural language queries directly
- Use `/help` to see available commands
- Use `/config` to configure LLM
- Use `/test` to run the test suite
- Use `/clear-dbase` to clear all records from the global D-Base
- Use `/new` / `/resume <id>` to start or resume a session
- Use `/quit` to exit

## Upgrading

```bash
uv tool upgrade dargus-cli
```

Upgrading never touches your Dargus home data.

## Uninstalling

```bash
dargus uninstall
```

Uninstall removes the program (via uv) and **preserves** your Dargus home
data — config, secrets, D-Base, and session archive — printing where they
remain. Nothing of yours is deleted; remove the Dargus home directory
yourself if you also want to delete the data.

## Where your data lives (Dargus home)

One per-user home holds everything (`$DARGUS_HOME`, default `~/.dargus/`):

| Data | Location |
| --- | --- |
| Configuration | `{home}/dargus_config.yaml` |
| Secrets (API keys) | `{home}/.env` |
| D-Base evidence store | `{home}/dbase` |
| Session archive | `{home}/sessions` |

## Architecture

- **CLI** (`dargus/cli/`) — User interface layer, includes both one-shot commands and REPL
- **API** (`dargus/api.py`) — Public facade, all UI access goes through here
- **Core** (`dargus/iris/`, `dargus/dbase/`, `dargus/runtime/`, `dargus/agents/`) — Internal runtime: the PRA (Perceive → Reason → Act) agent harness, Iris, and the D-Base evidence store

## Development

```bash
# Run tests
pytest

# Lint
ruff check dargus

# Format
black dargus dargus/tests
```
