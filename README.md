# Dargus

Clinical efficacy prediction system for drug-development researchers.

## Installation

```bash
pip install -e .
```

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
- Use `/quit` to exit

## Architecture

- **CLI** (`dargus/cli/`) — User interface layer, includes both one-shot commands and REPL
- **API** (`dargus/api.py`) — Public facade, all UI access goes through here
- **Core** (`dargus/iris/`, `dargus/dbase/`, `dargus/runtime/`, `dargus/agents/`) — Internal runtime: the PRA (Perceive → Reason → Act) agent harness, Iris, and the D-Base evidence store

## Development

```bash
# Run tests
pytest

# Lint
ruff check dargus dargus/tests

# Format
black dargus dargus/tests
```
