# Dargus

Clinical efficacy prediction system for drug-development researchers.

## Installation

```bash
pip install -e .
```

## Usage

Dargus provides a unified CLI with two modes:

### One-shot commands

```bash
# Send a query to Iris
dargus iris "predict aspirin for migraine"

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
- Use `/test` to run test suite
- Use `/quit` to exit

## Architecture

- **CLI** (`dargus/cli/`) — User interface layer, includes both one-shot commands and REPL
- **API** (`dargus/api.py`) — Public facade, all UI access goes through here
- **Core** (`dargus/iris/`, `dargus/dbase/`, etc.) — Internal runtime

## Development

```bash
# Run tests
pytest

# Lint
ruff check dargus dargus/tests

# Format
black dargus dargus/tests
```
