# Dargus CLI Design

> The Dargus CLI is the researcher's primary interface to the system. "CLI" refers to the whole command-line interface, which has two interaction modes: **one-shot commands** and the interactive **REPL**. The term "TUI" is not used.

## Two interaction modes

1. **REPL.** Running `dargus` with no subcommand launches a Rich-based REPL. The user types natural-language requests; anything that is not a slash command is sent to Iris, whose Perceive → Reason → Act loop decides how to handle it.
2. **One-shot commands.** `dargus iris <query>`, `dargus config`, and `dargus test` run a single task and exit.

Both modes reach the runtime exclusively through the public API layer.

## API boundary

`dargus.api` is the sole interaction interface to `DargusRuntime`. All CLI code — REPL, one-shot commands, menus — calls `dargus.api` functions and never instantiates `Iris`, `DargusRuntime`, `DBase`, or other internal classes directly.

The API exposes a **non-interactive core**: every function takes plain arguments, performs no prompting, and returns plain data. Interactive behavior — menus, prompts, confirmation display, output formatting — lives in the CLI as thin wrappers around the core. Destructive operations are guarded at the API layer: `api.clear_dbase()` requires a confirmation code issued by `api.generate_clear_dbase_code()` and verifies it before executing, so the safety gate travels with the operation rather than depending on the caller.

## Command surface

| Command | Purpose |
|---|---|
| `dargus` | Launch REPL |
| `dargus iris <question>` | Send a natural-language task to Iris |
| `dargus config` | Launch the Dargus configuration menu |
| `dargus test` | Launch the Dargus test menu |

There are no other one-shot commands. Capabilities formerly exposed as subcommands (predict, ingest, status) are reached by asking Iris, in either mode.

## REPL experience

- **No alternate screen buffer.** All output stays in the terminal scrollback.
- **Logo and greeting.** A boxed ASCII wordmark appears when the terminal is wide enough; a text fallback is shown on narrow terminals.
- **Prompt loop.** `> ` waits for input. Special commands start with `/`.
- **Built-in slash commands:** `/help`, `/quit`, `/model`, `/test`, `/clear-dbase`. `/model` opens the same configuration menu as `dargus config`; `/test` opens the same test menu as `dargus test`.
- **Configuration hint.** If no LLM API key is configured, the REPL shows setup instructions instead of failing silently.
- **Input routing.** Any input that is not a slash command goes to Iris.

## Configuration menu

`dargus config` opens a numbered menu:

1. Show current LLM configuration (key masked).
2. Set API key (writes to `.env`).
3. Run the LLM configuration wizard (base URL → model → key → connection test → save).
4. Clear D-Base (two-step confirmation code).
5. Back.

## Test menu

`dargus test` opens a numbered menu for internal verification: run the full pytest suite or a single module, write a single evidence record to the test D-Base, bulk-write evidence files, or run the Ingest workflow against a test directory with an optional report.

## Configuration file resolution

`dargus_config.yaml` resolves in order:

1. `DARGUS_CONFIG` environment variable (if set),
2. `~/.dargus/dargus_config.yaml` (if it exists),
3. the packaged default `dargus/config/dargus_config.yaml`.

The LLM API key is read from the `DARGUS_LLM_API_KEY` environment variable, typically loaded from the project `.env` at startup.

## Error handling

- Missing API key → clear setup instructions.
- Missing optional dependency → fallback message and exit code.
- Invalid arguments → argparse help.

## v1.0.0 scope

- Rich REPL with all non-command input routed to Iris.
- One-shot commands limited to `iris`, `config`, and `test`.
- `dargus.api` as the sole runtime interaction interface (non-interactive core + CLI interactive wrappers).
- Config and test menus shared between the one-shot commands and REPL slash commands.
- API-layer confirmation guard for destructive operations.

## Out of Scope

- **MCP server.** An optional MCP server could expose `dargus.api` and Dargus workflows to external agent platforms, letting Claude Code or other agents query D-Base and run Predict through a standardized protocol. The MCP adapter was removed from the repository in v0.19.0 and will be reintroduced after v1.0.0.
