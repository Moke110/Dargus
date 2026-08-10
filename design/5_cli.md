# CLI Design

> The Dargus CLI is the researcher's primary interface to the system. "CLI" refers to the whole command-line interface, which has two interaction surfaces: **one-shot commands** and the interactive **REPL**.

## Two interaction surfaces

1. **REPL.** Running `dargus` with no subcommand launches an interactive REPL. The user types natural-language requests; anything that is not a slash command is sent to Iris, whose Perceive → Reason → Act loop decides how to handle it.
2. **One-shot commands.** `dargus iris <query>`, `dargus config`, and `dargus test` run a single task and exit.

Both surfaces reach the runtime exclusively through `dargus.api`.

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

There are no other one-shot commands. Capabilities such as status are reached by asking Iris, in either surface.

## REPL input routing

- Any input that is not a slash command (starting with `/`) is forwarded to Iris.
- Slash commands are built-in REPL directives (help, quit, config, test, etc.).

## Configuration file resolution

`dargus_config.yaml` resolves in order:

1. `DARGUS_CONFIG` environment variable (if set),
2. `~/.dargus/dargus_config.yaml` (if it exists),
3. the packaged default `dargus/config/dargus_config.yaml`.

The LLM API key is read from the `DARGUS_LLM_API_KEY` environment variable.
