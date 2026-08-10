# Runtime Structure

> Dargus runs inside a single **DargusRuntime**. The runtime is the process-level container; it owns configuration, the shared model dependencies, the registries, the conversation store, and the AgentFactory. Agents receive their dependencies from the runtime.

## DargusRuntime

`DargusRuntime` is created at process startup and lives for the lifetime of the program. It is the single owner of every runtime singleton and the lifecycle boundary for a Dargus session.

### Core responsibilities

- Load configuration and expose resolved settings to every other component.
- Own the shared **reasoning LLM** and **embedding model** (the embedding model is shared with D-Base).
- Own the **ToolRegistry** and **SkillRegistry**.
- Own the **WorkspaceGuard**, which enforces the file-access boundary for the file Tools.
- Own the **conversation store** — every Agent's Conversation, keyed by `(session_id, agent)` — so the log survives agent churn and API turns (ADR-0003).
- Own the **AgentFactory**, which creates every Agent (Iris, Domain Experts, D4Expert). The factory injects runtime-provided dependencies, making the system testable: any dependency can be replaced with a fake or stub without changing Agent code.

## dargus.api — the sole interaction interface

`dargus.api` is the only interface through which external code (CLI, and in the future MCP servers or other frontends) may interact with `DargusRuntime`. No external code instantiates `Iris`, `DargusRuntime`, `DBase`, or other internal classes directly.

The API exposes a **non-interactive core**: every function takes plain arguments, performs no prompting, and returns plain data. Interactive behavior — menus, prompts, confirmation display, output formatting — lives in the CLI as thin wrappers around the core. Destructive operations are guarded at the API layer so the safety gate travels with the operation rather than depending on the caller.

## AgentFactory

`AgentFactory` creates every Agent and injects runtime-provided dependencies so that Agent code never reaches into the runtime directly. This is the sole creation path for Agents — no code may instantiate an Agent outside the factory.
