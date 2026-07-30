# Runtime Structure

> Dargus runs inside a single **DargusRuntime**. The runtime is the program container registered with the OS task manager; it owns configuration, the hook registry, the dependency graph, and D-Base access through tools. Agents receive their dependencies from the runtime.

## DargusRuntime

`DargusRuntime` is created at process startup and lives for the lifetime of the program. It is the single owner of every runtime singleton and the lifecycle boundary for a Dargus session.

### Core responsibilities

- Load configuration and expose resolved settings to every other component.
- Own the **HookRegistry**, which stores hook registrations and executes them at named lifecycle points.
- Own the **AgentFactory**, which creates and terminates every Agent (Iris, Domain Experts, D4Expert). The factory injects runtime-provided dependencies, making the system testable: any dependency can be replaced with a fake or stub without changing Agent code.

## dargus.api — the sole interaction interface

`dargus.api` is the only interface through which external code (CLI, and in the future MCP servers or other frontends) may interact with `DargusRuntime`. No external code instantiates `Iris`, `DargusRuntime`, `DBase`, or other internal classes directly.

The API exposes a **non-interactive core**: every function takes plain arguments, performs no prompting, and returns plain data. Interactive behavior — menus, prompts, confirmation display, output formatting — lives in the CLI as thin wrappers around the core. Destructive operations are guarded at the API layer so the safety gate travels with the operation rather than depending on the caller.

## HookRegistry

`HookRegistry` stores all hook registrations and executes them at the named lifecycle points defined in `4_harness.md`. The runtime registers core hooks at startup. Skills may register additional hooks at load time. Hook execution is sequential; non-observer hooks that raise exceptions abort the remaining hooks at that point and propagate to the runtime (fail-closed).

## AgentFactory

`AgentFactory` creates and terminates every Agent. It injects runtime-provided dependencies so that Agent code never reaches into the runtime directly. This is the sole creation path for Agents — no code may instantiate an Agent outside the factory.
