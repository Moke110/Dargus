# Dargus Design Overview

> Project Dargus is a **clinical efficacy prediction system for drug-development researchers**. It turns heterogeneous chemical, biological, and clinical evidence into traceable, quantitative predictions about whether a candidate drug will improve outcomes for a target disease.

## Core promise

Input a drug, a disease, and optionally endpoints. Dargus returns:

- a **Dargus efficacy score (DES)** and a **Dargus confidence score (DCS)** for the requested endpoint, expressed as **DES ± DCS**,
- the **supporting evidence records** behind every prediction,
- a transparent **reasoning trail** showing which Expert assessed what evidence.

The system is open-source, runs privately, and keeps all evidence in a single cumulative store: **D-Base**.

## Design principles

1. **One evidence store.** All evidence lives in D-Base; predictions never pull from hidden caches.
2. **Traceability by default.** Every prediction cites evidence and exposes its reasoning mode.
3. **Uncertainty is first-class.** Predictions are reported as a score pair (DES ± DCS), not a single point estimate.
4. **Human-in-the-loop.** Ingest and Predict pause for confirmation on duplicates or uncertain plans.
5. **Modular intelligence.** Agents, Tools, Skills, and Knowledge compose without rewriting orchestration.
6. **Multi-domain cooperation.** Domain Experts with complementary specializations delegate, assess, and synthesize evidence across molecular, cellular, animal, and clinical levels. No single Expert has to reason outside its domain.

## Top-level architecture

```
CLI (one-shot commands + REPL)
└── dargus.api                       ← sole interaction interface
    └── DargusRuntime
        ├── HookRegistry             ← lifecycle callbacks
        └── AgentFactory             ← creates and terminates all Agents
            ├── Iris                 ← top-level orchestrator Agent
            ├── Domain Experts
            └── D4Expert
                └── Tool / Skill / Knowledge layer
                    └── D-Base       ← cumulative evidence store
```

- **DargusRuntime** is the program entry registered with the OS task manager. It owns configuration, lifecycle, and all singletons.
- **CLI** is the user-facing interface: one-shot commands plus the REPL. All CLI code submits requests through **dargus.api**, the sole interaction interface to the runtime; nothing instantiates Agents or runtime internals directly.
- **HookRegistry** manages callbacks that observe and influence the agent lifecycle.
- **AgentFactory** creates and terminates every Agent, including Iris, Domain Experts, and D4Expert.
- **Iris** is an Agent that dispatches tasks to other Agents through agent communication protocols; it does not own or manage their lifecycle.
- **Domain Experts** and **D4Expert** are Agents created by AgentFactory.
- All Agents call **Tools / Skills / Knowledge** to read from and write to **D-Base**.
- **D-Base** is the cumulative evidence store. There is no separate D-BaseManager; all D-Base access happens through tools, skills, or hooks.

## v1.0.0

A complete, self-contained prediction appliance:

- D-Base stores structured evidence with vocabularies and dual deduplication.- Agents use a shared **Perceive → Reason → Act** harness with hook points.
- Ingest converts raw data into validated records.
- Predict routes requests to Experts, supports delegation, and produces a FinalReport.
- Benchmark evaluates Predict against held-out records without temporary D-Base copies.
- CLI provides a REPL and a minimal one-shot command surface (`iris`, `config`, `test`), all routed through `dargus.api`.

## Out of scope for v1.0.0

Multi-model ensembles, knowledge-graph routing, training pipelines, MCP server integration, and environment-aware deployments are deferred to post-v1.0.0 releases. See `x_prospect.md`.
