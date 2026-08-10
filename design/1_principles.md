# Dargus Principles

> Project Dargus is a **clinical efficacy prediction system for drug-development researchers**. It turns heterogeneous chemical, biological, and clinical evidence into traceable, quantitative predictions about whether a candidate drug will improve outcomes for a target disease.

## What Dargus is

Dargus is a foundation of two things:

- **D-Base** — a single, cumulative evidence store that unifies evidence across every stage of drug development.
- **An agent harness** — Iris, Domain Experts, and D4Expert built on a shared Perceive → Reason → Act loop, with Tools and Skills as the only way they reach beyond reasoning.

The task-specific working methods that gave Dargus its concrete prediction surface (evidence ingestion and drug–disease–endpoint assessment) were removed in the 2026-08 cleanup to leave a clean codebase for the redo. The goal above is unchanged; the mechanisms that implement it are to be rebuilt.

## What Dargus is not

- Dargus is **not a literature search system**. It does not perform keyword searches against PubMed or return paper lists.
- Dargus is **not a general-purpose data-analysis platform**. It does not support free-form statistical exploration, ad-hoc visualization, or interactive data mining.
- Dargus is **not a knowledge-graph browser**. It does not expose drug–target–disease relationship networks for interactive navigation.
- Dargus is **not a clinical trial management system**. It does not handle patient recruitment, trial progress tracking, or regulatory submission workflows.

## Design principles

1. **One evidence store.** All evidence lives in D-Base; nothing pulls from hidden caches.
2. **Traceability by default.** Every prediction cites evidence.
3. **Uncertainty is first-class.** Predictions are reported as a score pair, not a single point estimate.
4. **Human-in-the-loop.** Evidence entry and prediction pause for confirmation on duplicates or uncertain plans.
5. **Modular intelligence.** Agents, Tools, and Skills compose without rewriting orchestration.
6. **Multi-domain cooperation.** Domain Experts with complementary specializations cooperate across molecular, cellular, animal, and clinical levels. No single Expert reasons outside its domain.

## Top-level architecture

```
CLI (one-shot commands + REPL)
└── dargus.api                       ← sole interaction interface
    └── DargusRuntime
        └── AgentFactory             ← creates all Agents
            ├── Iris                 ← top-level orchestrator Agent
            ├── Domain Experts
            └── D4Expert
                └── Tool / Skill layer
                    └── D-Base       ← cumulative evidence store
```

- **DargusRuntime** is the process-level container. It owns configuration, the conversation store, the Tool registry, the WorkspaceGuard, and the AgentFactory.
- **CLI** is the user-facing interface: one-shot commands plus the REPL. All CLI code submits requests through **dargus.api**, the sole interaction interface to the runtime; nothing instantiates Agents or runtime internals directly.
- **AgentFactory** creates every Agent, including Iris, Domain Experts, and D4Expert.
- **Iris** is an Agent that answers the user directly and can call Tools (currently file access). It is the primary interface for dialogue.
- **Domain Experts** and **D4Expert** are Agents created by AgentFactory. Their task-specific methods were removed; they currently exist as skeletons declaring their domain scope.
- All Agents call **Tools / Skills** to reach beyond reasoning. D-Base access goes through its store.
- **D-Base** is the cumulative evidence store.
