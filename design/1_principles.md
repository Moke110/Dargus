# Dargus Principles

> Project Dargus is a **clinical efficacy prediction system for drug-development researchers**. It turns heterogeneous chemical, biological, and clinical evidence into traceable, quantitative predictions about whether a candidate drug will improve outcomes for a target disease.

## What Dargus is

Dargus has two core capabilities:

- **Ingest** — transform external data (files, tables, documents) into structured, validated evidence records stored in D-Base.
- **Predict** — assess all available evidence for a given drug–disease–endpoint combination and return a Dargus efficacy score (DES) with a Dargus confidence score (DCS), expressed as **DES ± DCS**.

## What Dargus is not

- Dargus is **not a literature search system**. It does not perform keyword searches against PubMed or return paper lists. Evidence must be ingested before it can inform predictions.
- Dargus is **not a general-purpose data-analysis platform**. It does not support free-form statistical exploration, ad-hoc visualization, or interactive data mining.
- Dargus is **not a knowledge-graph browser**. It does not expose drug–target–disease relationship networks for interactive navigation.
- Dargus is **not a clinical trial management system**. It does not handle patient recruitment, trial progress tracking, or regulatory submission workflows.

## Design principles

1. **One evidence store.** All evidence lives in D-Base; predictions never pull from hidden caches.
2. **Traceability by default.** Every prediction cites evidence and exposes its reasoning mode.
3. **Uncertainty is first-class.** Predictions are reported as a score pair (DES ± DCS), not a single point estimate.
4. **Human-in-the-loop.** Ingest and Predict pause for confirmation on duplicates or uncertain plans.
5. **Modular intelligence.** Agents, Tools, and Skills compose without rewriting orchestration.
6. **Multi-domain cooperation.** Domain Experts with complementary specializations delegate, assess, and synthesize evidence across molecular, cellular, animal, and clinical levels. No single Expert reasons outside its domain.

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
                └── Tool / Skill layer
                    └── D-Base       ← cumulative evidence store
```

- **DargusRuntime** is the program entry registered with the OS task manager. It owns configuration, lifecycle, and all singletons.
- **CLI** is the user-facing interface: one-shot commands plus the REPL. All CLI code submits requests through **dargus.api**, the sole interaction interface to the runtime; nothing instantiates Agents or runtime internals directly.
- **HookRegistry** manages callbacks that observe and influence the agent lifecycle.
- **AgentFactory** creates and terminates every Agent, including Iris, Domain Experts, and D4Expert.
- **Iris** is an Agent that dispatches tasks to other Agents through agent communication protocols; it does not own or manage their lifecycle.
- **Domain Experts** and **D4Expert** are Agents created by AgentFactory.
- All Agents call **Tools / Skills** to read from and write to **D-Base**.
- **D-Base** is the cumulative evidence store. All D-Base access happens through tools, skills, or hooks.

## Core workflows

### Ingest — from raw data to structured evidence

Ingest turns files, tables, and documents into validated D-Base records. It is the only way new evidence enters the system. After processing, Dargus presents a summary and any duplicate-review flags; the user confirms before records are written. Ingest is human-in-the-loop by design.

### Predict — from evidence to efficacy scores

Predict estimates the probability that a drug improves outcomes for a disease across one or more endpoints. Domain Experts retrieve relevant evidence from D-Base, produce assessments, and delegate when evidence falls outside their domain. D4Expert synthesizes all Expert reports into a FinalReport with a DES ± DCS score pair and supporting evidence citations.

## Universal prediction contract

All predictions return the same nested structure:

```json
{
  "drug_id": {
    "disease_id": {
      "endpoint": {
        "efficacy_score": 0.5,
        "confidence_score": 0.5,
        "supporting_records": ["ev_..."],
        "reasoning_mode": "Iris-expert",
        "confidence_level": "moderate"
      }
    }
  }
}
```

Every prediction must cite at least one supporting evidence ID from D-Base. The sole exception is a report with `confidence_level: insufficient_data`, which may cite zero records and leaves scores unset.

## Supporting records requirement

Every prediction must cite at least one supporting evidence ID from D-Base. Exception: a report with `confidence_level: insufficient_data` (scores unset) may cite zero supporting records.
