# Skills, Tools & Knowledge Design

> Agents extend their reasoning through three capability layers. **Tools** are typed executable functions, **Skills** are reusable methodologies, and **Knowledge** is reference information retrieved on demand.

## Tool system

Tools are typed, executable capabilities registered in a `ToolRegistry`. Each Tool declares:

- name,
- parameter schema (`ToolParam` with name, type, required, default, description, enum),
- an `execute(...)` method.

Agents declare `PERMITTED_TOOLS` as a whitelist. The Act step will only invoke Tools on that list. Dargus v1.0.0 ships with core Tools such as:

- `dbase_query` — read evidence from D-Base,
- `dbase_write` — write evidence through the single-writer D-Base API,
- `dbase_update_status` — append lifecycle status transitions (supersede, retract, holdout) to the status sidecar,
- `dbase_write_summary` — write or replace LLM summaries in the summary sidecar,
- `dbase_status` — report D-Base state,
- `delegate_to_expert` — hand records to another Expert,
- `embedding` — embed texts or batches with the project-level HuggingFace model; loads the model into the session `ToolCache` and supports `embed`, `test`, and `info` operations.

## Skill system

Skills are markdown+YAML frontmatter documents that describe multi-step methodologies. Each Skill declares:

- name and goal,
- `required_tools`,
- `supported_levels`,
- input and output schemas.

At startup, a BaseAgent validates that every Skill in `SUPPORTED_SKILLS` only requires Tools that the Agent is permitted to use.

## Routing Skill

A Routing Skill is a Skill whose job is to decide how an Expert retrieves relevant evidence from D-Base during Predict. In v1.0.0 the Routing Skill is:

> **Field match + semantic embedding search.**

The Expert filters D-Base by `biological_level`, `bg.drugs` entity IDs, `bg.disease_id`, and `y.type`, then asks the `embedding` tool to embed the query. It ranks the results by cosine similarity between the query vector and the vectors in the active embedding-model fingerprint sidecar (`sidecars/embeddings-{model_fp}.jsonl`). This keeps Iris from becoming a centralized data dispatcher: each Expert fetches what it needs, using its own domain lens.

## Knowledge system

KnowledgeRetrievers provide domain reference information through a uniform interface:

```python
search(query, domain, biological_level, top_k) -> list[KnowledgeItem]
lookup(entity_id, entity_type) -> KnowledgeItem | None
```

Examples of knowledge sources:

- `dbase` — structured summaries from D-Base,
- `disease_rag` — disease descriptions and background.

Agents declare `PERMITTED_KNOWLEDGE` to control which sources they may consult.

## v1.0.0 scope

- ToolRegistry with typed Tool registration.
- SkillRegistry loading markdown+YAML Skills.
- BaseAgent permission validation between Skills and Tools.
- Routing Skill: field match + semantic embedding search using the active model fingerprint sidecar.
- KnowledgeRetriever interface with D-Base and disease sources.
- `embedding` Tool with project-level HuggingFace model management.
- Hooks audit Tool calls and enforce allowlists.

## Beyond v1.0.0

Deferred: see `x_prospect.md`.
