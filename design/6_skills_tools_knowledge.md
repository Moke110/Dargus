# Skills & Tools Design

> Agents extend their reasoning through two capability layers. **Tools** are typed executable functions and **Skills** are reusable methodologies.

## Tool system

Tools are typed, executable capabilities registered in a `ToolRegistry`. Each Tool declares:

- name,
- parameter schema (`ToolParam` with name, type, required, default, description, enum),
- an `execute(...)` method.

Agents declare `PERMITTED_TOOLS` as a whitelist. The Act step will only invoke Tools on that list. Dargus v1.0.0 ships with core Tools such as:

- `dbase_query` — read evidence from D-Base,
- `dbase_write` — write evidence through the single-writer D-Base API,
- `dbase_update_status` — append lifecycle status transitions (supersede, retract) to the status sidecar,
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

## v1.0.0 scope

- ToolRegistry with typed Tool registration.
- SkillRegistry loading markdown+YAML Skills.
- BaseAgent permission validation between Skills and Tools.
- Routing Skill: field match + semantic embedding search using the active model fingerprint sidecar.
- `embedding` Tool with project-level HuggingFace model management.
- Hooks audit Tool calls and enforce allowlists.

## Out of Scope

- **Knowledge system.** A third capability layer alongside Tools and Skills: KnowledgeRetrievers providing domain reference information through a uniform interface (`search(query, domain, biological_level, top_k)` and `lookup(entity_id, entity_type)`), with sources such as `dbase` (structured summaries from D-Base) and `disease_rag` (disease descriptions and background), and a `PERMITTED_KNOWLEDGE` declaration controlling which sources an Agent may consult.
- **More professional skills, tools, and hooks.** Additional biomedical Tools and domain-specific Skills beyond the minimal set required for Ingest and Predict, and advanced Routing Skills (see `7_workflows.md`).
