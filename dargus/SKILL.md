# Dargus — Claude Code Skill

## Description

A clinical-efficacy prediction system that reads multi-level evidence and outputs drug-disease effect sizes with confidence intervals.

## Quick start

```
/dargus scan-v4 --drugs LRRK2-IN-1 --disease "Alzheimer's disease" --datadir ./data
/dargus status <project_id>
```

## Commands

- `/dargus scan-v4 --drugs <drugs> --disease <disease> [--datadir <path>]` — run a target-disease efficacy scan.
- `/dargus status <project_id>` — show project status.

## MCP Tools

When running as an MCP server, Dargus exposes:

- `dargus_start_project`
- `dargus_ingest_data`
- `dargus_search_literature`
- `dargus_predict`
- `dargus_query_dbase`
- `dargus_status`

## Configuration

See `dargus/config/dargus_config.yaml`.

## Disclaimer

Outputs are for research purposes only and do not constitute clinical advice.
