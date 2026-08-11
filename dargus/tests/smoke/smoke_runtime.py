"""Standalone smoke: Runtime — bootstrap a real DargusRuntime from a temp config.

Pins the assembly invariant: ``bootstrap()`` parses a real YAML config (via
``DARGUS_CONFIG``), wires the reasoning LLM and embedding model, and returns a
healthy DargusRuntime whose workspace guard is rooted at the config's
workspace_root. Runs fully offline — the wiring is constructed, never called.

Contract: prints a ``PASS`` / ``FAIL`` / ``SKIP`` verdict line and exits 0 on
pass/skip, non-zero on fail. Run directly:  python smoke_runtime.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import yaml
from _bootstrap import ensure_dargus_on_path

ensure_dargus_on_path()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "workspace"
        root.mkdir()

        config = {
            "models": {
                "reasoning": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4",
                    "api_key_env": "SMOKE_LLM_KEY",
                    "temperature": 0.0,
                    "max_tokens": 128,
                },
                "embedding": {"provider": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
            },
            "workspace_root": str(root),
        }
        config_path = Path(tmp) / "dargus_config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        os.environ["SMOKE_LLM_KEY"] = "sk-smoke-fake"  # wiring only — never called

        from dargus.runtime.bootstrap import bootstrap
        from dargus.runtime.context import health_check

        rt = bootstrap(str(config_path))

        # The runtime assembled with both model dependencies and is healthy.
        assert rt.healthy is True
        assert rt.reasoning_llm is not None
        assert rt.embedding_model is not None
        assert health_check(rt) is True

        # The workspace guard is rooted at the config's workspace_root.
        assert rt.workspace_root == str(root)
        assert rt.workspace_guard.root == str(root)

        # Entry points are wired: factory + file tools.
        assert rt.agent_factory is not None
        assert rt.tool_cache is not None
        names = {t.name for t in rt.tool_registry.list_all()}
        assert {"read_file", "write_file"} <= names

        # The reasoning LLM carries the config's model id (wiring, not a call).
        from dargus.models.reasoning import LiteLLMBackend

        assert isinstance(rt.reasoning_llm._backend, LiteLLMBackend)
        assert rt.reasoning_llm._backend._model == "claude-sonnet-4"

        rt.shutdown()

    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — smoke scripts report any failure as FAIL
        print(f"FAIL: {exc.__class__.__name__}: {exc}")
        sys.exit(1)
