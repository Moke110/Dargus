"""Skill system — Markdown-defined multi-step methodologies with typed I/O schemas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Skill:
    name: str
    description: str
    version: str
    supported_levels: list[str]
    required_tools: list[str]
    input_schema: dict
    output_schema: dict
    timeout_ms: int = 60_000
    fallback: str = "skip"
    body: str = ""  # Markdown body after frontmatter

    def validate_tools(self, permitted_tools: list[str]) -> list[str]:
        """Return required_tools that are NOT in permitted_tools."""
        return [t for t in self.required_tools if t not in permitted_tools]


class SkillRegistry:
    """Load .md files from skills/ directory, indexed by YAML frontmatter."""

    def __init__(self, skills_dir: Path | str | None = None):
        if skills_dir is None:
            skills_dir = Path(__file__).resolve().parent / "skills"
        self._dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}
        self._load()

    def _load(self) -> None:
        if not self._dir.exists():
            return
        for md_file in self._dir.glob("*.md"):
            skill = self._parse_skill(md_file)
            if skill:
                self._skills[skill.name] = skill

    def _parse_skill(self, path: Path) -> Skill | None:
        content = path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return None
        parts = content.split("---", 2)
        if len(parts) < 3:
            return None
        frontmatter = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()
        return Skill(
            name=frontmatter.get("name", path.stem),
            description=frontmatter.get("description", ""),
            version=frontmatter.get("version", "0.1.0"),
            supported_levels=frontmatter.get("supported_levels", []),
            required_tools=frontmatter.get("required_tools", []),
            input_schema=frontmatter.get("input_schema", {}),
            output_schema=frontmatter.get("output_schema", {}),
            timeout_ms=frontmatter.get("timeout_ms", 60_000),
            fallback=frontmatter.get("fallback", "skip"),
            body=body,
        )

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found in registry")
        return self._skills[name]

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def list_by_level(self, biological_level: str) -> list[Skill]:
        return [s for s in self._skills.values() if biological_level in s.supported_levels]

    def list_by_tool(self, tool_name: str) -> list[Skill]:
        return [s for s in self._skills.values() if tool_name in s.required_tools]
