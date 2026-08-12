"""Packaging tests — the distribution seam (T7, #114).

Builds the real wheel (offline, ``pip wheel --no-deps``) and inspects its
contents: the runtime data files ship via package-data, the distribution is
``dargus-cli`` at ``0.19.0``, the hard dependency list is slim, and the
packaged default config carries no machine-specific test paths. The full
fresh-install boot is exercised by the smoke suite (smoke_package.py) and the
CI publish flow — this file stays offline and fast.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_DATA_FILES = [
    "dargus/config/dargus_config.yaml",
    "dargus/dbase/field_registry.yaml",
    "dargus/dbase/vocabularies.json",
]

#: Leftover scientific deps that must NOT be in the base install.
SCIENTIFIC_DEPS = [
    "pymc",
    "pytensor",
    "rdkit",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "PyMuPDF",
    "biopython",
]

#: The slim base dependencies the package actually imports (T7).
CORE_DEPS = ["litellm", "sentence-transformers", "prompt-toolkit", "rich", "PyYAML", "pyarrow"]


@pytest.fixture(scope="module")
def wheel() -> Path:
    """Build the wheel once per module and return its path."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", tmp],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"wheel build failed:\n{proc.stdout}\n{proc.stderr}"
        wheels = list(Path(tmp).glob("dargus_cli-*.whl"))
        assert wheels, "no dargus_cli wheel produced"
        yield wheels[0]


@pytest.fixture(scope="module")
def wheel_zip(wheel: Path):
    with zipfile.ZipFile(wheel) as z:
        yield z


def _metadata(wheel_zip) -> str:
    name = [n for n in wheel_zip.namelist() if n.endswith("METADATA")][0]
    return wheel_zip.read(name).decode("utf-8")


class TestWheelContents:
    def test_distribution_name_is_dargus_cli(self, wheel: Path):
        assert wheel.name.startswith("dargus_cli-0.19.0")

    def test_package_data_files_shipped(self, wheel_zip):
        names = set(wheel_zip.namelist())
        for data_file in EXPECTED_DATA_FILES:
            assert data_file in names, f"missing packaged data file: {data_file}"

    def test_vocabularies_pending_not_packaged(self, wheel_zip):
        names = set(wheel_zip.namelist())
        assert not any("vocabularies_pending" in n for n in names)

    def test_console_script_entry_point(self, wheel_zip):
        entry = [n for n in wheel_zip.namelist() if n.endswith("entry_points.txt")]
        assert entry, "no entry_points.txt in wheel"
        text = wheel_zip.read(entry[0]).decode("utf-8")
        assert "dargus = dargus.cli:main" in text


class TestDependencySlim:
    def test_base_dependencies_are_slim(self, wheel_zip):
        meta = _metadata(wheel_zip)
        for dep in SCIENTIFIC_DEPS:
            assert (
                dep.lower() not in meta.lower().split()
            ), f"scientific dep {dep} must not be a base dependency"

    def test_core_dependencies_declared(self, wheel_zip):
        meta = _metadata(wheel_zip)
        for dep in CORE_DEPS:
            assert dep.lower() in meta.lower(), f"core dep {dep} missing from Requires-Dist"

    def test_all_extra_pulls_the_union(self, wheel_zip):
        meta = _metadata(wheel_zip)
        for extra in (
            "docking",
            "modeling",
            "analysis",
            "literature",
            "admet",
            "singlecell",
            "genetics",
            "llm",
            "dev",
            "all",
        ):
            assert f"Provides-Extra: {extra}" in meta, f"extra {extra} missing"

    def test_scientific_stack_lives_in_extras(self, wheel_zip):
        meta = _metadata(wheel_zip)
        # pymc/pytensor/rdkit must appear somewhere in the extras, not the base.
        for dep in ("pymc", "pytensor", "rdkit"):
            assert dep in meta, f"{dep} missing entirely — should be in an extra"


class TestCleanDefaultConfig:
    def test_packaged_config_has_no_machine_specific_test_paths(self, wheel_zip):
        import yaml

        name = [n for n in wheel_zip.namelist() if n.endswith("config/dargus_config.yaml")][0]
        cfg = yaml.safe_load(wheel_zip.read(name).decode("utf-8"))
        assert "test" not in cfg, "packaged default config must not carry dev/test paths"
        assert "ingest_dir" not in str(cfg)
        assert "/home/chang" not in str(cfg)

    def test_packaged_config_keeps_model_defaults(self, wheel_zip):
        import yaml

        name = [n for n in wheel_zip.namelist() if n.endswith("config/dargus_config.yaml")][0]
        cfg = yaml.safe_load(wheel_zip.read(name).decode("utf-8"))
        assert cfg["models"]["reasoning"]["model"] == "deepseek-v4-pro"
        assert cfg["models"]["embedding"]["model"] == "all-MiniLM-L6-v2"
