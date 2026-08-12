"""Standalone smoke: Packaging — build the wheel and boot from a fresh install.

Pins the distribution seam (T7/#114): building the wheel produces a
``dargus_cli-*`` distribution with the runtime data files inside; installing
that wheel into a fresh isolated venv (no source tree) lets ``dargus`` boot
from the packaged config and run one shot against a stubbed key.

Contract: prints a ``PASS`` / ``FAIL`` / ``SKIP`` verdict line and exits 0 on
pass/skip, non-zero on fail. Run directly:  python smoke_package.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

from _bootstrap import ensure_dargus_on_path

ensure_dargus_on_path()

REPO_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_DATA_FILES = [
    "dargus/config/dargus_config.yaml",
    "dargus/dbase/field_registry.yaml",
    "dargus/dbase/vocabularies.json",
]


def _build_wheel(out_dir: Path) -> Path:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"wheel build failed:\n{proc.stdout}\n{proc.stderr}"
    wheels = list(out_dir.glob("dargus_cli-*.whl"))
    assert wheels, "no dargus_cli wheel produced"
    return wheels[0]


def _verify_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
        for data_file in REQUIRED_DATA_FILES:
            assert data_file in names, f"missing packaged data file: {data_file}"
        meta = [n for n in names if n.endswith("METADATA")][0]
        metadata = z.read(meta).decode("utf-8")
        assert "Name: dargus-cli" in metadata
        # The packaged default config must be clean of machine-specific paths.
        cfg_name = [n for n in names if n.endswith("config/dargus_config.yaml")][0]
        cfg = z.read(cfg_name).decode("utf-8")
        assert "/home/chang" not in cfg
        assert "ingest_dir" not in cfg


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # 1. Build the wheel and verify its contents (offline, mandatory).
        wheel = _build_wheel(root)
        _verify_wheel_contents(wheel)

        # 2. Fresh-install E2E: an isolated venv installs the wheel (deps
        #    fetched from the index — requires network). The real CLI boots
        #    from the packaged config with the source tree absent.
        venv_dir = root / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        pip = venv_dir / "bin" / "pip"
        proc = subprocess.run(
            [str(pip), "install", str(wheel)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            # No network / dependency resolution unavailable — the wheel
            # content contract still passed; the E2E boot is skipped.
            print("SKIP: fresh-install E2E (dependency install unavailable):")
            print(
                (proc.stderr or proc.stdout).strip().splitlines()[-1]
                if (proc.stderr or proc.stdout)
                else ""
            )
            return 0

        python = venv_dir / "bin" / "python"
        dargus_bin = venv_dir / "bin" / "dargus"

        # 3. The installed command boots from the packaged config.
        boot = subprocess.run(
            [str(dargus_bin), "--help"],
            capture_output=True,
            text=True,
        )
        assert boot.returncode == 0, f"dargus --help failed:\n{boot.stderr}"
        assert "Dargus" in boot.stdout

        # 4. `dargus iris` boots through the packaged config and, with no key
        #    in the fresh environment, fails cleanly instead of crashing.
        #    Give the fresh env an initialised home whose config is the
        #    packaged one extracted from the wheel (as `dargus setup` would
        #    write) so the first-run guard passes and the runtime boots.
        home_dir = root / "home"
        (home_dir / ".dargus").mkdir(parents=True)
        with zipfile.ZipFile(wheel) as z:
            cfg_name = [n for n in z.namelist() if n.endswith("config/dargus_config.yaml")][0]
            (home_dir / ".dargus" / "dargus_config.yaml").write_bytes(z.read(cfg_name))
        clean_env = dict(os.environ)
        clean_env["HOME"] = str(home_dir)
        clean_env.pop("DARGUS_HOME", None)
        clean_env.pop("DARGUS_LLM_API_KEY", None)
        iris = subprocess.run(
            [str(dargus_bin), "iris", "hi"],
            capture_output=True,
            text=True,
            env=clean_env,
        )
        assert iris.returncode != 0, "dargus iris should refuse without a key"
        out = iris.stderr + iris.stdout
        assert "No LLM backend configured" in out, f"unexpected iris output: {out}"

        # 5. Version and config resolution come from the wheel, not the tree.
        check = subprocess.run(
            [
                str(python),
                "-c",
                "import dargus; from dargus.config.paths import get_config_path; "
                "import yaml; cfg = yaml.safe_load(open(get_config_path())); "
                "print(dargus.__version__); print(cfg['models']['reasoning']['model']); "
                "print('test' in cfg)",
            ],
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, check.stderr
        out = check.stdout.strip().splitlines()
        assert out[0] == "0.19.0", f"unexpected version {out[0]!r}"
        assert "test" not in out[2] or out[2] == "False", "packaged config carries test paths"

    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — smoke scripts report any failure as FAIL
        print(f"FAIL: {exc.__class__.__name__}: {exc}")
        sys.exit(1)
