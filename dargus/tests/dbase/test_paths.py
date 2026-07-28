"""Tests for D-Base path utilities."""

import os
import tempfile
from pathlib import Path

from dargus.dbase.paths import dbase_root, default_dargus_home, working_dbase


def test_default_dargus_home_uses_dot_dargus_under_home():
    old = os.environ.pop("DARGUS_HOME", None)
    try:
        home = os.environ.get("HOME", "/tmp")
        assert default_dargus_home() == Path(home) / ".dargus"
    finally:
        if old is not None:
            os.environ["DARGUS_HOME"] = old


def test_dargus_home_env_overrides_default():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DARGUS_HOME"] = tmp
        try:
            assert default_dargus_home() == Path(tmp)
            assert dbase_root() == Path(tmp) / "dbase"
        finally:
            del os.environ["DARGUS_HOME"]


def test_working_dbase_default():
    old = os.environ.pop("WORKING_DBASE", None)
    try:
        assert working_dbase() == "dbase"
    finally:
        if old is not None:
            os.environ["WORKING_DBASE"] = old


def test_working_dbase_env_overrides():
    os.environ["WORKING_DBASE"] = "dbase-test"
    try:
        assert working_dbase() == "dbase-test"
        assert dbase_root() == default_dargus_home() / "dbase-test"
    finally:
        del os.environ["WORKING_DBASE"]


def test_dbase_root_respects_both_env_vars():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DARGUS_HOME"] = tmp
        os.environ["WORKING_DBASE"] = "dbase-test"
        try:
            assert dbase_root() == Path(tmp) / "dbase-test"
        finally:
            del os.environ["DARGUS_HOME"]
            del os.environ["WORKING_DBASE"]
