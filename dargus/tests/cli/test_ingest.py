"""Tests for dargus ingest subcommand — argument parsing, dispatch, and API calls."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dargus.cli.main import main


def test_ingest_subcommand_exists(capsys):
    """'dargus ingest --help' shows usage and exits 0."""
    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--datadir" in captured.out


def test_ingest_requires_datadir():
    """'dargus ingest' without --datadir exits non-zero."""
    with pytest.raises(SystemExit) as exc:
        main(["ingest"])
    assert exc.value.code != 0


def test_train_subcommand_exists(capsys):
    """'dargus train --help' shows usage (backward compat alias for ingest)."""
    with pytest.raises(SystemExit) as exc:
        main(["train", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--datadir" in captured.out


def test_ingest_parses_args_and_calls_api(capsys):
    """Full argument parse -> API dispatch for ingest subcommand."""
    mock_report = _MockReport(n_records=42, n_skipped=3, dbase_size=99)

    with patch("dargus.api.ingest", return_value=mock_report) as mock_ingest:
        exit_code = main(["ingest", "--datadir", "/tmp/testdata"])
        assert exit_code == 0
        mock_ingest.assert_called_once_with("/tmp/testdata", reset=False)

    captured = capsys.readouterr()
    assert "Records added: 42" in captured.out
    assert "Duplicates skipped: 3" in captured.out
    assert "Global D-Base size: 99" in captured.out


def test_ingest_with_reset_flag(capsys):
    """--reset flag is forwarded to the API."""
    mock_report = _MockReport(n_records=10, n_skipped=0, dbase_size=10)

    with patch("dargus.api.ingest", return_value=mock_report) as mock_ingest:
        exit_code = main(["ingest", "--datadir", "/tmp/testdata", "--reset"])
        assert exit_code == 0
        mock_ingest.assert_called_once_with("/tmp/testdata", reset=True)


def test_train_alias_dispatches_to_ingest(capsys):
    """'dargus train' is a backward-compat alias for 'dargus ingest'."""
    mock_report = _MockReport(n_records=5, n_skipped=1, dbase_size=7)

    with patch("dargus.api.ingest", return_value=mock_report) as mock_ingest:
        exit_code = main(["train", "--datadir", "/tmp/traindata"])
        assert exit_code == 0
        mock_ingest.assert_called_once_with("/tmp/traindata", reset=False)


def test_ingest_has_disease_kb_flag(capsys):
    """--disease-kb-dir flag is accepted by the parser."""
    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--disease-kb-dir" in captured.out


def test_ingest_has_reset_flag(capsys):
    """--reset flag is accepted by the parser."""
    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--reset" in captured.out


class _MockReport:
    """Lightweight stub matching the attribute interface needed by the CLI handler."""

    def __init__(self, n_records: int, n_skipped: int, dbase_size: int) -> None:
        self.n_records = n_records
        self.n_skipped = n_skipped
        self.dbase_size = dbase_size
