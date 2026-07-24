"""Tests for dargus benchmark subcommand — argument parsing, dispatch, and API calls."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dargus.cli.main import main


def test_benchmark_subcommand_exists(capsys):
    """'dargus benchmark --help' shows usage and exits 0."""
    with pytest.raises(SystemExit) as exc:
        main(["benchmark", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--strip" in captured.out
    assert "--output-dir" in captured.out


def test_benchmark_default_strip_empty(capsys):
    """Benchmark with no --strip defaults to empty dict."""
    mock_result = {
        "metrics": {"accuracy": 0.75, "precision": 0.80, "recall": 0.70, "f1": 0.75},
        "n_test": 100,
    }
    with patch("dargus.api.benchmark", return_value=mock_result) as mock_benchmark:
        exit_code = main(["benchmark"])
        assert exit_code == 0
        call_args = mock_benchmark.call_args
        assert call_args.kwargs["strip"] == {}
        assert call_args.kwargs["split"] is None
        assert call_args.kwargs["output_dir"] is None


def test_benchmark_with_strip_json(capsys):
    """--strip with JSON dict is parsed and forwarded."""
    mock_result = {
        "metrics": {"accuracy": 0.90, "precision": 0.85, "recall": 0.88, "f1": 0.86},
        "n_test": 50,
    }
    with patch("dargus.api.benchmark", return_value=mock_result) as mock_benchmark:
        exit_code = main(
            [
                "benchmark",
                "--strip",
                '{"holdout_ids": ["rec-001", "rec-002"], "drug_ids": ["aspirin"]}',
            ]
        )
        assert exit_code == 0
        mock_benchmark.assert_called_once_with(
            strip={"holdout_ids": ["rec-001", "rec-002"], "drug_ids": ["aspirin"]},
            split=None,
            output_dir=None,
        )


def test_benchmark_with_split_and_output_dir(capsys):
    """--split and --output-dir are forwarded to the API."""
    mock_result = {
        "metrics": {"accuracy": 0.80, "precision": 0.75, "recall": 0.78, "f1": 0.76},
        "n_test": 30,
    }
    with patch("dargus.api.benchmark", return_value=mock_result) as mock_benchmark:
        exit_code = main(
            [
                "benchmark",
                "--strip",
                "{}",
                "--split",
                '{"test_size": 0.2, "random_state": 42}',
                "--output-dir",
                "/tmp/reports",
            ]
        )
        assert exit_code == 0
        mock_benchmark.assert_called_once_with(
            strip={},
            split={"test_size": 0.2, "random_state": 42},
            output_dir="/tmp/reports",
        )


def test_benchmark_prints_metrics(capsys):
    """Benchmark output displays metric values correctly."""
    mock_result = {
        "metrics": {"accuracy": 0.8312, "precision": 0.7900, "recall": 0.8456, "f1": 0.8169},
        "n_test": 42,
    }
    with patch("dargus.api.benchmark", return_value=mock_result):
        exit_code = main(["benchmark", "--strip", "{}"])
        assert exit_code == 0

    captured = capsys.readouterr()
    assert "Benchmark Results:" in captured.out
    assert "0.8312" in captured.out
    assert "0.7900" in captured.out
    assert "0.8456" in captured.out
    assert "0.8169" in captured.out
    assert "42" in captured.out
