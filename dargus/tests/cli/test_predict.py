"""Tests for dargus predict subcommand — argument parsing, dispatch, and API calls."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dargus.cli.main import main


def test_predict_subcommand_exists(capsys):
    """'dargus predict --help' shows usage and exits 0."""
    with pytest.raises(SystemExit) as exc:
        main(["predict", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--drugs" in captured.out
    assert "--disease" in captured.out


def test_predict_requires_drugs():
    """'dargus predict' without --drugs exits non-zero."""
    with pytest.raises(SystemExit) as exc:
        main(["predict"])
    assert exc.value.code != 0


def test_predict_requires_disease():
    """'dargus predict --drugs X' without --disease exits non-zero."""
    with pytest.raises(SystemExit) as exc:
        main(["predict", "--drugs", "aspirin"])
    assert exc.value.code != 0


def test_predict_parses_args_and_calls_api(capsys):
    """Full argument parse -> API dispatch for predict subcommand."""
    mock_result = {
        "aspirin": {
            "covid19": {
                "IC50": {
                    "efficacy_low": 0.3,
                    "efficacy_up": 0.7,
                    "confidence_level": "medium",
                    "reasoning_mode": "Iris-expert",
                    "supporting_records": [],
                    "expert_consensus": [],
                    "contradictions": [],
                    "data_gaps": [],
                }
            }
        }
    }
    with patch("dargus.api.predict", return_value=mock_result) as mock_predict:
        exit_code = main(["predict", "--drugs", "aspirin", "--disease", "covid19"])
        assert exit_code == 0
        mock_predict.assert_called_once_with(
            drug_ids=["aspirin"],
            disease_id="covid19",
            endpoints=None,
            max_rounds=5,
        )
    captured = capsys.readouterr()
    assert "aspirin:" in captured.out
    assert "0.300" in captured.out


def test_predict_multiple_drugs_comma_separated(capsys):
    """--drugs "aspirin,ibuprofen" splits into two drug IDs."""
    mock_result = {
        "aspirin": {"covid19": {}},
        "ibuprofen": {"covid19": {}},
    }
    with patch("dargus.api.predict", return_value=mock_result) as mock_predict:
        exit_code = main(["predict", "--drugs", "aspirin,ibuprofen", "--disease", "covid19"])
        assert exit_code == 0
        mock_predict.assert_called_once_with(
            drug_ids=["aspirin", "ibuprofen"],
            disease_id="covid19",
            endpoints=None,
            max_rounds=5,
        )


def test_predict_with_endpoints_and_max_rounds(capsys):
    """--endpoints and --max-rounds are forwarded to the API."""
    mock_result = {
        "aspirin": {
            "covid19": {
                "IC50": {"efficacy_low": 0.4, "efficacy_up": 0.9},
            }
        }
    }
    with patch("dargus.api.predict", return_value=mock_result) as mock_predict:
        exit_code = main(
            [
                "predict",
                "--drugs",
                "aspirin",
                "--disease",
                "covid19",
                "--endpoints",
                "IC50",
                "efficacy",
                "--max-rounds",
                "3",
            ]
        )
        assert exit_code == 0
        mock_predict.assert_called_once_with(
            drug_ids=["aspirin"],
            disease_id="covid19",
            endpoints=["IC50", "efficacy"],
            max_rounds=3,
        )
