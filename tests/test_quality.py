"""Unit tests for frozen QLT-001 policy boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.qw38_eval import EvalError, ScoreRequest, read_score_result

ROOT = Path(__file__).resolve().parents[1]


def test_quality_contract_is_versioned_and_has_all_cases() -> None:
    contract = json.loads((ROOT / "pins/quality_contract.json").read_text())
    assert contract["schema"] == "qw38.quality-contract"
    assert len(contract["tasks"]) == 16
    assert contract["thresholds"]["retrieval_frontier"] == 131072


def test_score_request_enforces_combined_frontier(tmp_path: Path) -> None:
    with pytest.raises(EvalError):
        ScoreRequest(
            tmp_path / "model",
            tuple([1] * 131072),
            tmp_path / "out",
            "rev",
            "dirty",
            (2,),
        )


def test_score_reader_rejects_top_two_ordering(tmp_path: Path) -> None:
    result = {
        "schema": "qw38.eval-result",
        "version": 1,
        "mode": "score",
        "status": "ok",
        "model": {},
        "tool": {},
        "runtime": {},
        "context_tokens": [1],
        "target_tokens": [2],
        "context_positions": [0],
        "target_positions": [1],
        "frontiers": {"context": 1, "final": 2},
        "steps": [
            {
                "position": 1,
                "target_token": 2,
                "log_probability": -1.0,
                "greedy_token": 3,
                "greedy_logit": 1.0,
                "runner_up_token": 4,
                "runner_up_logit": 2.0,
                "margin": -1.0,
            }
        ],
        "mean_nll": 1.0,
        "perplexity": 2.718281828459045,
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(EvalError):
        read_score_result(tmp_path)


def test_quality_contract_does_not_allow_post_run_exceptions() -> None:
    assert "near_tie" not in json.dumps(
        json.loads((ROOT / "pins/quality_contract.json").read_text())
    )
