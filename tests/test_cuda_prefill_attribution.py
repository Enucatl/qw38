from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/cuda_prefill_attribution_contract.json"
FIXTURE = ROOT / "fixtures/cuda_prefill_attribution.json"
CATEGORIES = (
    "embedding",
    "gdn",
    "attention",
    "ffn_mmq",
    "logits",
    "commit_sync",
    "graph",
    "other_idle",
)


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(contract["required_keys"])
    assert result["schema_version"] == 1 and result["task"] == "OPT-014"
    assert result["status"] == "measured"
    for key in contract["metadata_required"]:
        assert isinstance(result[key], str) and result[key].strip()
        assert "unknown" not in result[key].lower()
        assert "placeholder" not in result[key].lower()
    assert result["compute_capability"] == "12.0"
    assert result["toolkit"] == contract["toolkit"]
    assert result["pinned_image"] == contract["pinned_image"]
    assert result["prompt_tokens"] == contract["prompt_tokens"] == 2048
    assert result["evaluated_tokens"] == 2048
    assert result["chunk_count"] == 1
    assert result["prompt_graph_launches"] == 0
    assert result["cold"] is True
    assert result["cache_policy"] == "disabled"
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    proof = result["proof_limit"].casefold()
    assert "live cuda-event attribution" in proof
    assert "2048-token cold prefill" in proof
    assert "no nsight capture" in proof
    assert "not a throughput gate" in proof
    assert "not llama.cpp parity" in proof
    categories = result["categories_ms"]
    assert isinstance(categories, dict) and set(categories) == set(CATEGORIES)
    for name in CATEGORIES:
        value = categories[name]
        assert isinstance(value, (int, float))
        assert value >= 0.0
    assert categories["graph"] >= 0.0
    attributed = sum(float(categories[name]) for name in CATEGORIES)
    assert result["attributed_sum_ms"] == pytest.approx(attributed, rel=1e-6, abs=1e-6)
    assert math.isclose(
        attributed,
        float(result["wall_ms"]),
        rel_tol=contract["rel_tol"],
        abs_tol=contract["abs_tol_ms"],
    )
    assert isinstance(result["tok_s"], (int, float)) and result["tok_s"] > 0.0
    assert isinstance(result["wall_ms"], (int, float)) and result["wall_ms"] > 0.0


def test_cuda_prefill_attribution_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE.read_text()))
    chapter = (ROOT / "docs" / "51-runtime-timing-and-nvtx.md").read_text().casefold()
    for term in [
        "prefill attribution",
        "commit/sync",
        "other/idle",
        "without requiring a separate nsight capture",
    ]:
        assert term in chapter


def test_cuda_prefill_attribution_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []
    for mutate in (
        lambda x: x["categories_ms"].pop("logits"),
        lambda x: x["categories_ms"].__setitem__("graph", None),
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x["categories_ms"].__setitem__("embedding", x["wall_ms"] + 1000.0),
        lambda x: x.__setitem__("prompt_tokens", 2052),
        lambda x: x.__setitem__("prompt_graph_launches", 64),
        lambda x: x.__setitem__("driver", "placeholder"),
        lambda x: x.__setitem__(
            "proof_limit",
            "live CUDA-event attribution; 2048-token cold prefill; not a throughput gate",
        ),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)


def test_cuda_prefill_attribution_native() -> None:
    pytest.skip(
        "the live diagnostic now emits OPT-020 split JSON; "
        "the exclusive RTX 5090 gate lives in tests/test_opt020_prefill_split.py"
    )


def _regenerate() -> None:
    raise SystemExit(
        "usage error: historical OPT-014 fixture is frozen; "
        "live OPT-020 regeneration is tests/test_opt020_prefill_split.py "
        "--regenerate-fixture"
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--regenerate-fixture"]:
        raise SystemExit("usage: test_cuda_prefill_attribution.py --regenerate-fixture")
    _regenerate()
