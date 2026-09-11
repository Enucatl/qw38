"""Host tests for OPT-061 real-input streaming component replay."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.opt061_component_replay import (
    CALIBRATION_LAYERS,
    FAMILIES,
    FIXTURE,
    GGUF_SHA,
    HELD_OUT_LAYERS,
    HEADER,
    INVENTORY,
    ITERATION,
    LLAMA_REV,
    NATIVE,
    OPT060_FIXTURE,
    REPORT,
    SCHEDULER,
    capture_identity,
    guards_intact,
    independent_rounds,
    inventory_tensor,
    opt059_reference_identity,
    paired_ffn_ms,
    rank_next_families,
    refuse_hot_as_production,
    reject_wrong_role_shape,
    replay_command,
    sha256_floats,
    tensor_name,
    useful_byte_rate_gbps,
    validate_capture_record,
)
from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt061_component_replay_contract.json"
MAKEFILE = ROOT / "Makefile"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_and_native_target() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-061")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert HEADER.is_file()
    assert NATIVE.is_file()
    assert REPORT.is_file()
    assert contract["task"] == "OPT-061"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["accept_hot_as_production"] is False
    assert contract["gate_up_down_calls_per_token"] == 64
    assert contract["paired_gate_up_is_64_not_128"] is True
    assert contract["llama_revision"] == LLAMA_REV
    assert contract["gguf_sha256"] == GGUF_SHA
    assert contract["families"] == list(FAMILIES)
    assert iteration["task"] == "OPT-061"
    assert iteration["claims_throughput"] is False
    assert iteration["target"] == "build/qw38-cuda-component-replay"
    assert iteration["diagnostics_make_target"] == "cuda-opt061-diagnostics"
    assert fixture["accept_hot_as_production"] is False
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "qw38-cuda-component-replay" in makefile
    assert "cuda-opt061-diagnostics" in makefile
    native = NATIVE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    assert "CacheMode" in header
    assert "kHot" in header and "kRotating" in header
    assert "1908404576" not in native
    assert "blk.0.ffn_gate.weight" in native
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert "maybe_capture_residual" in scheduler
    assert "maybe_capture_prompt_rows" in scheduler


def test_capture_hashes_and_inventory_names() -> None:
    fixture = _json(FIXTURE)
    record = copy.deepcopy(fixture["synthetic_capture"])
    record["hash"] = sha256_floats(record["values"])
    checked = validate_capture_record(record)
    assert checked["tensor_name"] == "blk.0.ffn_gate.weight"
    assert tensor_name(0, "ffn_gate") == "blk.0.ffn_gate.weight"
    tensor = inventory_tensor("blk.0.ffn_gate.weight")
    assert tensor["role"] == "ffn_gate"
    assert tensor["dtype"] == "Q4_K"
    assert int(tensor["shape"][0]) == 5120
    identity = capture_identity(
        gguf_sha=GGUF_SHA,
        token_generator=fixture["token_generator"],
        stage="d128",
        layers=list(CALIBRATION_LAYERS + HELD_OUT_LAYERS),
        prompt_rows=1,
        source="opt043_extended",
    )
    assert len(identity) == 64
    mutated = copy.deepcopy(record)
    mutated["hash"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        validate_capture_record(mutated)


def test_wrong_role_shape_rejected() -> None:
    good = _json(FIXTURE)["synthetic_capture"]
    reject_wrong_role_shape(good)
    q8_gate = copy.deepcopy(good)
    q8_gate["dtype"] = "Q8_0"
    with pytest.raises(ValueError, match="cannot be Q8_0"):
        reject_wrong_role_shape(q8_gate)
    mixer = {
        "role": "gdn_packed_qkv",
        "shape": [5120, 10240],
        "dtype": "Q4_K",
    }
    with pytest.raises(ValueError, match="Q8_0"):
        reject_wrong_role_shape(mixer)
    residual = {"role": "residual", "shape": [1, 5120], "dtype": "BF16"}
    with pytest.raises(ValueError, match="FP32"):
        reject_wrong_role_shape(residual)


def test_double_counted_paired_ffn_rejected() -> None:
    fixture = _json(FIXTURE)
    charged = paired_ffn_ms(fixture["synthetic_paired_ffn"]["enclosing_only"])
    assert charged["charged_ms"] == 4.0
    assert charged["gate_up_calls"] == 64
    assert charged["paired_not_128"] is True
    with pytest.raises(ValueError, match="double-counted"):
        paired_ffn_ms(fixture["synthetic_paired_ffn"]["double_counted"])


def test_independent_timing_rounds() -> None:
    fixture = _json(FIXTURE)
    rounds = independent_rounds(fixture["synthetic_rounds"])
    assert len(rounds) == 3
    assert all(row["observation_unit"] == "independent_round" for row in rounds)
    correlated = copy.deepcopy(fixture["synthetic_rounds"][:1])
    correlated[0]["observation_unit"] = "inner_launch"
    with pytest.raises(ValueError, match="independent round"):
        independent_rounds(correlated)
    duplicate = copy.deepcopy(fixture["synthetic_rounds"][:1] * 2)
    with pytest.raises(ValueError, match="not independent"):
        independent_rounds(duplicate)


def test_hot_results_refused_as_sole_production_acceptance() -> None:
    fixture = _json(FIXTURE)
    refuse_hot_as_production(fixture)
    with pytest.raises(ValueError, match="hot cache_mode"):
        refuse_hot_as_production({"accept_hot_as_production": True})
    with pytest.raises(ValueError, match="hot cache_mode"):
        refuse_hot_as_production({"status": "component_accepted", "cache_mode": "hot"})
    with pytest.raises(ValueError, match="hot cache_mode"):
        refuse_hot_as_production({"accepted_cache_modes": ["hot"]})


def test_guards_and_useful_byte_estimate() -> None:
    guard = bytes([0xA5] * 8)
    payload = guard + b"inner-data" + guard
    assert guards_intact(payload, guard, 8, 8)
    altered = bytes([0xFF]) + payload[1:]
    assert not guards_intact(altered, guard, 8, 8)
    rate = useful_byte_rate_gbps(9_625_927_680, 15.43)
    assert rate["dram_counter_claim"] is False
    assert rate["kind"] == "estimate"
    assert rate["useful_byte_rate_gbps"] > 0.0


def test_iteration_loop_product_and_no_throughput() -> None:
    iteration = load_contract("OPT-061")
    decode = iteration["workloads"]["decode-ffn"]
    product = loop_product(decode)
    assert product == 1 * 1 * (1 + 3) * 1 * 2 * 1
    assert iteration["aggregate_deadline_s"] == 300
    report = REPORT.read_text(encoding="utf-8")
    lowered = report.lower()
    assert "no throughput" in lowered or "claims no performance improvement" in lowered
    native = NATIVE.read_text(encoding="utf-8")
    assert "claims_throughput" in native


def test_reference_export_and_ranking() -> None:
    identity = opt059_reference_identity()
    assert identity["identity"]
    fixture = _json(FIXTURE)
    command = replay_command(
        "decode-ffn",
        "rotating",
        capture_key="abc",
        rows=fixture["prompt_row_indices"],
    )
    assert "--workload decode-ffn" in command
    assert "--cache-mode rotating" in command
    assert "--rows 0,1024,2048,4095" in command
    attribution = fixture["opt060_attribution_ms_per_decode_token"]
    ranked = rank_next_families(
        attribution,
        {"decode-ffn": 12.0, "decode-mixer": 6.0, "prompt-ffn": 600.0},
    )
    assert ranked[0]["family"] == "prompt-ffn"
    assert OPT060_FIXTURE.is_file()
    assert INVENTORY.is_file()
    assert fixture["microbench_disagreement"]
