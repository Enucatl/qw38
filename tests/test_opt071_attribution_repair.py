"""Host tests for OPT-071 attribution window and capture-replay repair."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.opt071_attribution_repair import (
    ADAPTER,
    CONTRACT,
    DECODE_ATTENTION_CORES,
    DECODE_DOWNS,
    DECODE_GATE_UP_PAIRS,
    DECODE_GDN_CORES,
    DECODE_LOGITS,
    FIXTURE,
    HEADER,
    ITERATION,
    NATIVE_OPT060,
    NATIVE_REPLAY,
    OVERLAY,
    REPORT,
    SCHEDULER,
    accounting_split,
    cache_path,
    canonical_family,
    capture_identity,
    chargeable_records,
    expected_decode_counts,
    nested_interval_fixture,
    pool_overflow,
    rank_remaining_gaps,
    reject_wrong_role_layer_bundle,
    token_input_hash,
    two_token_epochs_share_origin,
    validate_call_counts,
    validate_capture_key,
    warmup_contamination,
)
from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_targets_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-071")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert OVERLAY.is_file()
    assert ADAPTER.is_file()
    assert HEADER.is_file()
    assert NATIVE_OPT060.is_file()
    assert NATIVE_REPLAY.is_file()
    assert contract["task"] == "OPT-071"
    assert contract["claims_throughput"] is False
    assert contract["no_new_profiling_backend"] is True
    assert iteration["task"] == "OPT-071"
    assert iteration["claims_throughput"] is False
    assert iteration["target"] == "build/qw38-cuda-opt060-engine-attribution-test"
    assert iteration["diagnostics_make_target"] == "cuda-opt071-diagnostics"
    assert iteration["modes"]["feedback"]["aggregate_deadline_s"] == 300
    assert iteration["modes"]["acceptance"]["aggregate_deadline_s"] == 300
    assert iteration["modes"]["feedback"]["repetitions"] == 1
    assert iteration["modes"]["acceptance"]["repetitions"] == 3
    assert "decode" in iteration["modes"]["feedback"]["tier_sequence"]
    assert "prefill" in iteration["modes"]["feedback"]["tier_sequence"]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert fixture["claims_throughput"] is False
    makefile = MAKEFILE.read_text(encoding="utf-8")
    overlay = OVERLAY.read_text(encoding="utf-8")
    adapter = ADAPTER.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    native = NATIVE_OPT060.read_text(encoding="utf-8")
    replay = NATIVE_REPLAY.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert "cuda-opt071-diagnostics" in makefile
    assert "qw38-cuda-opt060-engine-attribution-test" in makefile
    assert "qw38-cuda-component-replay" in makefile
    assert "begin_measured_window" in overlay
    assert "recording" in overlay
    assert "drain" in overlay
    assert "begin_measured_window" in adapter
    assert "parent_scope_id" in header
    assert "sequence_epoch" in header
    assert "record_sequence_epoch" in native or "sequence_epoch" in native
    assert "shipping_graph_wall" in native or "graph_wall" in native
    assert "capture_all_layers" in scheduler
    assert "maybe_capture_layer_input" in scheduler
    assert "maybe_capture_attn_output" in scheduler
    assert "EngineEventPool" in replay
    assert "capture_key" in replay
    assert "stale" in replay.lower() or "capture-key" in replay
    assert iteration["setup_host_commands"][0][1].endswith(
        "build_opt060_instrumented.sh"
    )


def test_nested_intervals_charge_enclosing_once() -> None:
    records = _json(FIXTURE)["synthetic_nested"]["records"]
    charged = chargeable_records(records)
    roles = {str(row["role"]) for row in charged}
    assert "ffn_mmv" in roles
    assert "ffn_gate" not in roles
    assert "ffn_gate_up_glu" not in roles
    nested = nested_interval_fixture(records)
    assert nested["summed_work_ms"] == 6.5
    assert nested["union_ms"] == 5.0
    assert nested["sum_exceeds_wall"] is True


def test_two_token_epochs_use_shared_origin() -> None:
    records = _json(FIXTURE)["two_token_epochs"]["records"]
    assert two_token_epochs_share_origin(records) is True
    reset = json.loads(json.dumps(records))
    reset[1]["start_ms"] = 0.2
    assert two_token_epochs_share_origin(reset) is False


def test_warmup_contamination_and_overflow_fail() -> None:
    fixture = _json(FIXTURE)
    assert warmup_contamination(fixture["warmup_contamination"]["records"]) is True
    assert warmup_contamination(fixture["synthetic_nested"]["records"]) is False
    assert pool_overflow(fixture["event_pool_overflow"]["records"]) is True
    assert pool_overflow(fixture["synthetic_nested"]["records"]) is False


def test_mul_mat_is_not_labeled_mixer() -> None:
    records = _json(FIXTURE)["mul_mat_not_mixer"]["records"]
    families = [canonical_family(row) for row in records]
    assert "mixer" not in families
    assert "mixer_projection" not in families
    assert families[0] == "ffn_down"
    assert families[1] == "logits_projection"


def test_expected_decode_family_counts() -> None:
    expected = expected_decode_counts(1)
    assert expected["ffn_gate_up_glu"] == DECODE_GATE_UP_PAIRS
    assert expected["ffn_down"] == DECODE_DOWNS
    assert expected["gdn_core"] == DECODE_GDN_CORES
    assert expected["attention_core"] == DECODE_ATTENTION_CORES
    assert expected["logits_projection"] == DECODE_LOGITS
    one_token = [
        {
            "engine": "quartz",
            "layer": layer,
            "phase": "decode",
            "role": "ffn_gate_up_glu",
            "attribution_role": "enclosing",
            "fused_member_ids": "ffn_gate,ffn_up,ffn_glu",
            "fused_member_count": 3,
            "complete_work_ms": 0.1,
        }
        for layer in range(64)
    ]
    one_token.extend(
        {
            "engine": "quartz",
            "layer": layer,
            "phase": "decode",
            "role": "ffn_down",
            "attribution_role": "member",
            "fused_member_count": 1,
            "complete_work_ms": 0.05,
        }
        for layer in range(64)
    )
    one_token.extend(
        {
            "engine": "quartz",
            "layer": layer,
            "phase": "decode",
            "role": "gdn_core",
            "attribution_role": "enclosing",
            "fused_member_count": 1,
            "complete_work_ms": 0.2,
        }
        for layer in range(48)
    )
    one_token.extend(
        {
            "engine": "quartz",
            "layer": 48 + index,
            "phase": "decode",
            "role": "attention_core",
            "attribution_role": "enclosing",
            "fused_member_count": 1,
            "complete_work_ms": 0.15,
        }
        for index in range(16)
    )
    one_token.append(
        {
            "engine": "quartz",
            "layer": -1,
            "phase": "decode",
            "role": "logits_projection",
            "attribution_role": "member",
            "fused_member_count": 1,
            "complete_work_ms": 0.3,
        }
    )
    checked = validate_call_counts(one_token, tokens=1)
    assert checked["ok"] is True


def test_capture_identity_cache_and_stale_key() -> None:
    fixture = _json(FIXTURE)
    tokens = [(42 + index * 997) % 248320 for index in range(129)]
    digest = token_input_hash(tokens)
    identity = capture_identity(
        gguf_sha=fixture["gguf_sha256"],
        token_generator=fixture["token_generator"],
        token_input_hash_hex=digest,
        stage="d128",
        source="production_graph",
        build_flags="QW38_DIAGNOSTIC_TRACE",
        selectors={"ffn_decode": "paired_staged", "q4_decode": "packed"},
        layer_role="all_64",
        staging="production_arithmetic",
        state="decode_d128",
        model_path="models/Qwen3.8-27B-Q4_K_M.gguf",
        prompt_rows=1,
    )
    assert len(identity) == 64
    reused = validate_capture_key(
        None, identity, {"capture_key": identity, "layers": []}
    )
    assert reused["recapture"] is False
    assert reused["action"] == "reuse"
    stale = fixture["stale_capture"]
    with pytest.raises(ValueError, match="stale capture key"):
        validate_capture_key(
            stale["provided_key"],
            stale["computed_key"],
            {"capture_key": stale["provided_key"]},
        )
    with pytest.raises(ValueError, match="bundle was not loaded"):
        validate_capture_key("c" * 64, "c" * 64, None)
    matched = validate_capture_key(
        identity, identity, {"capture_key": identity, "layers": []}
    )
    assert matched["action"] == "reuse"
    assert cache_path(identity).as_posix().endswith(f"{identity}/bundle.json")


def test_wrong_role_layer_bundle_rejected() -> None:
    fixture = _json(FIXTURE)
    reject_wrong_role_layer_bundle(fixture["good_layer_bundle"])
    with pytest.raises(ValueError, match="wrong role"):
        reject_wrong_role_layer_bundle(fixture["wrong_role_layer_bundle"])
    with pytest.raises(ValueError, match="missing layer"):
        reject_wrong_role_layer_bundle({"layers": []})


def test_accounting_split_and_honest_ranking() -> None:
    fixture = _json(FIXTURE)
    split = accounting_split(
        graph_wall_ms=22.0,
        eager_instrumented_wall_ms=23.5,
        eager_work_ms=18.0,
        attributed_union_ms=19.2,
        d2h_ms=0.2,
        commit_ms=0.1,
        prefix_ms=1.5,
        graph_create_ms=0.5,
    )
    assert split["shipping_graph_wall_ms"] == 20.0
    assert split["instrumentation_overhead_ms"] == 3.5
    assert split["prefix_excluded_ms"] == 1.5
    assert split["invented_overlap"] is False
    complete_counts = {"ok": True}
    ranked = rank_remaining_gaps(
        fixture["honest_gap"]["families"],
        fixture["honest_gap"]["accounting_complete"],
        complete_counts,
    )
    assert ranked["gap_attribution_complete"] is True
    assert ranked["ranked"][0]["family"] == "ffn_gate_up_glu"
    incomplete = rank_remaining_gaps(
        fixture["honest_gap"]["families"],
        fixture["honest_gap"]["accounting_incomplete"],
        complete_counts,
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["ranked"] == []
    invalid_counts = rank_remaining_gaps(
        fixture["honest_gap"]["families"],
        fixture["honest_gap"]["accounting_complete"],
        {"ok": False, "mismatches": {"ffn_down": {"observed": 2, "expected": 64}}},
    )
    assert invalid_counts["status"] == "incomplete"


def test_iteration_phases_and_no_oracles() -> None:
    iteration = load_contract("OPT-071")
    decode = iteration["workloads"]["decode"]
    assert decode["engines"] == 2
    assert decode["output_tokens"] == 16
    assert decode["prefix"] == 2048
    assert decode["execution_modes"] == 2
    assert loop_product(decode) == 64
    assert iteration["workloads"]["prefill"]["prompt"] == 4096
    assert "{repetitions}" in decode["args"]
    report = REPORT.read_text(encoding="utf-8").lower()
    assert "no throughput" in report or "no speedup" in report
    assert "eager" in report
    assert iteration["modes"]["release"]["historical_oracles"] is False
