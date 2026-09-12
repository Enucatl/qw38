"""Host tests for OPT-109 persistent col-major GDN state admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt109_persistent_gdn_state import (
    CANDIDATE_ID,
    CONFIGS,
    CONTRACT,
    CONTROL_ID,
    DEVICE_LAYOUT,
    DEVICE_SHA,
    ELEMENT_COUNT,
    FIXTURE,
    ITERATION,
    LOGICAL_LAYOUT,
    LOGICAL_SHA,
    MIN_SAVING_MS,
    REPORT,
    VERDICT_KEYS,
    decide_independent_verdicts,
    dispatch_matches,
    family_plan,
    layout_roundtrip_host,
    planned_observation,
    replay_command,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
GDN_PATH = ROOT / "cuda/gdn_decode_path.cuh"
GDN_REC = ROOT / "cuda/gdn_decode_recurrence.cuh"
GDN_STEP = ROOT / "cuda/gdn_step.cu"
NATIVE = ROOT / "cuda/opt109_persistent_gdn_state_test.cu"
TOOL = ROOT / "tools/opt109_persistent_gdn_state.py"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"
OPT077_FIXTURE = ROOT / "fixtures/opt077_gdn_decode.json"
OPT094_FIXTURE = ROOT / "fixtures/opt094_gdn_replication.json"
OPT101_FIXTURE = ROOT / "fixtures/opt101_gdn_transposed.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-109")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    path = GDN_PATH.read_text(encoding="utf-8")
    rec = GDN_REC.read_text(encoding="utf-8")
    step = GDN_STEP.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert TOOL.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-109"
    assert contract["claims_throughput"] is True
    assert contract["opt074_coverage_unadmitted_is_blocker"] is False
    assert contract["control"] == CONTROL_ID
    assert contract["candidate"] == CANDIDATE_ID
    assert contract["min_saving_ms_per_token"] == MIN_SAVING_MS
    assert contract["fp32_recurrence"] is True
    assert contract["no_timed_relayout"] is True
    assert contract["retain_sequential_control"] is True
    assert contract["checkpoint_layout"] == LOGICAL_LAYOUT
    assert contract["state_layout_device"] == DEVICE_LAYOUT
    assert contract["logical_layout_sha256"] == LOGICAL_SHA
    assert contract["device_layout_sha256"] == DEVICE_SHA
    assert contract["independent_verdicts"] == list(VERDICT_KEYS)
    assert [row["id"] for row in contract["configurations"]] == [
        CONTROL_ID,
        CANDIDATE_ID,
    ]
    assert iteration["task"] == "OPT-109"
    assert iteration["diagnostics_make_target"] == "cuda-opt109-diagnostics"
    assert iteration["target"] == "tools/opt109_persistent_gdn_state.py"
    assert iteration["case_ids"] == [
        "parity",
        "pilot",
        "gdn",
        "quality",
        "state",
        "d128",
        "d2048",
        "prefill-guard",
        "memory",
    ]
    assert "cuda-opt109-diagnostics" in makefile
    assert "qw38-cuda-opt109-persistent-gdn-state-test" in makefile
    assert 'kSelectedGdnDecodePath[] = "sequential"' in path
    assert "persistent_transposed" in path
    assert "launch_gdn_decode_persistent_transposed_recurrence" in rec
    assert "gdn_record_timed_relayout" in rec
    assert "gdn_decode_uses_persistent_transposed" in step
    assert "launch_gdn_convert_session_recurrent" in step
    assert "prepare_recurrence_window" in step
    assert "persistent_transposed" in replay
    assert "--gdn-decode sequential|tile16|tile32|transposed" in replay
    assert "--gdn-decode sequential|tile16|tile32|transposed" in probe
    assert "QW38_OPT109_RESULT=" in runner
    assert "QW38_OPT109_NATIVE_COUNTS=" in runner
    assert "QW38_OPT109_RESULT=" in native
    fixture = _json(FIXTURE)
    for key in contract["required_fixture_keys"]:
        assert key in fixture
    assert fixture["shipping_gdn_decode"] == "sequential"
    assert fixture["production_kept"] is False
    assert fixture["claims_throughput"] is False


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-109")
    parity = iteration["workloads"]["parity"]
    pilot = iteration["workloads"]["pilot"]
    gdn = iteration["workloads"]["gdn"]
    quality = iteration["workloads"]["quality"]
    state = iteration["workloads"]["state"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    prefill = iteration["workloads"]["prefill-guard"]
    memory = iteration["workloads"]["memory"]
    assert loop_product(workload_for_mode(parity, "feedback")) == 2
    assert loop_product(workload_for_mode(pilot, "feedback")) == 8
    assert loop_product(workload_for_mode(pilot, "acceptance")) == 26
    assert loop_product(workload_for_mode(gdn, "feedback")) == 8
    assert loop_product(workload_for_mode(gdn, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "release")) == 1
    assert loop_product(workload_for_mode(state, "acceptance")) == 2
    assert loop_product(workload_for_mode(d128, "acceptance")) == 10
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 10
    assert loop_product(workload_for_mode(prefill, "acceptance")) == 2
    assert loop_product(workload_for_mode(memory, "acceptance")) == 1
    screen = family_plan("pilot", "feedback")
    assert screen["warmups"] == 1
    assert screen["samples"] == 3
    accept = family_plan("pilot", "acceptance")
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    described = describe_plan("OPT-109", "feedback", iteration, "parity")
    assert "phase=parity" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "zero timed per-layer relayout" in proof
    assert "do not retry opt-077/094/101" in proof
    assert iteration["modes"]["release"]["historical_oracles"] is False


def test_layout_roundtrip_same_count_and_hashes() -> None:
    layout = layout_roundtrip_host()
    assert layout["pass"] is True
    assert layout["same_count"] is True
    assert layout["restored_ok"] is True
    assert layout["layouts_differ"] is True
    assert layout["element_count_production"] == ELEMENT_COUNT
    path = GDN_PATH.read_text(encoding="utf-8")
    assert LOGICAL_SHA in path
    assert DEVICE_SHA in path
    assert 'kGdnRecurrentLogicalLayout[] = "row_major_fp32"' in path
    assert 'kGdnRecurrentDeviceLayoutTransposed[] = "col_major_fp32"' in path


def test_two_configs_retain_sequential_no_tile_sweep() -> None:
    assert [row["id"] for row in CONFIGS] == [CONTROL_ID, CANDIDATE_ID]
    assert CONFIGS[0]["gdn_decode"] == "sequential"
    assert CONFIGS[1]["gdn_decode"] == "persistent_transposed"
    ids = [row["id"] for row in CONFIGS]
    assert "tile16" not in ids
    assert "tile32" not in ids


def test_dispatch_persistent_grid_48x32() -> None:
    sequential = CONFIGS[0]
    persistent = CONFIGS[1]
    observed_t = {
        "path": "persistent_transposed",
        "launch": "prepare_recurrence_decode_transposed",
        "value_tile": 0,
        "warps_per_cta": 4,
        "grid_x": 48,
        "grid_y": 32,
    }
    observed_s = {
        "path": "sequential",
        "launch": "prepare_recurrence_window",
        "value_tile": 0,
        "warps_per_cta": 4,
        "grid_x": 48,
        "grid_y": 1,
    }
    assert dispatch_matches(observed_t, persistent) is True
    assert dispatch_matches(observed_s, sequential) is True
    command = replay_command(family_plan("gdn", "feedback"), persistent, None)
    assert "--gdn-decode" in command
    assert "persistent_transposed" in command


def test_reject_keeps_sequential_and_leaves_opt077_094_101() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality={"model_quality_pass": True},
        pilot={"positive": False, "mean_diff_ms": -0.05, "ci95_low": -0.1},
        component=None,
        engine_by_prefix={},
        prefill=None,
        state={"pass": True},
        memory={"pass": True},
        mode="acceptance",
    )
    assert decided["shipping_gdn_decode"] == "sequential"
    assert decided["production_kept"] is False
    assert decided["claims_throughput"] is False
    assert decided["independent_verdicts"][CONTROL_ID]["production_kept"] is True
    assert decided["independent_verdicts"][CANDIDATE_ID]["performance_pass"] is False
    assert _json(OPT077_FIXTURE)["shipping_gdn_decode"] == "sequential"
    assert _json(OPT094_FIXTURE)["shipping_gdn_decode"] == "sequential"
    assert _json(OPT101_FIXTURE)["shipping_gdn_decode"] == "sequential"
    assert 'kSelectedGdnDecodePath[] = "sequential"' in GDN_PATH.read_text(
        encoding="utf-8"
    )


def test_keep_requires_pilot_complete_gdn_and_both_prefixes() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality={"model_quality_pass": True},
        pilot={"positive": True, "mean_diff_ms": 0.2, "ci95_low": 0.05},
        component={
            "positive": True,
            "mean_diff_ms": 0.25,
            "ci95_low": 0.12,
            "ci95_high": 0.40,
            "timed_relayout_launches": 0,
            "decode_conversions": 0,
        },
        engine_by_prefix={
            "d128": {
                "control_mean_ms": 800.0,
                "candidate_mean_ms": 780.0,
                "guards": {"pass": True},
            },
            "d2048": {
                "control_mean_ms": 900.0,
                "candidate_mean_ms": 870.0,
                "guards": {"pass": True},
            },
        },
        prefill={"throughput_ratio": 1.0},
        state={"pass": True},
        memory={"pass": True},
        mode="acceptance",
    )
    assert decided["production_kept"] is True
    assert decided["shipping_gdn_decode"] == CANDIDATE_ID
    assert decided["claims_throughput"] is True


def test_missing_complete_gdn_does_not_keep() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality={"model_quality_pass": True},
        pilot={"positive": True, "mean_diff_ms": 0.2, "ci95_low": 0.05},
        component=None,
        engine_by_prefix={},
        prefill=None,
        state={"pass": True},
        memory={"pass": True},
        mode="acceptance",
    )
    assert decided["production_kept"] is False
    assert decided["shipping_gdn_decode"] == "sequential"


def test_timed_relayout_blocks_keep() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality={"model_quality_pass": True},
        pilot={"positive": True, "mean_diff_ms": 0.2, "ci95_low": 0.05},
        component={
            "positive": True,
            "mean_diff_ms": 0.25,
            "ci95_low": 0.12,
            "timed_relayout_launches": 96,
            "decode_conversions": 0,
        },
        engine_by_prefix={
            "d128": {"guards": {"pass": True}},
            "d2048": {"guards": {"pass": True}},
        },
        prefill={"throughput_ratio": 1.0},
        state={"pass": True},
        memory={"pass": True},
        mode="acceptance",
    )
    assert decided["production_kept"] is False
    assert decided["independent_verdicts"][CANDIDATE_ID]["performance_pass"] is False


def test_opt109_result_prefix_is_parsed() -> None:
    stdout = (
        'QW38_OPT109_RESULT={"task":"OPT-109","phase":"parity","keep":false,'
        '"observed_samples":1,"observed_candidates":2,"observed_shapes":1,'
        '"observed_warmups":0,"observed_tier":"correctness","pairs":1,'
        '"sample_ids":[0],"acceptance_executed":false}\n'
        'QW38_OPT109_NATIVE_COUNTS={"observed_samples":1}\n'
    )
    observed = parse_native_observation(stdout)
    assert observed["keep"] is False
    assert observed["observed_candidates"] == 2
    validate_future_keep_policy("OPT-109", load_contract("OPT-109"))


def test_skip_gpu_parity_does_not_invent_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt109_persistent_gdn_state as mod

    monkeypatch.setattr(mod, "FIXTURE", tmp_path / "opt109_persistent_gdn_state.json")
    monkeypatch.setattr(mod, "REPORT", tmp_path / "REPORT.md")
    payload = mod.run(
        "feedback",
        "parity",
        tmp_path / "run",
        skip_gpu=True,
        synthetic={"pass": True, "stdout": "status=passed"},
    )
    assert payload.get("parity")
    assert payload.get("claims_throughput") is False
    assert payload.get("production_kept") in {False, None}
    counts = (payload.get("parity") or payload).get("native_counts") or {}
    planned = planned_observation(family_plan("parity", "feedback"))
    assert counts.get("keep") is False
    assert planned["keep"] is False
    assert (tmp_path / "opt109_persistent_gdn_state.json").is_file()
