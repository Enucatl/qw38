"""Host tests for OPT-101 transposed warp-column decode GDN admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt101_gdn_transposed import (
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
    opt099_expectation,
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
NATIVE = ROOT / "cuda/opt101_gdn_transposed_test.cu"
TOOL = ROOT / "tools/opt101_gdn_transposed.py"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"
OPT077_FIXTURE = ROOT / "fixtures/opt077_gdn_decode.json"
OPT094_FIXTURE = ROOT / "fixtures/opt094_gdn_replication.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fixture_shipping_is_sequential() -> bool:
    fixture = _json(FIXTURE)
    return fixture.get("shipping_gdn_decode") == "sequential"


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-101")
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
    assert contract["task"] == "OPT-101"
    assert contract["claims_throughput"] is True
    assert contract["opt074_coverage_unadmitted_is_blocker"] is False
    assert contract["control"] == CONTROL_ID
    assert contract["candidate"] == CANDIDATE_ID
    assert contract["min_saving_ms_per_token"] == MIN_SAVING_MS
    assert contract["fp32_recurrence"] is True
    assert contract["reduced_precision_forbidden"] is True
    assert contract["retain_sequential_control"] is True
    assert contract["opt077_replay_not_added_to_wall"] is True
    assert contract["checkpoint_layout"] == LOGICAL_LAYOUT
    assert contract["state_layout_device"] == DEVICE_LAYOUT
    assert contract["logical_layout_sha256"] == LOGICAL_SHA
    assert contract["device_layout_sha256"] == DEVICE_SHA
    assert contract["independent_verdicts"] == list(VERDICT_KEYS)
    assert [row["id"] for row in contract["configurations"]] == [
        CONTROL_ID,
        CANDIDATE_ID,
    ]
    assert iteration["task"] == "OPT-101"
    assert iteration["diagnostics_make_target"] == "cuda-opt101-diagnostics"
    assert iteration["target"] == "tools/opt101_gdn_transposed.py"
    assert iteration["case_ids"] == [
        "parity",
        "gdn",
        "quality",
        "d128",
        "d2048",
        "prefill-guard",
    ]
    assert "cuda-opt101-diagnostics" in makefile
    assert "qw38-cuda-opt101-gdn-transposed-test" in makefile
    assert 'kSelectedGdnDecodePath[] = "sequential"' in path
    assert "transposed" in path
    assert "kGdnDecodeLaunchVariantTransposed" in path
    assert "prepare_recurrence_decode_transposed" in rec
    assert "prepare_decode_qk_inverses" in rec
    assert "g_gdn_decode_qk_inverses" in rec
    assert "gdn_decode_uses_transposed" in step
    assert "launch_gdn_decode_transposed_recurrence" in step
    assert "prepare_recurrence_window" in step
    assert "transposed" in replay
    assert "--gdn-decode sequential|tile16|tile32|transposed" in replay
    assert "--gdn-decode sequential|tile16|tile32|transposed" in probe
    assert "QW38_OPT101_RESULT=" in runner
    assert "QW38_OPT101_NATIVE_COUNTS=" in runner
    assert "QW38_OPT101_RESULT=" in native
    assert fixture_shipping_is_sequential()
    fixture = _json(FIXTURE)
    for key in contract["required_fixture_keys"]:
        assert key in fixture
    assert fixture["shipping_gdn_decode"] == "sequential"
    assert fixture["production_kept"] is False
    assert fixture["claims_throughput"] is False


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-101")
    parity = iteration["workloads"]["parity"]
    gdn = iteration["workloads"]["gdn"]
    quality = iteration["workloads"]["quality"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    prefill = iteration["workloads"]["prefill-guard"]
    assert loop_product(workload_for_mode(parity, "feedback")) == 2
    assert loop_product(workload_for_mode(gdn, "feedback")) == 8
    assert loop_product(workload_for_mode(gdn, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "release")) == 1
    assert loop_product(workload_for_mode(d128, "acceptance")) == 10
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 10
    assert loop_product(workload_for_mode(prefill, "acceptance")) == 2
    screen = family_plan("gdn", "feedback")
    assert screen["warmups"] == 1
    assert screen["samples"] == 3
    accept = family_plan("gdn", "acceptance")
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    described = describe_plan("OPT-101", "feedback", iteration, "parity")
    assert "phase=parity" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "grid (48,32)" in proof
    assert "opt-077 replay" in proof
    assert "row-major" in proof
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


def test_two_configs_retain_sequential_and_tiles() -> None:
    assert [row["id"] for row in CONFIGS] == [CONTROL_ID, CANDIDATE_ID]
    assert CONFIGS[0]["gdn_decode"] == "sequential"
    assert CONFIGS[1]["gdn_decode"] == "transposed"
    path = GDN_PATH.read_text(encoding="utf-8")
    rec = GDN_REC.read_text(encoding="utf-8")
    assert "tile16" in path and "tile32" in path
    assert "prepare_recurrence_decode_tiled" in rec
    assert "prepare_recurrence_window" in (ROOT / "cuda/gdn_step.cu").read_text(
        encoding="utf-8"
    )


def test_dispatch_transposed_grid_48x32() -> None:
    sequential = CONFIGS[0]
    transposed = CONFIGS[1]
    observed_t = {
        "path": "transposed",
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
    assert dispatch_matches(observed_t, transposed) is True
    assert dispatch_matches(observed_s, sequential) is True
    wrong = dict(observed_t)
    wrong["grid_y"] = 4
    assert dispatch_matches(wrong, transposed) is False
    command = replay_command(family_plan("gdn", "feedback"), transposed, None)
    assert "--gdn-decode" in command
    assert "transposed" in command


def test_reject_keeps_sequential_and_leaves_opt077_094() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality={"model_quality_pass": True},
        component={
            "positive": False,
            "mean_diff_ms": -0.2,
            "ci95_low": -0.4,
            "ci95_high": -0.05,
        },
        engine_by_prefix={},
        prefill=None,
        mode="acceptance",
    )
    assert decided["shipping_gdn_decode"] == CONTROL_ID
    assert decided["production_kept"] is False
    assert decided["claims_throughput"] is False
    assert decided["independent_verdicts"][CONTROL_ID]["production_kept"] is True
    assert decided["independent_verdicts"][CANDIDATE_ID]["performance_pass"] is False
    opt077 = _json(OPT077_FIXTURE)
    opt094 = _json(OPT094_FIXTURE)
    assert opt077["shipping_gdn_decode"] == "sequential"
    assert opt094["shipping_gdn_decode"] == "sequential"
    assert 'kSelectedGdnDecodePath[] = "sequential"' in GDN_PATH.read_text(
        encoding="utf-8"
    )


def test_keep_requires_complete_gdn_and_both_prefixes() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality={"model_quality_pass": True},
        component={
            "positive": True,
            "mean_diff_ms": 0.25,
            "ci95_low": 0.12,
            "ci95_high": 0.40,
        },
        engine_by_prefix={
            "d128": {
                "control_mean_ms": 800.0,
                "candidate_mean_ms": 780.0,
                "regression_upper_ms": -10.0,
                "guards": {"pass": True},
            },
            "d2048": {
                "control_mean_ms": 900.0,
                "candidate_mean_ms": 870.0,
                "regression_upper_ms": -12.0,
                "guards": {"pass": True},
            },
        },
        prefill={"throughput_ratio": 1.0},
        mode="acceptance",
    )
    assert decided["production_kept"] is True
    assert decided["shipping_gdn_decode"] == CANDIDATE_ID
    assert decided["claims_throughput"] is True


def test_opt099_expectation_does_not_add_opt077_wall() -> None:
    expectation = opt099_expectation()
    assert expectation["family"] == "gdn_core"
    assert expectation["opt077_replay_not_added_to_wall"] is True
    assert expectation["used_as_expectation_only"] is True
    proof = " ".join(load_contract("OPT-101")["proof_limit"]).casefold()
    assert "do not add opt-077 replay values to wall time" in proof


def test_opt101_result_prefix_is_parsed() -> None:
    stdout = (
        'QW38_OPT101_RESULT={"task":"OPT-101","phase":"parity","keep":false,'
        '"observed_samples":1,"observed_candidates":2,"observed_shapes":1,'
        '"observed_warmups":0,"observed_tier":"correctness","pairs":1,'
        '"sample_ids":[0],"acceptance_executed":false}\n'
        'QW38_OPT101_NATIVE_COUNTS={"observed_samples":1}\n'
    )
    observed = parse_native_observation(stdout)
    assert observed["keep"] is False
    assert observed["observed_candidates"] == 2
    assert observed["observed_tier"] == "correctness"
    validate_future_keep_policy("OPT-101", load_contract("OPT-101"))


def test_skip_gpu_parity_does_not_invent_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt101_gdn_transposed as mod

    monkeypatch.setattr(mod, "FIXTURE", tmp_path / "opt101_gdn_transposed.json")
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
    assert (tmp_path / "opt101_gdn_transposed.json").is_file()
