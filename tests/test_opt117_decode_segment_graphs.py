"""Host tests for OPT-117 decode-segment graph repair and keep/reject."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt117_decode_segment_graphs import (
    CANDIDATE,
    CONTRACT,
    DECODE_TOKENS,
    FIXTURE,
    ITERATION,
    NATIVE,
    PAIR_COUNT,
    PARENT,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    WARMUPS,
    family_plan,
    pair_order,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
NATIVE_SRC = ROOT / "cuda/opt117_decode_segment_graphs_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-117")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-117"
    assert contract["parent"] == "post113_selected"
    assert contract["parent_execution_graphs"] == PARENT
    assert contract["candidate"] == CANDIDATE
    assert contract["aa_warmups"] == WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["throughput_ratio_lower_bound"] == 1.0
    assert contract["non_target_point_ratio_min"] == 0.98
    assert contract["decode_p95_ratio_max"] == 1.05
    assert contract["parity_positions"] == [0, 127, 1023, 1024, 2047, 131071]
    assert iteration["diagnostics_make_target"] == "cuda-opt117-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt117-diagnostics" in makefile
    assert "qw38-cuda-opt117-decode-segment-graphs-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "enqueue_decode_layer_eager" in scheduler
    assert "capture_decode_segment_graph" in scheduler
    assert "session->gdn_convolution_" in scheduler
    assert "workspace->gdn_candidate_convolution_" in scheduler
    assert "--workload capture-repro|positions|pointer-swaps|" in native
    assert "independently_restored=true" in native
    assert "decode_segments8" in native
    assert "QW38_OPT117_DECODE_SEGMENT_GRAPHS_RESULT=" in native
    assert "QW38_OPT117_NATIVE_COUNTS=" in runner
    assert "QW38_OPT117_DECODE_SEGMENT_GRAPHS_RESULT=" in runner
    validate_future_keep_policy("OPT-117", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-117")
    capture = iteration["workloads"]["capture-repro"]
    positions = iteration["workloads"]["positions"]
    pointers = iteration["workloads"]["pointer-swaps"]
    cancel = iteration["workloads"]["cancellation"]
    same = iteration["workloads"]["same-math"]
    d128 = iteration["workloads"]["d128"]
    quality = iteration["workloads"]["quality"]
    assert loop_product(workload_for_mode(capture, "feedback")) == 1
    assert loop_product(workload_for_mode(positions, "feedback")) == 12
    assert loop_product(workload_for_mode(pointers, "feedback")) == 2
    assert loop_product(workload_for_mode(cancel, "feedback")) == 2
    assert loop_product(workload_for_mode(same, "feedback")) == 16
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    described = describe_plan("OPT-117", "feedback", iteration, "capture-repro")
    assert "phase=capture-repro" in described
    assert "loop_product=1" in family_plan("feedback", "capture-repro")
    assert pair_order(0) == "AB"
    assert pair_order(1) == "BA"
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "3 warmups plus 10 interleaved ab/ba" in proof
    assert "7200 is a ceiling" in proof


def test_keep_requires_opt115_gates() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-117")
    assert contract["target_workloads"] == ["d128", "d2048"]
    assert contract["non_target_workloads"] == ["p4096"]
    assert iteration["workloads"]["quality"]["aggregate_deadline_s"] == 7200
    assert "opt116_generated_v1" in contract["opt116_quality_contract"]
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt117-decode-segment-graphs")
    assert FIXTURE.name == "opt117_decode_segment_graphs.json"
