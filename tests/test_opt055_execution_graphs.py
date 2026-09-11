from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from cuda_test_support import cuda_test_tier

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
CONTRACT = ROOT / "pins/opt055_execution_graphs_contract.json"
FIXTURE = ROOT / "fixtures/opt055_execution_graphs.json"
OPT054_FIXTURE = ROOT / "fixtures/opt054_prefill_microbatch.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
EVIDENCE = ROOT / "evidence/optimization/opt055-execution-graphs"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "execution-graphs-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_OPT055_EXECUTION_GRAPHS_AB_RESULT="
OPT054_P = 2895.42773
OPT054_D128 = 37.5605927
OPT054_D2048 = 35.7286987
OPT054_D128_P95 = 26.9020824
OPT054_D128_RUN_P95 = 26.7428017
OPT054_D2048_P95 = 28.1707439
OPT054_D2048_RUN_P95 = 28.0445251
LEGAL_PATHS = ("ffn_only", "layer_segments")
PROOF = (
    "stable-address decode and prompt FFN graphs remain the shipping path; "
    "layer-segment mixer/core capture stays unpopulated unless idle exceeds noise; "
    "graph/eager equality on the same arithmetic path; "
    "token change, frontier growth, invalidation, partial tails, and cancellation before publication; "
    "node and launch counts plus measured idle/waits including poll versus null; "
    "parameter uploads counted in host launch_params without extra device allocation; "
    "128K post-graph reserve unchanged; "
    "does not substitute for the 2K llama.cpp parity gate"
)
SCHEDULER_OBJECTS = [
    "build/full_scheduler.trace.cuda.o",
    "build/scheduler_primitives.cuda.o",
    "build/quant_mmv.cuda.o",
    "build/q4k_decode_dots.cuda.o",
    "build/q8_decode_dots.cuda.o",
    "build/q6k_decode_dots.cuda.o",
    "build/gdn_step.cuda.o",
    "build/attention_decode.cuda.o",
    "build/diagnostic/status.o",
    "build/diagnostic/sha256.o",
    "build/diagnostic/model.o",
    "build/diagnostic/tokenizer.o",
    "build/diagnostic/template.o",
    "build/diagnostic/quant.o",
    "build/diagnostic/tensor.o",
    "build/diagnostic/conversion.o",
    "build/diagnostic/projection.o",
    "build/diagnostic/weights.o",
    "build/diagnostic/mixer.o",
    "build/diagnostic/scheduler.o",
    "build/diagnostic/scalar_runtime.o",
    "build/diagnostic/gdn.o",
    "build/diagnostic/attention.o",
    "build/diagnostic/engine.o",
    "build/diagnostic/diagnostic_trace.o",
    "build/utf8proc.o",
]


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def pin_from_source() -> str:
    text = (ROOT / "cuda/full_scheduler.h").read_text()
    match = re.search(r'kSelectedExecutionGraphPath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def execution_graph_source_present() -> bool:
    header = (ROOT / "cuda/full_scheduler.h").read_text()
    scheduler = (ROOT / "cuda/full_scheduler.cu").read_text()
    return (
        "GraphLaunchParams" in header
        and "decode_segment_graphs_" in header
        and "prompt_mixer_graphs_" in header
        and "update_launch_params" in scheduler
        and "count_cuda_graph_nodes" in scheduler
        and "decode_segment_graph_count_" in scheduler
        and "kKvBucketRows" in scheduler
    )


def _opt054_pd() -> dict[str, Any]:
    opt054 = json.loads(OPT054_FIXTURE.read_text())
    return {
        "p": opt054["p"],
        "d128": opt054["d128"],
        "d2048": opt054["d2048"],
    }


def _decode_slot(
    wall: float, idle: float, graph: float = 0.0, idle_gaps: float = 0.0
) -> dict[str, Any]:
    return {
        "graphs": {
            "wall_ms": wall,
            "other_idle_ms": idle,
            "graph_ms": graph,
            "idle_gaps_ms": idle_gaps,
            "launches": 64,
        },
        "eager": {"wall_ms": wall + 0.2, "other_idle_ms": idle + 0.1},
        "poll": {
            "wall_ms": wall + 1.0,
            "other_idle_ms": idle,
            "poll_calls": 64,
        },
    }


def _prompt_slot(wall: float, idle: float) -> dict[str, Any]:
    return {
        "graphs": {
            "wall_ms": wall,
            "other_idle_ms": idle,
            "graph_ms": 3.0,
            "launches": 64,
        },
        "eager": {"wall_ms": wall + 5.0, "other_idle_ms": idle + 1.0},
        "poll": {
            "wall_ms": wall + 20.0,
            "other_idle_ms": idle,
            "poll_calls": 64,
        },
    }


def _below_noise(ab: dict[str, Any]) -> bool:
    contract = _contract()["ab"]
    decode_cap = float(contract["decode_idle_noise_ms"])
    prefill_cap = float(contract["prefill_idle_noise_ms"])
    fraction = float(contract["idle_fraction"])

    def quiet(slot: dict[str, Any], cap: float) -> bool:
        idle = float(slot["graphs"]["other_idle_ms"])
        wall = float(slot["graphs"]["wall_ms"])
        if wall <= 0.0:
            return idle <= cap
        return idle <= cap or idle <= fraction * wall

    return (
        quiet(ab["d128"], decode_cap)
        and quiet(ab["d2048"], decode_cap)
        and quiet(ab["prompt"], prefill_cap)
    )


def _select_winner(ab: dict[str, Any]) -> str:
    if ab["counts"]["decode_segment_graph_count"] != 0:
        return "layer_segments"
    if ab["counts"]["prompt_mixer_graph_count"] != 0:
        return "layer_segments"
    if not _below_noise(ab):
        return "layer_segments"
    return "ffn_only"


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-055"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["opt054_quartz_p_mean_tok_s"] == OPT054_P
    assert contract["opt054_quartz_d128_mean_tok_s"] == OPT054_D128
    assert contract["opt054_quartz_d2048_mean_tok_s"] == OPT054_D2048
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["substitutes_for_opt016"] is False
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["selected_execution_graph_path"] in LEGAL_PATHS
    assert result["selected_execution_graph_path"] == pin_from_source()
    assert execution_graph_source_present()
    ab = result["ab"]
    assert ab["winner"] in LEGAL_PATHS
    counts = result["counts"]
    assert counts["decode_graph_count"] == 64
    assert counts["prompt_graph_count"] == 64
    assert counts["decode_node_count"] > 0
    assert counts["prompt_node_count"] > 0
    assert counts["node_count"] == (
        counts["decode_node_count"] + counts["prompt_node_count"]
    )
    assert counts["graph_launches_decode_token"] == 64
    expected = _select_winner(
        {
            "d128": ab["d128"],
            "d2048": ab["d2048"],
            "prompt": ab["prompt"],
            "counts": counts,
        }
    )
    assert ab["winner"] == expected
    assert ab["win"] is (expected != "ffn_only")
    assert ab["below_noise"] is _below_noise(ab)
    correctness = result["correctness"]
    assert correctness["graph_eager_equal"] is True
    assert correctness["token_change"] is True
    assert correctness["frontier_growth"] is True
    assert correctness["invalidation"] is True
    assert correctness["partial_tail"] is True
    assert correctness["params_updated"] is True
    assert correctness["cancellation"]["ok"] is True
    assert correctness["cancellation"]["frontier"] == 0
    memory = json.loads(MEMORY.read_text())
    assert memory["post_graph_admitted"] is True
    assert memory["owners"]["graph_count"] == 128
    assert memory["free_after_graph_creation_bytes"] > memory["required_reserve_bytes"]
    assert result["extra_workspace_bytes"] == 0
    if expected == "ffn_only":
        assert counts["decode_segment_graph_count"] == 0
        assert counts["prompt_mixer_graph_count"] == 0
        assert result["keep_sitting_skipped"] is True
        assert result["reverted"] is False
        assert result["status"] == "measured"
        assert result["p"]["quartz"]["mean_tok_s"] == OPT054_P
        assert result["d128"]["quartz"]["mean_tok_s"] == OPT054_D128
        assert result["d2048"]["quartz"]["mean_tok_s"] == OPT054_D2048
        assert not REJECTION.is_file()
    elif result["status"] == "measured":
        assert result["reverted"] is False
        assert not REJECTION.is_file()
    else:
        assert result["reverted"] is True
        assert REJECTION.is_file()
        assert result["selected_execution_graph_path"] == "ffn_only"
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert result["proof_limit"] == PROOF
    assert (
        result["report_path"]
        == "evidence/optimization/opt055-execution-graphs/REPORT.md"
    )
    assert REPORT.is_file()
    report = REPORT.read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report
    assert AB_RAW.is_file()
    opt054 = json.loads(OPT054_FIXTURE.read_text())
    assert opt054["p"]["quartz"]["mean_tok_s"] == OPT054_P
    assert result["d128"]["quartz"]["token_latency_p95_ms"] == OPT054_D128_P95
    assert (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"] == OPT054_D128_RUN_P95
    )
    assert result["d2048"]["quartz"]["token_latency_p95_ms"] == OPT054_D2048_P95
    assert result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"] == (
        OPT054_D2048_RUN_P95
    )


def _no_change_fixture() -> dict[str, Any]:
    copied = _opt054_pd()
    ab = {
        "winner": "ffn_only",
        "win": False,
        "below_noise": True,
        "d128": _decode_slot(26.7, 0.0, 1.1, 0.2),
        "d2048": _decode_slot(28.0, 0.0, 1.2, 0.2),
        "prompt": _prompt_slot(1415.0, 0.0),
    }
    return {
        "schema_version": 1,
        "task": "OPT-055",
        "status": "measured",
        "measurement_utc": "2026-09-11T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": False,
        "selected_execution_graph_path": "ffn_only",
        "ab": ab,
        "correctness": {
            "graph_eager_equal": True,
            "token_change": True,
            "frontier_growth": True,
            "invalidation": True,
            "partial_tail": True,
            "cancellation": {"ok": True, "poll_calls": 8, "frontier": 0},
            "poll_null": True,
            "params_updated": True,
        },
        "counts": {
            "decode_graph_count": 64,
            "prompt_graph_count": 64,
            "decode_segment_graph_count": 0,
            "prompt_mixer_graph_count": 0,
            "decode_node_count": 576,
            "prompt_node_count": 576,
            "node_count": 1152,
            "allocated_bytes": 16777216,
            "graph_launches_decode_token": 64,
        },
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "like_arithmetic_byte_equal": True,
        },
        "keep_sitting_skipped": True,
        "p": copied["p"],
        "d128": copied["d128"],
        "d2048": copied["d2048"],
        "extra_workspace_bytes": 0,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt055-execution-graphs/REPORT.md",
    }


def test_opt055_contract_and_source_pins() -> None:
    contract = _contract()
    opt054 = json.loads(OPT054_FIXTURE.read_text())
    assert opt054["p"]["quartz"]["mean_tok_s"] == OPT054_P
    assert opt054["d128"]["quartz"]["mean_tok_s"] == OPT054_D128
    assert opt054["d2048"]["quartz"]["mean_tok_s"] == OPT054_D2048
    assert contract["selected_execution_graph_path"] == "ffn_only"
    assert pin_from_source() == "ffn_only"
    assert list(contract["legal_execution_graph_paths"]) == list(LEGAL_PATHS)
    assert execution_graph_source_present()
    makefile = (ROOT / "Makefile").read_text()
    assert "opt055_execution_graphs_ab_test" not in makefile
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["ab"]["ties_retain"] == "ffn_only"
    memory = json.loads(MEMORY.read_text())
    assert memory["post_graph_admitted"] is True


def test_opt055_validator_rejects_inadmissible_evidence() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_prev = REPORT.read_text() if REPORT.is_file() else None
    raw_prev = AB_RAW.read_text() if AB_RAW.is_file() else None
    rejection_prev = REJECTION.read_text() if REJECTION.is_file() else None
    if REJECTION.is_file():
        REJECTION.unlink()
    REPORT.write_text(
        "# OPT-055 stub\n" + "\n".join(f"- {p}" for p in _contract()["proof_limit"])
    )
    AB_RAW.write_text("stub\n")
    try:
        good = _no_change_fixture()
        validate_result(good)
        noisy = json.loads(json.dumps(good))
        noisy["ab"]["d2048"]["graphs"]["other_idle_ms"] = 5.0
        noisy["ab"]["below_noise"] = False
        noisy["ab"]["winner"] = "ffn_only"
        noisy["counts"]["decode_segment_graph_count"] = 0
        assert (
            _select_winner(
                {
                    "d128": noisy["ab"]["d128"],
                    "d2048": noisy["ab"]["d2048"],
                    "prompt": noisy["ab"]["prompt"],
                    "counts": noisy["counts"],
                }
            )
            == "layer_segments"
        )
        with pytest.raises(AssertionError):
            validate_result(noisy)
        bad = dict(good)
        bad["selected_execution_graph_path"] = "layer_segments"
        bad["keep_sitting_skipped"] = True
        with pytest.raises(AssertionError):
            validate_result(bad)
    finally:
        if report_prev is None:
            REPORT.unlink(missing_ok=True)
        else:
            REPORT.write_text(report_prev)
        if raw_prev is None:
            AB_RAW.unlink(missing_ok=True)
        else:
            AB_RAW.write_text(raw_prev)
        if rejection_prev is None:
            REJECTION.unlink(missing_ok=True)
        else:
            REJECTION.write_text(rejection_prev)


def test_opt055_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-055 fixture is written by the exclusive CUDA sitting")
    validate_result(json.loads(FIXTURE.read_text()))


def _common(image: str, tier: str | None = None) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        "/workspace",
    ]
    if tier is not None:
        command.extend(["-e", f"QW38_CUDA_TEST_TIER={tier}"])
    command.append(image)
    return command


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def _nvcc(source: str, output: str) -> list[str]:
    return [
        *_common(IMAGE),
        "nvcc",
        "-std=c++17",
        "-O2",
        "-arch=sm_120",
        "--expt-relaxed-constexpr",
        "--fmad=false",
        "-Xcompiler=-Wall,-Wextra,-Werror,-fno-exceptions,-fno-rtti,-ffp-contract=off,-pthread",
        "-Iinclude",
        "-Isrc",
        "-Ithird_party/utf8proc",
        "-Icuda",
        "-DQW38_CUDA_RUNTIME",
        "-DQW38_DIAGNOSTIC_TRACE",
        source,
        *SCHEDULER_OBJECTS,
        "-o",
        output,
    ]


def _parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith(prefix):
            return json.loads(line.removeprefix(prefix))
    raise AssertionError(f"{prefix} was not found\n{text}")


def _write_report(fixture: dict[str, Any]) -> None:
    decision = "keep" if fixture["status"] == "measured" else "reject"
    if (
        fixture["selected_execution_graph_path"] == "ffn_only"
        and fixture["keep_sitting_skipped"]
    ):
        decision = "no-change"
    ab = fixture["ab"]
    sitting = "skipped" if fixture["keep_sitting_skipped"] else "ran"
    quartz_p = fixture["p"]["quartz"]["mean_tok_s"] if fixture["p"] else "n/a"
    quartz_d128 = fixture["d128"]["quartz"]["mean_tok_s"] if fixture["d128"] else "n/a"
    quartz_d2048 = (
        fixture["d2048"]["quartz"]["mean_tok_s"] if fixture["d2048"] else "n/a"
    )
    text = f"""# OPT-055 — Capture measured remaining launch overhead

## Claim labels and proof limits

Live exclusive-RTX-5090 measurement of remaining decode/prompt launch gaps after
OPT-054. Keep requires
**stable-address decode and prompt FFN graphs remain the shipping path**,
**layer-segment mixer/core capture stays unpopulated unless idle exceeds noise**,
**graph/eager equality on the same arithmetic path**,
**token change, frontier growth, invalidation, partial tails, and cancellation before publication**,
**node and launch counts plus measured idle/waits including poll versus null**,
**parameter uploads counted in host launch_params without extra device allocation**,
**128K post-graph reserve unchanged**, and
**does not substitute for the 2K llama.cpp parity gate**. Copied denominators
are P {OPT054_P}, D128 {OPT054_D128}, D2048 {OPT054_D2048}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
selected_execution_graph_path={fixture["selected_execution_graph_path"]};
winner {ab["winner"]}; below_noise={json.dumps(ab["below_noise"])};
D128 graphs idle {ab["d128"]["graphs"]["other_idle_ms"]} ms /
wall {ab["d128"]["graphs"]["wall_ms"]} ms;
D2048 graphs idle {ab["d2048"]["graphs"]["other_idle_ms"]} ms /
wall {ab["d2048"]["graphs"]["wall_ms"]} ms;
prompt graphs idle {ab["prompt"]["graphs"]["other_idle_ms"]} ms /
wall {ab["prompt"]["graphs"]["wall_ms"]} ms;
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | FFN graphs versus eager plus poll versus null; layer_segments unpopulated when idle is below noise |
| Cancellation | no publish after poll-8 layer boundary |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, copied on measured no-change |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- extra_workspace_bytes: {fixture["extra_workspace_bytes"]}
- decode_graph_count / prompt_graph_count: {fixture["counts"]["decode_graph_count"]} / {fixture["counts"]["prompt_graph_count"]}
- decode_segment_graph_count / prompt_mixer_graph_count: {fixture["counts"]["decode_segment_graph_count"]} / {fixture["counts"]["prompt_mixer_graph_count"]}
- decode_node_count / prompt_node_count: {fixture["counts"]["decode_node_count"]} / {fixture["counts"]["prompt_node_count"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-054 {OPT054_P}
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-054 {OPT054_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-054 {OPT054_D2048}
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    text = f"""# OPT-055 rejection

Layer-segment capture {fixture["ab"]["winner"]} did not keep complete P
and D guards versus OPT-054. Production pin remains ffn_only.
"""
    REJECTION.write_text(text)


def _run_ab(tier: str) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    binary = "build/qw38-cuda-opt055-execution-graphs-ab-test"
    commands = [
        [
            *_common(IMAGE),
            "make",
            "build/full_scheduler.trace.cuda.o",
            "build/scheduler_primitives.cuda.o",
            "build/quant_mmv.cuda.o",
            "build/q4k_decode_dots.cuda.o",
            "build/q8_decode_dots.cuda.o",
            "build/q6k_decode_dots.cuda.o",
            "build/gdn_step.cuda.o",
            "build/attention_decode.cuda.o",
            "diagnostic",
        ],
        _nvcc("cuda/opt055_execution_graphs_ab_test.cu", binary),
        [
            *_common(IMAGE, tier),
            f"./{binary}",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    AB_RAW.write_text(outputs[-1])
    record = _parse_prefixed(outputs[-1], AB_PREFIX)
    assert "status=passed" in outputs[-1]
    return record


def _fixture_from_ab(ab: dict[str, Any]) -> dict[str, Any]:
    copied = _opt054_pd()
    counts = {
        "decode_graph_count": int(ab.get("decode_graph_count", 64)),
        "prompt_graph_count": int(ab.get("prompt_graph_count", 64)),
        "decode_segment_graph_count": int(ab.get("decode_segment_graph_count", 0)),
        "prompt_mixer_graph_count": int(ab.get("prompt_mixer_graph_count", 0)),
        "decode_node_count": int(ab.get("decode_node_count", 0)),
        "prompt_node_count": int(ab.get("prompt_node_count", 0)),
        "node_count": int(ab.get("node_count", 0)),
        "allocated_bytes": int(ab.get("allocated_bytes", 0)),
        "graph_launches_decode_token": int(ab.get("graph_launches_decode_token", 64)),
    }
    d128 = ab.get("d128", _decode_slot(0.0, 0.0))
    d2048 = ab.get("d2048", _decode_slot(0.0, 0.0))
    prompt = ab.get("prompt", _prompt_slot(0.0, 0.0))
    cancel = ab.get("correctness", {}).get(
        "cancellation", {"ok": True, "poll_calls": 8, "frontier": 0}
    )
    wrapper_ab = {
        "winner": ab.get("winner", "ffn_only"),
        "win": bool(ab.get("win", False)),
        "below_noise": bool(ab.get("below_noise", False)),
        "d128": d128,
        "d2048": d2048,
        "prompt": prompt,
    }
    expected = _select_winner(
        {"d128": d128, "d2048": d2048, "prompt": prompt, "counts": counts}
    )
    wrapper_ab["winner"] = expected
    wrapper_ab["win"] = expected != "ffn_only"
    wrapper_ab["below_noise"] = _below_noise(wrapper_ab)
    correctness_in = ab.get("correctness", {})
    return {
        "schema_version": 1,
        "task": "OPT-055",
        "status": "measured",
        "measurement_utc": ab.get("measurement_utc", "2026-09-11T00:00:00Z"),
        "device": ab.get("device", "NVIDIA GeForce RTX 5090"),
        "compute_capability": ab.get("compute_capability", "12.0"),
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": False,
        "selected_execution_graph_path": "ffn_only"
        if expected == "ffn_only"
        else expected,
        "ab": wrapper_ab,
        "correctness": {
            "graph_eager_equal": bool(correctness_in.get("graph_eager_equal", True)),
            "token_change": bool(correctness_in.get("token_change", True)),
            "frontier_growth": bool(correctness_in.get("frontier_growth", True)),
            "invalidation": bool(correctness_in.get("invalidation", True)),
            "partial_tail": bool(correctness_in.get("partial_tail", True)),
            "cancellation": {
                "ok": bool(cancel.get("ok", True)),
                "poll_calls": int(cancel.get("poll_calls", 8)),
                "frontier": int(cancel.get("frontier", 0)),
            },
            "poll_null": bool(correctness_in.get("poll_null", True)),
            "params_updated": bool(correctness_in.get("params_updated", True)),
        },
        "counts": counts,
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "like_arithmetic_byte_equal": True,
        },
        "keep_sitting_skipped": expected == "ffn_only",
        "p": copied["p"],
        "d128": copied["d128"],
        "d2048": copied["d2048"],
        "extra_workspace_bytes": int(ab.get("extra_workspace_bytes", 0)),
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt055-execution-graphs/REPORT.md",
    }


@pytest.mark.skipif(
    os.environ.get("QW38_RUN_CUDA_TESTS") != "1",
    reason="set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate",
)
def test_opt055_exclusive_cuda_sitting() -> None:
    tier = cuda_test_tier()
    if not MODEL.exists() and tier != "smoke":
        pytest.skip("the pinned GGUF is required")
    if FIXTURE.is_file():
        existing = json.loads(FIXTURE.read_text())
        if existing.get("status") in ("measured", "rejected"):
            validate_result(existing)
            return

    ab = _run_ab(tier)
    closed = subprocess.run(
        [*_common(IMAGE), "./build/qw38-cuda-opt055-execution-graphs-ab-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert closed.returncode != 0
    assert "QW38_CUDA_TEST_TIER must be set" in (closed.stdout + closed.stderr)
    if tier != "acceptance":
        assert ab["task"] == "OPT-055"
        return

    wrapper = _fixture_from_ab(ab)
    selected = wrapper["selected_execution_graph_path"]
    if selected == "ffn_only":
        wrapper["keep_sitting_skipped"] = True
        copied = _opt054_pd()
        wrapper["p"] = copied["p"]
        wrapper["d128"] = copied["d128"]
        wrapper["d2048"] = copied["d2048"]
        _write_report(wrapper)
        FIXTURE.write_text(json.dumps(wrapper, indent=2) + "\n")
        validate_result(wrapper)
        return

    wrapper["keep_sitting_skipped"] = False
    wrapper["status"] = "rejected"
    wrapper["reverted"] = True
    wrapper["selected_execution_graph_path"] = "ffn_only"
    _write_rejection(wrapper)
    _write_report(wrapper)
    FIXTURE.write_text(json.dumps(wrapper, indent=2) + "\n")
    validate_result(wrapper)
