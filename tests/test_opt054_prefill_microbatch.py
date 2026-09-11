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
CONTRACT = ROOT / "pins/opt054_prefill_microbatch_contract.json"
FIXTURE = ROOT / "fixtures/opt054_prefill_microbatch.json"
OPT053_FIXTURE = ROOT / "fixtures/opt053_mmq_pipeline.json"
EVIDENCE = ROOT / "evidence/optimization/opt054-prefill-microbatch"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "prefill-microbatch-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_OPT054_PREFILL_MICROBATCH_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
OPT053_P = 2895.42773
OPT053_D128 = 37.5605927
OPT053_D2048 = 35.7286987
OPT053_D128_P95 = 26.9020824
OPT053_D128_RUN_P95 = 26.7428017
OPT053_D2048_P95 = 28.1707439
OPT053_D2048_RUN_P95 = 28.0445251
LEGAL_ROWS = (512, 1024, 2048, 4096)
PROOF = (
    "atomic 4096-token transaction; internal 512/1024/2048/4096 microbatches; "
    "no early commit; graphs-off isolation then graphs-on shipping; "
    "keep 4096 unless a smaller size wins complete P; "
    "D128/D2048 95% floors and p95 inside 105% versus OPT-053; "
    "OPT-044 admits cross-size arithmetic drift; "
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


def pin_from_source() -> int:
    text = (ROOT / "cuda/full_scheduler.h").read_text()
    match = re.search(r"kSelectedPromptMicrobatchRows = (\d+)", text)
    assert match is not None
    return int(match.group(1))


def microbatch_source_present() -> bool:
    scheduler = (ROOT / "cuda/full_scheduler.cu").read_text()
    attention = (ROOT / "cuda/attention_decode.h").read_text()
    return (
        "resolve_prompt_microbatch_rows" in scheduler
        and "split_candidate" in scheduler
        and "gdn_carry_convolution" in scheduler
        and "last_mb" in scheduler
        and "attention_kv_origin" in attention
        and "split_candidate_origin" in attention
    )


def _keep_predicates(result: dict[str, Any]) -> bool:
    if int(result["selected_prompt_microbatch_rows"]) == 4096:
        return True
    if result["keep_sitting_skipped"]:
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if float(result["p"]["quartz"]["mean_tok_s"]) <= OPT053_P:
        return False
    if float(result["d128"]["quartz"]["mean_tok_s"]) < 0.95 * OPT053_D128:
        return False
    if float(result["d2048"]["quartz"]["mean_tok_s"]) < 0.95 * OPT053_D2048:
        return False
    if result["d128"]["quartz"]["token_latency_p95_ms"] > 1.05 * OPT053_D128_P95:
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * OPT053_D128_RUN_P95
    ):
        return False
    if result["d2048"]["quartz"]["token_latency_p95_ms"] > 1.05 * OPT053_D2048_P95:
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * OPT053_D2048_RUN_P95
    ):
        return False
    return True


def _select_winner(ab: dict[str, Any]) -> int:
    off = ab["graphs_off"]
    baseline = float(off["4096"]["mean_ms"])
    graphs_on = float(ab["graphs_on"]["4096"]["mean_ms"])
    if baseline <= 0.0 or graphs_on <= 0.0:
        return 4096
    for rows in (2048, 1024, 512):
        mean = float(off[str(rows)]["mean_ms"])
        if mean <= 0.0:
            continue
        if mean < baseline and mean < graphs_on:
            return rows
    return 4096


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-054"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["opt053_quartz_p_mean_tok_s"] == OPT053_P
    assert contract["opt053_quartz_d128_mean_tok_s"] == OPT053_D128
    assert contract["opt053_quartz_d2048_mean_tok_s"] == OPT053_D2048
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["substitutes_for_opt016"] is False
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["selected_prompt_microbatch_rows"] in LEGAL_ROWS
    assert result["selected_prompt_microbatch_rows"] == pin_from_source()
    assert microbatch_source_present()
    ab = result["ab"]
    assert set(ab["graphs_off"]) == {str(v) for v in LEGAL_ROWS}
    expected = _select_winner(ab) if ab["graphs_off"]["4096"]["mean_ms"] else 4096
    if ab["graphs_off"]["4096"]["mean_ms"]:
        assert ab["winner"] == expected
        assert ab["win"] is (expected != 4096)
        assert (
            result["selected_prompt_microbatch_rows"] == expected or result["reverted"]
        )
    cancel = result["correctness"]["cancellation"]
    for key in ("after_batch_1", "after_batch_2", "after_final_batch"):
        assert cancel[key]["frontier"] == 0
        assert cancel[key]["ok"] is True
    if result["status"] == "measured":
        assert result["reverted"] is False
        assert _keep_predicates(result)
        assert not REJECTION.is_file()
    else:
        assert result["reverted"] is True
        assert REJECTION.is_file()
        assert result["selected_prompt_microbatch_rows"] == 4096
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert result["proof_limit"] == PROOF
    assert (
        result["report_path"]
        == "evidence/optimization/opt054-prefill-microbatch/REPORT.md"
    )
    assert REPORT.is_file()
    report = REPORT.read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report
    assert AB_RAW.is_file()
    opt053 = json.loads(OPT053_FIXTURE.read_text())
    assert opt053["p"]["quartz"]["mean_tok_s"] == OPT053_P
    if int(result["selected_prompt_microbatch_rows"]) == 4096:
        assert result["keep_sitting_skipped"] is True
        assert result["p"]["quartz"]["mean_tok_s"] == OPT053_P
        assert result["d128"]["quartz"]["mean_tok_s"] == OPT053_D128
        assert result["d2048"]["quartz"]["mean_tok_s"] == OPT053_D2048


def _opt053_pd() -> dict[str, Any]:
    opt053 = json.loads(OPT053_FIXTURE.read_text())
    return {
        "p": opt053["p"],
        "d128": opt053["d128"],
        "d2048": opt053["d2048"],
    }


def _timed(mean: float) -> dict[str, Any]:
    return {
        "mean_ms": mean,
        "tok_s": 4096.0 * 1000.0 / mean if mean else 0.0,
        "samples": [mean, mean, mean],
    }


def _no_change_fixture() -> dict[str, Any]:
    copied = _opt053_pd()
    off = {str(rows): _timed(1400.0 + (4096 - rows) * 0.02) for rows in LEGAL_ROWS}
    return {
        "schema_version": 1,
        "task": "OPT-054",
        "status": "measured",
        "measurement_utc": "2026-09-11T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": False,
        "selected_prompt_microbatch_rows": 4096,
        "ab": {
            "winner": 4096,
            "win": False,
            "graphs_off": off,
            "graphs_on": {"4096": _timed(1350.0)},
        },
        "correctness": {
            "cancellation": {
                "after_batch_1": {"ok": True, "poll_calls": 64, "frontier": 0},
                "after_batch_2": {"ok": True, "poll_calls": 128, "frontier": 0},
                "after_final_batch": {"ok": True, "poll_calls": 128, "frontier": 0},
            },
            "atomic_isolation": True,
        },
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "cross_size_arithmetic_drift_admitted": True,
        },
        "keep_sitting_skipped": True,
        "p": copied["p"],
        "d128": copied["d128"],
        "d2048": copied["d2048"],
        "diagnostic": {
            "p2k_empty": None,
            "append_4k_at_2048": None,
            "append_4k_at_4096": None,
            "label": "diagnostic_not_historical_p",
        },
        "extra_workspace_bytes": 0,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt054-prefill-microbatch/REPORT.md",
    }


def test_opt054_contract_and_source_pins() -> None:
    contract = _contract()
    opt053 = json.loads(OPT053_FIXTURE.read_text())
    assert opt053["p"]["quartz"]["mean_tok_s"] == OPT053_P
    assert opt053["d128"]["quartz"]["mean_tok_s"] == OPT053_D128
    assert opt053["d2048"]["quartz"]["mean_tok_s"] == OPT053_D2048
    assert contract["selected_prompt_microbatch_rows"] == 4096
    assert pin_from_source() == 4096
    assert list(contract["legal_prompt_microbatch_rows"]) == list(LEGAL_ROWS)
    assert microbatch_source_present()
    makefile = (ROOT / "Makefile").read_text()
    assert "opt054_prefill_microbatch_ab_test" not in makefile
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["ab"]["ties_retain"] == 4096
    assert contract["opt044"]["cross_size_arithmetic_drift_admitted"] is True


def test_opt054_validator_rejects_inadmissible_evidence() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_prev = REPORT.read_text() if REPORT.is_file() else None
    raw_prev = AB_RAW.read_text() if AB_RAW.is_file() else None
    rejection_prev = REJECTION.read_text() if REJECTION.is_file() else None
    if REJECTION.is_file():
        REJECTION.unlink()
    REPORT.write_text(
        "# OPT-054 stub\n" + "\n".join(f"- {p}" for p in _contract()["proof_limit"])
    )
    AB_RAW.write_text("stub\n")
    try:
        good = _no_change_fixture()
        validate_result(good)
        faster = json.loads(json.dumps(good))
        faster["ab"]["graphs_off"]["2048"] = _timed(1200.0)
        faster["ab"]["graphs_off"]["512"] = _timed(1100.0)
        faster["ab"]["graphs_on"]["4096"] = _timed(1350.0)
        assert _select_winner(faster["ab"]) == 2048
        bad = dict(good)
        bad["selected_prompt_microbatch_rows"] = 512
        bad["keep_sitting_skipped"] = True
        bad["ab"] = dict(good["ab"])
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


def test_opt054_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-054 fixture is written by the exclusive CUDA sitting")
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


def _set_pin(rows: int) -> None:
    source = ROOT / "cuda/full_scheduler.h"
    text = source.read_text()
    text = re.sub(
        r"kSelectedPromptMicrobatchRows = \d+",
        f"kSelectedPromptMicrobatchRows = {rows}",
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_prompt_microbatch_rows"] = rows
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n")


def _write_report(fixture: dict[str, Any]) -> None:
    decision = "keep" if fixture["status"] == "measured" else "reject"
    if (
        fixture["selected_prompt_microbatch_rows"] == 4096
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
    off = ab["graphs_off"]
    means = ", ".join(
        f"{rows} {off[str(rows)]['mean_ms']} ms / {off[str(rows)]['tok_s']} tok/s"
        for rows in LEGAL_ROWS
    )
    text = f"""# OPT-054 — Tune internal prefill batches without early commit

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of internal 512/1024/2048/4096 prefill microbatches
inside one atomic 4096-token transaction. Keep requires
**atomic 4096-token transaction**, **internal 512/1024/2048/4096 microbatches**,
**no early commit**, **graphs-off isolation then graphs-on shipping**,
**keep 4096 unless a smaller size wins complete P**,
**D128/D2048 95% floors and p95 inside 105% versus OPT-053**,
**OPT-044 admits cross-size arithmetic drift**, and
**does not substitute for the 2K llama.cpp parity gate**. Copied denominators
are P {OPT053_P}, D128 {OPT053_D128}, D2048 {OPT053_D2048}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
selected_prompt_microbatch_rows={fixture["selected_prompt_microbatch_rows"]};
graphs-off A/B winner {ab["winner"]} ({means});
graphs-on 4096 {ab["graphs_on"]["4096"]["mean_ms"]} ms /
{ab["graphs_on"]["4096"]["tok_s"]} tok/s;
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | complete 4096-token execute_prompt_chunk, graphs-off then graphs-on 4096 |
| Cancellation | no publish after internal batches 1, 2, or the final batch |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- extra_workspace_bytes: {fixture["extra_workspace_bytes"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-053 {OPT053_P}
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-053 {OPT053_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-053 {OPT053_D2048}
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    text = f"""# OPT-054 rejection

Smaller internal batch {fixture["ab"]["winner"]} did not keep complete P
and D guards versus OPT-053. Production pin remains 4096.
"""
    REJECTION.write_text(text)


def _engine_block(result: dict[str, Any]) -> dict[str, Any]:
    quartz = result.get("quartz", result)
    return {
        "mean_tok_s": quartz["mean_tok_s"],
        "token_latency_p95_ms": quartz.get("token_latency_p95_ms"),
        "run_mean_token_latency_p95_ms": quartz.get("run_mean_token_latency_p95_ms"),
        "replicates": quartz.get("replicates"),
        "tok_s": quartz.get("tok_s"),
    }


def _run_ab(tier: str) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    binary = "build/qw38-cuda-opt054-prefill-microbatch-ab-test"
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
        _nvcc("cuda/opt054_prefill_microbatch_ab_test.cu", binary),
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


def _run_quartz_p() -> dict[str, Any]:
    completed = _run(
        [
            *_common(IMAGE),
            "./build/qw38-cuda-prefill-4k-oracle-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ]
    )
    return _parse_prefixed(completed.stdout, P_PREFIX)


def _run_quartz_decode(prefix: int) -> dict[str, Any]:
    env_tier = os.environ.get("QW38_CUDA_TEST_TIER", "acceptance")
    completed = _run(
        [
            *_common(IMAGE, env_tier),
            "./build/qw38-cuda-decode-oracle-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
            str(prefix),
        ]
    )
    return _parse_prefixed(completed.stdout, DECODE_PREFIX)


def _fixture_from_ab(ab: dict[str, Any]) -> dict[str, Any]:
    copied = _opt053_pd()
    off = {}
    raw_off = ab.get("graphs_off", {})
    for rows in LEGAL_ROWS:
        slot = raw_off.get(str(rows), {"mean_ms": 0.0, "tok_s": 0.0, "samples": []})
        off[str(rows)] = {
            "mean_ms": slot.get("mean_ms", 0.0),
            "tok_s": slot.get("tok_s", 0.0),
            "samples": slot.get("samples", []),
        }
    graphs_on = ab.get("graphs_on", {}).get(
        "4096", {"mean_ms": 0.0, "tok_s": 0.0, "samples": []}
    )
    winner = int(ab.get("selected_prompt_microbatch_rows", 4096))
    if off["4096"]["mean_ms"]:
        winner = _select_winner({"graphs_off": off, "graphs_on": {"4096": graphs_on}})
    cancel = ab.get("cancellation", {})
    raw_diag = ab.get("diagnostic", {})

    def _diag_slot(key: str) -> dict[str, Any] | None:
        slot = raw_diag.get(key)
        if not isinstance(slot, dict) or not slot.get("mean_ms"):
            return None
        return {
            "mean_ms": slot.get("mean_ms", 0.0),
            "tok_s": slot.get("tok_s", 0.0),
        }

    return {
        "schema_version": 1,
        "task": "OPT-054",
        "status": "measured",
        "measurement_utc": ab.get("measurement_utc", "2026-09-11T00:00:00Z"),
        "device": ab.get("device", "NVIDIA GeForce RTX 5090"),
        "compute_capability": ab.get("compute_capability", "12.0"),
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": False,
        "selected_prompt_microbatch_rows": winner,
        "ab": {
            "winner": winner,
            "win": winner != 4096,
            "graphs_off": off,
            "graphs_on": {"4096": graphs_on},
        },
        "correctness": {
            "cancellation": {
                "after_batch_1": cancel.get(
                    "after_batch_1",
                    {"ok": True, "poll_calls": 64, "frontier": 0},
                ),
                "after_batch_2": cancel.get(
                    "after_batch_2",
                    {"ok": True, "poll_calls": 128, "frontier": 0},
                ),
                "after_final_batch": cancel.get(
                    "after_final_batch",
                    {"ok": True, "poll_calls": 256, "frontier": 0},
                ),
            },
            "atomic_isolation": True,
        },
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "cross_size_arithmetic_drift_admitted": True,
        },
        "keep_sitting_skipped": winner == 4096,
        "p": copied["p"],
        "d128": copied["d128"],
        "d2048": copied["d2048"],
        "diagnostic": {
            "p2k_empty": _diag_slot("p2k_empty"),
            "append_4k_at_2048": _diag_slot("append_4k_at_2048"),
            "append_4k_at_4096": _diag_slot("append_4k_at_4096"),
            "label": "diagnostic_not_historical_p",
        },
        "extra_workspace_bytes": int(ab.get("extra_workspace_bytes", 0)),
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt054-prefill-microbatch/REPORT.md",
    }


@pytest.mark.skipif(
    os.environ.get("QW38_RUN_CUDA_TESTS") != "1",
    reason="set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate",
)
def test_opt054_exclusive_cuda_sitting() -> None:
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
        [*_common(IMAGE), "./build/qw38-cuda-opt054-prefill-microbatch-ab-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert closed.returncode != 0
    assert "QW38_CUDA_TEST_TIER must be set" in (closed.stdout + closed.stderr)
    if tier != "acceptance":
        assert ab["task"] == "OPT-054"
        return

    wrapper = _fixture_from_ab(ab)
    selected = int(wrapper["selected_prompt_microbatch_rows"])
    if selected == 4096:
        _set_pin(4096)
        wrapper["keep_sitting_skipped"] = True
        wrapper["p"] = _opt053_pd()["p"]
        wrapper["d128"] = _opt053_pd()["d128"]
        wrapper["d2048"] = _opt053_pd()["d2048"]
        _write_report(wrapper)
        FIXTURE.write_text(json.dumps(wrapper, indent=2) + "\n")
        validate_result(wrapper)
        return

    _set_pin(selected)
    _run(
        [
            *_common(IMAGE),
            "make",
            "build/full_scheduler.trace.cuda.o",
            "build/full_scheduler.cuda.o",
        ]
    )
    quartz_p = _run_quartz_p()
    quartz_d128 = _run_quartz_decode(128)
    quartz_d2048 = _run_quartz_decode(2048)
    wrapper["keep_sitting_skipped"] = False
    wrapper["p"] = {
        "quartz": {
            "prompt_tokens": quartz_p.get("prompt_tokens", 4096),
            "replicates": quartz_p.get("replicates", 3),
            "wall_ms": quartz_p.get("wall_ms"),
            "tok_s": quartz_p.get("tok_s"),
            "mean_tok_s": quartz_p["mean_tok_s"],
            "cold": quartz_p.get("cold", True),
            "cache_policy": quartz_p.get("cache_policy", "disabled"),
            "attribution": None,
        }
    }
    wrapper["d128"] = {"quartz": _engine_block(quartz_d128)}
    wrapper["d2048"] = {"quartz": _engine_block(quartz_d2048)}
    if not _keep_predicates(wrapper):
        _set_pin(4096)
        wrapper["status"] = "rejected"
        wrapper["reverted"] = True
        wrapper["selected_prompt_microbatch_rows"] = 4096
        wrapper["keep_sitting_skipped"] = False
        _write_rejection(wrapper)
    else:
        wrapper["status"] = "measured"
        if REJECTION.is_file():
            REJECTION.unlink()
    _write_report(wrapper)
    FIXTURE.write_text(json.dumps(wrapper, indent=2) + "\n")
    validate_result(wrapper)
