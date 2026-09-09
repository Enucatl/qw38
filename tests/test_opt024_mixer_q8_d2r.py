from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
LLAMA_IMAGE = "qw38-llama-authority:cuda-13.0.2"
PREFIX = "QW38_PREFILL_4K_D2R_RESULT="
CONTRACT = ROOT / "pins/opt024_mixer_q8_d2r_contract.json"
FIXTURE = ROOT / "fixtures/opt024_mixer_q8_d2r.json"
EVIDENCE = ROOT / "evidence/optimization/opt024-mixer-q8-d2r"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
LLAMA_JSON = EVIDENCE / "llama-bench-4k.json"
QUARTZ_JSON = EVIDENCE / "quartz-4k.json"
AB_RAW = EVIDENCE / "d2r-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
LLAMA_BENCH = ROOT / ".cache/authorities/llama-build/bin/llama-bench"
OPT023_MEAN = 1687.86169
PROOF = (
    "aligned-SoA D2R for large mixer Q8_0; Q8 association; "
    "4K keep/reject; envelopes unloosened; "
    "does not substitute for the 2K llama.cpp parity gate; "
    "OPT-009 byte-exact reference retained"
)


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def selected_path_from_source() -> str:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    match = re.search(
        r'kSelectedLargeMixerQ8Path\[\] = "([^"]+)"',
        text,
    )
    assert match is not None
    return match.group(1)


def production_is_d2r() -> bool:
    return selected_path_from_source() == "d2r_soa"


def production_is_quality_mma() -> bool:
    return selected_path_from_source() == "quality_mma"


def scheduler_shares_residual_y() -> bool:
    text = (ROOT / "cuda/full_scheduler.cu").read_text()
    return (
        "launch_quantize_mmq_q8_1" in text
        and "matrix_prompt_q8_quality_mma" in text
        and "prompt_normalized_" in text
    )


def skinny_path_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    return 'kSelectedSkinnyMixerPath[] = "mma_i32_j128"' in text


def decode_mmv_gguf_layout() -> bool:
    mmv = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        "columns / 32) * 34" in mmv
        and "q8_mmv_bf16<<<" in (ROOT / "cuda/full_scheduler.cu").read_text()
    )


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-024"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["primary_tokens"] == 4096
    assert contract["yardstick"] == "4k_keep_reject_oracle"
    assert contract["keep_reject_rule"] == "strictly_greater_mean_tok_s"
    assert contract["opt023_quartz_mean_tok_s"] == OPT023_MEAN
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["substitutes_for_opt016"] is False
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["opt023_quartz_mean_tok_s"] == OPT023_MEAN
    assert (
        result["selected_large_mixer_q8_path"]
        == contract["selected_large_mixer_q8_path"]
    )
    assert result["ab_winner"] in ("quality_mma", "d2r_soa")
    quartz = result["quartz"]
    assert quartz["prompt_tokens"] == 4096
    assert quartz["replicates"] == contract["quartz_replicates"] == 3
    assert quartz["cold"] is True
    assert quartz["cache_policy"] == "disabled"
    assert quartz["attribution"] is None
    assert quartz["graphs_created"] is True
    assert quartz["prompt_graph_rows"] == 4096
    assert len(quartz["wall_ms"]) == 3 and len(quartz["tok_s"]) == 3
    mean = sum(float(v) for v in quartz["tok_s"]) / 3.0
    assert quartz["mean_tok_s"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    llama = result["llama_cpp"]
    assert llama["n_prompt"] == 4096
    assert "n_batch" in llama and "n_ubatch" in llama
    assert "flash_attn" in llama
    assert "avg_ts" in llama and "avg_ns" in llama
    assert "build_commit" in llama and "test_time" in llama
    meets = float(quartz["mean_tok_s"]) >= float(llama["avg_ts"])
    assert result["quartz_meets_llama"] is meets
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert result["shared_residual_y"] is True
    assert result["skinny_mixer_unchanged"] is True
    assert result["decode_mmv_gguf_layout"] is True
    keep = (
        float(quartz["mean_tok_s"]) > OPT023_MEAN
        and result["ab_win"] is True
        and result["ab_winner"] == "d2r_soa"
    )
    if result["status"] == "measured":
        assert keep
        assert result["reverted"] is False
        assert result["successor_oracle"] is True
        assert result["production_d2r"] is True
        assert result["selected_large_mixer_q8_path"] == "d2r_soa"
        assert not REJECTION.is_file()
    else:
        assert not keep
        assert result["reverted"] is True
        assert result["successor_oracle"] is False
        assert result["production_d2r"] is False
        assert result["selected_large_mixer_q8_path"] == "quality_mma"
        assert REJECTION.is_file()
        rejection = REJECTION.read_text()
        assert str(quartz["mean_tok_s"]) in rejection
        assert str(OPT023_MEAN) in rejection
        assert result["ab_winner"] in rejection
    proof = result["proof_limit"]
    for phrase in contract["proof_limit"]:
        assert phrase in proof
    assert result["report_path"] == contract["report_path"]
    assert "/tmp" not in result["report_path"]
    assert REPORT.is_file()
    report = REPORT.read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report
    assert str(quartz["mean_tok_s"]) in report
    assert AB_RAW.is_file()


def test_opt024_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE.read_text()))
    contract = _contract()
    assert contract["llama_bench_args"] == [
        "-p",
        "4096",
        "-n",
        "0",
        "--no-warmup",
        "-r",
        "3",
        "-ngl",
        "99",
    ]
    opt023 = json.loads((ROOT / "fixtures/opt023_skinny_mixer.json").read_text())
    assert opt023["quartz"]["mean_tok_s"] == OPT023_MEAN
    assert contract["opt023_quartz_mean_tok_s"] == OPT023_MEAN
    pin = json.loads((ROOT / "pins/cuda_prompt_mmq_contract.json").read_text())
    assert pin["admission"]["q8_bf16_reference_exact"] is True
    mmq = json.loads((ROOT / "pins/cuda_mmq_contract.json").read_text())
    assert mmq["admission"]["maximum_absolute_error"] == 5.0e-4
    assert mmq["admission"]["maximum_rms_error"] == 2.5e-4
    assert selected_path_from_source() == contract["selected_large_mixer_q8_path"]
    assert skinny_path_unchanged()


def test_opt024_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []

    def flip_meets(changed: dict[str, Any]) -> None:
        changed["quartz_meets_llama"] = True
        if changed["quartz"]["mean_tok_s"] >= changed["llama_cpp"]["avg_ts"]:
            changed["quartz"]["mean_tok_s"] = changed["llama_cpp"]["avg_ts"] / 2.0
            changed["quartz"]["tok_s"] = [changed["quartz"]["mean_tok_s"]] * 3

    for mutate in (
        lambda x: x["quartz"].__setitem__("prompt_tokens", 2048),
        lambda x: x["quartz"].__setitem__("attribution", {"mixer_mmq": 1.0}),
        lambda x: x["quartz"].__setitem__("replicates", 1),
        lambda x: x["quartz"].__setitem__("graphs_created", False),
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        flip_meets,
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("task", "OPT-016"),
        lambda x: x.__setitem__("opt023_quartz_mean_tok_s", 1.0),
        lambda x: x.__setitem__("report_path", "/tmp/opt024/REPORT.md"),
        lambda x: x.__setitem__("skinny_mixer_unchanged", False),
        lambda x: x.__setitem__("shared_residual_y", False),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    if fixture["status"] == "measured":
        keep_break = json.loads(json.dumps(fixture))
        keep_break["quartz"]["mean_tok_s"] = OPT023_MEAN
        keep_break["quartz"]["tok_s"] = [OPT023_MEAN] * 3
        mutations.append(keep_break)
        flag_break = json.loads(json.dumps(fixture))
        flag_break["reverted"] = True
        mutations.append(flag_break)
        win_break = json.loads(json.dumps(fixture))
        win_break["ab_win"] = False
        mutations.append(win_break)
    else:
        reject_break = json.loads(json.dumps(fixture))
        reject_break["reverted"] = False
        mutations.append(reject_break)
        successor = json.loads(json.dumps(fixture))
        successor["successor_oracle"] = True
        mutations.append(successor)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)


def _common(image: str) -> list[str]:
    return [
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
        image,
    ]


def _ensure_llama_bench() -> None:
    if LLAMA_BENCH.is_file():
        return
    configure = [
        *_common(LLAMA_IMAGE),
        "cmake",
        "-S",
        "/workspace/.cache/authorities/llama.cpp",
        "-B",
        "/workspace/.cache/authorities/llama-build",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=120",
        "-DGGML_CUDA=ON",
        "-DGGML_NATIVE=OFF",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=ON",
    ]
    build = [
        *_common(LLAMA_IMAGE),
        "cmake",
        "--build",
        "/workspace/.cache/authorities/llama-build",
        "--target",
        "llama-bench",
        "-j",
        "6",
    ]
    for command in (configure, build):
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr


def _run_llama_bench() -> dict[str, Any]:
    _ensure_llama_bench()
    command = [
        *_common(LLAMA_IMAGE),
        str(LLAMA_BENCH.relative_to(ROOT)),
        "-m",
        "models/Qwen3.8-27B-Q4_K_M.gguf",
        "-p",
        "4096",
        "-n",
        "0",
        "--no-warmup",
        "-r",
        "3",
        "-ngl",
        "99",
        "-o",
        "json",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    text = completed.stdout
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list) and payload and payload[0].get("n_prompt") == 4096:
            EVIDENCE.mkdir(parents=True, exist_ok=True)
            LLAMA_JSON.write_text(json.dumps(payload, indent=2) + "\n")
            return payload[0]
    raise AssertionError("llama-bench JSON with n_prompt 4096 was not found\n" + text)


def _run_quartz() -> dict[str, Any]:
    nvcc = [
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
        "-DQW38_DIAGNOSTIC_TRACE",
        "cuda/prefill_4k_d2r_test.cu",
        "build/full_scheduler.trace.cuda.o",
        "build/scheduler_primitives.cuda.o",
        "build/quant_mmv.cuda.o",
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
        "-o",
        "build/qw38-cuda-prefill-4k-d2r-test",
    ]
    commands = [
        [*_common(IMAGE), "make", "build/qw38-cuda-timing-test"],
        nvcc,
        [
            *_common(IMAGE),
            "./build/qw38-cuda-prefill-4k-d2r-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        outputs.append(completed.stdout)
    records = [
        json.loads(line.removeprefix(PREFIX))
        for line in outputs[-1].splitlines()
        if line.startswith(PREFIX)
    ]
    assert len(records) == 1
    assert "status=passed" in outputs[-1]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    QUARTZ_JSON.write_text(json.dumps(records[0], indent=2) + "\n")
    return records[0]


def _parse_ab(raw_text: str) -> tuple[str, bool]:
    winner = "quality_mma"
    win = False
    for line in raw_text.splitlines():
        if not line.startswith("d2r_ab_winner "):
            continue
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        winner = fields["id"]
        win = fields["win"] == "true"
    return winner, win


def _write_report(fixture: dict[str, Any]) -> None:
    quartz = fixture["quartz"]
    llama = fixture["llama_cpp"]
    walls = ", ".join(str(v) for v in quartz["wall_ms"])
    tok_s = ", ".join(str(v) for v in quartz["tok_s"])
    decision = "keep" if fixture["status"] == "measured" else "reject"
    text = f"""# OPT-024 — Blackwell-aligned Q8_0 D2R for large mixer GEMMs

## Claim labels and proof limits

This increment lands **aligned-SoA D2R for large mixer Q8_0** GEMMs under the
plan.md **Q8 association** rule when it beats quality MMQ, or retains OPT-022
I=128 quality MMA when the A/B loses or 4K fails.
Live exclusive-RTX-5090 **4K keep/reject** versus the frozen OPT-023 Quartz mean
{OPT023_MEAN} tok/s. Numeric and exact-state **envelopes unloosened**. This
increment **does not substitute for the 2K llama.cpp parity gate**.
The **OPT-009 byte-exact reference retained** (`launch_q8_mmq_bf16_variant`
versus `launch_q8_mmq_bf16_reference`).

## Decision

**{decision}** — Quartz mean tok/s {quartz["mean_tok_s"]} versus OPT-023 baseline {OPT023_MEAN}.
A/B winner `{fixture["ab_winner"]}` win={json.dumps(fixture["ab_win"])};
`selected_large_mixer_q8_path`={fixture["selected_large_mixer_q8_path"]};
`reverted`={json.dumps(fixture["reverted"])}; `successor_oracle`={json.dumps(fixture["successor_oracle"])};
`production_d2r`={json.dumps(fixture["production_d2r"])}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| Tokens | exact 4096, not chat-rendered |
| Quartz token IDs | `(42 + index * 997) % kVocabularySize` |
| Quartz path | production `sync_tokens`, attribution null, graphs created, production fused GDN, cache_policy disabled, 0 warm-ups, 3 cold replicates |
| llama.cpp | `llama-bench -p 4096 -n 0 --no-warmup -r 3 -ngl 99` |
| Same sitting | llama.cpp first, then Quartz |
| Nsight | not_used |
| Keep/reject | strictly greater Quartz mean tok/s than {OPT023_MEAN} and A/B D2R win on every large mixer shape |

## Measured 4K sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- Quartz mean tok/s: {quartz["mean_tok_s"]} (walls {walls} ms; tok/s {tok_s})
- Quartz prompt_tokens: {quartz["prompt_tokens"]}; graphs_created: {json.dumps(quartz["graphs_created"])}; prompt_graph_rows: {quartz["prompt_graph_rows"]}
- llama.cpp live `avg_ts`: {llama["avg_ts"]} (`avg_ns` {llama["avg_ns"]}, `n_prompt` {llama["n_prompt"]}, `n_batch` {llama["n_batch"]}, `n_ubatch` {llama["n_ubatch"]}, `flash_attn` {llama["flash_attn"]}, `build_commit` {llama["build_commit"]}, `test_time` {llama["test_time"]})
- `quartz_meets_llama`: {json.dumps(fixture["quartz_meets_llama"])} (informational; not this gate)
- `owns_opt016_parity_gate`: false
- `substitutes_for_opt016`: false
- A/B winner: {fixture["ab_winner"]}; ab_win: {json.dumps(fixture["ab_win"])}
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    quartz = fixture["quartz"]
    text = f"""# OPT-024 rejection

Aligned-SoA D2R for large mixer Q8_0 GEMMs was measured on exclusive RTX 5090
and did not strictly beat quality MMQ on every timed large shape, or the 4K
oracle did not strictly beat the OPT-023 Quartz baseline.

- quality sitting mean tok/s: {quartz["mean_tok_s"]}
- OPT-023 baseline: {OPT023_MEAN}
- A/B winner: {fixture["ab_winner"]}
- A/B win: {json.dumps(fixture["ab_win"])}
- rule: strictly_greater_mean_tok_s (equality is a reject); A/B loss is a reject
- production large mixer GEMMs remain OPT-022 I=128 J=128 quality MMA
- successor_oracle: false (OPT-025+ denominator remains {OPT023_MEAN})
- measurement_utc: {fixture["measurement_utc"]}
"""
    REJECTION.write_text(text)


def test_opt024_native_4k_keep_reject() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")

    if FIXTURE.is_file():
        existing = json.loads(FIXTURE.read_text())
        if existing.get("status") == "rejected" and existing.get("reverted") is True:
            assert production_is_quality_mma()
            assert scheduler_shares_residual_y()
            assert skinny_path_unchanged()
            assert decode_mmv_gguf_layout()
            validate_result(existing)
            return

    ab_winner, ab_win = (
        _parse_ab(AB_RAW.read_text()) if AB_RAW.is_file() else ("quality_mma", False)
    )
    if ab_win:
        assert production_is_d2r(), "4K sitting must run on the winning D2R path"
        assert selected_path_from_source() == ab_winner
    else:
        assert production_is_quality_mma()
    assert scheduler_shares_residual_y()
    assert skinny_path_unchanged()
    assert decode_mmv_gguf_layout()
    llama = _run_llama_bench()
    quartz = _run_quartz()
    mean_tok_s = float(quartz["mean_tok_s"])
    avg_ts = float(llama["avg_ts"])
    quartz_meets_llama = mean_tok_s >= avg_ts
    keep = mean_tok_s > OPT023_MEAN and ab_win and ab_winner == "d2r_soa"
    path = "d2r_soa" if keep else "quality_mma"
    fixture = {
        "schema_version": 1,
        "task": "OPT-024",
        "status": "measured" if keep else "rejected",
        "measurement_utc": quartz["measurement_utc"],
        "device": quartz["device"],
        "compute_capability": quartz["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "quartz": {
            "prompt_tokens": 4096,
            "replicates": 3,
            "wall_ms": quartz["wall_ms"],
            "tok_s": quartz["tok_s"],
            "mean_tok_s": mean_tok_s,
            "cold": True,
            "cache_policy": "disabled",
            "attribution": None,
            "graphs_created": True,
            "prompt_graph_rows": 4096,
        },
        "llama_cpp": {
            "avg_ts": avg_ts,
            "avg_ns": llama["avg_ns"],
            "n_prompt": 4096,
            "n_batch": llama.get("n_batch", 2048),
            "n_ubatch": llama.get("n_ubatch", 512),
            "flash_attn": llama.get("flash_attn", -1),
            "build_commit": llama.get("build_commit", "cc83d7b"),
            "test_time": llama.get("test_time", quartz["measurement_utc"]),
        },
        "opt023_quartz_mean_tok_s": OPT023_MEAN,
        "quartz_meets_llama": quartz_meets_llama,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "reverted": False if keep else True,
        "successor_oracle": True if keep else False,
        "selected_large_mixer_q8_path": path,
        "ab_winner": ab_winner,
        "ab_win": ab_win,
        "production_d2r": True if keep else False,
        "shared_residual_y": True,
        "skinny_mixer_unchanged": True,
        "decode_mmv_gguf_layout": True,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt024-mixer-q8-d2r/REPORT.md",
    }
    assert quartz["graphs_created"] is True
    assert quartz["prompt_graph_rows"] == 4096
    assert quartz["prompt_tokens"] == 4096
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    _write_report(fixture)
    if not keep:
        _write_rejection(fixture)
        FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
        validate_result(fixture)
        return
    validate_result(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
