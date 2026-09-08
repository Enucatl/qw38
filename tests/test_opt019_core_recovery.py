from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
LLAMA_IMAGE = "qw38-llama-authority:cuda-13.0.2"
PARITY_PREFIX = "QW38_PREFILL_2K_PARITY_RESULT="
ATTRIBUTION_PREFIX = "QW38_PREFILL_ATTRIBUTION_RESULT="
CONTRACT = ROOT / "pins/opt019_core_recovery_contract.json"
FIXTURE = ROOT / "fixtures/opt019_core_recovery.json"
EVIDENCE = ROOT / "evidence/optimization/opt019-gdn-attention-core"
REPORT = EVIDENCE / "REPORT.md"
LLAMA_JSON = EVIDENCE / "llama-bench-2k.json"
QUARTZ_JSON = EVIDENCE / "quartz-2k.json"
ATTRIBUTION = EVIDENCE / "attribution.json"
LOCKED_BEFORE = EVIDENCE / "opt018-before-attribution.json"
LOCKED_BEFORE_UTC = "2026-09-08T18:33:22Z"
GDN_RAW = EVIDENCE / "gdn-fused-quality-ab-raw.txt"
ATTN_RAW = EVIDENCE / "attention-mma-ncols-sweep-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GDN_BEFORE_MS = 1643.46082
ATTENTION_BEFORE_MS = 1148.47461
COMBINED_BEFORE_MS = 2791.93543
FFN_REFERENCE_MS = 399.287018
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
PROOF = (
    "envelopes unloosened; sequential GDN remains the reference; "
    "tiled attention remains the reference; parity gate owner remains blocked; "
    "not an end-to-end 2K tok/s gate; no 8K/32K/128K throughput gate; "
    "mixer versus core split is not this increment"
)


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def _gdn_scan() -> dict[str, Any]:
    return json.loads((ROOT / "pins" / "cuda_gdn_scan_contract.json").read_text())


def _tiled() -> dict[str, Any]:
    return json.loads(
        (ROOT / "pins" / "cuda_tiled_attention_contract.json").read_text()
    )


def _attribution_contract() -> dict[str, Any]:
    return json.loads(
        (ROOT / "pins" / "cuda_prefill_attribution_contract.json").read_text()
    )


def _locked_before_attribution() -> dict[str, Any]:
    return json.loads((ROOT / _contract()["gdn_before_source"]).read_text())


def _core_addressed(gdn_after: float, attention_after: float) -> bool:
    contract = _contract()
    return (
        gdn_after < contract["per_category_max_ratio"] * GDN_BEFORE_MS
        and attention_after < contract["per_category_max_ratio"] * ATTENTION_BEFORE_MS
        and (gdn_after + attention_after)
        < contract["combined_max_ratio"] * COMBINED_BEFORE_MS
    )


def validate_result(
    result: Any,
    *,
    contract: dict[str, Any] | None = None,
    attribution_path: Path = ATTRIBUTION,
    report_path: Path = REPORT,
) -> None:
    if contract is None:
        contract = _contract()
    gdn = _gdn_scan()
    tiled = _tiled()
    attr_contract = _attribution_contract()
    before = _locked_before_attribution()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-019"
    assert result["status"] == "measured"
    assert result["status"] != "source_inspected"
    assert contract["primary_tokens"] == 2048
    assert contract["owns_opt016_parity_gate"] is False
    assert result["owns_opt016_parity_gate"] is False
    assert contract["opt016_status"] == "blocked"
    assert result["opt016_status"] == "blocked"
    assert contract["gdn_production"] == "fused_warp_column_quality"
    assert result["gdn_production"] == "fused_warp_column_quality"
    assert contract["gdn_reference"] == "kSequentialWindows"
    assert contract["gdn_rank2_retained"] is True
    assert contract["attention_production"] == "fattn_mma_f16_for_token_count_ge_16"
    assert result["attention_production"] == "fattn_mma_f16_for_token_count_ge_16"
    assert contract["attention_reference"] == "launch_attention_prepare_chunk_tiled"
    assert contract["attention_rank3_retained"] is True
    assert contract["envelopes_unloosened"] is True
    assert result["envelopes_unloosened"] is True
    assert contract["gdn_max_abs"] == gdn["admission"]["maximum_absolute_error"]
    assert contract["gdn_max_rms"] == gdn["admission"]["maximum_rms_error"]
    assert contract["atn_bf16_max_abs"] == tiled["proof_limits"]["bf16_max_abs"]
    assert contract["atn_bf16_rms"] == tiled["proof_limits"]["bf16_rms"]
    assert contract["gdn_max_abs"] == 5.0e-8
    assert contract["atn_bf16_max_abs"] == 0.00005
    assert contract["attention_ncols2"] == 2
    assert contract["legal_attention_ncols1"] == [8, 16, 32]
    assert (
        result["attention_mma_query_rows_2048"]
        == contract["attention_mma_query_rows_2048"]
    )
    assert result["attention_mma_query_rows_2048"] in contract["legal_attention_ncols1"]
    assert contract["gdn_block"] == {"x": 32, "y": 4}
    assert contract["gdn_grid_z_for_width_128"] == 32
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["gdn_before_ms"] == GDN_BEFORE_MS == contract["gdn_before_ms"]
    assert (
        result["attention_before_ms"]
        == ATTENTION_BEFORE_MS
        == contract["attention_before_ms"]
    )
    assert (
        result["combined_before_ms"]
        == COMBINED_BEFORE_MS
        == contract["combined_before_ms"]
    )
    assert contract["gdn_before_source"] == (
        "evidence/optimization/opt019-gdn-attention-core/opt018-before-attribution.json"
    )
    assert contract["attention_before_source"] == contract["gdn_before_source"]
    assert Path(ROOT / contract["gdn_before_source"]) == LOCKED_BEFORE
    assert before["measurement_utc"] == LOCKED_BEFORE_UTC
    assert before["categories_ms"]["gdn"] == GDN_BEFORE_MS
    assert before["categories_ms"]["attention"] == ATTENTION_BEFORE_MS
    quartz = result["quartz"]
    assert quartz["prompt_tokens"] == 2048
    assert quartz["replicates"] == contract["quartz_replicates"] == 3
    assert quartz["cold"] is True
    assert quartz["cache_policy"] == "disabled"
    assert quartz["attribution"] is None
    assert len(quartz["wall_ms"]) == 3 and len(quartz["tok_s"]) == 3
    mean = sum(float(v) for v in quartz["tok_s"]) / 3.0
    assert quartz["mean_tok_s"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    llama = result["llama_cpp"]
    assert llama["n_prompt"] == 2048
    would = float(quartz["mean_tok_s"]) >= float(llama["avg_ts"])
    assert result["would_pass_opt016"] is would
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    proof = result["proof_limit"]
    for phrase in contract["proof_limit"]:
        assert phrase in proof
    assert result["report_path"] == contract["report_path"]
    assert report_path.is_file()
    report = report_path.read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report
    assert attribution_path.is_file()
    attribution = json.loads(attribution_path.read_text())
    nested = result["attribution"]
    assert nested["categories_ms"] == attribution["categories_ms"]
    assert nested["wall_ms"] == attribution["wall_ms"]
    categories = nested["categories_ms"]
    assert set(categories) == set(CATEGORIES)
    attributed = sum(float(categories[name]) for name in CATEGORIES)
    assert math.isclose(
        attributed,
        float(nested["wall_ms"]),
        rel_tol=attr_contract["rel_tol"],
        abs_tol=attr_contract["abs_tol_ms"],
    )
    gdn_after = float(result["gdn_after_ms"])
    attention_after = float(result["attention_after_ms"])
    assert gdn_after == categories["gdn"]
    assert attention_after == categories["attention"]
    assert result["combined_after_ms"] == pytest.approx(
        gdn_after + attention_after, rel=1e-12, abs=1e-9
    )
    addressed = _core_addressed(gdn_after, attention_after)
    assert result["core_addressed"] is addressed
    assert result["core_addressed"] is True


def test_opt019_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE.read_text()))
    contract = _contract()
    assert contract["llama_bench_args"] == [
        "-p",
        "2048",
        "-n",
        "0",
        "--no-warmup",
        "-r",
        "3",
        "-ngl",
        "99",
    ]
    assert contract["quartz_warmups"] == 0
    assert contract["quartz_attribution_for_remasurement"] == "null_unperturbed"
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["attention_ncols2"] == 2
    assert contract["gdn_rank2_retained"] is True
    assert contract["attention_rank3_retained"] is True
    gdn = _gdn_scan()
    tiled = _tiled()
    assert contract["gdn_max_abs"] == gdn["admission"]["maximum_absolute_error"]
    assert contract["atn_bf16_max_abs"] == tiled["proof_limits"]["bf16_max_abs"]


def test_opt019_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []
    for mutate in (
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        lambda x: x["quartz"].__setitem__("prompt_tokens", 2052),
        lambda x: x.__setitem__("gdn_production", "rank2_state_128"),
        lambda x: x.__setitem__(
            "attention_production", "rank3_warp0_mma_grouped_chunk"
        ),
        lambda x: x.__setitem__("gdn_before_ms", 1646.56665),
        lambda x: x.update(
            {
                "core_addressed": True,
                "gdn_after_ms": GDN_BEFORE_MS,
                "attention_after_ms": ATTENTION_BEFORE_MS,
                "combined_after_ms": COMBINED_BEFORE_MS,
            }
        ),
        lambda x: x["llama_cpp"].__setitem__("n_prompt", 8192),
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x.__setitem__("status", "source_inspected"),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)
    with pytest.raises(AssertionError):
        validate_result(fixture, attribution_path=Path("/tmp/missing-opt019.json"))
    loosened = json.loads(json.dumps(_contract()))
    loosened["gdn_max_abs"] = 5.0e-7
    with pytest.raises(AssertionError):
        validate_result(fixture, contract=loosened)
    loosened_atn = json.loads(json.dumps(_contract()))
    loosened_atn["atn_bf16_max_abs"] = 0.0005
    with pytest.raises(AssertionError):
        validate_result(fixture, contract=loosened_atn)
    ncols2 = json.loads(json.dumps(_contract()))
    ncols2["attention_ncols2"] = 8
    with pytest.raises(AssertionError):
        validate_result(fixture, contract=ncols2)


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


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def _nvcc(sources: list[str], output: str) -> list[str]:
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
        "-DQW38_DIAGNOSTIC_TRACE",
        *sources,
        "-o",
        output,
    ]


def _diagnostic_objects() -> list[str]:
    return [
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
    ]


def _run_component_diagnostics() -> tuple[str, str]:
    _run([*_common(IMAGE), "make", "cuda-products", "cuda-native"])
    _run([*_common(IMAGE), "make", "build/qw38-cuda-gdn-chunk-test"])
    _run(
        [
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
            "cuda/gdn_scan_test.cu",
            "build/gdn_step.cuda.o",
            "build/gdn.o",
            "build/status.o",
            "-o",
            "build/qw38-cuda-gdn-scan-test",
        ]
    )
    gdn = _run([*_common(IMAGE), "./build/qw38-cuda-gdn-scan-test"])
    _run(
        [
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
            "cuda/tiled_attention_test.cu",
            "build/attention_decode.cuda.o",
            "-o",
            "build/qw38-cuda-tiled-attention-test",
        ]
    )
    attn = _run([*_common(IMAGE), "./build/qw38-cuda-tiled-attention-test"])
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    gdn_lines = [
        line
        for line in gdn.stdout.splitlines()
        if line.startswith(("opt019_gdn_", "fused_ab "))
    ]
    attn_lines = [
        line
        for line in attn.stdout.splitlines()
        if line.startswith("opt019_attention_")
    ]
    GDN_RAW.write_text("\n".join(gdn_lines) + "\n")
    ATTN_RAW.write_text("\n".join(attn_lines) + "\n")
    assert any(line.startswith("opt019_gdn_2048 ") for line in gdn_lines)
    assert any("passed=true" in line for line in gdn_lines if "opt019_gdn_2048" in line)
    assert any(
        "faster=true" in line for line in gdn_lines if line.startswith("fused_ab ")
    )
    assert any(line.startswith("opt019_attention_2048 ") for line in attn_lines)
    assert any(
        "passed=true" in line for line in attn_lines if "opt019_attention_2048" in line
    )
    assert any(
        "faster=true" in line
        for line in attn_lines
        if line.startswith("opt019_attention_ab ")
    )
    return gdn.stdout, attn.stdout


def _run_llama_bench() -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
        "-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf -p 2048 -n 0 --no-warmup "
        "-r 3 -ngl 99 -o json",
    ]
    completed = _run(command)
    text = completed.stdout + completed.stderr
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list) and payload and payload[0].get("n_prompt") == 2048:
            LLAMA_JSON.write_text(json.dumps(payload, indent=2) + "\n")
            return payload[0]
    raise AssertionError("llama-bench JSON with n_prompt 2048 was not found\n" + text)


def _run_quartz() -> dict[str, Any]:
    _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    _run(
        _nvcc(
            ["cuda/prefill_2k_parity_test.cu", *_diagnostic_objects()],
            "build/qw38-cuda-prefill-2k-parity-test",
        )
    )
    run = _run(
        [
            *_common(IMAGE),
            "./build/qw38-cuda-prefill-2k-parity-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ]
    )
    records = [
        json.loads(line.removeprefix(PARITY_PREFIX))
        for line in run.stdout.splitlines()
        if line.startswith(PARITY_PREFIX)
    ]
    assert len(records) == 1
    assert "status=passed" in run.stdout
    QUARTZ_JSON.write_text(json.dumps(records[0], indent=2) + "\n")
    return records[0]


def _run_attribution() -> dict[str, Any]:
    _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    _run(
        _nvcc(
            ["cuda/prefill_attribution_test.cu", *_diagnostic_objects()],
            "build/qw38-cuda-prefill-attribution-test",
        )
    )
    run = _run(
        [
            *_common(IMAGE),
            "./build/qw38-cuda-prefill-attribution-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ]
    )
    records = [
        json.loads(line.removeprefix(ATTRIBUTION_PREFIX))
        for line in run.stdout.splitlines()
        if line.startswith(ATTRIBUTION_PREFIX)
    ]
    assert len(records) == 1
    assert "status=passed" in run.stdout
    ATTRIBUTION.write_text(json.dumps(records[0], indent=2) + "\n")
    return records[0]


def _write_report(fixture: dict[str, Any], gdn_stdout: str, attn_stdout: str) -> None:
    quartz = fixture["quartz"]
    gdn_ab = next(
        line for line in gdn_stdout.splitlines() if line.startswith("fused_ab ")
    )
    gdn_2048 = next(
        line for line in gdn_stdout.splitlines() if line.startswith("opt019_gdn_2048 ")
    )
    attn_ab = next(
        line
        for line in attn_stdout.splitlines()
        if line.startswith("opt019_attention_ab ")
    )
    attn_2048 = next(
        line
        for line in attn_stdout.splitlines()
        if line.startswith("opt019_attention_2048 ")
    )
    text = f"""# OPT-019 — GDN and attention core quality

## Claim labels and proof limits

Production prompt GDN recurrence is the warp-column fused quality path.
Production prompt attention (`token_count >= 16`) is the fattn-mma-f16 analog.
The envelopes unloosened. sequential GDN remains the reference.
tiled attention remains the reference. The parity gate owner remains blocked.
This remasurement is not an end-to-end 2K tok/s gate. There is
no 8K/32K/128K throughput gate. A mixer versus core split is not this increment.

Live GDN/attention envelopes, ncols1 pin, category before/after, attribution,
and remasurement are **Measured**. llama.cpp `gated_delta_net.cu` /
`fattn-mma-f16.cuh` provenance at `{LLAMA_REV}` is **External**.

## Production path

`GdnScanPath::kFusedTokenLoop` dispatches warp-column quality recurrence after
the existing parallel convolution. Rank-2 fused remains `launch_gdn_fused_rank2`.
`launch_attention_prepare_chunk` uses fattn-mma quality when
`token_count >= 16` (ncols1={fixture["attention_mma_query_rows_2048"]}, ncols2=2)
and tiled otherwise. Rank-3 MMA remains `launch_attention_prepare_chunk_mma_rank3`.

## Component envelopes (**Measured**)

GDN quality versus sequential at 2048 tokens:
`{gdn_2048}`
`{gdn_ab}`

Attention quality versus tiled at 2048 rows:
`{attn_2048}`
`{attn_ab}`

Raw samples: `gdn-fused-quality-ab-raw.txt`, `attention-mma-ncols-sweep-raw.txt`.

## Residual sinks addressed (**Measured**, not the parity gate)

- GDN before (OPT-018 attribution): {GDN_BEFORE_MS} ms
- Attention before: {ATTENTION_BEFORE_MS} ms
- Combined before: {COMBINED_BEFORE_MS} ms
- GDN after: {fixture["gdn_after_ms"]} ms
- Attention after: {fixture["attention_after_ms"]} ms
- Combined after: {fixture["combined_after_ms"]} ms
- `core_addressed`: {fixture["core_addressed"]}
- Quartz mean tok/s: {quartz["mean_tok_s"]} (walls {quartz["wall_ms"]})
- llama.cpp `avg_ts`: {fixture["llama_cpp"]["avg_ts"]}
- `would_pass_opt016`: {fixture["would_pass_opt016"]} (informational)
- `owns_opt016_parity_gate`: false; `opt016_status`: blocked
"""
    REPORT.write_text(text)


def test_opt019_native_core_recovery() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")

    gdn_stdout, attn_stdout = _run_component_diagnostics()
    winner_line = next(
        line
        for line in attn_stdout.splitlines()
        if line.startswith("opt019_attention_2048 ")
    )
    fields = dict(part.split("=", 1) for part in winner_line.split()[1:])
    winner = int(fields["winner_ncols1"])
    selected = int(fields["selected_ncols1"])
    assert winner in (8, 16, 32)
    assert selected == _contract()["attention_mma_query_rows_2048"]
    if winner != selected:
        pytest.fail(
            f"ncols1 sweep winner {winner} differs from selected {selected}; "
            "update kSelectedAttentionMmaQueryRows and the OPT-019 pin"
        )

    llama = _run_llama_bench()
    quartz_raw = _run_quartz()
    attribution = _run_attribution()
    gdn_after = float(attribution["categories_ms"]["gdn"])
    attention_after = float(attribution["categories_ms"]["attention"])
    ffn_after = float(attribution["categories_ms"]["ffn_mmq"])
    assert abs(ffn_after - FFN_REFERENCE_MS) / FFN_REFERENCE_MS <= 0.10
    mean_tok_s = float(quartz_raw["mean_tok_s"])
    avg_ts = float(llama["avg_ts"])
    addressed = _core_addressed(gdn_after, attention_after)
    assert addressed
    fixture = {
        "schema_version": 1,
        "task": "OPT-019",
        "status": "measured",
        "measurement_utc": quartz_raw["measurement_utc"],
        "device": quartz_raw["device"],
        "compute_capability": quartz_raw["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "gdn_production": "fused_warp_column_quality",
        "attention_production": "fattn_mma_f16_for_token_count_ge_16",
        "attention_mma_query_rows_2048": selected,
        "owns_opt016_parity_gate": False,
        "opt016_status": "blocked",
        "envelopes_unloosened": True,
        "gdn_before_ms": GDN_BEFORE_MS,
        "attention_before_ms": ATTENTION_BEFORE_MS,
        "combined_before_ms": COMBINED_BEFORE_MS,
        "gdn_after_ms": gdn_after,
        "attention_after_ms": attention_after,
        "combined_after_ms": gdn_after + attention_after,
        "core_addressed": addressed,
        "quartz": {
            "prompt_tokens": 2048,
            "replicates": 3,
            "wall_ms": quartz_raw["wall_ms"],
            "tok_s": quartz_raw["tok_s"],
            "mean_tok_s": mean_tok_s,
            "cold": True,
            "cache_policy": "disabled",
            "attribution": None,
        },
        "llama_cpp": {
            "avg_ts": avg_ts,
            "avg_ns": llama["avg_ns"],
            "n_prompt": 2048,
            "n_ubatch": llama.get("n_ubatch", 512),
            "flash_attn": llama.get("flash_attn", -1),
            "build_commit": llama.get("build_commit", "cc83d7b"),
            "test_time": llama.get("test_time", quartz_raw["measurement_utc"]),
        },
        "would_pass_opt016": mean_tok_s >= avg_ts,
        "attribution": attribution,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt019-gdn-attention-core/REPORT.md",
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    _write_report(fixture, gdn_stdout, attn_stdout)
    validate_result(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
