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
CONTRACT = ROOT / "pins/opt037_ffn_tile_contract.json"
FIXTURE = ROOT / "fixtures/opt037_ffn_tiles.json"
OPT034_FIXTURE = ROOT / "fixtures/opt034_packed_mmv.json"
EVIDENCE = ROOT / "evidence/optimization/opt037-ffn-tiles"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "ffn-tile-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_FFN_TILE_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
LLAMA_DECODE_PREFIX = "QW38_LLAMA_DECODE_ORACLE_RESULT="
OPT034_P = 1869.84412
OPT034_D128 = 25.3816128
OPT034_D2048 = 20.169548
OPT034_D128_P95 = 39.9736366
OPT034_D128_RUN_P95 = 39.4155655
OPT034_D2048_P95 = 50.2872772
OPT034_D2048_RUN_P95 = 49.590683
HISTORICAL_P = 1746.71973
OPT032_P = 1637.58594
OPT033_P = 1745.10315
OPT035_P = 1865.21155
OPT036_P = 1644.04822
LEGAL_TILES = {
    "i64_j32",
    "i64_j64",
    "i64_j128",
    "i128_j32",
    "i128_j64",
    "i128_j128",
}
PROJECTIONS = ("gate", "up", "down")
CANDIDATES = [
    "i64_j32",
    "i64_j64",
    "i64_j128",
    "i128_j32",
    "i128_j64",
    "i128_j128",
]
PROOF = (
    "admitted component wins; improved P; frozen MMQ envelopes; "
    "cross-workload guard; does not substitute for the 2K llama.cpp parity gate; "
    "OPT-034 P D128 D2048 are the keep denominators"
)
AB_OBJECTS = [
    "build/quant_mmv.cuda.o",
    "build/quant.o",
    "build/status.o",
]
NVCC_OBJECTS = [
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


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def _parse_tile(ident: str) -> tuple[int, int]:
    match = re.fullmatch(r"i(\d+)_j(\d+)", ident)
    assert match is not None, ident
    return int(match.group(1)), int(match.group(2))


def _source_u32(name: str) -> int:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    match = re.search(rf"{name} = (\d+)U", text)
    assert match is not None, name
    return int(match.group(1))


def selected_pins_from_source() -> dict[str, int]:
    return {
        "gate_i": _source_u32("kSelectedFfnGateQualityI"),
        "gate_j": _source_u32("kSelectedFfnGatePromptTile"),
        "up_i": _source_u32("kSelectedFfnUpQualityI"),
        "up_j": _source_u32("kSelectedFfnUpPromptTile"),
        "down_i": _source_u32("kSelectedFfnDownQualityI"),
        "down_j": _source_u32("kSelectedFfnDownPromptTile"),
    }


def frozen_neighbors_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    mmv = (ROOT / "cuda/quant_mmv.cu").read_text()
    fattn = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    decode = (ROOT / "cuda/attention_decode.cu").read_text()
    return (
        'kSelectedFfnPath[] = "shared_y_swiglu_q8"' in text
        and "unsigned int selected_mma_mmq_prompt_tile() noexcept { return 128U; }"
        in text
        and 'kSelectedMmqStreamKPath[] = "off"' in text
        and 'kSelectedSkinnyMixerPath[] = "mma_i32_j128"' in text
        and 'kSelectedMmvLoadPath[] = "packed"' in mmv
        and 'kSelectedFattnPath[] = "stream_k"' in fattn
        and 'kSelectedPvPath[] = "mma"' in fattn
        and 'kSelectedVkqAccum[] = "registers"' in fattn
        and "kSelectedDecodeKvPartsLow = 16" in decode
        and "kSelectedDecodeKvPartsHigh = 16" in decode
    )


def _ab_candidate(proj: dict[str, Any], ident: str) -> None:
    cand = proj["candidates"][ident]
    assert cand["id"] == ident
    quality_i, prompt_tile = _parse_tile(ident)
    assert cand["quality_i"] == quality_i
    assert cand["prompt_tile"] == prompt_tile
    assert len(cand["warmup_ms"]) == 3
    if cand["launch_ok"] and cand["eligible"]:
        assert len(cand["samples"]) == 30
        mean = sum(float(v) for v in cand["samples"]) / 30.0
        assert cand["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
        assert cand["occupancy"] >= 1
        assert cand["timed_nonfinite"] == 0
        assert cand["envelope"]["ok"] is True
        assert cand["envelope"]["nonfinite"] == 0
        assert cand["envelope"]["bad"] == 0


def _select_winner(proj: dict[str, Any]) -> str:
    best_id = "i128_j128"
    best = 1.0e30
    for ident in CANDIDATES:
        cand = proj["candidates"][ident]
        if cand["eligible"] and float(cand["mean_ms"]) < best:
            best = float(cand["mean_ms"])
            best_id = ident
    baseline = float(proj["candidates"]["i128_j128"]["mean_ms"])
    if best_id == "i128_j128" or not (
        proj["candidates"][best_id]["eligible"]
        and proj["candidates"]["i128_j128"]["eligible"]
        and float(proj["candidates"][best_id]["mean_ms"]) < baseline
    ):
        return "i128_j128"
    return best_id


def _keep_predicates(result: dict[str, Any]) -> bool:
    contract = _contract()
    if result["reverted"] is True or result["keep_sitting_skipped"] is True:
        return False
    if result["status"] != "measured":
        return False
    if result["ab"]["any_win"] is not True:
        return False
    wins = 0
    for name in PROJECTIONS:
        proj = result["ab"]["projections"][name]
        if proj["win"] is True:
            wins += 1
            assert proj["winner"] != "i128_j128"
        installed_i = result[f"selected_ffn_{name}_quality_i"]
        installed_j = result[f"selected_ffn_{name}_prompt_tile"]
        expected_i, expected_j = _parse_tile(proj["installed"])
        if installed_i != expected_i or installed_j != expected_j:
            return False
    if wins < 1:
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if (
        float(result["p"]["quartz"]["mean_tok_s"])
        <= contract["opt034_quartz_p_mean_tok_s"]
    ):
        return False
    if (
        float(result["d128"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt034_quartz_d128_mean_tok_s"]
    ):
        return False
    if (
        result["d128"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt034_quartz_d128_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt034_quartz_d128_run_mean_token_latency_p95_ms"]
    ):
        return False
    if (
        float(result["d2048"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt034_quartz_d2048_mean_tok_s"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt034_quartz_d2048_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt034_quartz_d2048_run_mean_token_latency_p95_ms"]
    ):
        return False
    return True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-037"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["yardstick"] == "prefill_ffn_4k_tiles"
    assert contract["opt034_quartz_p_mean_tok_s"] == OPT034_P
    assert contract["opt034_quartz_d128_mean_tok_s"] == OPT034_D128
    assert contract["opt034_quartz_d2048_mean_tok_s"] == OPT034_D2048
    assert contract["historical_opt026_quartz_mean_tok_s"] == HISTORICAL_P
    assert contract["use_as_keep_denominator"] is False
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert contract["ab"]["candidates"] == CANDIDATES
    assert contract["ab"]["byte_equal_required"] is False
    assert contract["ab"]["time_mma_only"] is True
    assert contract["selected_mma_mmq_prompt_tile"] == 128
    pins = selected_pins_from_source()
    assert (
        pins["gate_i"]
        == result["selected_ffn_gate_quality_i"]
        == contract["selected_ffn_gate_quality_i"]
    )
    assert (
        pins["gate_j"]
        == result["selected_ffn_gate_prompt_tile"]
        == contract["selected_ffn_gate_prompt_tile"]
    )
    assert (
        pins["up_i"]
        == result["selected_ffn_up_quality_i"]
        == contract["selected_ffn_up_quality_i"]
    )
    assert (
        pins["up_j"]
        == result["selected_ffn_up_prompt_tile"]
        == contract["selected_ffn_up_prompt_tile"]
    )
    assert (
        pins["down_i"]
        == result["selected_ffn_down_quality_i"]
        == contract["selected_ffn_down_quality_i"]
    )
    assert (
        pins["down_j"]
        == result["selected_ffn_down_prompt_tile"]
        == contract["selected_ffn_down_prompt_tile"]
    )
    assert set(result["ab"]["projections"]) == set(PROJECTIONS)
    assert result["ab"]["any_win"] is (
        any(result["ab"]["projections"][name]["win"] for name in PROJECTIONS)
    )
    for name in PROJECTIONS:
        proj = result["ab"]["projections"][name]
        assert proj["id"] == name
        assert proj["prompt_rows"] == 4096
        assert proj["winner"] in LEGAL_TILES
        assert proj["installed"] in LEGAL_TILES
        assert set(proj["candidates"]) == set(CANDIDATES)
        for ident in CANDIDATES:
            _ab_candidate(proj, ident)
        selected = _select_winner(proj)
        assert selected == proj["winner"] or (
            selected == "i128_j128" and proj["win"] is False
        )
        if proj["win"]:
            assert proj["installed"] == proj["winner"] != "i128_j128"
        else:
            assert proj["installed"] == "i128_j128"
    assert frozen_neighbors_unchanged()
    keep = _keep_predicates(result)
    if result["status"] == "measured":
        assert keep
        assert result["reverted"] is False
        assert result["keep_sitting_skipped"] is False
        assert result["p"]["quartz"]["mean_tok_s"] > OPT034_P
        assert not REJECTION.is_file()
    else:
        assert not keep
        assert result["reverted"] is True
        assert result["selected_ffn_gate_quality_i"] == 128
        assert result["selected_ffn_gate_prompt_tile"] == 128
        assert result["selected_ffn_up_quality_i"] == 128
        assert result["selected_ffn_up_prompt_tile"] == 128
        assert result["selected_ffn_down_quality_i"] == 128
        assert result["selected_ffn_down_prompt_tile"] == 128
        assert REJECTION.is_file()
        rejection = REJECTION.read_text()
        assert "i128_j128" in rejection
        if result["keep_sitting_skipped"] is False:
            assert str(result["p"]["quartz"]["mean_tok_s"]) in rejection
        else:
            assert result["p"] is None and result["d128"] is None
            assert result["d2048"] is None
    proof = result["proof_limit"]
    for phrase in contract["proof_limit"]:
        assert phrase in proof
        assert phrase in PROOF
    assert result["report_path"] == contract["report_path"]
    assert "/tmp" not in result["report_path"]
    assert REPORT.is_file()
    report = REPORT.read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report
    assert AB_RAW.is_file()
    opt034 = json.loads(OPT034_FIXTURE.read_text())
    assert opt034["p"]["quartz"]["mean_tok_s"] == OPT034_P
    assert opt034["d128"]["quartz"]["mean_tok_s"] == OPT034_D128
    assert opt034["d2048"]["quartz"]["mean_tok_s"] == OPT034_D2048


def test_opt037_contract_and_source_pins() -> None:
    contract = _contract()
    opt034 = json.loads(OPT034_FIXTURE.read_text())
    assert contract["opt034_quartz_p_mean_tok_s"] == OPT034_P
    assert contract["opt034_quartz_d128_mean_tok_s"] == OPT034_D128
    assert contract["opt034_quartz_d2048_mean_tok_s"] == OPT034_D2048
    assert contract["historical_opt026_quartz_mean_tok_s"] == HISTORICAL_P
    assert contract["use_as_keep_denominator"] is False
    assert contract["ab"]["candidates"] == CANDIDATES
    assert contract["owns_opt016_parity_gate"] is False
    assert frozen_neighbors_unchanged()
    pins = selected_pins_from_source()
    assert pins["gate_i"] in (64, 128)
    assert pins["gate_j"] in (32, 64, 128)
    assert contract["opt034_quartz_p_mean_tok_s"] == opt034["p"]["quartz"]["mean_tok_s"]
    mmq = json.loads((ROOT / "pins/cuda_mmq_contract.json").read_text())
    assert mmq["admission"]["q4k_q6k_abs_scale"] == 0.2
    assert mmq["admission"]["q4k_q6k_rel_tol"] == 0.05


def test_opt037_rejects_historical_p_as_keep_denominator() -> None:
    contract = _contract()
    keep_denominators = {
        contract["opt034_quartz_p_mean_tok_s"],
        contract["opt034_quartz_d128_mean_tok_s"],
        contract["opt034_quartz_d2048_mean_tok_s"],
    }
    assert HISTORICAL_P not in keep_denominators
    assert OPT032_P not in keep_denominators
    assert OPT033_P not in keep_denominators
    assert OPT035_P not in keep_denominators
    assert OPT036_P not in keep_denominators
    assert contract["use_as_keep_denominator"] is False


def test_opt037_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-037 fixture is written by the exclusive CUDA sitting")
    validate_result(json.loads(FIXTURE.read_text()))


def _common(image: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "QW38_CUDA_TEST_TIER=acceptance",
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


def _nvcc(source: str, output: str, extra: list[str] | None = None) -> list[str]:
    objects = extra if extra is not None else NVCC_OBJECTS
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
        source,
        *objects,
        "-o",
        output,
    ]


def _parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith(prefix):
            return json.loads(line.removeprefix(prefix))
    raise AssertionError(f"{prefix} was not found\n{text}")


def _set_pins(pins: dict[str, int]) -> None:
    source = ROOT / "cuda/quant_mmq_mma.cuh"
    text = source.read_text()
    replacements = {
        "kSelectedFfnGateQualityI": pins["gate_i"],
        "kSelectedFfnGatePromptTile": pins["gate_j"],
        "kSelectedFfnUpQualityI": pins["up_i"],
        "kSelectedFfnUpPromptTile": pins["up_j"],
        "kSelectedFfnDownQualityI": pins["down_i"],
        "kSelectedFfnDownPromptTile": pins["down_j"],
    }
    for name, value in replacements.items():
        text, count = re.subn(rf"{name} = \d+U", f"{name} = {value}U", text, count=1)
        assert count == 1, name
    source.write_text(text)
    contract = _contract()
    contract["selected_ffn_gate_quality_i"] = pins["gate_i"]
    contract["selected_ffn_gate_prompt_tile"] = pins["gate_j"]
    contract["selected_ffn_up_quality_i"] = pins["up_i"]
    contract["selected_ffn_up_prompt_tile"] = pins["up_j"]
    contract["selected_ffn_down_quality_i"] = pins["down_i"]
    contract["selected_ffn_down_prompt_tile"] = pins["down_j"]
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n")


def _pins_from_ab(ab: dict[str, Any]) -> dict[str, int]:
    pins = {
        "gate_i": 128,
        "gate_j": 128,
        "up_i": 128,
        "up_j": 128,
        "down_i": 128,
        "down_j": 128,
    }
    for name in PROJECTIONS:
        ident = ab["projections"][name]["installed"]
        quality_i, prompt_tile = _parse_tile(ident)
        pins[f"{name}_i"] = quality_i
        pins[f"{name}_j"] = prompt_tile
    return pins


def _write_report(fixture: dict[str, Any]) -> None:
    decision = "keep" if fixture["status"] == "measured" else "reject"
    sitting = "skipped" if fixture["keep_sitting_skipped"] else "ran"
    quartz_p = fixture["p"]["quartz"]["mean_tok_s"] if fixture["p"] else "n/a"
    quartz_d128 = fixture["d128"]["quartz"]["mean_tok_s"] if fixture["d128"] else "n/a"
    quartz_d2048 = (
        fixture["d2048"]["quartz"]["mean_tok_s"] if fixture["d2048"] else "n/a"
    )
    legs = []
    for name in PROJECTIONS:
        proj = fixture["ab"]["projections"][name]
        legs.append(
            f"{name} winner {proj['winner']} win={json.dumps(proj['win'])} "
            f"installed {proj['installed']}"
        )
    text = f"""# OPT-037 — Select 4K FFN tiles per projection

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of 4096-row Q4_K FFN quality MMA I∈{{64,128}} ×
J∈{{32,64,128}} independently for gate, up, and down, with shared-Y staging
preserved and 2D scheduling. Keep requires **admitted component wins**,
**improved P**, **frozen MMQ envelopes**,
and the **cross-workload guard**. This increment does not substitute for the 2K llama.cpp parity gate. The OPT-034 P D128 D2048 are the keep denominators
(P {OPT034_P}, D128 {OPT034_D128}, D2048 {OPT034_D2048}), not
historical OPT-026 {HISTORICAL_P}, not OPT-032 {OPT032_P}, not OPT-033 {OPT033_P},
not OPT-035 {OPT035_P}, and not OPT-036 {OPT036_P}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
any_win={json.dumps(fixture["ab"]["any_win"])};
{"; ".join(legs)};
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | 4096-row gate/up/down independently, I/J sweep, 3 warm + 30 measured, MMA only |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-034 {OPT034_P} (must improve)
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-034 {OPT034_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-034 {OPT034_D2048} (guard ≥95%)
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    reason = (
        "component A/B retained I=128/J=128 on every FFN projection"
        if fixture["keep_sitting_skipped"]
        else "keep sitting failed P improvement or the cross-workload guard"
    )
    tok = (
        ""
        if fixture["keep_sitting_skipped"]
        else f"\n- P Quartz mean tok/s: {fixture['p']['quartz']['mean_tok_s']}\n"
        f"- OPT-034 P baseline: {OPT034_P}\n"
        f"- D128 Quartz mean tok/s: {fixture['d128']['quartz']['mean_tok_s']}\n"
        f"- D2048 Quartz mean tok/s: {fixture['d2048']['quartz']['mean_tok_s']}"
    )
    legs = "\n".join(
        f"- {name}: winner {fixture['ab']['projections'][name]['winner']} "
        f"win={json.dumps(fixture['ab']['projections'][name]['win'])}"
        for name in PROJECTIONS
    )
    text = f"""# OPT-037 rejection

{reason}. Production FFN tiles are i128_j128
on every projection.

{legs}
- keep_sitting_skipped: {json.dumps(fixture["keep_sitting_skipped"])}
- measurement_utc: {fixture["measurement_utc"]}{tok}
"""
    REJECTION.write_text(text)


def _run_ab() -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commands = [
        [*_common(IMAGE), "make", "build/qw38-cuda-quant-test"],
        _nvcc(
            "cuda/ffn_tile_ab_test.cu",
            "build/qw38-cuda-ffn-tile-ab-test",
            AB_OBJECTS,
        ),
        [
            *_common(IMAGE),
            "./build/qw38-cuda-ffn-tile-ab-test",
            "evidence/optimization/opt037-ffn-tiles/ffn-tile-ab-raw.txt",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], AB_PREFIX)
    assert "status=passed" in outputs[-1]
    return record


def _parse_llama_bench(text: str, predicate, what: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list) and payload and predicate(payload[0]):
            return payload
    raise AssertionError(what + "\n" + text)


def _run_llama_bench_p() -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
        "-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf "
        "-p 4096 -n 0 --no-warmup -r 3 -ngl 99 -o json",
    ]
    completed = _run(command)
    payload = _parse_llama_bench(
        completed.stdout + completed.stderr,
        lambda row: row.get("n_prompt") == 4096,
        "llama-bench JSON with n_prompt 4096",
    )
    (EVIDENCE / "llama-bench-4k.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload[0]


def _run_quartz_p() -> dict[str, Any]:
    commands = [
        _nvcc(
            "cuda/prefill_4k_oracle_test.cu", "build/qw38-cuda-prefill-4k-oracle-test"
        ),
        [
            *_common(IMAGE),
            "./build/qw38-cuda-prefill-4k-oracle-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], P_PREFIX)
    (EVIDENCE / "quartz-p.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _run_quartz_decode(prefix: int) -> dict[str, Any]:
    binary = "build/qw38-cuda-decode-oracle-test"
    commands = [
        _nvcc("cuda/decode_oracle_test.cu", binary),
        [*_common(IMAGE), f"./{binary}", "models/Qwen3.8-27B-Q4_K_M.gguf", str(prefix)],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], DECODE_PREFIX)
    (EVIDENCE / f"quartz-d{prefix}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


def _run_llama_decode(prefix: int) -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/qw38-llama-decode-oracle "
        f"/workspace/models/Qwen3.8-27B-Q4_K_M.gguf {prefix}",
    ]
    completed = _run(command)
    record = _parse_prefixed(completed.stdout + completed.stderr, LLAMA_DECODE_PREFIX)
    (EVIDENCE / f"llama-decode-d{prefix}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


def _engine_block(record: dict[str, Any], sidecar: str | None = None) -> dict[str, Any]:
    block = {
        "prefix": record["prefix"],
        "decode_tokens": record["decode_tokens"],
        "warmups": record["warmups"],
        "runs": record["runs"],
        "warmup_tok_s": record["warmup_tok_s"],
        "tok_s": record["tok_s"],
        "run_wall_ms": record["run_wall_ms"],
        "mean_tok_s": record["mean_tok_s"],
        "token_latency_p50_ms": record["token_latency_p50_ms"],
        "token_latency_p95_ms": record["token_latency_p95_ms"],
        "run_mean_token_latency_p95_ms": record["run_mean_token_latency_p95_ms"],
        "graphs_created": record.get("graphs_created", True),
        "attribution": record.get("attribution"),
        "cache_policy": record.get("cache_policy", "disabled"),
    }
    if sidecar is not None:
        path = EVIDENCE / sidecar
        path.write_text(json.dumps(record["token_latency_ms"]) + "\n")
        block["token_latency_sidecar"] = (
            f"evidence/optimization/opt037-ffn-tiles/{sidecar}"
        )
    for key in ("n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"):
        if key in record:
            block[key] = record[key]
    return block


def _base_fixture(
    ab: dict[str, Any], pins: dict[str, int], **extra: Any
) -> dict[str, Any]:
    fixture = {
        "schema_version": 1,
        "task": "OPT-037",
        "status": extra["status"],
        "measurement_utc": extra.get("measurement_utc", ab["measurement_utc"]),
        "device": ab["device"],
        "compute_capability": ab["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": extra["reverted"],
        "selected_ffn_gate_quality_i": pins["gate_i"],
        "selected_ffn_gate_prompt_tile": pins["gate_j"],
        "selected_ffn_up_quality_i": pins["up_i"],
        "selected_ffn_up_prompt_tile": pins["up_j"],
        "selected_ffn_down_quality_i": pins["down_i"],
        "selected_ffn_down_prompt_tile": pins["down_j"],
        "ab": {
            "any_win": ab["any_win"],
            "projections": ab["projections"],
        },
        "keep_sitting_skipped": extra["keep_sitting_skipped"],
        "p": extra.get("p"),
        "d128": extra.get("d128"),
        "d2048": extra.get("d2048"),
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt037-ffn-tiles/REPORT.md",
    }
    return fixture


def test_opt037_native_keep_reject() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")
    if FIXTURE.is_file():
        existing = json.loads(FIXTURE.read_text())
        if existing.get("status") in ("measured", "rejected"):
            validate_result(existing)
            return

    ab = _run_ab()
    if not ab["any_win"]:
        baseline = {
            "gate_i": 128,
            "gate_j": 128,
            "up_i": 128,
            "up_j": 128,
            "down_i": 128,
            "down_j": 128,
        }
        _set_pins(baseline)
        fixture = _base_fixture(
            ab,
            baseline,
            status="rejected",
            reverted=True,
            keep_sitting_skipped=True,
            p=None,
            d128=None,
            d2048=None,
        )
        _write_report(fixture)
        _write_rejection(fixture)
        FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
        validate_result(fixture)
        return

    pins = _pins_from_ab(ab)
    _set_pins(pins)
    _run([*_common(IMAGE), "make", "build/qw38-cuda-quant-test"])
    llama_p = _run_llama_bench_p()
    quartz_p = _run_quartz_p()
    llama_d128 = _run_llama_decode(128)
    quartz_d128 = _run_quartz_decode(128)
    llama_d2048 = _run_llama_decode(2048)
    quartz_d2048 = _run_quartz_decode(2048)
    fixture = _base_fixture(
        ab,
        pins,
        status="measured",
        reverted=False,
        keep_sitting_skipped=False,
        measurement_utc=quartz_d2048.get("measurement_utc", ab["measurement_utc"]),
        p={
            "quartz": {
                "prompt_tokens": quartz_p["prompt_tokens"],
                "replicates": quartz_p["replicates"],
                "wall_ms": quartz_p["wall_ms"],
                "tok_s": quartz_p["tok_s"],
                "mean_tok_s": quartz_p["mean_tok_s"],
                "cold": quartz_p.get("cold", True),
                "cache_policy": quartz_p.get("cache_policy", "disabled"),
                "attribution": quartz_p.get("attribution"),
                "graphs_created": quartz_p.get("graphs_created", True),
                "prompt_graph_rows": quartz_p.get("prompt_graph_rows", 4096),
            },
            "llama_cpp": {
                "avg_ts": llama_p.get("avg_ts"),
                "avg_ns": llama_p.get("avg_ns"),
                "n_prompt": llama_p.get("n_prompt"),
                "n_batch": llama_p.get("n_batch"),
                "n_ubatch": llama_p.get("n_ubatch"),
                "flash_attn": llama_p.get("flash_attn"),
                "build_commit": llama_p.get("build_commit"),
                "test_time": llama_p.get("test_time"),
            },
        },
        d128={
            "quartz": _engine_block(quartz_d128, "quartz-d128-tokens.json"),
            "llama_cpp": _engine_block(llama_d128),
        },
        d2048={
            "quartz": _engine_block(quartz_d2048, "quartz-d2048-tokens.json"),
            "llama_cpp": _engine_block(llama_d2048),
        },
    )
    if not _keep_predicates(fixture):
        baseline = {
            "gate_i": 128,
            "gate_j": 128,
            "up_i": 128,
            "up_j": 128,
            "down_i": 128,
            "down_j": 128,
        }
        _set_pins(baseline)
        fixture["status"] = "rejected"
        fixture["reverted"] = True
        fixture["selected_ffn_gate_quality_i"] = 128
        fixture["selected_ffn_gate_prompt_tile"] = 128
        fixture["selected_ffn_up_quality_i"] = 128
        fixture["selected_ffn_up_prompt_tile"] = 128
        fixture["selected_ffn_down_quality_i"] = 128
        fixture["selected_ffn_down_prompt_tile"] = 128
        _write_report(fixture)
        _write_rejection(fixture)
        FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
        validate_result(fixture)
        return
    _write_report(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(fixture)
