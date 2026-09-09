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
CONTRACT = ROOT / "pins/opt032_decode_oracle_contract.json"
FIXTURE = ROOT / "fixtures/opt032_decode_oracle.json"
EVIDENCE = ROOT / "evidence/optimization/opt032-decode-oracle"
REPORT = EVIDENCE / "REPORT.md"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
LLAMA_BENCH = ROOT / ".cache/authorities/llama-build/bin/llama-bench"
LLAMA_DECODE = ROOT / ".cache/authorities/llama-build/bin/qw38-llama-decode-oracle"
HISTORICAL_P = 1746.71973
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
LLAMA_DECODE_PREFIX = "QW38_LLAMA_DECODE_ORACLE_RESULT="
PREFILL_ATTR_PREFIX = "QW38_PREFILL_4K_ATTRIBUTION_RESULT="
DECODE_ATTR_PREFIX = "QW38_DECODE_ATTRIBUTION_RESULT="
PREFILL_CATEGORIES = (
    "embedding",
    "mixer_mmq",
    "gdn_core",
    "attention_core",
    "ffn_mmq",
    "logits",
    "commit_sync",
    "graph",
    "other_idle",
)
DECODE_CATEGORIES = (
    "embedding",
    "mixer_mmv",
    "gdn_core",
    "attention_core",
    "ffn_mmv",
    "logits",
    "state_commit",
    "graph",
    "other_idle",
)
PROOF = (
    "claims no performance improvement; D128 and D2048 oracles; "
    "exclusive decode categories; matched pinned llama.cpp; "
    "recorded next-task order; "
    "does not substitute for the 2K llama.cpp parity gate; "
    "llama-bench random decode is informational"
)
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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(v) for v in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def next_task_order(
    p_gap: float, d_gap: float, mmv_ms: float, attn_ms: float
) -> list[str]:
    decode_deficit_larger = d_gap > p_gap
    decode_chain = (
        ["OPT-034", "OPT-036"] if mmv_ms >= attn_ms else ["OPT-036", "OPT-034"]
    )
    prefill_chain = ["OPT-033", "OPT-035", "OPT-037"]
    order: list[str] = []
    if decode_deficit_larger:
        order.append(decode_chain.pop(0))
    else:
        order.append(prefill_chain.pop(0))
    while decode_chain or prefill_chain:
        trailing_prefill = 0
        for task_id in reversed(order):
            if task_id in {"OPT-033", "OPT-035", "OPT-037"}:
                trailing_prefill += 1
            else:
                break
        if decode_deficit_larger:
            if decode_chain and (trailing_prefill >= 2 or not prefill_chain):
                order.append(decode_chain.pop(0))
            elif prefill_chain:
                order.append(prefill_chain.pop(0))
            else:
                order.append(decode_chain.pop(0))
        else:
            if decode_chain and (trailing_prefill >= 1 or not prefill_chain):
                order.append(decode_chain.pop(0))
            elif prefill_chain:
                order.append(prefill_chain.pop(0))
            else:
                order.append(decode_chain.pop(0))
    return order


def _reconstructs(
    categories: dict[str, Any], wall: float, names: tuple[str, ...]
) -> None:
    contract = _contract()
    for name in names:
        value = categories[name]
        assert isinstance(value, (int, float))
        assert value >= 0.0
    attributed = sum(float(categories[name]) for name in names)
    assert math.isclose(
        attributed,
        float(wall),
        rel_tol=contract["rel_tol"],
        abs_tol=contract["abs_tol_ms"],
    )


def _engine_block(block: dict[str, Any], prefix: int) -> None:
    assert block["warmups"] == 3
    assert block["runs"] == 30
    assert block["decode_tokens"] == 256
    assert block["prefix"] == prefix
    assert len(block["tok_s"]) == 30
    assert len(block["warmup_tok_s"]) == 3
    assert len(block["run_wall_ms"]) == 30
    mean = sum(float(v) for v in block["tok_s"]) / 30.0
    assert block["mean_tok_s"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    assert isinstance(block["token_latency_p50_ms"], (int, float))
    assert isinstance(block["token_latency_p95_ms"], (int, float))
    assert isinstance(block["run_mean_token_latency_p95_ms"], (int, float))
    sidecar = ROOT / block["token_latency_sidecar"]
    assert sidecar.is_file()
    latencies = json.loads(sidecar.read_text())
    assert len(latencies) == 30 * 256
    assert block["token_latency_p50_ms"] == pytest.approx(
        percentile(latencies, 0.50), rel=1e-5, abs=1e-5
    )
    assert block["token_latency_p95_ms"] == pytest.approx(
        percentile(latencies, 0.95), rel=1e-5, abs=1e-5
    )


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-032"
    assert result["status"] == "measured"
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["yardstick"] == "p_d128_d2048_oracles"
    assert contract["claims_performance_improvement"] is False
    assert result["claims_performance_improvement"] is False
    assert result["historical_opt026_quartz_mean_tok_s"] == HISTORICAL_P
    assert contract["historical_opt026_quartz_mean_tok_s"] == HISTORICAL_P
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["substitutes_for_opt016"] is False
    assert contract["llama_bench_decode_is_informational"] is True
    assert set(contract["exclusive_decode_categories"]) == set(DECODE_CATEGORIES)
    assert contract["abs_tol_ms"] == 0.05
    assert contract["rel_tol"] == 1e-4
    assert contract["decode_output_tokens"] == 256
    assert contract["decode_warmups"] == 3
    assert contract["decode_measured_runs"] == 30
    assert contract["decode_prefixes"] == [128, 2048]
    assert contract["quartz_graphs"] == "created"
    assert contract["attribution_on_throughput"] == "null"
    assert contract["next_task_order_algorithm"] == (
        "p_vs_d2048_gap_then_transferable_sinks"
    )

    quartz_p = result["p"]["quartz"]
    llama_p = result["p"]["llama_cpp"]
    assert quartz_p["prompt_tokens"] == 4096
    assert quartz_p["replicates"] == 3
    assert quartz_p["cold"] is True
    assert quartz_p["cache_policy"] == "disabled"
    assert quartz_p["attribution"] is None
    assert quartz_p["graphs_created"] is True
    assert quartz_p["prompt_graph_rows"] == 4096
    assert len(quartz_p["wall_ms"]) == 3 and len(quartz_p["tok_s"]) == 3
    mean_p = sum(float(v) for v in quartz_p["tok_s"]) / 3.0
    assert quartz_p["mean_tok_s"] == pytest.approx(mean_p, rel=1e-6, abs=1e-6)
    assert llama_p["n_prompt"] == 4096
    assert "avg_ts" in llama_p

    for prefix, key in ((128, "d128"), (2048, "d2048")):
        block = result[key]
        assert "n_prompt" not in block["llama_cpp"]
        assert block["quartz"]["graphs_created"] is True
        assert block["quartz"]["attribution"] is None
        assert block["llama_cpp"]["n_gpu_layers"] == 99
        assert block["llama_cpp"]["n_ctx"] == 4096
        assert block["llama_cpp"]["n_batch"] == 2048
        assert block["llama_cpp"]["n_ubatch"] == 512
        _engine_block(block["quartz"], prefix)
        _engine_block(block["llama_cpp"], prefix)

    prefill = result["prefill_attribution"]
    assert prefill["task"] == "OPT-032"
    assert prefill["prompt_tokens"] == 4096
    assert prefill["prompt_graph_launches"] == 64
    assert set(prefill["categories_ms"]) == set(PREFILL_CATEGORIES)
    assert "gdn" not in prefill["categories_ms"]
    _reconstructs(
        prefill["categories_ms"], float(prefill["wall_ms"]), PREFILL_CATEGORIES
    )

    for key, prefix in (
        ("decode_attribution_d128", 128),
        ("decode_attribution_d2048", 2048),
    ):
        decoded = result[key]
        assert decoded["prefix"] == prefix
        assert set(decoded["categories_ms"]) == set(DECODE_CATEGORIES)
        assert "gdn" not in decoded["categories_ms"]
        assert "attention" not in decoded["categories_ms"]
        cats = decoded["categories_ms"]
        assert cats["mixer_mmv"] > 0.0
        assert cats["gdn_core"] > 0.0
        assert cats["attention_core"] > 0.0
        assert cats["ffn_mmv"] > 0.0
        _reconstructs(cats, float(decoded["wall_ms"]), DECODE_CATEGORIES)

    p_gap = float(llama_p["avg_ts"]) / float(quartz_p["mean_tok_s"])
    d_gap = float(result["d2048"]["llama_cpp"]["mean_tok_s"]) / float(
        result["d2048"]["quartz"]["mean_tok_s"]
    )
    assert result["p_gap"] == pytest.approx(p_gap, rel=1e-9, abs=1e-9)
    assert result["d2048_gap"] == pytest.approx(d_gap, rel=1e-9, abs=1e-9)
    informational = result["llama_bench_informational"]
    assert "d128" in informational and "d2048" in informational
    assert informational["d2048"].get("n_prompt") == 0
    decode_deficit_larger = d_gap > p_gap
    assert result["decode_deficit_larger"] is decode_deficit_larger
    mmv_ms = float(result["mmv_ms"])
    attn_ms = float(result["attention_core_ms_d2048"])
    d2048_cats = result["decode_attribution_d2048"]["categories_ms"]
    assert mmv_ms == pytest.approx(
        float(d2048_cats["mixer_mmv"]) + float(d2048_cats["ffn_mmv"]),
        rel=1e-9,
        abs=1e-9,
    )
    assert attn_ms == pytest.approx(
        float(d2048_cats["attention_core"]), rel=1e-9, abs=1e-9
    )
    expected_order = next_task_order(p_gap, d_gap, mmv_ms, attn_ms)
    assert result["next_task_order"] == expected_order
    assert result["next_task"] == expected_order[0]
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    proof = result["proof_limit"]
    for phrase in contract["proof_limit"]:
        assert phrase in proof
    assert result["report_path"] == contract["report_path"]
    assert "/tmp" not in result["report_path"]
    assert REPORT.is_file()
    report = REPORT.read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report


def test_opt032_contract_and_fixture_are_connected() -> None:
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
    assert contract["quartz_warmups"] == 0
    assert contract["claims_performance_improvement"] is False


def test_opt032_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []

    def use_llama_bench_denominator(changed: dict[str, Any]) -> None:
        changed["d2048"]["llama_cpp"] = dict(
            changed["llama_bench_informational"]["d2048"]
        )

    for mutate in (
        lambda x: x["d128"]["quartz"].__setitem__("prefix", 127),
        lambda x: x["d2048"]["quartz"].__setitem__("decode_tokens", 255),
        lambda x: x["d128"]["quartz"].__setitem__("warmups", 0),
        use_llama_bench_denominator,
        lambda x: x.__setitem__("claims_performance_improvement", True),
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x["decode_attribution_d2048"]["categories_ms"].__setitem__(
            "gdn", 1.0
        ),
        lambda x: x.__setitem__(
            "next_task_order", list(reversed(x["next_task_order"]))
        ),
        lambda x: x.__setitem__("task", "OPT-021"),
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x.__setitem__("report_path", "/tmp/opt032/REPORT.md"),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
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


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def _parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    records = [
        json.loads(line.removeprefix(prefix))
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    assert len(records) == 1
    return records[0]


def _parse_llama_bench(blob: str, predicate: Any, label: str) -> list[Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(blob):
        if char != "[":
            continue
        try:
            payload, _ = decoder.raw_decode(blob[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list) and payload and predicate(payload[0]):
            return payload
    raise AssertionError(f"{label} was not found\n" + blob)


def _ensure_llama_tools() -> None:
    if not LLAMA_BENCH.is_file():
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
        _run(configure)
        _run(build)
    if LLAMA_DECODE.is_file():
        return
    configure_adapter = [
        *_common(LLAMA_IMAGE),
        "cmake",
        "-S",
        "/workspace/tools/llama_authority",
        "-B",
        "/workspace/.cache/authorities/llama-adapter-build",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_SOURCE=/workspace/.cache/authorities/llama.cpp",
        "-DLLAMA_BUILD=/workspace/.cache/authorities/llama-build",
    ]
    build_adapter = [
        *_common(LLAMA_IMAGE),
        "cmake",
        "--build",
        "/workspace/.cache/authorities/llama-adapter-build",
        "--target",
        "qw38-llama-decode-oracle",
        "-j",
        "6",
    ]
    _run(configure_adapter)
    _run(build_adapter)
    assert LLAMA_DECODE.is_file(), "qw38-llama-decode-oracle was not built"


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
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "llama-bench-4k.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload[0]


def _run_llama_bench_decode(depth: int) -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
        "-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf "
        f"-p 0 -n 256 -d {depth} --no-warmup -r 30 -ngl 99 -o json",
    ]
    completed = _run(command)
    payload = _parse_llama_bench(
        completed.stdout + completed.stderr,
        lambda row: True,
        f"llama-bench JSON for depth {depth}",
    )
    (EVIDENCE / f"llama-bench-d{depth}.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return payload[0]


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
        "-DQW38_DIAGNOSTIC_TRACE",
        source,
        *NVCC_OBJECTS,
        "-o",
        output,
    ]


def _ensure_quartz_objects() -> None:
    _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])


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
    assert "status=passed" in outputs[-1]
    (EVIDENCE / "quartz-p.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _run_quartz_decode(prefix: int) -> dict[str, Any]:
    binary = "build/qw38-cuda-decode-oracle-test"
    commands = [
        _nvcc("cuda/decode_oracle_test.cu", binary),
        [
            *_common(IMAGE),
            f"./{binary}",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
            str(prefix),
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], DECODE_PREFIX)
    assert "status=passed" in outputs[-1]
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


def _run_prefill_attribution() -> dict[str, Any]:
    binary = "build/qw38-cuda-prefill-4k-attribution-test"
    commands = [
        _nvcc("cuda/prefill_4k_attribution_test.cu", binary),
        [*_common(IMAGE), f"./{binary}", "models/Qwen3.8-27B-Q4_K_M.gguf"],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], PREFILL_ATTR_PREFIX)
    assert "status=passed" in outputs[-1]
    (EVIDENCE / "quartz-prefill-attribution-4k.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


def _run_decode_attribution(prefix: int) -> dict[str, Any]:
    binary = "build/qw38-cuda-decode-attribution-test"
    commands = [
        _nvcc("cuda/decode_attribution_test.cu", binary),
        [
            *_common(IMAGE),
            f"./{binary}",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
            str(prefix),
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], DECODE_ATTR_PREFIX)
    assert "status=passed" in outputs[-1]
    (EVIDENCE / f"quartz-decode-attribution-d{prefix}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


def _write_token_sidecar(name: str, latencies: list[Any]) -> str:
    path = EVIDENCE / name
    path.write_text(json.dumps(latencies) + "\n")
    return f"evidence/optimization/opt032-decode-oracle/{name}"


def _engine_from_live(record: dict[str, Any], sidecar_name: str) -> dict[str, Any]:
    sidecar = _write_token_sidecar(sidecar_name, record["token_latency_ms"])
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
        "token_latency_sidecar": sidecar,
    }
    for key in ("n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"):
        if key in record:
            block[key] = record[key]
    if "graphs_created" in record:
        block["graphs_created"] = record["graphs_created"]
        block["attribution"] = record.get("attribution")
        block["cache_policy"] = record.get("cache_policy", "disabled")
    return block


def _write_report(fixture: dict[str, Any]) -> None:
    quartz_p = fixture["p"]["quartz"]
    llama_p = fixture["p"]["llama_cpp"]
    d128_q = fixture["d128"]["quartz"]
    d128_l = fixture["d128"]["llama_cpp"]
    d2048_q = fixture["d2048"]["quartz"]
    d2048_l = fixture["d2048"]["llama_cpp"]
    order = ", ".join(fixture["next_task_order"])
    text = f"""# OPT-032 — Freeze decode oracle and refresh sink attribution

## Claim labels and proof limits

This increment **claims no performance improvement**. It freezes **D128 and D2048 oracles**
with **exclusive decode categories** and **matched pinned llama.cpp** public-API
measurements, plus a **recorded next-task order**. This protocol
**does not substitute for the 2K llama.cpp parity gate**.
**llama-bench random decode is informational**.

Live numbers in `fixtures/opt032_decode_oracle.json` and this directory are the
Measured same-sitting exclusive RTX 5090 record. Historical OPT-026 Quartz mean
{HISTORICAL_P} tok/s is contract transparency, not a keep.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| P | exact 4096, attribution null, graphs created, 0 warm-ups, 3 cold replicates |
| D128/D2048 | prefix 128 or 2048 then 256 predetermined tokens, 3 warm + 30 measured |
| llama.cpp P | `llama-bench -p 4096 -n 0 --no-warmup -r 3 -ngl 99` |
| llama.cpp D | `qw38-llama-decode-oracle` via pinned `llama.h` and `llama_time_us` |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- P Quartz mean tok/s: {quartz_p["mean_tok_s"]} ; llama.cpp avg_ts: {llama_p["avg_ts"]}
- D128 Quartz mean tok/s: {d128_q["mean_tok_s"]} ; llama.cpp mean tok/s: {d128_l["mean_tok_s"]}
- D2048 Quartz mean tok/s: {d2048_q["mean_tok_s"]} ; llama.cpp mean tok/s: {d2048_l["mean_tok_s"]}
- p_gap: {fixture["p_gap"]} ; d2048_gap: {fixture["d2048_gap"]} ; decode_deficit_larger: {json.dumps(fixture["decode_deficit_larger"])}
- mmv_ms: {fixture["mmv_ms"]} ; attention_core_ms_d2048: {fixture["attention_core_ms_d2048"]}
- next_task: {fixture["next_task"]}
- next_task_order: {order}
- claims_performance_improvement: false
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def test_opt032_native_decode_oracle() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    _ensure_llama_tools()
    _ensure_quartz_objects()

    llama_p = _run_llama_bench_p()
    quartz_p = _run_quartz_p()
    llama_d128 = _run_llama_decode(128)
    quartz_d128 = _run_quartz_decode(128)
    llama_d2048 = _run_llama_decode(2048)
    quartz_d2048 = _run_quartz_decode(2048)
    bench_d128 = _run_llama_bench_decode(128)
    bench_d2048 = _run_llama_bench_decode(2048)
    prefill_attr = _run_prefill_attribution()
    decode_attr_d128 = _run_decode_attribution(128)
    decode_attr_d2048 = _run_decode_attribution(2048)

    p_gap = float(llama_p["avg_ts"]) / float(quartz_p["mean_tok_s"])
    d_gap = float(llama_d2048["mean_tok_s"]) / float(quartz_d2048["mean_tok_s"])
    mmv_ms = float(decode_attr_d2048["categories_ms"]["mixer_mmv"]) + float(
        decode_attr_d2048["categories_ms"]["ffn_mmv"]
    )
    attn_ms = float(decode_attr_d2048["categories_ms"]["attention_core"])
    order = next_task_order(p_gap, d_gap, mmv_ms, attn_ms)
    fixture = {
        "schema_version": 1,
        "task": "OPT-032",
        "status": "measured",
        "measurement_utc": quartz_p["measurement_utc"],
        "device": quartz_p["device"],
        "compute_capability": quartz_p["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "historical_opt026_quartz_mean_tok_s": HISTORICAL_P,
        "claims_performance_improvement": False,
        "p": {
            "quartz": {
                "prompt_tokens": 4096,
                "replicates": 3,
                "wall_ms": quartz_p["wall_ms"],
                "tok_s": quartz_p["tok_s"],
                "mean_tok_s": float(quartz_p["mean_tok_s"]),
                "cold": True,
                "cache_policy": "disabled",
                "attribution": None,
                "graphs_created": True,
                "prompt_graph_rows": 4096,
            },
            "llama_cpp": {
                "avg_ts": float(llama_p["avg_ts"]),
                "avg_ns": llama_p["avg_ns"],
                "n_prompt": 4096,
                "n_batch": llama_p.get("n_batch", 2048),
                "n_ubatch": llama_p.get("n_ubatch", 512),
                "flash_attn": llama_p.get("flash_attn", -1),
                "build_commit": llama_p.get("build_commit", "cc83d7b"),
                "test_time": llama_p.get("test_time", quartz_p["measurement_utc"]),
            },
        },
        "d128": {
            "quartz": _engine_from_live(quartz_d128, "quartz-d128-tokens.json"),
            "llama_cpp": _engine_from_live(llama_d128, "llama-decode-d128-tokens.json"),
        },
        "d2048": {
            "quartz": _engine_from_live(quartz_d2048, "quartz-d2048-tokens.json"),
            "llama_cpp": _engine_from_live(
                llama_d2048, "llama-decode-d2048-tokens.json"
            ),
        },
        "prefill_attribution": {
            "task": "OPT-032",
            "prompt_tokens": 4096,
            "prompt_graph_launches": prefill_attr["prompt_graph_launches"],
            "categories_ms": prefill_attr["categories_ms"],
            "attributed_sum_ms": prefill_attr["attributed_sum_ms"],
            "wall_ms": prefill_attr["wall_ms"],
            "tok_s": prefill_attr["tok_s"],
        },
        "decode_attribution_d128": {
            "prefix": 128,
            "categories_ms": decode_attr_d128["categories_ms"],
            "attributed_sum_ms": decode_attr_d128["attributed_sum_ms"],
            "wall_ms": decode_attr_d128["wall_ms"],
        },
        "decode_attribution_d2048": {
            "prefix": 2048,
            "categories_ms": decode_attr_d2048["categories_ms"],
            "attributed_sum_ms": decode_attr_d2048["attributed_sum_ms"],
            "wall_ms": decode_attr_d2048["wall_ms"],
        },
        "p_gap": p_gap,
        "d2048_gap": d_gap,
        "decode_deficit_larger": d_gap > p_gap,
        "mmv_ms": mmv_ms,
        "attention_core_ms_d2048": attn_ms,
        "next_task": order[0],
        "next_task_order": order,
        "llama_bench_informational": {
            "d128": bench_d128,
            "d2048": bench_d2048,
        },
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt032-decode-oracle/REPORT.md",
    }
    _write_report(fixture)
    validate_result(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
