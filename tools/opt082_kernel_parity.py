"""OPT-082 bounded CUDA kernel-parity suite runner.

Host catalog, association envelopes, and optional native GPU execution.
No production keep and no throughput claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.kernel_parity import (  # noqa: E402
    ASSOCIATION,
    CUD001_HISTORICAL_ABS,
    FAMILIES,
    OPT074_FAMILY_ADMISSION_REQUIRED,
    POLICY_STATEMENT,
    SAME_MATH,
    STAGING_SUM_Q,
    STAGING_SUM_X,
    KernelParityError,
    abs_tolerance,
    assert_cpp_matches_python,
    check_close,
    family_envelope,
)
from tools.run_optimization_task import (  # noqa: E402
    parse_native_observation,
)

ITERATION = ROOT / "pins/opt082_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt082_kernel_parity.json"
REPORT = ROOT / "evidence/optimization/opt082-kernel-parity/REPORT.md"
NATIVE = ROOT / "build/qw38-cuda-opt082-kernel-parity-test"
NATIVE_REL = "build/qw38-cuda-opt082-kernel-parity-test"
MAKEFILE = ROOT / "Makefile"
IMAGE = "qw38-cuda:13.0.2"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
PHASES = ("q4", "q8-q6", "parity")
MMV_M = (1, 3, 17)
MMQ_N = (4, 8)
K_TINY_MEDIUM = (256, 512)
K_CORE = (256, 512, 2048)
K_PROD = (4096, 5120, 6144)
Q4_MMV = ("packed", "integer_q8", "integer_q8_late")
Q4_MMQ = ("fma_async", "fma_async_x")
Q8_LAYOUTS = ("r1_w4", "r2_w2")
PROOF = (
    "no throughput claim",
    "no production pin change",
    "kernel admission is not model quality",
    "no full production M",
    "generated quantized blocks only",
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def docker_common(tier: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        f"QW38_CUDA_TEST_TIER={tier}",
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        "/workspace",
        IMAGE,
    ]


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    if listed and not listed[0].startswith("docker"):
        listed = [*docker_common(tier), *listed]
    return subprocess.run(listed, cwd=ROOT, capture_output=True, text=True)


def catalog(phase: str) -> list[dict[str, Any]]:
    """Explicit product of families, candidates, shapes, and classes."""
    rows: list[dict[str, Any]] = []
    q4 = phase in {"smoke", "q4", "parity"}
    q8 = phase in {"q8-q6", "parity"}
    ks = K_CORE if phase == "parity" else K_TINY_MEDIUM
    if phase == "smoke":
        ks = (256,)
    ms = (1,) if phase == "smoke" else MMV_M
    cands = ("packed",) if phase == "smoke" else Q4_MMV

    def add(**kwargs: Any) -> None:
        family = str(kwargs["family"])
        envelope = family_envelope(family)
        columns = int(kwargs["K"])
        rows.append(
            {
                **kwargs,
                "abs_scale": envelope.get("abs_scale"),
                "rel_tol": envelope.get("rel_tol"),
                "abs_tol": (
                    abs_tolerance(float(envelope["abs_scale"]), columns)
                    if envelope.get("applies") and kwargs["class_name"] == ASSOCIATION
                    else 0.0
                ),
            }
        )

    if q4:
        for candidate in cands:
            for m in ms:
                for k in ks if phase != "parity" else K_CORE:
                    add(
                        id=f"Q4_K_mmv_{candidate}_M{m}_N1_K{k}_random_assoc",
                        family="Q4_K",
                        class_name=ASSOCIATION,
                        candidate=candidate,
                        op="mmv",
                        pattern="random",
                        M=m,
                        N=1,
                        K=k,
                        expected_launch={
                            "packed": "quant_mmv_packed",
                            "integer_q8": "q4k_coop_mmv_q8",
                            "integer_q8_late": "q4k_coop_mmv_late_q8",
                        }[candidate],
                        expect_fallback=False,
                    )
                    if phase == "smoke":
                        return rows
        if phase != "smoke":
            for pattern in ("zero", "cancel", "minmax", "half"):
                add(
                    id=f"Q4_K_mmv_packed_M1_N1_K256_{pattern}_assoc",
                    family="Q4_K",
                    class_name=ASSOCIATION,
                    candidate="packed",
                    op="mmv",
                    pattern=pattern,
                    M=1,
                    N=1,
                    K=256,
                    expected_launch="quant_mmv_packed",
                    expect_fallback=False,
                )
            mmq_k = K_CORE if phase == "parity" else K_TINY_MEDIUM
            for candidate in Q4_MMQ:
                for m in MMV_M:
                    for n in MMQ_N:
                        for k in mmq_k:
                            add(
                                id=(
                                    f"Q4_K_mmq_{candidate}_M{m}_N{n}_K{k}"
                                    "_unaligned_assoc"
                                ),
                                family="Q4_K",
                                class_name=ASSOCIATION,
                                candidate=candidate,
                                op="mmq",
                                pattern="unaligned",
                                M=m,
                                N=n,
                                K=k,
                                expected_launch="sync_fallback",
                                expect_fallback=True,
                            )
                expected = "fma_async_x" if candidate == "fma_async_x" else "fma_async"
                add(
                    id=f"Q4_K_mmq_{candidate}_M128_N128_K256_aligned_assoc",
                    family="Q4_K",
                    class_name=ASSOCIATION,
                    candidate=candidate,
                    op="mmq",
                    pattern="aligned",
                    M=128,
                    N=128,
                    K=256,
                    expected_launch=expected,
                    expect_fallback=False,
                )
            if phase == "parity":
                for k in K_PROD:
                    add(
                        id=f"Q4_K_mmv_packed_M1_N1_K{k}_random_assoc",
                        family="Q4_K",
                        class_name=ASSOCIATION,
                        candidate="packed",
                        op="mmv",
                        pattern="random",
                        M=1,
                        N=1,
                        K=k,
                        expected_launch="quant_mmv_packed",
                        expect_fallback=False,
                    )
            add(
                id="Q4_K_mmv_packed_down_M3_N1_K256_random_assoc",
                family="Q4_K",
                class_name=ASSOCIATION,
                candidate="packed",
                op="down",
                pattern="random",
                M=3,
                N=1,
                K=256,
                expected_launch="quant_mmv_packed",
                expect_fallback=False,
            )
            for spec in (
                {
                    "id": "Q4_K_mmv_two_consumers_M3_N1_K256_random_same",
                    "op": "staging_two_consumers",
                    "expected_launch": "quant_mmv_prequant_packed",
                },
                {
                    "id": "Q4_K_mmv_eager_vs_graph_M1_N1_K256_random_same",
                    "op": "eager_vs_captured",
                    "expected_launch": "quant_mmv_packed",
                    "M": 1,
                },
                {
                    "id": "Q4_K_fused_vs_independent_M3_N1_K256_random_same",
                    "op": "fused_gate_up",
                    "expected_launch": "q4k_gate_up_swiglu_prequant",
                },
            ):
                add(
                    id=spec["id"],
                    family="Q4_K",
                    class_name=SAME_MATH,
                    candidate="packed",
                    op=spec["op"],
                    pattern="random",
                    M=int(spec.get("M", 3)),
                    N=1,
                    K=256,
                    expected_launch=spec["expected_launch"],
                    expect_fallback=False,
                )
    if q8:
        qk = K_CORE if phase == "parity" else K_TINY_MEDIUM
        for candidate in Q8_LAYOUTS:
            for m in MMV_M:
                for k in qk:
                    add(
                        id=f"Q8_0_mmv_{candidate}_M{m}_N1_K{k}_random_assoc",
                        family="Q8_0",
                        class_name=ASSOCIATION,
                        candidate=candidate,
                        op="mmv",
                        pattern="random",
                        M=m,
                        N=1,
                        K=k,
                        expected_launch=candidate,
                        expect_fallback=False,
                    )
        for m in MMV_M:
            for k in qk:
                add(
                    id=f"Q6_K_mmv_integer_q8_1_M{m}_N1_K{k}_random_assoc",
                    family="Q6_K",
                    class_name=ASSOCIATION,
                    candidate="integer_q8_1",
                    op="mmv",
                    pattern="random",
                    M=m,
                    N=1,
                    K=k,
                    expected_launch="integer_q8_1",
                    expect_fallback=False,
                )
        add(
            id="Q8_0_q8_1_sum_q_producer_K256_typed",
            family="Q8_0",
            class_name=ASSOCIATION,
            candidate="q8_1_sum_q",
            op="staging_typed",
            pattern="random",
            M=1,
            N=1,
            K=256,
            expected_launch="quantize_bf16_q8_1",
            expect_fallback=False,
        )
        add(
            id="Q8_0_q8_1_cannot_equate_sum_q_sum_x",
            family="Q8_0",
            class_name=ASSOCIATION,
            candidate="q8_1_typed",
            op="staging_typed",
            pattern="host",
            M=1,
            N=1,
            K=256,
            expected_launch="reject_sum_q_vs_sum_x",
            expect_fallback=False,
        )
    return rows


def required_coverage() -> dict[str, Any]:
    cases = catalog("parity")
    families = sorted({row["family"] for row in cases})
    candidates = sorted({row["candidate"] for row in cases})
    ops = sorted({row["op"] for row in cases})
    return {
        "families": families,
        "candidates": candidates,
        "ops": ops,
        "q4_mmv": list(Q4_MMV),
        "q4_mmq": list(Q4_MMQ),
        "q8": list(Q8_LAYOUTS),
        "q6": ["integer_q8_1"],
        "M": list(MMV_M),
        "N_mmq": list(MMQ_N),
        "K_core": list(K_CORE),
        "K_prod": list(K_PROD),
        "has_same_math": any(row["class_name"] == SAME_MATH for row in cases),
        "has_fused": any(row["op"] == "fused_gate_up" for row in cases),
        "has_down": any(row["op"] == "down" for row in cases),
        "has_staging": any("staging" in row["op"] for row in cases),
        "case_count": len(cases),
    }


def host_synthetic_cases() -> list[dict[str, Any]]:
    columns = 256
    return [
        {
            "id": "host_q4_or_abs",
            "family": "Q4_K",
            "class_name": ASSOCIATION,
            "columns": columns,
            "got": [1.0, 1.0],
            "ref": [0.5, 0.5],
            "expect_pass": True,
        },
        {
            "id": "host_q8_both_fail",
            "family": "Q8_0",
            "class_name": ASSOCIATION,
            "columns": columns,
            "got": [20.0, 20.0],
            "ref": [10.0, 10.0],
            "expect_pass": False,
        },
        {
            "id": "host_q6_reuses_q4",
            "family": "Q6_K",
            "class_name": ASSOCIATION,
            "columns": columns,
            "got": [11.0],
            "ref": [10.0],
            "expect_pass": True,
        },
        {
            "id": "host_same_math_identity",
            "family": "Q8_0",
            "class_name": SAME_MATH,
            "columns": columns,
            "got": [1.25, -2.0],
            "ref": [1.25, -2.0],
            "expect_pass": True,
        },
        {
            "id": "host_nonfinite",
            "family": "Q4_K",
            "class_name": ASSOCIATION,
            "columns": columns,
            "got": [1.0, math.nan],
            "ref": [1.0, 1.0],
            "expect_pass": False,
        },
    ]


def evaluate_host_cases() -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for case in host_synthetic_cases():
        python = check_close(
            list(case["got"]),
            list(case["ref"]),
            family=str(case["family"]),
            columns=int(case["columns"]),
            class_name=str(case["class_name"]),
        )
        evaluated.append(
            {
                "id": case["id"],
                "family": case["family"],
                "class": case["class_name"],
                "pass": python["pass"],
                "reason": python["reason"],
                "max_abs": python["max_abs"],
                "max_rel": python["max_rel"],
                "matched_expectation": python["pass"] is bool(case["expect_pass"]),
            }
        )
    return evaluated


def q8_1_typed_reject() -> dict[str, Any]:
    rejected = False
    message = ""
    try:
        check_close(
            [1.0, 2.0],
            [1.0, 2.0],
            family="Q8_0",
            columns=256,
            class_name=ASSOCIATION,
            candidate_field=STAGING_SUM_Q,
            reference_field=STAGING_SUM_X,
        )
    except KernelParityError as exc:
        rejected = True
        message = str(exc)
    return {"typed_comparison_rejected": rejected, "message": message, "pass": rejected}


def source_uses_canonical_checker() -> dict[str, Any]:
    files = (
        ROOT / "cuda/quant_mmv_test.cu",
        ROOT / "cuda/prompt_mmq_test.cu",
        ROOT / "cuda/ffn_tile_ab_test.cu",
        ROOT / "cuda/opt082_kernel_parity_test.cu",
    )
    missing_include = [
        str(path.relative_to(ROOT))
        for path in files
        if '#include "kernel_parity.cuh"' not in path.read_text(encoding="utf-8")
    ]
    third_copy = []
    for path in files[:3]:
        text = path.read_text(encoding="utf-8")
        if "constexpr float kAbsScale" in text:
            third_copy.append(str(path.relative_to(ROOT)))
    makefile = MAKEFILE.read_text(encoding="utf-8")
    return {
        "missing_include": missing_include,
        "duplicate_abs_scale": third_copy,
        "makefile_target": "qw38-cuda-opt082-kernel-parity-test" in makefile,
        "makefile_diagnostics": "cuda-opt082-diagnostics" in makefile,
        "pass": not missing_include
        and not third_copy
        and "qw38-cuda-opt082-kernel-parity-test" in makefile,
    }


def gpu_available() -> tuple[bool, str]:
    inspect = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode != 0:
        return False, f"docker image {IMAGE} missing"
    smi = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
    )
    if smi.returncode != 0 or "GPU" not in (smi.stdout or ""):
        return False, "nvidia-smi did not list a GPU"
    return True, ""


def run_native(
    phase: str,
    *,
    runner: NativeRunner | None = None,
) -> dict[str, Any]:
    execute = runner or default_native_runner
    available, blocker = gpu_available()
    record: dict[str, Any] = {
        "available": available,
        "blocker": blocker,
        "ran": False,
        "success": False,
        "cases": [],
        "stdout": "",
        "stderr": "",
    }
    if not available:
        return record
    if not NATIVE.is_file():
        built = execute(["make", "cuda-opt082-diagnostics"], "smoke")
        if built.returncode != 0:
            record["blocker"] = "native binary failed to compile:\n" + (
                built.stderr or built.stdout
            )
            return record
    completed = execute([f"./{NATIVE_REL}", "--phase", phase], "correctness")
    record["ran"] = True
    record["stdout"] = completed.stdout
    record["stderr"] = completed.stderr
    record["returncode"] = completed.returncode
    observation = parse_native_observation(completed.stdout)
    cases = []
    for line in (completed.stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("QW38_OPT082_CASE="):
            try:
                cases.append(json.loads(stripped.split("=", 1)[1]))
            except json.JSONDecodeError:
                continue
    record["cases"] = cases
    record["observation"] = {
        key: observation.get(key)
        for key in ("success", "case_count", "passed", "failed", "phase")
        if key in observation
    }
    record["success"] = completed.returncode == 0 and bool(
        observation.get("success", completed.returncode == 0)
    )
    if not record["success"] and not record["blocker"]:
        record["blocker"] = f"native phase {phase} failed rc={completed.returncode}"
    return record


def build_fixture(
    phase: str,
    host_cases: Sequence[Mapping[str, Any]],
    typed: Mapping[str, Any],
    sources: Mapping[str, Any],
    native: Mapping[str, Any],
    constants: Mapping[str, Any],
) -> dict[str, Any]:
    cases = catalog(phase)
    native_by_id = {row.get("id"): row for row in native.get("cases", [])}
    merged = []
    for row in cases:
        gpu = native_by_id.get(row["id"], {})
        merged.append(
            {
                **row,
                "class": row["class_name"],
                "gpu_pass": gpu.get("pass"),
                "gpu_reason": gpu.get("reason"),
                "gpu_launched": gpu.get("launched"),
                "gpu_fallback": gpu.get("fallback"),
                "gpu_max_abs": gpu.get("max_abs"),
                "gpu_max_rel": gpu.get("max_rel"),
                "gpu_rms": gpu.get("rms"),
                "gpu_failing_count": gpu.get("failing_count"),
                "gpu_nonfinite_count": gpu.get("nonfinite_count"),
            }
        )
    shipping = [
        row
        for row in merged
        if row["candidate"] in {"packed", "r2_w2", "fma_async_x", "integer_q8_1"}
        and row["class_name"] == ASSOCIATION
        and not row.get("expect_fallback")
    ]
    return {
        "schema_version": 1,
        "task": "OPT-082",
        "status": "kernel_parity_suite",
        "phase": phase,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "opt074_family_admission_required": OPT074_FAMILY_ADMISSION_REQUIRED,
        "policy_statement": POLICY_STATEMENT,
        "families": FAMILIES,
        "association": {
            "Q4_K": "0.20 * sqrt(K) OR 5%",
            "Q6_K": "0.20 * sqrt(K) OR 5%",
            "Q8_0": "0.05 * sqrt(K) OR 5%",
        },
        "host_cases": list(host_cases),
        "q8_1_typed": dict(typed),
        "sources": dict(sources),
        "gpu": {
            "available": native.get("available"),
            "ran": native.get("ran"),
            "success": native.get("success"),
            "blocker": native.get("blocker") or "",
            "observation": native.get("observation", {}),
        },
        "catalog_count": len(cases),
        "gpu_case_count": len(native.get("cases", [])),
        "cases": merged,
        "shipping_measured": [
            {
                "id": row["id"],
                "candidate": row["candidate"],
                "gpu_pass": row.get("gpu_pass"),
                "gpu_reason": row.get("gpu_reason"),
                "gpu_launched": row.get("gpu_launched"),
            }
            for row in shipping
        ],
        "coverage": required_coverage(),
        "cpp_constants": dict(constants),
        "historical_cud001_abs": CUD001_HISTORICAL_ABS,
        "proof_limit": list(PROOF),
        "report_path": "evidence/optimization/opt082-kernel-parity/REPORT.md",
    }


def write_report(fixture: Mapping[str, Any]) -> None:
    host_lines = [
        f"| {row['id']} | {row['family']} | {row['class']} | "
        f"{'pass' if row['pass'] else 'fail'} | {row['reason']} |"
        for row in fixture["host_cases"]
    ]
    gpu = fixture["gpu"]
    shipping_lines = [
        f"| {row['candidate']} | {row['id']} | {row.get('gpu_pass')} | "
        f"{row.get('gpu_launched') or ''} | {row.get('gpu_reason') or ''} |"
        for row in fixture["shipping_measured"]
    ]
    documented = [
        row for row in fixture.get("cases", []) if row.get("gpu_pass") is False
    ]
    documented_lines = [
        f"| {row['id']} | {row.get('candidate')} | {row.get('gpu_reason')} | "
        f"{row.get('gpu_max_abs')} | {row.get('abs_tol')} |"
        for row in documented
    ]
    blocker = gpu.get("blocker") or "none"
    text = f"""# OPT-082 — CUDA kernel-parity suite

Status: **kernel-parity suite**. `claims_throughput: false`. No production pin
or selector change. Authority remains llama.cpp `{LLAMA_REV}` and GGUF SHA-256
`{GGUF_SHA}`.

## Policy

{fixture["policy_statement"]}

Reference = independent CPU/dequant of the same quantized weights and
staged/input activations. Classes: `quantized_operation_association`
(`abs > abs_scale*sqrt(K)` **and** `rel > rel_tol` fails an element) and
`same_math_equivalence` (bit identity). Nonfinite count must be 0.

| Family | abs_scale | rel_tol |
|---|---:|---:|
| Q8_0 | 0.05 | 0.05 |
| Q4_K | 0.20 | 0.05 |
| Q6_K | 0.20 | 0.05 |

Canonical checkers: `cuda/kernel_parity.cuh` and `tools/kernel_parity.py`.
Existing `ds4_q4k_association_ok` / `ds4_q8_association_ok` wrappers now call
the canonical helper.

## Host cases

| Case | Family | Class | Verdict | Reason |
|---|---|---|---|---|
{chr(10).join(host_lines)}

Q8_1 typed `sum_q` versus `sum_x` rejected:
{fixture["q8_1_typed"]["typed_comparison_rejected"]}.

## GPU

available={gpu.get("available")} ran={gpu.get("ran")} success={gpu.get("success")}
blocker={blocker}
catalog_count={fixture["catalog_count"]} gpu_case_count={fixture["gpu_case_count"]}

Aligned MMQ uses M=128 N=128 so `fma_async` / `fma_async_x` actually launch.
Listed N∈{{4,8}} MMQ shapes are unaligned fallback association cases and are
not counted as the candidate kernel. Down projection is packed Q4_K MMV
(`Q4_K_mmv_packed_down_M3_N1_K256_random_assoc`). Native success fail-closes
on wrong selector, nonfinite, fallback-measured-as-candidate, and CUDA
errors. Association and same-math misses are recorded as documented fails.

## Shipping paths (pass or documented fail)

| Candidate | Case | gpu_pass | launched | reason |
|---|---|---|---|---|
{chr(10).join(shipping_lines) if shipping_lines else "| none | | | | |"}

## Documented GPU fails

| Case | Candidate | Reason | max_abs | abs_tol |
|---|---|---|---:|---:|
{chr(10).join(documented_lines) if documented_lines else "| none | | | | |"}

## Proof limit

- no throughput claim
- no production pin change
- kernel admission is not model quality
- no full production M
- generated quantized blocks only
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def run_phase(
    phase: str,
    run_dir: Path | None = None,
    *,
    runner: NativeRunner | None = None,
    skip_gpu: bool = False,
) -> dict[str, Any]:
    if phase not in {"smoke", *PHASES}:
        raise ValueError(f"unknown OPT-082 phase {phase}")
    host_cases = evaluate_host_cases()
    typed = q8_1_typed_reject()
    sources = source_uses_canonical_checker()
    constants = assert_cpp_matches_python()
    host_ok = (
        all(row["matched_expectation"] for row in host_cases)
        and typed["pass"]
        and sources["pass"]
        and constants["q4_k_abs_scale"] == 0.20
        and constants["q8_0_abs_scale"] == 0.05
    )
    native: dict[str, Any]
    if skip_gpu:
        native = {
            "available": False,
            "blocker": "skip_gpu",
            "ran": False,
            "success": False,
            "cases": [],
        }
    else:
        native = run_native(phase, runner=runner)
    fixture = build_fixture(phase, host_cases, typed, sources, native, constants)
    write_json(FIXTURE, fixture)
    write_report(fixture)
    gpu_ok = bool(native.get("success")) if native.get("ran") else True
    success = host_ok and gpu_ok
    payload: dict[str, Any] = {
        "success": success,
        "result_class": "ok" if success else "kernel_parity_failure",
        "gpu_work": bool(native.get("ran")),
        "task": "OPT-082",
        "phase": phase,
        "claims_throughput": False,
        "catalog_count": len(catalog(phase)),
        "host_ok": host_ok,
        "gpu_available": native.get("available"),
        "gpu_ran": native.get("ran"),
        "gpu_success": native.get("success"),
        "gpu_blocker": native.get("blocker") or "",
        "gpu_case_count": len(native.get("cases", [])),
        "sources_ok": sources["pass"],
        "typed_reject": typed["pass"],
        "fixture": "fixtures/opt082_kernel_parity.json",
        "report": "evidence/optimization/opt082-kernel-parity/REPORT.md",
    }
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "opt082-result.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("smoke", *PHASES))
    parser.add_argument("--mode", default="feedback")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--skip-gpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_phase(args.phase, args.run_dir, skip_gpu=bool(args.skip_gpu))
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
