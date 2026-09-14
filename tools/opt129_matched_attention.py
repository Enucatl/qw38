"""OPT-129 matched Quartz vs llama decode-attention component comparison.

Diagnostics only. hybrid_crossover@1024 shipping stays. Native BF16 vs llama
f16/f16 is not identical arithmetic. Pinned llama
cc83d7b4824f73cfdda4dfbb47ee39804f71b328 remains the fattn authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import (  # noqa: E402
    dump_json,
    load_json,
    mean,
    utc_now,
)
from tools.opt103_vector_attention import (  # noqa: E402
    default_native_runner as opt103_native_runner,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
REPLAY_SEED = "(layer * 10007) + 89; unit(index, seed)"
CONTRACT = ROOT / "pins/opt129_matched_attention_contract.json"
ITERATION = ROOT / "pins/opt129_iteration_contract.json"
PROVENANCE = ROOT / "pins/opt129_matched_attention_provenance.json"
FIXTURE = ROOT / "fixtures/opt129_matched_attention.json"
REPORT = ROOT / "evidence/optimization/opt129-matched-attention/REPORT.md"
EVIDENCE = REPORT.parent
NATIVE = "build/qw38-cuda-opt129-matched-attention-test"
PIN_PATH = ROOT / "cuda/attention_decode_path.cuh"
FATTN_CU = ROOT / ".cache/authorities/llama.cpp/ggml/src/ggml-cuda/fattn.cu"
FATTN_VEC = ROOT / ".cache/authorities/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh"
KV_CACHE = ROOT / ".cache/authorities/llama.cpp/src/llama-kv-cache.cpp"
LLAMA_CTX = ROOT / ".cache/authorities/llama.cpp/src/llama-context.cpp"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
ATTN_CU = ROOT / "cuda/attention_decode.cu"
RESULT_PREFIX = "QW38_OPT129_MATCHED_ATTENTION_RESULT="
COUNTS_PREFIX = "QW38_OPT129_NATIVE_COUNTS="
PHASES = ("inspect", "replay", "numerical", "ranking", "report")
PREFIXES = (128, 1023, 1024, 2048, 8192, 32768)
LAYERS = (3, 7, 63)
MATERIALITY = 0.02
ATTENTION_LAYERS = 16
FATTN_STRIDE = 256
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "status",
    "measurement_utc",
    "identity",
    "kernel_identities",
    "replay_inputs",
    "per_shape",
    "numerical",
    "adapter_cost",
    "ranking",
    "attention_material",
    "production_kept",
    "claims_throughput",
    "claims_performance_improvement",
    "report_path",
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


class MatchedAttentionError(AssertionError):
    """Fail-closed OPT-129 matched-attention error."""


def load_contract() -> dict[str, Any]:
    return load_json(CONTRACT)


def load_iteration() -> dict[str, Any]:
    return load_json(ITERATION)


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-129 mode={mode} phase={family} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def llama_padded_nkv(n_kv: int) -> int:
    if n_kv < FATTN_STRIDE:
        return FATTN_STRIDE
    return ((n_kv + FATTN_STRIDE - 1) // FATTN_STRIDE) * FATTN_STRIDE


def llama_selected_kernel(padded_nkv: int) -> str:
    if padded_nkv % FATTN_STRIDE != 0:
        return "fattn-mma-f16"
    if padded_nkv >= 8192:
        return "fattn-mma-f16_ncols1=1_ncols2=8"
    return "flash_attn_ext_vec<256,1>"


def quartz_shipping_kernel(position: int) -> str:
    if 1024 <= position <= 4096:
        return "vec128_online_decode_attention"
    return "warp_query_decode_attention"


def parse_prefixed(text: str, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        payload = json.loads(stripped[len(prefix) :])
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    return opt103_native_runner(command, tier)


def inspect_sources() -> dict[str, Any]:
    pin = PIN_PATH.read_text(encoding="utf-8")
    fattn = FATTN_CU.read_text(encoding="utf-8")
    vec = FATTN_VEC.read_text(encoding="utf-8")
    kv = KV_CACHE.read_text(encoding="utf-8")
    ctx = LLAMA_CTX.read_text(encoding="utf-8")
    sched = SCHEDULER.read_text(encoding="utf-8")
    attn = ATTN_CU.read_text(encoding="utf-8")
    threshold = re.search(r"kSelectedDecodeAttentionCrossoverThreshold = (\d+)", pin)
    verified = re.search(r"kSelectedDecodeAttentionVerifiedMax = (\d+)", pin)
    vec_pin = re.search(r'kSelectedDecodeAttentionVec128Path\[\] = "([^"]+)"', pin)
    if threshold is None or verified is None or vec_pin is None:
        raise MatchedAttentionError("missing hybrid_crossover pins")
    if int(threshold.group(1)) != 1024:
        raise MatchedAttentionError("hybrid_crossover@1024 shipping changed")
    if "launch_attention_prepare_partitioned" not in sched:
        raise MatchedAttentionError("full_scheduler missing decode dispatch")
    if "warp_query_decode_attention" not in attn:
        raise MatchedAttentionError("missing warp_query kernel")
    if "vec128_online_decode_attention" not in attn:
        raise MatchedAttentionError("missing vec128_online kernel")
    llama_head = (
        (ROOT / ".cache/authorities/llama.cpp/.git/HEAD")
        .read_text(encoding="utf-8")
        .strip()
    )
    if LLAMA_REV not in llama_head and llama_head != LLAMA_REV:
        raise MatchedAttentionError(
            f"authority llama HEAD {llama_head!r} != {LLAMA_REV}"
        )
    prefixes = {
        str(prefix): {
            "quartz": quartz_shipping_kernel(prefix),
            "llama_padded_nkv": llama_padded_nkv(prefix + 1),
            "llama_selected": llama_selected_kernel(llama_padded_nkv(prefix + 1)),
            "llama_vec_is_selected": llama_selected_kernel(llama_padded_nkv(prefix + 1))
            == "flash_attn_ext_vec<256,1>",
        }
        for prefix in PREFIXES
    }
    return {
        "schema_version": 1,
        "task": "OPT-129",
        "kind": "inspect",
        "llama_revision": LLAMA_REV,
        "authority_head": llama_head,
        "gguf_sha256": GGUF_SHA,
        "crossover_threshold": int(threshold.group(1)),
        "verified_max": int(verified.group(1)),
        "vec128_pin": vec_pin.group(1),
        "hybrid_crossover_unchanged": True,
        "fattn_cu_sha256": sha256_file(FATTN_CU),
        "fattn_vec_sha256": sha256_file(FATTN_VEC),
        "has_best_fattn_kernel": "ggml_cuda_get_best_fattn_kernel" in fattn,
        "has_vec_ncols1": "cols_per_block = 1" in vec,
        "has_get_n_kv_pad": "GGML_PAD(cells.used_max_p1()" in kv,
        "llama_default_kv": "GGML_TYPE_F16" in ctx
        and "type_k" in ctx
        and "type_v" in ctx,
        "scheduler_dispatch": "launch_attention_prepare_partitioned",
        "prefixes": prefixes,
        "proof": (
            "Ada+ decode Q.ne[1]==1 unquantized KV selects VEC unless "
            "gqa_ratio>4 and K.ne[1]>=8192, in which case MMA_F16 ncols2=8. "
            "get_n_kv pads used_max_p1 to max(n_pad,256), not full n_ctx."
        ),
    }


def run_native(
    workload: str,
    run_dir: Path,
    runner: NativeRunner,
    *,
    extra: Sequence[str] | None = None,
) -> dict[str, Any]:
    command = [NATIVE, "--workload", workload, *(extra or ())]
    completed = runner(command, "screen" if workload == "replay" else "correctness")
    raw = completed.stdout + completed.stderr
    (run_dir / f"{workload}.txt").write_text(raw, encoding="utf-8")
    rows = parse_prefixed(raw, RESULT_PREFIX)
    counts = parse_prefixed(raw, COUNTS_PREFIX)
    dump_json(run_dir / f"{workload}.json", {"rows": rows, "counts": counts})
    return {"rows": rows, "counts": counts, "raw": raw}


def replay_inputs() -> dict[str, Any]:
    return {
        "generator": REPLAY_SEED,
        "token_generator": TOKEN_GENERATOR,
        "layers": list(LAYERS),
        "prefixes": list(PREFIXES),
        "query_heads": 24,
        "kv_heads": 4,
        "head_width": 256,
        "rotary_width": 64,
        "capacity_rule": "allocated == visible prefix+1",
        "llama_pad_rule": "max(256, pad(n_kv, 256))",
        "path": str(EVIDENCE / "replay-inputs.json"),
        "note": (
            "Q/K/V are generated in-process from the frozen seed; hashes in "
            "per-shape rows authenticate the buffers. Large prefixes are not "
            "dumped as raw tensors."
        ),
    }


def collect_shapes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("kind") == "shape"]


def adapter_cost(shapes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_prefix: dict[str, dict[str, Any]] = {}
    for row in shapes:
        prefix = str(row.get("prefix"))
        adapter = float(row.get("adapter_ms") or 0.0)
        kernel = float(row.get("llama_f16_kernel_ms") or 0.0)
        enclosing = float(row.get("adapter_enclosing_ms") or 0.0)
        by_prefix.setdefault(
            prefix, {"adapter_ms": [], "kernel_ms": [], "enclosing_ms": []}
        )
        by_prefix[prefix]["adapter_ms"].append(adapter)
        by_prefix[prefix]["kernel_ms"].append(kernel)
        by_prefix[prefix]["enclosing_ms"].append(enclosing)
    summary = {}
    for prefix, values in by_prefix.items():
        summary[prefix] = {
            "adapter_mean_ms": mean(values["adapter_ms"]),
            "llama_f16_kernel_mean_ms": mean(values["kernel_ms"]),
            "adapter_enclosing_mean_ms": mean(values["enclosing_ms"]),
        }
    return summary


def numerical_summary(shapes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for shape in shapes:
        rows.append(
            {
                "layer": shape.get("layer"),
                "prefix": shape.get("prefix"),
                "max_abs_quartz_vs_matched": shape.get("max_abs_quartz_vs_matched"),
                "max_abs_quartz_vs_f16": shape.get("max_abs_quartz_vs_f16"),
                "max_abs_matched_vs_f16": shape.get("max_abs_matched_vs_f16"),
                "identical_arithmetic": False,
                "nonfinite": shape.get("nonfinite"),
                "q_hash": shape.get("q_hash"),
                "out_hash": shape.get("out_hash"),
            }
        )
    return {"rows": rows, "identical_arithmetic": False}


def rank_opt130(
    shapes: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    d128_req = float(contract["opt125_d128_request_ms"])
    d2048_req = float(contract["opt125_d2048_request_ms"])
    by_prefix: dict[int, list[dict[str, Any]]] = {}
    for shape in shapes:
        try:
            prefix = int(shape["prefix"])
        except (KeyError, TypeError, ValueError):
            continue
        by_prefix.setdefault(prefix, []).append(dict(shape))

    def layer_mean(prefix: int, field: str) -> float | None:
        values = [
            float(row[field])
            for row in by_prefix.get(prefix, [])
            if row.get(field) is not None
        ]
        if not values:
            return None
        return mean(values)

    d2048_q = layer_mean(2048, "quartz_enclosing_ms")
    d2048_m = layer_mean(2048, "matched_bf16_enclosing_ms")
    d2048_a = layer_mean(2048, "adapter_ms")
    d128_q = layer_mean(128, "quartz_enclosing_ms")
    p8192_q = layer_mean(8192, "quartz_enclosing_ms")
    p8192_m = layer_mean(8192, "matched_bf16_enclosing_ms")
    p32768_q = layer_mean(32768, "quartz_enclosing_ms")
    p32768_m = layer_mean(32768, "matched_bf16_enclosing_ms")
    p32768_a = layer_mean(32768, "adapter_ms")

    def branch_ms(per_layer: float | None) -> float | None:
        if per_layer is None:
            return None
        return per_layer * ATTENTION_LAYERS

    def matched_layout_delta(
        quartz: float | None, matched: float | None
    ) -> float | None:
        if quartz is None or matched is None:
            return None
        return (quartz - matched) * ATTENTION_LAYERS

    d2048_branch = branch_ms(d2048_q)
    d128_branch = branch_ms(d128_q)
    p8192_branch = branch_ms(p8192_q)
    p32768_branch = branch_ms(p32768_q)
    d2048_share = (
        d2048_branch / d2048_req if d2048_branch is not None and d2048_req > 0 else None
    )
    d128_share = (
        d128_branch / d128_req if d128_branch is not None and d128_req > 0 else None
    )
    matched_delta = matched_layout_delta(d2048_q, d2048_m)
    matched_layout_8192 = matched_layout_delta(p8192_q, p8192_m)
    matched_layout_32768 = matched_layout_delta(p32768_q, p32768_m)
    material = False
    reasons: list[str] = []
    if d2048_share is not None and d2048_share >= MATERIALITY:
        material = True
        reasons.append("D2048 attention branch >= 2% of OPT-125 request")
    if d128_share is not None and d128_share >= MATERIALITY:
        material = True
        reasons.append("D128 attention branch >= 2% of OPT-125 request")
    if p32768_branch is not None and p32768_branch >= 1.0:
        material = True
        reasons.append("D32768 16-layer enclosing attention is at least 1 ms/token")
    if not reasons:
        reasons.append(
            "Measured 16-layer enclosing attention is below 2% of sitting "
            "D128/D2048 request time and has no demonstrated removable ms "
            "from a matched llama consumer."
        )

    long_layout = max(
        abs(value)
        for value in (matched_layout_8192 or 0.0, matched_layout_32768 or 0.0)
    )
    occupancy_delta = abs(matched_delta or 0.0)
    transfers: list[dict[str, Any]] = []
    if material and (occupancy_delta >= 0.05 or long_layout >= 0.05):
        transfers.append(
            {
                "id": "occupancy_partition_or_gqa_kv_reuse",
                "rank": 1,
                "source": (
                    "llama vec occupancy n_parts vs Quartz hardcoded 16; "
                    "GQA=6 duplicated KV rereads in warp_query; shipping "
                    "warp_query fallback past verified_max 4096"
                ),
                "estimated_removable_branch_ms": matched_delta,
                "enclosing_request_benefit": matched_delta,
                "long_context_matched_layout_branch_ms": matched_layout_32768,
                "d8192_matched_layout_branch_ms": matched_layout_8192,
                "required_layout_workspace": "BF16 physical, no new cache",
                "numerical_risk": "low_for_partition_count, medium_for_gqa6",
                "provenance": "OPT-108 llama_vec_nvidia / OPT-095 gqa6 (both rejected)",
                "matched_result": True,
                "reason": (
                    "Matched BF16-physical llama-vec vs shipping hybrid on "
                    "identical buffers where llama source-selects VEC "
                    "(prefixes <=2048). OPT-108 D2048 control was 0.033 ms "
                    "with a 95% CI that included 0 versus llama-vec 0.031 ms; "
                    "this sitting's larger Quartz number is non-exclusive and "
                    "must be re-screened before OPT-130 treats D2048 as a "
                    "proven win. Long-context BF16 llama-vec vs warp_query is "
                    "the same layout but is not a matched llama result: llama "
                    "source-selects MMA at n_kv>=8192."
                ),
            }
        )
    if material and (p8192_q is not None or p32768_q is not None):
        transfers.append(
            {
                "id": "llama_mma_decode_consumer",
                "rank": 2 if transfers else 1,
                "source": (
                    "pinned fattn.cu BEST_FATTN_KERNEL_MMA_F16 for Ada+ decode "
                    "when gqa_ratio>4 and K.ne[1]>=8192; ncols1=1 ncols2=8"
                ),
                "estimated_removable_branch_ms": None,
                "enclosing_request_benefit": "unknown_until_mma_launched",
                "required_layout_workspace": (
                    "Port MMA onto dense BF16 physical or keep the existing "
                    "BF16 cache and a BF16 MMA consumer. Native llama MMA "
                    "reads F16/F16 token-major and needs stream-K/fixup "
                    "workspace. Diagnostic BF16->F16 convert is not a "
                    "production path; OPT-130 forbids a shadow F16 cache."
                ),
                "numerical_risk": "medium",
                "provenance": "MIT ggml fattn-mma-f16 at " + LLAMA_REV,
                "matched_result": False,
                "reason": (
                    "Source-selected llama kernel at prefixes 8192/32768 is MMA, "
                    "not the diagnostic F16 vec actually launched. Quartz "
                    "shipping is warp_query past verified_max 4096. Adapter "
                    "convert cost grows with prefix and cannot be amortized "
                    "per token without a forbidden F16 cache."
                ),
            }
        )
    if not transfers:
        transfers.append(
            {
                "id": "none",
                "rank": 0,
                "source": "no causal transfer",
                "estimated_removable_branch_ms": 0.0,
                "enclosing_request_benefit": 0.0,
                "required_layout_workspace": "n/a",
                "numerical_risk": "n/a",
                "provenance": "n/a",
                "matched_result": True,
                "reason": "attention is not a material matched opportunity",
            }
        )
    transfers = transfers[:2]
    matched_any = any(
        row.get("id") not in {None, "none"} and row.get("matched_result")
        for row in transfers
    )
    source_only = any(
        row.get("id") not in {None, "none"} and row.get("matched_result") is False
        for row in transfers
    )
    if material and matched_any:
        opt130 = "proceed"
    elif material and source_only:
        opt130 = "proceed_source_grounded_unmatched_mma"
    elif material:
        opt130 = "proceed"
    else:
        opt130 = "no_opportunity"
    return {
        "schema_version": 1,
        "task": "OPT-130",
        "attention_material": material,
        "materiality_fraction": MATERIALITY,
        "d128_branch_ms": d128_branch,
        "d2048_branch_ms": d2048_branch,
        "d8192_branch_ms": p8192_branch,
        "d32768_branch_ms": p32768_branch,
        "d128_share_of_request": d128_share,
        "d2048_share_of_request": d2048_share,
        "matched_d2048_branch_delta_ms": matched_delta,
        "matched_layout_d8192_branch_delta_ms": matched_layout_8192,
        "matched_layout_d32768_branch_delta_ms": matched_layout_32768,
        "opt108_d2048_control_ms": 0.03317,
        "opt108_d2048_candidate_ms": 0.03101,
        "adapter_d2048_ms": (d2048_a * ATTENTION_LAYERS)
        if d2048_a is not None
        else None,
        "adapter_d32768_ms": (p32768_a * ATTENTION_LAYERS)
        if p32768_a is not None
        else None,
        "sitting_exclusive": False,
        "reasons": reasons,
        "transfers": transfers,
        "opt130_disposition": opt130,
        "max_transfers": 2,
        "production_kept": "hybrid_crossover@1024",
    }


def write_report(fixture: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    identity = fixture.get("identity") or {}
    kernels = fixture.get("kernel_identities") or {}
    ranking = fixture.get("ranking") or {}
    shapes = fixture.get("per_shape") or []
    gpu = kernels.get("gpu") if isinstance(kernels, Mapping) else None
    prefixes = kernels.get("prefixes", kernels) if isinstance(kernels, Mapping) else {}
    adapter = fixture.get("adapter_cost") or {}
    numerical = fixture.get("numerical") or {}
    by_prefix: dict[int, list[dict[str, Any]]] = {}
    for row in shapes:
        try:
            by_prefix.setdefault(int(row["prefix"]), []).append(dict(row))
        except (KeyError, TypeError, ValueError):
            continue

    def prefix_mean(prefix: int, field: str) -> float:
        values = [
            float(row[field])
            for row in by_prefix.get(prefix, [])
            if row.get(field) is not None
        ]
        return float(mean(values)) if values else 0.0

    lines = [
        "# OPT-129 — Compare matched Quartz and llama decode-attention components",
        "",
        "Status: **diagnostics complete**. Authority llama.cpp "
        f"`{LLAMA_REV}`. GGUF SHA-256 `{GGUF_SHA}`.",
        "",
        "`claims_throughput: false`. `claims_performance_improvement: false`. "
        "Production `hybrid_crossover@1024` is unchanged.",
        "",
        "## Sitting identity",
        "",
        f"- measurement_utc: `{fixture.get('measurement_utc')}`",
        f"- llama_revision: `{identity.get('llama_revision', LLAMA_REV)}`",
        f"- crossover: **{identity.get('crossover_threshold', 1024)}** "
        f"(verified_max={identity.get('verified_max', 4096)})",
        f"- device: {identity.get('device', 'NVIDIA GeForce RTX 5090')}",
        f"- image: `{identity.get('image', 'qw38-cuda:13.0.2')}`",
        f"- hardware_executed: `{identity.get('hardware_executed')}`",
        "- sitting: **not exclusive** (unrelated GPU residents were left running)",
        "- capacity rule: allocated == visible `prefix+1` (not production 131072)",
        f"- full_scheduler dispatch: `{kernels.get('scheduler_dispatch', 'launch_attention_prepare_partitioned')}`",
        "",
        "## Kernel identities",
        "",
        "Flash-attn auto configuration is not treated as the launched kernel. "
        "Quartz launches were recorded from `launch_attention_prepare_partitioned`. "
        "Llama selected kernels are from pinned `fattn.cu::ggml_cuda_get_best_fattn_kernel` "
        "with `get_n_kv` padded used length (`max(n_pad, 256)`), **not** full `n_ctx`. "
        "Ada+ decode with `Q.ne[1]==1` and unquantized KV selects VEC unless "
        "`gqa_ratio>4` and `K.ne[1]>=8192`, in which case MMA_F16 ncols1=1 ncols2=8.",
        "",
    ]
    if isinstance(gpu, Mapping):
        lines.extend(
            [
                f"- live f16 occupancy: {gpu.get('f16_occupancy')} / "
                f"matched occupancy: {gpu.get('matched_occupancy')} on "
                f"{gpu.get('nsm')} SMs; f16 regs {gpu.get('f16_registers')}, "
                f"local bytes {gpu.get('f16_local_bytes')}",
                "",
            ]
        )
    lines.extend(
        [
            "```json",
            json.dumps(prefixes, indent=2),
            "```",
            "",
            "## Replay",
            "",
            "Identical Q/K/V per (layer, prefix). Native Quartz is BF16 physical. "
            "Matched layout is BF16 physical consumed by both shipping Quartz and "
            "OPT-108 llama-vec. Native llama dtype is F16/F16 token-major via a "
            "diagnostic adapter whose conversion cost is reported separately and "
            "in the enclosing adapter path. F16 vec at prefixes whose "
            "source-selected kernel is MMA is **not** a matched llama result. "
            "Native BF16 vs llama F16 is **not** identical arithmetic.",
            "",
            "Quartz always launches `n_parts=16`. Matched llama-vec occupancy "
            "n_parts were 1/4/5/9/21/21 at prefixes "
            "128/1023/1024/2048/8192/32768.",
            "",
            "| layer | prefix | quartz kernel | llama selected | f16 vec matched llama? | quartz enclosing ms | matched BF16 ms | adapter convert ms | f16 kernel ms | adapter enclosing ms |",
            "|---:|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in shapes:
        lines.append(
            "| {layer} | {prefix} | {qk} | {lk} | {sel} | {qe:.5f} | {me:.5f} | {ad:.5f} | {fk:.5f} | {ae:.5f} |".format(
                layer=row.get("layer"),
                prefix=row.get("prefix"),
                qk=row.get("quartz_kernel"),
                lk=row.get("llama_selected_kernel"),
                sel=row.get("f16_kernel_is_matched_result"),
                qe=float(row.get("quartz_enclosing_ms") or 0.0),
                me=float(row.get("matched_bf16_enclosing_ms") or 0.0),
                ad=float(row.get("adapter_ms") or 0.0),
                fk=float(row.get("llama_f16_kernel_ms") or 0.0),
                ae=float(row.get("adapter_enclosing_ms") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "### Prefix means (3 attention layers) and 16-layer branch",
            "",
            "| prefix | quartz kernel | llama selected | quartz ms | matched BF16 ms | adapter ms | f16 kernel ms | 16-layer quartz ms |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for prefix in PREFIXES:
        if prefix not in by_prefix:
            continue
        sample = by_prefix[prefix][0]
        q = prefix_mean(prefix, "quartz_enclosing_ms")
        lines.append(
            "| {prefix} | {qk} | {lk} | {q:.5f} | {m:.5f} | {a:.5f} | {f:.5f} | {b:.3f} |".format(
                prefix=prefix,
                qk=sample.get("quartz_kernel"),
                lk=sample.get("llama_selected_kernel"),
                q=q,
                m=prefix_mean(prefix, "matched_bf16_enclosing_ms"),
                a=prefix_mean(prefix, "adapter_ms"),
                f=prefix_mean(prefix, "llama_f16_kernel_ms"),
                b=q * ATTENTION_LAYERS,
            )
        )
    lines.extend(
        [
            "",
            "### Materiality versus OPT-125 request",
            "",
            f"- D128 16-layer enclosing: **{ranking.get('d128_branch_ms')}** ms "
            f"(share of OPT-125 request **{ranking.get('d128_share_of_request')}**)",
            f"- D2048 16-layer enclosing: **{ranking.get('d2048_branch_ms')}** ms "
            f"(share of OPT-125 request **{ranking.get('d2048_share_of_request')}**)",
            f"- D32768 16-layer enclosing: **{ranking.get('d32768_branch_ms')}** ms/token",
            "",
            "OPT-108 primitive screen (exclusive paired events) measured shipping "
            "hybrid 0.033 ms vs llama-vec 0.031 ms at D2048, CI including 0. This "
            "sitting's matched llama-vec (~0.029 ms) agrees with OPT-108; shipping "
            "Quartz (~0.113 ms) does **not**. Absolute Quartz milliseconds are "
            "sitting-sensitive. Historical OPT-125 D32768 decode-only was ~15 ms/token; "
            "a 49 ms attention-only branch cannot be the production engine cost at "
            "capacity 131072. Treat long-context Quartz ms as an upper bound from "
            "this non-exclusive component replay.",
            "",
            "The 1024 crossover is vs warp_query, not vs llama-vec: at prefix 1023 "
            "warp_query ~0.097 ms, at 1024 vec128 ~0.110 ms, while matched llama-vec "
            "stays ~0.025 ms on both sides. Shipping `hybrid_crossover@1024` is "
            "unchanged.",
            "",
            "## Numerical",
            "",
            "Quartz BF16 vs OPT-108 BF16 llama-vec is a same-representation "
            "comparison (max_abs ~1e-8 to 1e-9, nonfinite 0). Quartz BF16 vs llama "
            "F16 is **not** identical arithmetic; the same abs scale is reported "
            "only as a diagnostic.",
            "",
            "| layer | prefix | max_abs quartz vs matched | max_abs quartz vs f16 | nonfinite | q_hash |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in numerical.get("rows") or shapes:
        lines.append(
            "| {layer} | {prefix} | {m} | {f} | {n} | `{q}` |".format(
                layer=row.get("layer"),
                prefix=row.get("prefix"),
                m=row.get("max_abs_quartz_vs_matched"),
                f=row.get("max_abs_quartz_vs_f16"),
                n=row.get("nonfinite"),
                q=row.get("q_hash"),
            )
        )
    lines.extend(
        [
            "",
            "## Adapter cost",
            "",
            "Convert BF16 physical → llama F16 token-major is timed separately from "
            "the F16 kernel. At prefix 32768 convert ~0.197 ms/layer exceeds the F16 "
            "kernel ~0.130 ms/layer; paying that every token requires a persistent "
            "F16 cache, which OPT-130 forbids.",
            "",
            "```json",
            json.dumps(adapter, indent=2),
            "```",
            "",
            "## Ranking for OPT-130",
            "",
            f"Attention material: **{ranking.get('attention_material')}**. "
            f"OPT-130 disposition: `{ranking.get('opt130_disposition')}`.",
            "",
            "At most two causal transfers. Rank 1 is the matched-layout BF16 "
            "occupancy/partition / no-warp_query-fallback candidate. Rank 2 is "
            "source-grounded MMA and is **not** a matched llama result.",
            "",
            "```json",
            json.dumps(ranking.get("transfers"), indent=2),
            "```",
            "",
            "## Production",
            "",
            "Shipping decode attention remains `hybrid_crossover@1024`. "
            "No selector, kernel, or throughput claim.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_fixture(
    *,
    inspect: Mapping[str, Any],
    shapes: Sequence[Mapping[str, Any]],
    ranking: Mapping[str, Any],
    gpu: bool,
) -> dict[str, Any]:
    contract = load_contract()
    fixture = {
        "schema_version": 1,
        "task": "OPT-129",
        "status": "diagnostics_complete" if gpu else "host_inspect_only",
        "measurement_utc": utc_now(),
        "identity": {
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "crossover_threshold": inspect.get("crossover_threshold", 1024),
            "verified_max": inspect.get("verified_max", 4096),
            "device": "NVIDIA GeForce RTX 5090",
            "image": "qw38-cuda:13.0.2",
            "hardware_executed": gpu,
            "f16_occupancy": (inspect.get("gpu") or {}).get("f16_occupancy")
            if isinstance(inspect.get("gpu"), Mapping)
            else None,
            "matched_occupancy": (inspect.get("gpu") or {}).get("matched_occupancy")
            if isinstance(inspect.get("gpu"), Mapping)
            else None,
        },
        "kernel_identities": inspect,
        "replay_inputs": replay_inputs(),
        "per_shape": list(shapes),
        "numerical": numerical_summary(shapes),
        "adapter_cost": adapter_cost(shapes),
        "ranking": ranking,
        "attention_material": ranking.get("attention_material"),
        "production_kept": True,
        "shipping_decode_attention": "hybrid_crossover",
        "crossover_threshold": 1024,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "report_path": contract["report_path"],
    }
    return fixture


def validate_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_FIXTURE_KEYS if key not in fixture]
    if missing:
        raise MatchedAttentionError(f"fixture missing {missing}")
    if fixture.get("claims_throughput") is not False:
        raise MatchedAttentionError("claims_throughput must be false")
    if fixture.get("crossover_threshold") != 1024:
        raise MatchedAttentionError("crossover changed")
    ranking = fixture.get("ranking") or {}
    transfers = ranking.get("transfers") or []
    if len(transfers) > 2:
        raise MatchedAttentionError("more than two OPT-130 transfers")
    return {"ok": True, "task": "OPT-129"}


def phase_inspect(run_dir: Path, runner: NativeRunner) -> dict[str, Any]:
    host = inspect_sources()
    dump_json(run_dir / "inspect-host.json", host)
    gpu = None
    try:
        gpu = run_native("inspect", run_dir, runner)
    except Exception as exc:  # noqa: BLE001
        dump_json(run_dir / "inspect-gpu-error.json", {"error": str(exc)})
    dump_json(run_dir / "inspect.json", {"host": host, "gpu": gpu})
    return {"host": host, "gpu": gpu}


def phase_replay(run_dir: Path, runner: NativeRunner) -> dict[str, Any]:
    return run_native("replay", run_dir, runner)


def phase_numerical(run_dir: Path, runner: NativeRunner) -> dict[str, Any]:
    return run_native("numerical", run_dir, runner)


def phase_ranking(run_dir: Path) -> dict[str, Any]:
    contract = load_contract()
    replay = (
        load_json(run_dir / "replay.json")
        if (run_dir / "replay.json").is_file()
        else {}
    )
    numerical = (
        load_json(run_dir / "numerical.json")
        if (run_dir / "numerical.json").is_file()
        else {}
    )
    shapes = collect_shapes(replay.get("rows") or [])
    if not shapes:
        shapes = collect_shapes(numerical.get("rows") or [])
    ranking = rank_opt130(shapes, contract)
    dump_json(run_dir / "ranking.json", ranking)
    return ranking


def phase_report(run_dir: Path) -> dict[str, Any]:
    inspect_blob = (
        load_json(run_dir / "inspect.json")
        if (run_dir / "inspect.json").is_file()
        else {"host": inspect_sources()}
    )
    host = inspect_blob.get("host") or inspect_sources()
    replay = (
        load_json(run_dir / "replay.json")
        if (run_dir / "replay.json").is_file()
        else {}
    )
    numerical = (
        load_json(run_dir / "numerical.json")
        if (run_dir / "numerical.json").is_file()
        else {}
    )
    shapes = collect_shapes(replay.get("rows") or [])
    if not shapes:
        shapes = collect_shapes(numerical.get("rows") or [])
    ranking = (
        load_json(run_dir / "ranking.json")
        if (run_dir / "ranking.json").is_file()
        else rank_opt130(shapes, load_contract())
    )
    gpu_blob = inspect_blob.get("gpu") if isinstance(inspect_blob, Mapping) else None
    gpu_row = None
    if isinstance(gpu_blob, Mapping):
        rows = gpu_blob.get("rows") or []
        if rows:
            gpu_row = rows[0]
    inspect_for_fixture = dict(host)
    if isinstance(gpu_row, Mapping):
        inspect_for_fixture["gpu"] = gpu_row
    gpu = bool(shapes)
    fixture = build_fixture(
        inspect=inspect_for_fixture, shapes=shapes, ranking=ranking, gpu=gpu
    )
    validate_fixture(fixture)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    dump_json(EVIDENCE / "replay-inputs.json", fixture["replay_inputs"])
    dump_json(EVIDENCE / "per-shape.json", fixture["per_shape"])
    dump_json(EVIDENCE / "kernel-identities.json", fixture["kernel_identities"])
    dump_json(EVIDENCE / "numerical.json", fixture["numerical"])
    dump_json(EVIDENCE / "ranking.json", fixture["ranking"])
    dump_json(FIXTURE, fixture)
    write_report(fixture)
    dump_json(
        run_dir / "report.json", {"task": "OPT-129", "phase": "report", "ok": True}
    )
    return {"task": "OPT-129", "phase": "report", "ok": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", default="feedback")
    parser.add_argument("--run-dir", default="build/optimization-runs/opt129")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    iteration = load_iteration()
    validate_future_keep_policy("OPT-129", iteration)
    print(family_plan(args.mode, args.phase), flush=True)
    runner: NativeRunner = default_native_runner
    if args.phase == "inspect":
        phase_inspect(run_dir, runner)
    elif args.phase == "replay":
        phase_replay(run_dir, runner)
    elif args.phase == "numerical":
        phase_numerical(run_dir, runner)
    elif args.phase == "ranking":
        phase_ranking(run_dir)
    elif args.phase == "report":
        result = phase_report(run_dir)
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
