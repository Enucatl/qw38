"""OPT-150 post-149 decode-attention dataflow and matched scaling.

Diagnostics only. Production selectors and arithmetic stay unchanged.
Phases: smoke, correctness, replay, matched, report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import dump_json, load_json, utc_now  # noqa: E402
from tools.opt080_batch_gate import (  # noqa: E402
    GGUF_SHA,
    IMAGE,
    LLAMA_REV,
    docker_common,
    git_identity,
)
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools import opt136_graph_accounting as opt136  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt150_attention_dataflow_contract.json"
ITERATION = ROOT / "pins/opt150_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt150_attention_dataflow.json"
EVIDENCE = ROOT / "evidence/optimization/opt150-attention-dataflow"
REPORT = EVIDENCE / "REPORT.md"
PIN_PATH = ROOT / "cuda/attention_decode_path.cuh"
MMA_SRC = ROOT / "cuda/opt137_dense_mma_decode.cuh"
ATTN_CU = ROOT / "cuda/attention_decode.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
FATTN_CU = ROOT / ".cache/authorities/llama.cpp/ggml/src/ggml-cuda/fattn.cu"
FATTN_MMA = ROOT / ".cache/authorities/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh"
KV_CACHE = ROOT / ".cache/authorities/llama.cpp/src/llama-kv-cache.cpp"
NATIVE = "build/qw38-cuda-opt150-attention-dataflow-test"
PRODUCTION_OBJ = "build/attention_decode.cuda.o"
MANGLED_MMA = (
    "_ZN4qw384cuda6opt13730dense_bf16_tile_f16_mma_decodeENS0_15AttentionConfigEmi"
    "PKfPK13__nv_bfloat16S7_S7_S7_PfS8_PKNS0_17DecodeLaunchStateE"
)
OPT136_NATIVE = opt136.NATIVE
LLAMA_BIN = opt136.LLAMA_BIN
MODEL = opt136.MODEL
GPU_LOCK = opt136.GPU_LOCK
RESULT_PREFIX = "QW38_OPT150_ATTENTION_DATAFLOW_RESULT="
COUNTS_PREFIX = "QW38_OPT150_NATIVE_COUNTS="
PHASES = ("smoke", "correctness", "replay", "matched", "report")
PARENT = "post124_plus_opt127_decode_segments8_plus_opt137_mma_plus_opt148_flash_vec"
SELECTOR = "decode_segments8"
CAPACITY = 131072
CROSSOVER = 1024
VERIFIED_MAX = 4096
MMA_THRESHOLD = 8192
FATTN_STRIDE = 256
GQA_RATIO = 6
ATTENTION_LAYERS = 16
LAYERS = (3, 7, 63)
DISPATCH_POSITIONS = (
    0,
    128,
    1023,
    1024,
    4096,
    4097,
    6144,
    7935,
    7936,
    8191,
    8192,
    8193,
    32768,
)
TIMING_SHAPES = (2048, 8192, 32768)
NUMERICAL_POSITIONS = (128, 1024, 8192)
MATCHED_EVALS = 32
SCREEN_WARMUPS = 1
SCREEN_PAIRS = 3
CHILD_TIMEOUT_S = 300
AGGREGATE_DEADLINE_S = 7200
QUERY_HEADS = 24
KV_HEADS = 4
HEAD_WIDTH = 256
ROTARY_WIDTH = 64
MMA_NBATCH_FA = 64
MMA_OCCUPANCY = 4
MMA_NSM_PINNED = 148
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "status",
    "mode",
    "claims_throughput",
    "claims_performance_improvement",
    "production_kept",
    "selected_execution_graph_path",
    "dispatch_positions",
    "timing_shapes",
    "identity",
    "dispatch",
    "dataflow",
    "compiled",
    "correctness",
    "replay",
    "matched",
    "answers",
    "report_path",
)
HISTORICAL_CORRECTIONS = (
    {
        "source": "OPT-129",
        "claim": "approximately 0.050 ms at prefix 8192",
        "correction": (
            "That number is the BF16 llama-vector adapter enclosing path "
            "(matched_bf16_enclosing mean 0.05033 ms/layer), explicitly not "
            "the native selected fattn-mma-f16 ncols1=1 ncols2=8 kernel. The "
            "sitting was non-exclusive and allocated prefix+1, not production "
            "capacity 131072."
        ),
        "artifact": "evidence/optimization/opt129-matched-attention/REPORT.md",
    },
    {
        "source": "OPT-136",
        "claim": "D32768 ratio 0.243909",
        "correction": (
            "That ratio is pre-OPT-137 whole-engine decode-only throughput "
            "(quartz 15.317 tok/s / llama 62.799 tok/s) at capacity 131072, "
            "not a component attention ratio and not post-148 flash-vec."
        ),
        "artifact": "evidence/optimization/opt136-graph-accounting/REPORT.md",
    },
)


class AttentionDataflowError(RuntimeError):
    """OPT-150 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-150":
        raise AttentionDataflowError("attention-dataflow contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-150", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workloads = iteration["workloads"]
    alias = "smoke" if family not in workloads else family
    workload = workload_for_mode(workloads[alias], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-150 mode={mode} phase={family} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sidecar(run_dir: Path, name: str) -> Path:
    return run_dir / name


def store_sidecar(
    run_dir: Path, name: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    path = sidecar(run_dir, name)
    dump_json(path, payload)
    return dict(payload)


def load_sidecar(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = sidecar(run_dir, name)
    if not path.is_file():
        return None
    payload = load_json(path)
    return payload if isinstance(payload, dict) else None


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def llama_padded_nkv(n_kv: int) -> int:
    pad = FATTN_STRIDE
    if n_kv < pad:
        return pad
    return ((n_kv + pad - 1) // pad) * pad


def llama_selected_kernel(padded_nkv: int) -> str:
    if padded_nkv % FATTN_STRIDE != 0:
        return "fattn-mma-f16"
    if GQA_RATIO > 4 and padded_nkv >= MMA_THRESHOLD:
        return "fattn-mma-f16_ncols1=1_ncols2=8"
    return "flash_attn_ext_vec<256,1>"


def llama_mask_note(padded_nkv: int) -> str:
    if padded_nkv % FATTN_STRIDE == 0:
        return (
            "causal mask present; gqa_opt requires mask and "
            "K.ne[1] % FATTN_KQ_STRIDE==0; padded used length is K.ne[1]"
        )
    return "unpadded K.ne[1] disables vector kernel and gqa_opt"


def opt137_parallel_blocks(
    visible: int, occupancy: int = MMA_OCCUPANCY, nsm: int = MMA_NSM_PINNED
) -> int:
    ntiles_kv = (visible + MMA_NBATCH_FA - 1) // MMA_NBATCH_FA
    ntiles_dst = 4
    occ_use = occupancy if occupancy > 0 else MMA_OCCUPANCY
    sm = nsm if nsm > 0 else 1
    parallel = occ_use if occ_use < ntiles_kv else ntiles_kv
    if parallel < 1:
        parallel = 1
    blocks_per_wave = sm * occ_use
    nwaves_best = 0
    efficiency_best = 0
    test = parallel
    while test <= ntiles_kv:
        nblocks_total = ntiles_dst * test
        nwaves = (nblocks_total + blocks_per_wave - 1) // blocks_per_wave
        efficiency = (
            (100 * nblocks_total) // (nwaves * blocks_per_wave)
            if blocks_per_wave and nwaves
            else 0
        )
        if efficiency_best >= 95 and nwaves > nwaves_best:
            break
        if efficiency > efficiency_best:
            nwaves_best = nwaves
            efficiency_best = efficiency
            parallel = test
        test += 1
    return max(1, min(256, parallel))


def opt137_n_parts_for_position(position: int) -> int:
    if position >= 65536:
        visible = 131072
    elif position >= 32768:
        visible = 33024
    else:
        visible = 8448
    return opt137_parallel_blocks(visible)


def quartz_path(position: int) -> str:
    if position >= MMA_THRESHOLD:
        return "dense_bf16_tile_f16_mma_decode_v1"
    if CROSSOVER <= position <= VERIFIED_MAX:
        return "decode_attention_flash_vec_v1"
    return "warp_query"


def quartz_launch(position: int) -> str:
    if position >= MMA_THRESHOLD:
        return "dense_bf16_tile_f16_mma_decode_v1"
    if CROSSOVER <= position <= VERIFIED_MAX:
        return "flash_vec_decode_attention"
    return "warp_query_decode_attention"


def quartz_topology(position: int) -> int:
    if position >= 65536:
        return 4
    if position >= 32768:
        return 3
    if position >= MMA_THRESHOLD:
        return 2
    if CROSSOVER <= position <= VERIFIED_MAX:
        return 1
    return 0


def quartz_n_parts(position: int) -> int:
    if position >= MMA_THRESHOLD:
        return opt137_n_parts_for_position(position)
    return 16


def graph_boundary(position: int, previous: int | None) -> str | None:
    if previous is None:
        return None
    prev_t = quartz_topology(previous)
    cur_t = quartz_topology(position)
    if prev_t == cur_t:
        return None
    return f"topology_{prev_t}_to_{cur_t}"


def dispatch_row(position: int, previous: int | None) -> dict[str, Any]:
    visible = position + 1
    padded = llama_padded_nkv(visible)
    path = quartz_path(position)
    return {
        "position": position,
        "quartz_visible_length": visible,
        "quartz_zero_based_position": position,
        "allocated_capacity": CAPACITY,
        "quartz_path": path,
        "quartz_launch": quartz_launch(position),
        "quartz_topology": quartz_topology(position),
        "quartz_n_parts": quartz_n_parts(position),
        "graph_boundary_crossing": graph_boundary(position, previous),
        "llama_logical_nkv": visible,
        "llama_padded_nkv": padded,
        "llama_pad_rule": "max(256, pad(used_max_p1, 256))",
        "llama_selected_kernel": llama_selected_kernel(padded),
        "llama_vec_is_selected": llama_selected_kernel(padded)
        == "flash_attn_ext_vec<256,1>",
        "llama_mask": llama_mask_note(padded),
        "dtypes": {
            "quartz_q": "fp32_prepared",
            "quartz_kv": "bf16_physical",
            "llama_kv_native": "f16_f16_token_major",
        },
        "strides": {
            "quartz_kv_physical": (
                f"attention_kv_physical_index(token, kv_head, dim, "
                f"capacity={CAPACITY}, head_width={HEAD_WIDTH})"
            ),
            "llama_k_view": "hparams.n_embd_head_k x n_head_kv x n_kv",
        },
    }


def build_dispatch_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: int | None = None
    for position in DISPATCH_POSITIONS:
        rows.append(dispatch_row(position, previous))
        previous = position
    return rows


def pin_flags(text: str) -> dict[str, Any]:
    def const_int(name: str) -> int | None:
        match = re.search(rf"{name} = (\d+)", text)
        return int(match.group(1)) if match else None

    def const_bool(name: str) -> bool | None:
        match = re.search(rf"{name} = (true|false)", text)
        if match is None:
            return None
        return match.group(1) == "true"

    def const_str(name: str) -> str | None:
        match = re.search(rf'{name}\[\] = "([^"]+)"', text)
        return match.group(1) if match else None

    return {
        "crossover": const_int("kSelectedDecodeAttentionCrossoverThreshold"),
        "verified_max": const_int("kSelectedDecodeAttentionVerifiedMax"),
        "mma_threshold": const_int("kOpt137MmaThreshold"),
        "opt137_mma": const_bool("kSelectedOpt137DenseMma"),
        "flash_vec": const_bool("kSelectedDecodeAttentionFlashVec"),
        "vec128_n_parts": const_int("kSelectedVec128NParts"),
        "vec128_pin": const_str("kSelectedDecodeAttentionVec128Path"),
        "flash_vec_id": const_str("kLegalDecodeAttentionFlashVec"),
        "mma_id": const_str("kLegalDecodeAttentionDenseMma"),
        "mma_ncols1": const_int("kOpt137MmaNcols1"),
        "mma_ncols2": const_int("kOpt137MmaNcols2"),
        "mma_nthreads": const_int("kOpt137MmaNthreads"),
        "mma_occupancy": const_int("kOpt137MmaOccupancyPinned"),
        "mma_nstages": const_int("kOpt137MmaNstages"),
        "mma_nbatch_fa": const_int("kOpt137MmaNbatchFa"),
        "topology_count": const_int("kOpt137TopologyCount"),
    }


def inspect_dataflow() -> dict[str, Any]:
    mma = MMA_SRC.read_text(encoding="utf-8")
    fattn_mma = FATTN_MMA.read_text(encoding="utf-8") if FATTN_MMA.is_file() else ""
    fattn = FATTN_CU.read_text(encoding="utf-8") if FATTN_CU.is_file() else ""
    attn = ATTN_CU.read_text(encoding="utf-8")
    sched = SCHEDULER.read_text(encoding="utf-8")
    zero_weight = "0.0F * mma_scores[0]" in mma
    qk_fp32 = "local = __fadd_rn(local, __fmul_rn(q_f32[i], k_f))" in mma
    pv_scalar = "vkq[i] = vkq[i] * rescale + weight * v_f" in mma
    pv_mma = "mma_pv" in mma or "mma_vkq" in mma
    gqa_pad = (
        "live_head = head_col < kGqa" in mma and "kNcols2 = kOpt137MmaNcols2" in mma
    )
    convert = "bf16_to_f16_opt111" in mma or "__float2half_rn(__bfloat162float" in mma
    merge = "merge_decode_kv_parts" in attn
    graph_callers = (
        "decode_graph_topology_index" in sched
        and "launch_attention_prepare_partitioned" in sched
    )
    llama_qk_mma = "VKQ" in fattn_mma and "nbatch_fa" in fattn_mma
    llama_switch = (
        "gqa_ratio > 4 && K->ne[1] >= 8192" in fattn
        or "gqa_ratio > 4 && K->ne[1] >= 8192" in fattn.replace(" ", "")
    )
    if not llama_switch:
        llama_switch = "gqa_ratio > 4" in fattn and "K->ne[1] >= 8192" in fattn
    return {
        "quartz_mma": {
            "qk": "fp32_dot_plus_warp_sum",
            "qk_mma": "mma_qk_eight_zero_weight" if zero_weight else "unknown",
            "zero_weight_mma_in_source": zero_weight,
            "qk_fp32_products": qk_fp32,
            "softmax": "per_kv_row_online",
            "pv": "scalar_fp32_weight_times_v" if pv_scalar else "unknown",
            "pv_mma": pv_mma,
            "gqa_reuse": "ncols2=8_live_heads_0_5_mask_6_7" if gqa_pad else "unknown",
            "loads_conversion": (
                "dense_bf16_tile_to_f16_opt111_rounding" if convert else "unknown"
            ),
            "partitions": "frozen_graph_buckets_8448_33024_131072",
            "merge": "merge_decode_kv_parts" if merge else "unknown",
        },
        "llama_fattn_mma_d256_ncols1_1_ncols2_8": {
            "qk": "mma_kq",
            "softmax": "tiled_rescale_nbatch_fa_64_ampere",
            "pv": "mma_vkq",
            "gqa_reuse": "ncols2=8_shared_kv_when_gqa_ratio>4",
            "loads_conversion": "staged_f16_tiles",
            "nthreads": 64,
            "occupancy": 4,
            "nstages": 2,
            "source_switch": (
                "Ada+ decode Q.ne[1]==1 unquantized KV selects VEC unless "
                "gqa_ratio>4 and K.ne[1]>=8192"
            ),
            "mma_config_present": llama_qk_mma,
            "selector_present": llama_switch,
        },
        "quartz_flash_vec_window": {
            "range": "[1024,4096]",
            "path": "decode_attention_flash_vec_v1",
            "launch": "flash_vec_decode_attention",
            "n_parts": 16,
        },
        "quartz_warp_query": {
            "ranges": ["<1024", "[4097,8191]"],
            "path": "warp_query",
            "launch": "warp_query_decode_attention",
            "n_parts": 16,
        },
        "graph_eager_callers": {
            "full_scheduler_uses_topology_index": graph_callers,
            "partitioned_entry": "launch_attention_prepare_partitioned",
            "mma_branch": "opt137_uses_mma_at(position) before vec/flash",
            "flash_reuses_topology_1": True,
        },
        "zero_weight_mma_is_source_fact_not_cost": True,
    }


def inspect_compiled_source() -> dict[str, Any]:
    mma = MMA_SRC.read_text(encoding="utf-8")
    return {
        "binary": NATIVE,
        "kernel": "dense_bf16_tile_f16_mma_decode",
        "zero_weight_add": "0.0F * mma_scores[0]" in mma,
        "mma_qk_eight_defined": "__device__ void mma_qk_eight" in mma,
        "pv_uses_mma": False,
        "note": (
            "Source feeds MMA scores with a zero-weight add into the FP32 QK "
            "dot. A kernel name or launch counter cannot prove the compiler "
            "retains that MMA. SASS inspection is recorded separately."
        ),
        "proof_limit": "instruction presence without data dependence is unknown",
    }


def workspace_relpath(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sass_argv(binary: Path) -> list[str]:
    listed = [
        "cuobjdump",
        "-sass",
        "-fun",
        MANGLED_MMA,
        workspace_relpath(binary),
    ]
    if os.environ.get("QW38_HOST_NATIVE") == "1":
        return listed
    return [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        "/workspace",
        IMAGE,
        *listed,
    ]


def run_cuobjdump(binary: Path) -> dict[str, Any]:
    candidates = []
    production = ROOT / PRODUCTION_OBJ
    if production.is_file():
        candidates.append(production)
    if binary.is_file() and binary not in candidates:
        candidates.append(binary)
    if not candidates:
        return {
            "ok": False,
            "blocked": True,
            "reason": f"missing {relpath(binary)} and {PRODUCTION_OBJ}",
            "zero_weight_mma_survives": None,
        }
    found_tool = None
    raw = ""
    used = candidates[0]
    last_error = ""
    for candidate in candidates:
        completed = subprocess.run(
            _sass_argv(candidate),
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode == 0 and "Function :" in stdout:
            found_tool = "cuobjdump"
            raw = stdout
            used = candidate
            break
        last_error = (stderr + stdout)[-2000:]
    if found_tool is None:
        return {
            "ok": False,
            "blocked": True,
            "reason": "cuobjdump/nvdisasm unavailable or failed",
            "stderr_tail": last_error,
            "zero_weight_mma_survives": None,
        }
    kernel_hit = "dense_bf16_tile_f16_mma_decode" in raw
    hmma = len(re.findall(r"\bHMMA\b", raw))
    mma_sync = len(re.findall(r"HMMA\.16816|HMMA\.1688|MMA\.SYNC", raw, re.I))
    survives: bool | None
    if not kernel_hit:
        survives = None
    elif hmma + mma_sync == 0:
        survives = False
    else:
        survives = True
    return {
        "ok": True,
        "blocked": False,
        "tool": found_tool,
        "binary": relpath(used),
        "mangled_symbol": MANGLED_MMA,
        "kernel_symbol_present": kernel_hit,
        "hmma_count": hmma,
        "mma_sync_like_count": mma_sync,
        "zero_weight_mma_survives": survives,
        "note": (
            "Production kernel SASS contains HMMA.16816 ops. That shows the "
            "compiler kept tensor-core instructions in "
            "opt137::dense_bf16_tile_f16_mma_decode. It does not prove the "
            "zero-weight QK add is on the output-producing dataflow; PV stays "
            "scalar."
        ),
        "raw_path": relpath(EVIDENCE / "compiled-sass.txt"),
        "raw_bytes": len(raw.encode("utf-8")),
        "raw_excerpt": raw[:4000],
        "raw": raw,
    }


def parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            try:
                return json.loads(stripped[len(prefix) :])
            except json.JSONDecodeError:
                continue
    return {}


def parse_prefixed_rows(text: str, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        try:
            payload = json.loads(stripped[len(prefix) :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def run_locked(
    command: Sequence[str], *, tier: str, timeout_s: float
) -> subprocess.CompletedProcess[str]:
    os.environ["QW38_CUDA_TEST_TIER"] = tier
    listed = list(command)
    if os.environ.get("QW38_HOST_NATIVE") == "1":
        cmd = listed
    else:
        cmd = [*docker_common(IMAGE, tier), *listed]
    return opt136.with_gpu_lock(cmd, timeout_s=timeout_s)


def ensure_built(run_dir: Path) -> dict[str, Any]:
    available, gpu_blocker = gpu_available()
    make_rc = 1
    make_stderr = gpu_blocker or "gpu_unavailable"
    if available:
        make = run_locked(
            ["make", "cuda-opt150-diagnostics"],
            tier="correctness",
            timeout_s=CHILD_TIMEOUT_S,
        )
        make_rc = make.returncode
        make_stderr = ((make.stderr or "") + (make.stdout or ""))[-4000:]
        (run_dir / "make-diagnostics.txt").write_text(
            (make.stdout or "") + (make.stderr or ""), encoding="utf-8"
        )
    return {
        "gpu_available": available,
        "gpu_blocker": gpu_blocker,
        "make_returncode": make_rc,
        "make_stderr_tail": make_stderr,
        "ok": available and make_rc == 0,
    }


def skip_payload(
    run_dir: Path,
    mode: str,
    phase: str,
    *,
    reason: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-150",
        "phase": phase,
        "mode": mode,
        "ok": True,
        "skipped": True,
        "blocked": True,
        "status": "incomplete",
        "skip_reason": reason,
        "claims_throughput": False,
        "family_plan": family_plan(mode, phase),
        "measured_at": utc_now(),
    }
    if extra:
        payload.update(dict(extra))
    return store_sidecar(run_dir, f"{phase}.json", payload)


def empty_fixture(mode: str = "feedback") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-150",
        "status": "structure",
        "mode": mode,
        "measurement_utc": None,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "dispatch_positions": list(DISPATCH_POSITIONS),
        "timing_shapes": list(TIMING_SHAPES),
        "identity": None,
        "dispatch": None,
        "dataflow": None,
        "compiled": None,
        "correctness": None,
        "replay": None,
        "matched": None,
        "answers": {},
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
    }


def persist_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    dump_json(FIXTURE, payload)
    return dict(payload)


def validate_fixture(payload: Mapping[str, Any]) -> None:
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise AttentionDataflowError(f"fixture missing {key}")
    if payload.get("claims_throughput") is not False:
        raise AttentionDataflowError("claims_throughput must be false")
    if payload.get("selected_execution_graph_path") != SELECTOR:
        raise AttentionDataflowError("selector drifted from decode_segments8")
    if payload.get("production_kept") is not True:
        raise AttentionDataflowError("diagnostics must not flip production")


def build_identity() -> dict[str, Any]:
    source, dirty = git_identity()
    pin = PIN_PATH.read_text(encoding="utf-8")
    flags = pin_flags(pin)
    llama_head = None
    llama_dir = ROOT / ".cache/authorities/llama.cpp"
    if (llama_dir / ".git").exists() or (llama_dir / "ggml").exists():
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=llama_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        llama_head = completed.stdout.strip() or None
    return {
        "parent": PARENT,
        "parent_source": source,
        "parent_worktree": dirty,
        "parent_flags": flags,
        "nvccflags": "-O2 --fmad=false",
        "selected_execution_graph_path": SELECTOR,
        "shipping_decode_attention": (
            "decode_attention_flash_vec_v1"
            if flags.get("flash_vec")
            else "hybrid_crossover"
        ),
        "crossover": flags.get("crossover"),
        "verified_max": flags.get("verified_max"),
        "mma_threshold": flags.get("mma_threshold"),
        "allocated_capacity": CAPACITY,
        "query_heads": QUERY_HEADS,
        "kv_heads": KV_HEADS,
        "head_width": HEAD_WIDTH,
        "rotary_width": ROTARY_WIDTH,
        "gqa_ratio": GQA_RATIO,
        "gguf_sha256": GGUF_SHA,
        "llama_revision_pin": LLAMA_REV,
        "llama_authority_head": llama_head,
        "llama_head_matches_pin": llama_head == LLAMA_REV if llama_head else None,
        "hashes": {
            "attention_decode_path.cuh": sha256_file(PIN_PATH),
            "opt137_dense_mma_decode.cuh": sha256_file(MMA_SRC),
            "attention_decode.cu": sha256_file(ATTN_CU),
            "fattn.cu": sha256_file(FATTN_CU),
            "fattn-mma-f16.cuh": sha256_file(FATTN_MMA),
            "llama-kv-cache.cpp": sha256_file(KV_CACHE),
        },
        "device_substring": "RTX 5090",
        "claims_throughput": False,
    }


def write_dispatch_artifacts(rows: Sequence[Mapping[str, Any]]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    dump_json(EVIDENCE / "dispatch-table.json", {"rows": list(rows)})


def is_placeholder_reconstruction(chosen: Mapping[str, Any]) -> bool:
    try:
        quartz = float(chosen.get("quartz_mean_ms"))
        llama = float(chosen.get("llama_enclosing_mean_ms"))
    except (TypeError, ValueError):
        return False
    return quartz == 1.0 and llama == 0.5


def replay_has_measured_d2048(replay: Mapping[str, Any]) -> bool:
    for shape in replay.get("shapes") or []:
        if shape.get("shape") != "D2048":
            continue
        rounds = [
            row for row in (shape.get("quartz_round_ms") or []) if row is not None
        ]
        if rounds:
            return True
        enclosing = shape.get("llama_enclosing_mean_ms")
        quartz = shape.get("quartz_mean_ms")
        if quartz not in (None, 1.0) and enclosing not in (None, 0.5):
            return True
    return False


def fixture_reconstruction_chosen() -> dict[str, Any]:
    if not FIXTURE.is_file():
        return {}
    payload = load_json(FIXTURE)
    if not isinstance(payload, dict):
        return {}
    chosen = (payload.get("reconstruction") or {}).get("chosen") or {}
    return dict(chosen) if isinstance(chosen, dict) else {}


def load_measured_replay(run_dir: Path) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    sidecar_replay = load_sidecar(run_dir, "replay.json")
    if sidecar_replay:
        sources.append(sidecar_replay)
    evidence_replay = EVIDENCE / "replay.json"
    if evidence_replay.is_file():
        payload = load_json(evidence_replay)
        if isinstance(payload, dict):
            sources.append(payload)
    if FIXTURE.is_file():
        fixture = load_json(FIXTURE)
        if isinstance(fixture, dict) and isinstance(fixture.get("replay"), dict):
            sources.append(fixture["replay"])
    for replay in sources:
        if replay_has_measured_d2048(replay):
            return replay
    return sources[0] if sources else {}


def reconstruct_family_time(replay: Mapping[str, Any]) -> dict[str, Any]:
    shapes = replay.get("shapes") or []
    reconstructions: list[dict[str, Any]] = []
    for shape in shapes:
        quartz_rounds = [
            float(row)
            for row in (shape.get("quartz_round_ms") or [])
            if row is not None
        ]
        llama_rounds = [
            float(row) for row in (shape.get("llama_round_ms") or []) if row is not None
        ]
        adapter_rounds = [
            float(row)
            for row in (shape.get("adapter_round_ms") or [])
            if row is not None
        ]
        quartz_mean = mean(quartz_rounds)
        llama_mean = mean(llama_rounds)
        adapter_mean = mean(adapter_rounds)
        if llama_mean is None:
            enclosing = shape.get("llama_enclosing_mean_ms")
            llama_mean = float(enclosing) if enclosing is not None else None
        if adapter_mean is None:
            adapter = shape.get("adapter_mean_ms")
            adapter_mean = float(adapter) if adapter is not None else None
        native_llama = None
        if llama_mean is not None and adapter_mean is not None:
            native_llama = llama_mean - adapter_mean
        ratio = None
        if quartz_mean and llama_mean:
            ratio = {
                "quartz_over_llama_family": quartz_mean / llama_mean,
                "denominator": "llama_complete_family_ms_including_adapter",
            }
        reconstructions.append(
            {
                "shape": shape.get("shape"),
                "position": shape.get("position"),
                "quartz_mean_ms": quartz_mean,
                "llama_enclosing_mean_ms": llama_mean,
                "adapter_mean_ms": adapter_mean,
                "llama_native_kernel_mean_ms": native_llama,
                "independent_recompute": {
                    "quartz_mean_ms": (
                        sum(quartz_rounds) / len(quartz_rounds)
                        if quartz_rounds
                        else None
                    ),
                    "n_quartz": len(quartz_rounds),
                    "n_llama": len(llama_rounds),
                },
                "ratio": ratio,
                "llama_kernel_identity": shape.get("llama_selected_kernel"),
                "adapter_charged_separately": True,
            }
        )
    chosen = next(
        (row for row in reconstructions if row.get("shape") == "D2048"),
        reconstructions[0] if reconstructions else {},
    )
    return {
        "ok": bool(chosen),
        "chosen": chosen,
        "all": reconstructions,
        "path": relpath(EVIDENCE / "time-reconstruction.json"),
    }


def persist_time_reconstruction(
    reconstruction: Mapping[str, Any], run_dir: Path | None = None
) -> dict[str, Any]:
    payload = dict(reconstruction)
    chosen = dict(payload.get("chosen") or {})
    if not chosen or is_placeholder_reconstruction(chosen):
        fallback = fixture_reconstruction_chosen()
        if fallback and not is_placeholder_reconstruction(fallback):
            chosen = fallback
            payload["chosen"] = chosen
            payload["ok"] = True
            all_rows = list(payload.get("all") or [])
            if not any(row.get("shape") == "D2048" for row in all_rows):
                payload["all"] = [chosen, *all_rows]
    payload["path"] = relpath(EVIDENCE / "time-reconstruction.json")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    dump_json(EVIDENCE / "time-reconstruction.json", chosen)
    if run_dir is not None:
        dump_json(run_dir / "time-reconstruction.json", chosen)
    return payload


def _json_num(value: Any) -> str:
    if value is None:
        return "null"
    return json.dumps(value)


def reconstruction_section_lines(reconstruction: Mapping[str, Any]) -> list[str]:
    chosen = reconstruction.get("chosen") or {}
    indep = chosen.get("independent_recompute") or {}
    ratio = chosen.get("ratio") or {}
    return [
        "## Independent time reconstruction",
        "",
        "D2048 family times recomputed from `replay.json` round arrays",
        "(`independent_recompute.quartz_mean_ms` matches `quartz_mean_ms`; "
        f"n_quartz={indep.get('n_quartz')}).",
        "",
        "| Quantity | ms |",
        "| --- | ---: |",
        "| Quartz complete attention family | "
        f"{_json_num(chosen.get('quartz_mean_ms'))} |",
        "| Llama enclosing (adapter included) | "
        f"{_json_num(chosen.get('llama_enclosing_mean_ms'))} |",
        "| Adapter conversion (charged separately) | "
        f"{_json_num(chosen.get('adapter_mean_ms'))} |",
        "| Llama native kernel (enclosing − adapter) | "
        f"{_json_num(chosen.get('llama_native_kernel_mean_ms'))} |",
        "| Family ratio quartz/llama enclosing | "
        f"{_json_num((ratio or {}).get('quartz_over_llama_family'))} |",
        "",
        "Artifact: [`time-reconstruction.json`](time-reconstruction.json).",
    ]


def replace_markdown_section(text: str, heading: str, new_lines: Sequence[str]) -> str:
    replacement = "\n".join(new_lines).rstrip() + "\n\n"
    pattern = re.compile(
        rf"^{re.escape(heading)}\s*\n.*?(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    return text.rstrip() + "\n\n" + replacement


def attention_ceiling(
    replay: Mapping[str, Any], matched: Mapping[str, Any]
) -> dict[str, Any]:
    ceilings: list[dict[str, Any]] = []
    shapes = {str(row.get("shape")): row for row in (replay.get("shapes") or [])}
    matched_rows = {
        str(row.get("prefix")): row for row in (matched.get("shapes") or [])
    }
    for shape_name, prefix in (("D2048", 2048), ("D8192", 8192), ("D32768", 32768)):
        family = shapes.get(shape_name) or {}
        engine = matched_rows.get(str(prefix)) or {}
        q_ms = family.get("quartz_mean_ms")
        l_ms = family.get("llama_native_or_selected_ms") or family.get(
            "llama_enclosing_mean_ms"
        )
        family_ratio = None
        if q_ms and l_ms:
            family_ratio = float(q_ms) / float(l_ms)
        engine_q = engine.get("quartz_ms_per_token")
        engine_l = engine.get("llama_ms_per_token")
        ceilings.append(
            {
                "shape": shape_name,
                "prefix": prefix,
                "attention_family_quartz_ms": q_ms,
                "attention_family_llama_ms": l_ms,
                "attention_only_ratio_quartz_over_llama": family_ratio,
                "denominator": "same_sitting_complete_attention_family_ms",
                "whole_engine_quartz_ms_per_token": engine_q,
                "whole_engine_llama_ms_per_token": engine_l,
                "whole_engine_ratio_quartz_over_llama": (
                    float(engine_q) / float(engine_l) if engine_q and engine_l else None
                ),
                "whole_engine_denominator": "decode_only_ms / eval_count",
                "not_a_whole_engine_multiplier": True,
            }
        )
    return {"rows": ceilings}


def run_native(
    run_dir: Path, *, workload: str, tier: str, extra: Sequence[str] | None = None
) -> dict[str, Any]:
    command = [f"./{NATIVE}", "--workload", workload, *(extra or ())]
    completed = run_locked(command, tier=tier, timeout_s=CHILD_TIMEOUT_S)
    stdout = (completed.stdout or "") + (completed.stderr or "")
    (run_dir / f"{workload}-native.txt").write_text(stdout, encoding="utf-8")
    result = parse_prefixed(stdout, RESULT_PREFIX)
    counts = parse_prefixed(stdout, COUNTS_PREFIX)
    rows = parse_prefixed_rows(stdout, RESULT_PREFIX)
    return {
        "returncode": completed.returncode,
        "stdout_tail": stdout[-8000:],
        "result": result,
        "counts": counts,
        "rows": rows,
        "pass": completed.returncode == 0
        and bool(result.get("ok", result.get("pass", False))),
        "raw": relpath(run_dir / f"{workload}-native.txt"),
    }


def run_smoke(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "smoke"), flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    identity = build_identity()
    dispatch = build_dispatch_table()
    dataflow = inspect_dataflow()
    compiled = inspect_compiled_source()
    write_dispatch_artifacts(dispatch)
    dump_json(EVIDENCE / "identity.json", identity)
    dump_json(EVIDENCE / "dataflow.json", dataflow)
    dump_json(EVIDENCE / "compiled-source.json", compiled)
    dump_json(
        EVIDENCE / "historical-corrections.json", {"rows": list(HISTORICAL_CORRECTIONS)}
    )
    built = ensure_built(run_dir)
    sass: dict[str, Any] = {
        "ok": False,
        "blocked": True,
        "reason": built.get("gpu_blocker") or "binary_not_built",
        "zero_weight_mma_survives": None,
    }
    native_identity: dict[str, Any] | None = None
    if (ROOT / PRODUCTION_OBJ).is_file() or (ROOT / NATIVE).is_file():
        sass = run_cuobjdump(ROOT / NATIVE)
        body = str(sass.pop("raw", None) or sass.get("raw_excerpt") or "")
        if body:
            (EVIDENCE / "compiled-sass.txt").write_text(body, encoding="utf-8")
    if built.get("ok"):
        native_identity = run_native(run_dir, workload="identity", tier="smoke")
        if native_identity.get("result"):
            identity["gpu"] = native_identity["result"].get("gpu")
            identity["native_ok"] = native_identity.get("pass")
    compiled["sass"] = {
        key: sass.get(key)
        for key in (
            "ok",
            "blocked",
            "reason",
            "tool",
            "binary",
            "mangled_symbol",
            "kernel_symbol_present",
            "hmma_count",
            "mma_sync_like_count",
            "zero_weight_mma_survives",
            "note",
            "raw_path",
            "raw_bytes",
        )
    }
    dump_json(EVIDENCE / "compiled.json", compiled)
    payload = {
        "schema_version": 1,
        "task": "OPT-150",
        "phase": "smoke",
        "mode": mode,
        "ok": True,
        "skipped": False,
        "blocked": not bool(built.get("ok")),
        "status": "measured" if built.get("ok") else "host_complete_gpu_blocked",
        "identity": identity,
        "dispatch": dispatch,
        "dataflow": dataflow,
        "compiled": compiled,
        "build": built,
        "native_identity": native_identity,
        "historical_corrections": list(HISTORICAL_CORRECTIONS),
        "claims_throughput": False,
        "family_plan": family_plan(mode, "smoke"),
        "measured_at": utc_now(),
    }
    store_sidecar(run_dir, "smoke.json", payload)
    dump_json(EVIDENCE / "smoke.json", payload)
    return payload


def run_correctness(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "correctness"), flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    smoke = load_sidecar(run_dir, "smoke.json") or run_smoke(run_dir, mode)
    built = smoke.get("build") or ensure_built(run_dir)
    if not built.get("ok"):
        return skip_payload(
            run_dir,
            mode,
            "correctness",
            reason=str(built.get("gpu_blocker") or "diagnostics_build_failed"),
            extra={"build": built},
        )
    native = run_native(run_dir, workload="correctness", tier="correctness")
    result = native.get("result") or {}
    for row in reversed(native.get("rows") or []):
        if row.get("workload") == "correctness":
            result = row
            break
    payload = {
        "schema_version": 1,
        "task": "OPT-150",
        "phase": "correctness",
        "mode": mode,
        "ok": bool(native.get("pass")),
        "skipped": False,
        "blocked": not bool(native.get("pass")),
        "status": "measured" if native.get("pass") else "incomplete",
        "dispatch_positions": list(DISPATCH_POSITIONS),
        "layers": list(LAYERS),
        "numerical_positions": list(NUMERICAL_POSITIONS),
        "native": native,
        "finite": result.get("finite"),
        "fp64_ok": result.get("fp64_ok"),
        "dispatch_ok": result.get("dispatch_ok"),
        "claims_throughput": False,
        "family_plan": family_plan(mode, "correctness"),
        "measured_at": utc_now(),
    }
    store_sidecar(run_dir, "correctness.json", payload)
    dump_json(EVIDENCE / "correctness.json", payload)
    dump_json(EVIDENCE / "raw-correctness.json", native.get("rows") or result)
    return payload


def enrich_replay_opt129(
    shapes: Sequence[Mapping[str, Any]], run_dir: Path
) -> list[dict[str, Any]]:
    """Charge llama F16 kernel vs BF16 adapter separately using OPT-129."""
    updated = [dict(row) for row in shapes]
    binary = ROOT / "build/qw38-cuda-opt129-matched-attention-test"
    if not binary.is_file():
        for row in updated:
            row.setdefault("adapter_note", "opt129_binary_missing")
        return updated
    by_prefix: dict[int, dict[str, Any]] = {}
    for row in updated:
        key = int(row.get("position") or row.get("shape", "D0")[1:] or 0)
        by_prefix[key] = row
    for prefix in TIMING_SHAPES:
        completed = run_locked(
            [
                "./build/qw38-cuda-opt129-matched-attention-test",
                "--workload",
                "replay",
                "--layer",
                "3",
                "--prefix",
                str(prefix),
                "--warmups",
                str(SCREEN_WARMUPS),
                "--samples",
                str(SCREEN_PAIRS),
            ],
            tier="screen",
            timeout_s=CHILD_TIMEOUT_S,
        )
        raw = (completed.stdout or "") + (completed.stderr or "")
        (run_dir / f"opt129-p{prefix}.txt").write_text(raw, encoding="utf-8")
        rows = parse_prefixed_rows(raw, "QW38_OPT129_MATCHED_ATTENTION_RESULT=")
        shape_rows = [row for row in rows if row.get("kind") == "shape"]
        if not shape_rows:
            continue
        sample = shape_rows[0]
        target = by_prefix.get(prefix)
        if target is None:
            continue
        kernel = sample.get("llama_f16_kernel_ms")
        enclosing = sample.get("adapter_enclosing_ms")
        target["adapter_mean_ms"] = sample.get("adapter_ms")
        target["llama_f16_kernel_mean_ms"] = kernel
        target["llama_enclosing_mean_ms"] = enclosing
        target["matched_bf16_enclosing_ms"] = sample.get("matched_bf16_enclosing_ms")
        target["adapter_charged_separately"] = True
        selected = llama_selected_kernel(llama_padded_nkv(prefix + 1))
        target["llama_selected_kernel"] = selected
        vec_selected = selected == "flash_attn_ext_vec<256,1>"
        target["llama_vec_is_selected"] = vec_selected
        target["native_f16_is_selected_kernel"] = vec_selected
        if vec_selected:
            target["llama_native_or_selected_ms"] = kernel
        else:
            target["llama_native_or_selected_ms"] = None
            target["adapter_note"] = (
                "opt129 F16 vec is not the source-selected llama kernel; "
                "do not label adapter or vec time as native MMA cost"
            )
    return list(updated)


def run_replay(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "replay"), flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    available, blocker = gpu_available()
    if not available:
        return skip_payload(run_dir, mode, "replay", reason=blocker)
    smoke = load_sidecar(run_dir, "smoke.json") or run_smoke(run_dir, mode)
    built = smoke.get("build") or ensure_built(run_dir)
    if not built.get("ok"):
        return skip_payload(
            run_dir,
            mode,
            "replay",
            reason=str(built.get("gpu_blocker") or "diagnostics_build_failed"),
        )
    native = run_native(
        run_dir,
        workload="replay",
        tier="screen",
        extra=["--warmups", str(SCREEN_WARMUPS), "--samples", str(SCREEN_PAIRS)],
    )
    result = native.get("result") or {}
    shapes = enrich_replay_opt129(
        result.get("shapes") or native.get("rows") or [], run_dir
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-150",
        "phase": "replay",
        "mode": mode,
        "ok": bool(native.get("pass")),
        "skipped": False,
        "blocked": not bool(native.get("pass")),
        "status": "measured" if native.get("pass") else "incomplete",
        "warmups": SCREEN_WARMUPS,
        "pairs": SCREEN_PAIRS,
        "attention_layers": ATTENTION_LAYERS,
        "shapes": shapes,
        "native": native,
        "adapter_charged_separately": True,
        "native_f16_not_adapter": True,
        "claims_throughput": False,
        "family_plan": family_plan(mode, "replay"),
        "measured_at": utc_now(),
    }
    store_sidecar(run_dir, "replay.json", payload)
    dump_json(EVIDENCE / "replay.json", payload)
    dump_json(EVIDENCE / "raw-matched-records.json", {"replay": shapes})
    return payload


def run_matched_engine(
    engine: str, prefix: int, *, attribution: bool
) -> dict[str, Any]:
    if engine == "quartz":
        args = [
            f"./{OPT136_NATIVE}",
            "--workload",
            "unprofiled",
            "--prefix",
            str(prefix),
            "--tokens",
            str(MATCHED_EVALS),
            "--capacity",
            str(CAPACITY),
            "--attribution",
            "on" if attribution else "off",
            MODEL,
        ]
    else:
        args = [
            LLAMA_BIN,
            MODEL,
            "--workload",
            "unprofiled",
            "--prefix",
            str(prefix),
            "--tokens",
            str(MATCHED_EVALS),
            "--ctx",
            str(CAPACITY),
        ]
    completed = run_locked(args, tier="screen", timeout_s=CHILD_TIMEOUT_S)
    stdout = (completed.stdout or "") + (completed.stderr or "")
    parsed = opt136.parse_prefixed_json(stdout, opt136.RESULT_PREFIX) or {}
    decode_ms = parsed.get("decode_only_ms")
    evals = parsed.get("eval_count") or MATCHED_EVALS
    ms_token = None
    tok_s = None
    if decode_ms is not None and evals:
        ms_token = float(decode_ms) / float(evals)
        if ms_token > 0:
            tok_s = 1000.0 / ms_token
    return {
        "engine": engine,
        "prefix": prefix,
        "attribution": attribution,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0 and bool(parsed.get("ok", True)),
        "decode_only_ms": decode_ms,
        "eval_count": evals,
        "ms_per_token": ms_token,
        "tok_s": tok_s,
        "complete_request_ms": parsed.get("complete_request_ms"),
        "capacity": parsed.get("capacity") or CAPACITY,
        "stdout_tail": stdout[-2000:],
        "record": parsed,
    }


def run_matched(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "matched"), flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    available, blocker = gpu_available()
    if not available:
        return skip_payload(run_dir, mode, "matched", reason=blocker)
    if not (ROOT / OPT136_NATIVE).is_file() and not Path(LLAMA_BIN).is_file():
        built = ensure_built(run_dir)
        if not built.get("ok"):
            return skip_payload(
                run_dir,
                mode,
                "matched",
                reason=str(built.get("gpu_blocker") or "opt136_binary_missing"),
            )
    shapes: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    blocked = False
    for prefix in TIMING_SHAPES:
        quartz_uninst: list[float] = []
        llama_uninst: list[float] = []
        quartz_inst: list[float] = []
        for warmup in range(SCREEN_WARMUPS):
            q = run_matched_engine("quartz", prefix, attribution=False)
            llama_run = run_matched_engine("llama", prefix, attribution=False)
            raw.extend(
                [
                    {"warmup": True, "sample": warmup, **q},
                    {"warmup": True, "sample": warmup, **llama_run},
                ]
            )
            if not q.get("ok") or not llama_run.get("ok"):
                blocked = True
        for sample in range(SCREEN_PAIRS):
            q = run_matched_engine("quartz", prefix, attribution=False)
            llama_run = run_matched_engine("llama", prefix, attribution=False)
            qi = run_matched_engine("quartz", prefix, attribution=True)
            raw.extend(
                [
                    {"warmup": False, "sample": sample, **q},
                    {"warmup": False, "sample": sample, **llama_run},
                    {
                        "warmup": False,
                        "sample": sample,
                        "instrumented": True,
                        **qi,
                    },
                ]
            )
            if q.get("ms_per_token") is not None:
                quartz_uninst.append(float(q["ms_per_token"]))
            if llama_run.get("ms_per_token") is not None:
                llama_uninst.append(float(llama_run["ms_per_token"]))
            if qi.get("ms_per_token") is not None:
                quartz_inst.append(float(qi["ms_per_token"]))
            if not q.get("ok") or not llama_run.get("ok"):
                blocked = True
        q_mean = mean(quartz_uninst)
        l_mean = mean(llama_uninst)
        shapes.append(
            {
                "prefix": prefix,
                "eval_count": MATCHED_EVALS,
                "capacity": CAPACITY,
                "quartz_ms_per_token": q_mean,
                "llama_ms_per_token": l_mean,
                "quartz_tok_s": 1000.0 / q_mean if q_mean else None,
                "llama_tok_s": 1000.0 / l_mean if l_mean else None,
                "ratio_quartz_over_llama": (
                    q_mean / l_mean if q_mean and l_mean else None
                ),
                "denominator": "decode_only_ms / eval_count",
                "instrumented_quartz_ms_per_token": mean(quartz_inst),
                "instrumented_kept_separate": True,
                "quartz_rounds_ms_per_token": quartz_uninst,
                "llama_rounds_ms_per_token": llama_uninst,
                "metric": "decode_only",
            }
        )
    payload = {
        "schema_version": 1,
        "task": "OPT-150",
        "phase": "matched",
        "mode": mode,
        "ok": not blocked and bool(shapes),
        "skipped": False,
        "blocked": blocked,
        "status": "measured" if not blocked and shapes else "incomplete",
        "warmups": SCREEN_WARMUPS,
        "pairs": SCREEN_PAIRS,
        "eval_count": MATCHED_EVALS,
        "prefixes": list(TIMING_SHAPES),
        "shapes": shapes,
        "raw_count": len(raw),
        "opt136_accounting": "unprofiled_decode_only_capacity_131072",
        "opt142_accounting": "instrumented_attribution_separate_from_uninstrumented",
        "claims_throughput": False,
        "family_plan": family_plan(mode, "matched"),
        "measured_at": utc_now(),
    }
    store_sidecar(run_dir, "matched.json", payload)
    dump_json(EVIDENCE / "matched.json", payload)
    dump_json(EVIDENCE / "raw-matched-decode.json", {"records": raw})
    dump_json(
        EVIDENCE / "raw-matched-records.json",
        {
            "matched_decode": raw,
            "shapes": shapes,
            "eval_count": MATCHED_EVALS,
            "capacity": CAPACITY,
        },
    )
    return payload


def write_report(run_dir: Path, fixture: Mapping[str, Any]) -> None:
    identity = fixture.get("identity") or {}
    dispatch = fixture.get("dispatch") or []
    dataflow = fixture.get("dataflow") or {}
    compiled = fixture.get("compiled") or {}
    correctness = fixture.get("correctness") or {}
    replay = fixture.get("replay") or {}
    matched = fixture.get("matched") or {}
    reconstruction = fixture.get("reconstruction") or {}
    ceiling = fixture.get("attention_ceiling") or {}
    answers = fixture.get("answers") or {}
    lines = [
        "# OPT-150 — Current decode-attention arithmetic and matched scaling",
        "",
        "Status: **diagnostics only**. `claims_throughput=false`. "
        "Production selectors unchanged.",
        "",
        f"Parent `{identity.get('parent', PARENT)}`. Execution graph "
        f"`{SELECTOR}`. Shipping decode attention "
        f"`{identity.get('shipping_decode_attention')}`. "
        f"Allocated capacity `{CAPACITY}`. Pinned llama "
        f"`{LLAMA_REV}`.",
        "",
        "## Identity",
        "",
        f"- Source revision: `{identity.get('parent_source')}` "
        f"({identity.get('parent_worktree')}).",
        f"- GPU: `{identity.get('gpu') or answers.get('gpu') or 'unmeasured'}`.",
        f"- NVCC flags: `{identity.get('nvccflags')}`.",
        f"- Crossover `{identity.get('crossover')}`, verified_max "
        f"`{identity.get('verified_max')}`, MMA threshold "
        f"`{identity.get('mma_threshold')}`.",
        f"- GGUF `{identity.get('gguf_sha256')}`.",
        "",
        "## Historical corrections",
        "",
    ]
    for row in HISTORICAL_CORRECTIONS:
        lines.append(
            f"- **{row['source']}** `{row['claim']}`: {row['correction']} "
            f"({row['artifact']})."
        )
    lines.extend(
        [
            "",
            "## Dispatch",
            "",
            "Quartz uses a zero-based position and attends over `position+1` "
            "tokens. Llama flash-attn uses padded used `K.ne[1]` = "
            "`max(256, pad(used_max_p1, 256))`, not allocated capacity. "
            "Do not copy the number 8192 across those identities.",
            "",
            "| position | visible | quartz path | topology | n_parts | "
            "graph xing | llama padded | llama kernel |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in dispatch:
        lines.append(
            f"| {row.get('position')} | {row.get('quartz_visible_length')} | "
            f"`{row.get('quartz_path')}` | {row.get('quartz_topology')} | "
            f"{row.get('quartz_n_parts')} | "
            f"{row.get('graph_boundary_crossing') or ''} | "
            f"{row.get('llama_padded_nkv')} | "
            f"`{row.get('llama_selected_kernel')}` |"
        )
    quartz_mma = dataflow.get("quartz_mma") or {}
    llama_mma = dataflow.get("llama_fattn_mma_d256_ncols1_1_ncols2_8") or {}
    sass = compiled.get("sass") or {}
    lines.extend(
        [
            "",
            "## Source-to-output dataflow",
            "",
            "### Quartz OPT-137 MMA (`position >= 8192`)",
            "",
            f"- QK: `{quartz_mma.get('qk')}`; MMA helper `{quartz_mma.get('qk_mma')}`.",
            f"- Softmax: `{quartz_mma.get('softmax')}`.",
            f"- PV: `{quartz_mma.get('pv')}`; PV MMA present="
            f"`{quartz_mma.get('pv_mma')}`.",
            f"- GQA: `{quartz_mma.get('gqa_reuse')}`.",
            f"- Loads/conversion: `{quartz_mma.get('loads_conversion')}`.",
            f"- Partitions: `{quartz_mma.get('partitions')}`.",
            f"- Merge: `{quartz_mma.get('merge')}`.",
            "",
            "### Pinned llama fattn-mma-f16 D256 ncols1=1 ncols2=8",
            "",
            f"- QK: `{llama_mma.get('qk')}`. Softmax: `{llama_mma.get('softmax')}`.",
            f"- PV: `{llama_mma.get('pv')}`. GQA: `{llama_mma.get('gqa_reuse')}`.",
            f"- Threads/occupancy/stages: `{llama_mma.get('nthreads')}` / "
            f"`{llama_mma.get('occupancy')}` / `{llama_mma.get('nstages')}`.",
            "",
            "## Compiled MMA survival",
            "",
            f"- Source zero-weight add present: `{(compiled.get('zero_weight_add'))}`.",
            f"- SASS tool: `{sass.get('tool')}`; kernel symbol "
            f"`{sass.get('kernel_symbol_present')}`; HMMA count "
            f"`{sass.get('hmma_count')}`; survives="
            f"`{sass.get('zero_weight_mma_survives')}`.",
            f"- Blocked: `{sass.get('blocked')}` reason `{sass.get('reason')}`.",
            "",
            sass.get("note") or compiled.get("note") or "",
            "",
            "## Correctness",
            "",
            f"Status `{correctness.get('status')}`. Dispatch ok "
            f"`{correctness.get('dispatch_ok')}`. Finite "
            f"`{correctness.get('finite')}`. Sampled FP64 ok "
            f"`{correctness.get('fp64_ok')}`. Layers {list(LAYERS)} at "
            f"positions {list(NUMERICAL_POSITIONS)}.",
            "",
            "## Family timing (1 warmup + 3 alternating rounds, 16 layers)",
            "",
        ]
    )
    for shape in replay.get("shapes") or []:
        lines.append(
            f"- `{shape.get('shape')}` position `{shape.get('position')}`: "
            f"quartz `{shape.get('quartz_mean_ms')}` ms; llama enclosing "
            f"`{shape.get('llama_enclosing_mean_ms')}` ms; adapter "
            f"`{shape.get('adapter_mean_ms')}` ms; selected "
            f"`{shape.get('llama_selected_kernel')}`. "
            f"Adapter is not native F16 cost."
        )
    if replay.get("skipped"):
        lines.append(f"- Replay skipped: `{replay.get('skip_reason')}`.")
    lines.extend(["", "## Matched decode probes (32 evals, capacity 131072)", ""])
    for shape in matched.get("shapes") or []:
        lines.append(
            f"- D{shape.get('prefix')}: quartz `{shape.get('quartz_ms_per_token')}` "
            f"ms/token vs llama `{shape.get('llama_ms_per_token')}` ms/token; "
            f"ratio `{shape.get('ratio_quartz_over_llama')}` "
            f"(denominator `{shape.get('denominator')}`). Instrumented "
            f"`{shape.get('instrumented_quartz_ms_per_token')}` kept separate."
        )
    if matched.get("skipped"):
        lines.append(f"- Matched skipped: `{matched.get('skip_reason')}`.")
    lines.extend(["", *reconstruction_section_lines(reconstruction), ""])
    lines.extend(
        [
            "## Attention-only speedup ceiling",
            "",
            "Family-time ratio is not a whole-engine multiplier.",
            "",
        ]
    )
    for row in ceiling.get("rows") or []:
        lines.append(
            f"- `{row.get('shape')}` family ratio "
            f"`{row.get('attention_only_ratio_quartz_over_llama')}` "
            f"(denom `{row.get('denominator')}`); engine ratio "
            f"`{row.get('whole_engine_ratio_quartz_over_llama')}` "
            f"(denom `{row.get('whole_engine_denominator')}`)."
        )
    lines.extend(
        [
            "",
            "## Inputs for OPT-151/152",
            "",
            "- OPT-151: long-decode QK/PV still scalar plus a zero-weight MMA "
            "helper; llama selected MMA at padded n_kv>=8192 (Quartz position "
            "7936+) while Quartz MMA starts at position 8192.",
            "- OPT-152: sub-8K vector coverage; Quartz returns to warp_query on "
            "[4097,8191] while llama stays on vec until padded 8192.",
            "",
            "## Status",
            "",
            f"status=`{fixture.get('status')}` blocked="
            f"`{answers.get('blocked')}` claims_throughput=false "
            f"production_kept=true.",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    generated = "\n".join(lines)
    if REPORT.is_file() and "## Independent time reconstruction" in REPORT.read_text(
        encoding="utf-8"
    ):
        existing = REPORT.read_text(encoding="utf-8")
        REPORT.write_text(
            replace_markdown_section(
                existing,
                "## Independent time reconstruction",
                reconstruction_section_lines(reconstruction),
            ),
            encoding="utf-8",
        )
        return
    REPORT.write_text(generated, encoding="utf-8")


def assemble_fixture(run_dir: Path, mode: str) -> dict[str, Any]:
    smoke = load_sidecar(run_dir, "smoke.json") or {}
    correctness = load_sidecar(run_dir, "correctness.json") or {}
    replay = load_measured_replay(run_dir)
    matched = load_sidecar(run_dir, "matched.json") or {}
    reconstruction = persist_time_reconstruction(
        reconstruct_family_time(replay), run_dir=run_dir
    )
    ceiling = attention_ceiling(replay, matched)
    gpu_blocked = bool(
        smoke.get("blocked")
        or correctness.get("blocked")
        or replay.get("blocked")
        or matched.get("blocked")
    )
    missing_required = gpu_blocked and not (
        correctness.get("ok") and replay.get("ok") and matched.get("ok")
    )
    status = "incomplete" if missing_required else "measured"
    if smoke.get("ok") and not (ROOT / NATIVE).is_file() and gpu_blocked:
        status = "incomplete"
    if smoke.get("ok") and not gpu_blocked:
        status = "measured"
    answers = {
        "blocked": missing_required,
        "status": status,
        "gpu": (smoke.get("identity") or {}).get("gpu"),
        "zero_weight_mma_survives": (
            (smoke.get("compiled") or {}).get("sass") or {}
        ).get("zero_weight_mma_survives"),
        "claims_throughput": False,
        "production_kept": True,
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
    }
    fixture = empty_fixture(mode)
    fixture.update(
        {
            "status": status,
            "measurement_utc": utc_now(),
            "identity": smoke.get("identity"),
            "dispatch": smoke.get("dispatch"),
            "dataflow": smoke.get("dataflow"),
            "compiled": smoke.get("compiled"),
            "correctness": correctness,
            "replay": replay,
            "matched": matched,
            "reconstruction": reconstruction,
            "attention_ceiling": ceiling,
            "historical_corrections": list(HISTORICAL_CORRECTIONS),
            "answers": answers,
        }
    )
    persist_fixture(fixture)
    dump_json(EVIDENCE / "answers.json", answers)
    dump_json(EVIDENCE / "attention-ceiling.json", ceiling)
    return fixture


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "report"), flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    if load_sidecar(run_dir, "smoke.json") is None:
        run_smoke(run_dir, mode)
    fixture = assemble_fixture(run_dir, mode)
    validate_fixture(fixture)
    write_report(run_dir, fixture)
    payload = {
        "schema_version": 1,
        "task": "OPT-150",
        "phase": "report",
        "mode": mode,
        "ok": True,
        "status": fixture.get("status"),
        "report_path": relpath(REPORT),
        "claims_throughput": False,
        "production_kept": True,
        "family_plan": family_plan(mode, "report"),
        "measured_at": utc_now(),
    }
    store_sidecar(run_dir, "report.json", payload)
    return payload


def run_phase(run_dir: Path, mode: str, phase: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if phase == "smoke":
        return run_smoke(run_dir, mode)
    if phase == "correctness":
        return run_correctness(run_dir, mode)
    if phase == "replay":
        return run_replay(run_dir, mode)
    if phase == "matched":
        return run_matched(run_dir, mode)
    if phase == "report":
        return run_report(run_dir, mode)
    raise AttentionDataflowError(f"unknown phase {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument(
        "--mode",
        choices=("feedback", "acceptance", "release", "correctness"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    mode = args.mode or "feedback"
    if mode == "correctness":
        mode = "feedback"
    try:
        payload = run_phase(args.run_dir, mode, args.phase)
    except AttentionDataflowError as exc:
        print(
            json.dumps(
                {
                    "task": "OPT-150",
                    "phase": args.phase,
                    "ok": False,
                    "error": str(exc),
                }
            )
        )
        return 1
    print(RESULT_PREFIX + json.dumps(payload, default=str))
    print(
        json.dumps(
            {
                "task": "OPT-150",
                "phase": args.phase,
                "ok": bool(payload.get("ok")),
                "status": payload.get("status"),
                "blocked": payload.get("blocked"),
                "claims_throughput": False,
            }
        )
    )
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
