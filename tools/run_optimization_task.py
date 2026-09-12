"""Run a bounded optimization feedback, acceptance, or release loop."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
FEEDBACK_BUDGET_S = 300.0
MODES = ("feedback", "acceptance", "release")
TIERS = frozenset({"smoke", "correctness", "screen", "acceptance"})
HISTORICAL_ORACLE_TARGETS = (
    "build/qw38-cuda-prefill-4k-oracle-test",
    "build/qw38-cuda-decode-oracle-test",
)
HISTORICAL_OPT074_KEEP_TASKS = frozenset({"OPT-070", "OPT-075", "OPT-076"})
OPT074_UNADMITTED_REASON = "opt074_coverage_unadmitted"
OPT074_PROMOTION_BLOCKER_KEYS = frozenset(
    {
        "promotion_blockers",
        "keep_blockers",
        "required_blockers",
        "required_promotion_blockers",
        "active_keep_blockers",
        "admission_blockers",
        "keep_promotion_blockers",
    }
)
SETUP_IMAGE_COMMAND = "docker build -f docker/cuda.Dockerfile -t qw38-cuda:13.0.2 ."


class SetupError(RuntimeError):
    """Image, model, GPU lock, or foreign-compute setup failed."""


class KernelParityPolicyError(SetupError):
    """A future keep contract still treats OPT-074 unadmitted as a blocker."""


class BudgetExhausted(RuntimeError):
    """The 300-second aggregate deadline expired."""


@dataclass
class Clock:
    """Monotonic and wall clocks; tests inject a fake."""

    def monotonic(self) -> float:
        return time.monotonic()

    def time(self) -> float:
        return time.time()


@dataclass
class Completed:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    compiled: bool = False
    pid: int | None = None
    pgid: int | None = None
    docker_name: str | None = None
    terminated: bool = False


@dataclass
class OwnedChild:
    pid: int
    pgid: int
    docker_name: str | None = None


class ProcessLauncher:
    """Launch and tear down owned subprocesses and containers."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        timeout_s: float | None,
        docker_name: str | None = None,
    ) -> Completed:
        raise NotImplementedError

    def terminate(self, child: OwnedChild) -> None:
        raise NotImplementedError


class SubprocessLauncher(ProcessLauncher):
    def __init__(self) -> None:
        self.owned: list[OwnedChild] = []

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        timeout_s: float | None,
        docker_name: str | None = None,
    ) -> Completed:
        popen_env = os.environ.copy()
        if env:
            popen_env.update(env)
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=popen_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        pgid = os.getpgid(process.pid)
        child = OwnedChild(pid=process.pid, pgid=pgid, docker_name=docker_name)
        self.owned.append(child)
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
            returncode = process.returncode
            terminated = False
        except subprocess.TimeoutExpired:
            self.terminate(child)
            stdout, stderr = process.communicate()
            returncode = process.returncode if process.returncode is not None else 1
            terminated = True
        compiled = "nvcc" in (stdout + stderr) or "g++" in (stdout + stderr)
        return Completed(
            command=list(command),
            returncode=returncode or 0,
            stdout=stdout or "",
            stderr=stderr or "",
            compiled=compiled,
            pid=process.pid,
            pgid=pgid,
            docker_name=docker_name,
            terminated=terminated,
        )

    def terminate(self, child: OwnedChild) -> None:
        if child.docker_name:
            subprocess.run(
                ["docker", "kill", child.docker_name],
                check=False,
                capture_output=True,
                text=True,
            )
        try:
            os.killpg(child.pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            os.killpg(child.pgid, signal.SIGKILL)
        except ProcessLookupError:
            return


@dataclass
class GpuLock:
    path: Path
    _fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._fd)
            self._fd = None
            raise SetupError(f"exclusive GPU lock is held at {self.path}") from exc

    def release(self) -> None:
        if self._fd is None:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None


def validate_tier(value: str | None) -> str:
    if value is None or not str(value).strip():
        raise ValueError(
            "QW38_CUDA_TEST_TIER must be set to smoke, correctness, "
            "screen, or acceptance"
        )
    tier = str(value).strip().lower()
    if tier not in TIERS:
        raise ValueError(
            f"QW38_CUDA_TEST_TIER must be one of {sorted(TIERS)}, got {tier!r}"
        )
    return tier


def contract_path(task: str) -> Path:
    number = task.split("-", 1)[1].lower()
    return ROOT / "pins" / f"opt{number}_iteration_contract.json"


def _opt074_unadmitted_blockers(node: Any) -> list[str]:
    found: list[str] = []
    if isinstance(node, Mapping):
        if node.get("opt074_family_admission_required") is True:
            found.append("opt074_family_admission_required")
        for key, value in node.items():
            if key in OPT074_PROMOTION_BLOCKER_KEYS:
                items = value if isinstance(value, (list, tuple)) else [value]
                for item in items:
                    text = str(item)
                    if OPT074_UNADMITTED_REASON in text:
                        found.append(text)
            else:
                found.extend(_opt074_unadmitted_blockers(value))
    elif isinstance(node, (list, tuple)):
        for item in node:
            found.extend(_opt074_unadmitted_blockers(item))
    return found


def validate_future_keep_policy(task: str, contract: Mapping[str, Any]) -> None:
    """Fail closed if a non-historical keep still requires OPT-074 unadmitted."""
    if task in HISTORICAL_OPT074_KEEP_TASKS:
        return
    blockers = _opt074_unadmitted_blockers(contract)
    if not blockers:
        return
    raise KernelParityPolicyError(
        "policy error: future keep contract still requires "
        f"{OPT074_UNADMITTED_REASON} as a promotion blocker "
        f"(task={task}, fields={blockers}); kernel_parity_v1 is the active "
        "kernel-admission authority and OPT-074 unadmitted is not an active "
        "keep blocker"
    )


def load_contract(task: str) -> dict[str, Any]:
    path = contract_path(task)
    if not path.is_file():
        raise SetupError(f"missing task contract {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("task") != task:
        raise SetupError(f"contract task {contract.get('task')!r} != {task}")
    validate_future_keep_policy(task, contract)
    return contract


def loop_product(workload: Mapping[str, Any]) -> int:
    cases = max(1, int(workload.get("cases", 1)))
    candidates = max(1, int(workload.get("candidates", 1)))
    warmups = int(workload.get("warmups", 0))
    samples = max(1, int(workload.get("samples", 1)))
    tokens = max(1, int(workload.get("tokens", 1)))
    modes = max(1, int(workload.get("execution_modes", 1)))
    pairs = max(1, int(workload.get("control_candidate_pairs", 1)))
    return cases * candidates * (warmups + samples) * tokens * modes * pairs


def workload_for_mode(workload: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Apply optional per-mode warmup/sample/tier overrides for one family."""
    merged = dict(workload)
    extra = (workload.get("mode_overrides") or {}).get(mode)
    if isinstance(extra, Mapping):
        merged.update(dict(extra))
    return merged


def gpu_setup_required(contract: Mapping[str, Any]) -> bool:
    if contract.get("host_only") or contract.get("skip_gpu_setup"):
        return False
    return True


def compile_required(contract: Mapping[str, Any], mode: str) -> bool:
    if contract.get("skip_compile"):
        return False
    mode_spec = contract.get("modes", {}).get(mode, {})
    if mode_spec.get("skip_compile"):
        return False
    return True


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            payload, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(payload, dict):
            objects.append(payload)
        index = start + consumed
    return objects


def parse_native_observation(stdout: str) -> dict[str, Any]:
    """Read actual native warmup/round/candidate/shape/tier counts from stdout."""
    text = stdout or ""
    records = _json_objects(text)
    counts: dict[str, Any] = {}
    for prefix in (
        "QW38_OPT070_NATIVE_COUNTS=",
        "QW38_OPT070_RESULT=",
        "QW38_OPT075_NATIVE_COUNTS=",
        "QW38_OPT075_RESULT=",
        "QW38_OPT076_NATIVE_COUNTS=",
        "QW38_OPT076_RESULT=",
        "QW38_OPT077_NATIVE_COUNTS=",
        "QW38_OPT077_RESULT=",
        "QW38_OPT078_NATIVE_COUNTS=",
        "QW38_OPT078_RESULT=",
        "QW38_OPT078_DECODE_ATTENTION_RESULT=",
        "QW38_OPT079_NATIVE_COUNTS=",
        "QW38_OPT079_RESULT=",
        "QW38_OPT079_ATTENTION_KV_OPERANDS_RESULT=",
        "QW38_OPT082_NATIVE_COUNTS=",
        "QW38_OPT082_KERNEL_PARITY_RESULT=",
        "QW38_OPT082_CASE=",
        "QW38_OPT085_NATIVE_COUNTS=",
        "QW38_OPT085_RESULT=",
        "QW38_OPT086_NATIVE_COUNTS=",
        "QW38_OPT086_RESULT=",
        "QW38_OPT087_NATIVE_COUNTS=",
        "QW38_OPT087_RESULT=",
        "QW38_OPT089_NATIVE_COUNTS=",
        "QW38_OPT089_RESULT=",
        "QW38_OPT090_NATIVE_COUNTS=",
        "QW38_OPT090_RESULT=",
        "QW38_OPT090_ATTRIBUTION_RESULT=",
        "QW38_OPT091_NATIVE_COUNTS=",
        "QW38_OPT091_RESULT=",
        "QW38_OPT061_COMPONENT_REPLAY_RESULT=",
        "QW38_OPT057_PROBE_RESULT=",
    ):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                try:
                    payload = json.loads(stripped[len(prefix) :])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    for record in records:
        for key, value in record.items():
            if key not in counts or counts[key] in (None, "", [], {}):
                counts[key] = value
        nested = record.get("native_counts")
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                counts.setdefault(key, value)
    keep = counts.get("keep")
    if keep is None:
        if re.search(r"\bkeep\s*=\s*true\b", text, re.IGNORECASE):
            keep = True
        elif re.search(r"\bkeep\s*=\s*false\b", text, re.IGNORECASE):
            keep = False
    counts["keep"] = keep
    rounds = [
        record
        for record in records
        if record.get("observation_unit") == "independent_round"
        or "sample_index" in record
        or record.get("round") is not None
    ]
    if not rounds:
        rounds = list(counts.get("rounds") or counts.get("paired_rounds") or [])
    sample_ids: list[Any] = []
    for record in rounds:
        if isinstance(record, Mapping):
            if "sample_index" in record:
                sample_ids.append(record.get("sample_index"))
            elif "sample_id" in record:
                sample_ids.append(record.get("sample_id"))
        elif isinstance(record, (int, float, str)):
            sample_ids.append(record)
    if not sample_ids and counts.get("sample_ids"):
        sample_ids = list(counts["sample_ids"])
    counts["rounds"] = rounds
    counts["sample_ids"] = sample_ids
    counts["observed_warmups"] = int(
        counts.get("observed_warmups", counts.get("warmups", 0)) or 0
    )
    counts["observed_samples"] = int(
        counts.get("observed_samples", counts.get("samples", len(sample_ids))) or 0
    )
    counts["observed_candidates"] = int(
        counts.get("observed_candidates", counts.get("candidates", 0)) or 0
    )
    counts["observed_shapes"] = int(
        counts.get("observed_shapes", counts.get("shapes", counts.get("cases", 0))) or 0
    )
    counts["observed_tier"] = str(
        counts.get("observed_tier", counts.get("tier", counts.get("executed_tier", "")))
        or ""
    )
    counts["acceptance_executed"] = bool(
        counts.get("acceptance_executed", counts.get("observed_tier") == "acceptance")
    )
    counts["pairs"] = int(
        counts.get("pairs", counts.get("control_candidate_pairs", 0)) or 0
    )
    return counts


def validate_performance_admission(
    contract: Mapping[str, Any],
    *,
    mode: str,
    workload_name: str,
    workload: Mapping[str, Any],
    stdout: str,
    success: bool,
) -> dict[str, Any]:
    """Reject screen-only keeps, missing pairs, reused samples, and count drift."""
    spec = contract.get("performance_admission")
    if not spec:
        return {"ok": True, "result_class": "ok", "message": ""}
    instrumentation_only = bool(
        spec.get("instrumentation_only", contract.get("instrumentation_only", False))
    )
    observed = parse_native_observation(stdout)
    keep = observed.get("keep")
    if instrumentation_only:
        if keep is True:
            return {
                "ok": False,
                "result_class": "instrumentation_keep_forbidden",
                "message": "instrumentation-only tasks cannot claim a keep",
                "observed": observed,
            }
        return {
            "ok": True,
            "result_class": "ok",
            "message": "instrumentation-only: no speed admission",
            "observed": observed,
        }
    if not success:
        return {
            "ok": False,
            "result_class": "native_failed",
            "message": "native workload failed before admission",
            "observed": observed,
        }
    plan_warmups = int(workload.get("warmups", 0))
    plan_samples = int(workload.get("samples", 0) or 0)
    plan_candidates = int(workload.get("candidates", 0) or 0)
    plan_shapes = int(workload.get("cases", 0) or 0)
    plan_tier = str(workload.get("tier", workload_name))
    mismatches: list[str] = []
    if observed["observed_warmups"] != plan_warmups:
        mismatches.append(
            f"warmups observed={observed['observed_warmups']} plan={plan_warmups}"
        )
    if observed["observed_samples"] != plan_samples:
        mismatches.append(
            f"rounds observed={observed['observed_samples']} plan={plan_samples}"
        )
    if observed["observed_candidates"] != plan_candidates:
        mismatches.append(
            f"candidates observed={observed['observed_candidates']} "
            f"plan={plan_candidates}"
        )
    if observed["observed_shapes"] != plan_shapes:
        mismatches.append(
            f"shapes observed={observed['observed_shapes']} plan={plan_shapes}"
        )
    observed_tier = observed["observed_tier"]
    if observed_tier and observed_tier != plan_tier:
        mismatches.append(f"tier observed={observed_tier} plan={plan_tier}")
    if mismatches:
        return {
            "ok": False,
            "result_class": "native_count_mismatch",
            "message": "native counts disagree with the plan: " + "; ".join(mismatches),
            "observed": observed,
        }
    sample_ids = [item for item in observed.get("sample_ids", []) if item is not None]
    if len(sample_ids) != len(set(sample_ids)) and sample_ids:
        return {
            "ok": False,
            "result_class": "reused_sample_ids",
            "message": "paired rounds reused sample IDs",
            "observed": observed,
        }
    planned_pairs = int(workload.get("control_candidate_pairs", 1) or 1)
    if keep is True or mode == "acceptance":
        if planned_pairs > 0 and observed.get("pairs", 0) < planned_pairs:
            return {
                "ok": False,
                "result_class": "missing_pairs",
                "message": "missing control/candidate pairs for admission",
                "observed": observed,
            }
    if mode == "acceptance" and not observed.get("acceptance_executed"):
        return {
            "ok": False,
            "result_class": "unexecuted_acceptance",
            "message": "acceptance mode did not execute the acceptance tier",
            "observed": observed,
        }
    if keep is True and (
        observed_tier == "screen"
        or (plan_tier == "screen" and not observed.get("acceptance_executed"))
    ):
        return {
            "ok": False,
            "result_class": "screen_only_keep",
            "message": "screen-only keep=true is not production admission",
            "observed": observed,
        }
    return {
        "ok": True,
        "result_class": "ok",
        "message": "",
        "observed": observed,
    }


def describe_plan(
    task: str,
    mode: str,
    contract: Mapping[str, Any],
    phase: str | None,
) -> str:
    mode_spec = contract["modes"][mode]
    deadline = float(
        mode_spec.get(
            "aggregate_deadline_s",
            contract.get("aggregate_deadline_s", FEEDBACK_BUDGET_S),
        )
    )
    target = str(mode_spec.get("target", contract["target"]))
    tiers = list(mode_spec.get("tier_sequence", []))
    case_ids = [str(item) for item in contract.get("case_ids", [])]
    lines = [
        f"task={task}",
        f"mode={mode}",
        f"target={target}",
        f"tier={' '.join(tiers) if tiers else mode_spec.get('tier', 'n/a')}",
        f"deadline_s={deadline:g}",
        f"phase={phase or 'all'}",
        f"host_only={str(bool(contract.get('host_only', False))).lower()}",
        f"skip_gpu_setup={str(not gpu_setup_required(contract)).lower()}",
        f"skip_compile={str(not compile_required(contract, mode)).lower()}",
    ]
    if case_ids:
        lines.append("case_ids=" + ",".join(case_ids))
        lines.append(f"case_id_count={len(case_ids)}")
    workloads = contract.get("workloads", {})
    selected = mode_spec.get("workloads", list(workloads))
    if phase:
        selected = [phase] if phase in workloads else selected
    total = 0
    for name in selected:
        if name not in workloads:
            continue
        workload = workload_for_mode(workloads[name], mode)
        product = loop_product(workload)
        total += product
        extras: list[str] = []
        for key in (
            "engines",
            "configurations",
            "prefixes",
            "replicates",
            "output_tokens",
            "held_out_targets",
            "functional_cases",
            "functional_tokens",
        ):
            if key in workload:
                extras.append(f"{key}={workload[key]}")
        extra = (" " + " ".join(str(item) for item in extras)) if extras else ""
        lines.append(
            f"{name}: target={workload.get('target', target)} "
            f"tier={workload.get('tier', name)} "
            f"cases={workload.get('cases', 1)} "
            f"candidates={workload.get('candidates', 0)} "
            f"warmups={workload.get('warmups', 0)} "
            f"samples={workload.get('samples', 1)} "
            f"tokens={workload.get('tokens', 0)} "
            f"modes={workload.get('execution_modes', 1)} "
            f"pairs={workload.get('control_candidate_pairs', 1)} "
            f"product={product}"
            f"{extra}"
        )
    lines.append(f"loop_product_total={total}")
    if mode == "release":
        lines.append("historical_oracles=" + ",".join(HISTORICAL_ORACLE_TARGETS))
    else:
        lines.append("historical_oracles=none")
    return "\n".join(lines) + "\n"


def docker_image_present(
    image: str,
    launcher: ProcessLauncher | None = None,
    cwd: Path = ROOT,
) -> bool:
    command = ["docker", "image", "inspect", image]
    if launcher is None:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        return result.returncode == 0
    completed = launcher.run(command, cwd=cwd, env=None, timeout_s=30)
    return completed.returncode == 0


def list_compute_apps(
    launcher: ProcessLauncher | None = None,
    cwd: Path = ROOT,
) -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader",
    ]
    if launcher is None:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        text = result.stdout
        if result.returncode != 0:
            return []
    else:
        completed = launcher.run(command, cwd=cwd, env=None, timeout_s=15)
        if completed.returncode != 0:
            return []
        text = completed.stdout
    apps: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) >= 2:
            apps.append({"pid": parts[0], "process_name": parts[1]})
    return apps


@dataclass
class OptimizationRunner:
    root: Path = ROOT
    clock: Clock = field(default_factory=Clock)
    launcher: ProcessLauncher = field(default_factory=SubprocessLauncher)
    image_exists: Callable[[str], bool] | None = None
    compute_apps: Callable[[], list[dict[str, str]]] | None = None
    lock_factory: Callable[[Path], Any] | None = None
    owned: list[OwnedChild] = field(default_factory=list)

    def run(
        self,
        task: str,
        mode: str,
        *,
        phase: str | None = None,
        output_dir: Path | None = None,
        dry_run: bool = False,
        contract_data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        contract = (
            dict(contract_data) if contract_data is not None else load_contract(task)
        )
        if contract_data is not None:
            validate_future_keep_policy(task, contract)
        plan = describe_plan(task, mode, contract, phase)
        if dry_run:
            return {
                "status": "dry_run",
                "result_class": "dry_run",
                "plan": plan,
                "task": task,
                "mode": mode,
            }
        run_id = (
            datetime.fromtimestamp(self.clock.time(), tz=timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            + f"-{uuid.uuid4().hex[:8]}"
        )
        destination = output_dir or (
            self.root / "build" / "optimization-runs" / task / mode
        )
        run_dir = destination / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plan.txt").write_text(plan, encoding="utf-8")
        phases_path = run_dir / "phases.jsonl"
        result = self._execute(task, mode, contract, phase, run_dir, phases_path)
        (run_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        result["run_dir"] = str(run_dir)
        result["plan"] = plan
        return result

    def terminate_owned(self) -> None:
        for child in list(self.owned):
            self.launcher.terminate(child)
        self.owned.clear()

    def _execute(
        self,
        task: str,
        mode: str,
        contract: Mapping[str, Any],
        phase: str | None,
        run_dir: Path,
        phases_path: Path,
    ) -> dict[str, Any]:
        started = self.clock.monotonic()
        mode_spec = contract["modes"][mode]
        deadline = float(
            mode_spec.get(
                "aggregate_deadline_s",
                contract.get("aggregate_deadline_s", FEEDBACK_BUDGET_S),
            )
        )
        records: list[dict[str, Any]] = []
        lock: GpuLock | None = None
        durations: dict[str, float] = {}
        compile_count = 0
        link_count = 0
        cache_hits = 0
        cache_misses = 0
        load_count = 0
        reference_count = 0
        result_class = "ok"
        success = True
        message = ""

        def remaining() -> float:
            return deadline - (self.clock.monotonic() - started)

        def emit(record: dict[str, Any]) -> None:
            records.append(record)
            with phases_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

        def phase_wrap(name: str, body: Callable[[], dict[str, Any]]) -> dict[str, Any]:
            nonlocal success, result_class, message
            if remaining() <= 0:
                raise BudgetExhausted(name)
            begin = self.clock.monotonic()
            epoch_start = self.clock.time()
            emit(
                {
                    "phase": name,
                    "event": "start",
                    "epoch_s": epoch_start,
                    "monotonic_s": begin,
                }
            )
            payload = body()
            end = self.clock.monotonic()
            payload.setdefault("elapsed_s", end - begin)
            record = {
                "phase": name,
                "event": "end",
                "epoch_s": self.clock.time(),
                "elapsed_s": end - begin,
                "success": payload.get("success", True),
                "cache_hits": payload.get("cache_hits", 0),
                "cache_misses": payload.get("cache_misses", 0),
                "compile": payload.get("compile", 0),
                "link": payload.get("link", 0),
                "load": payload.get("load", 0),
                "reference": payload.get("reference", 0),
                "tokens": payload.get("tokens", 0),
                "selected_paths": payload.get("selected_paths", []),
                "result_class": payload.get("result_class", "ok"),
            }
            emit(record)
            if not record["success"]:
                success = False
                result_class = str(payload.get("result_class", "numerical_failure"))
                message = str(payload.get("message", ""))
            return payload

        try:
            setup = phase_wrap(
                "setup",
                lambda: self._setup(contract, run_dir),
            )
            lock = setup.get("lock")
            if phase == "setup":
                return self._finish(
                    task,
                    mode,
                    run_dir,
                    started,
                    True,
                    "ok",
                    "",
                    durations,
                    compile_count,
                    cache_hits,
                    cache_misses,
                )

            mode_spec = contract["modes"][mode]
            workloads = contract.get("workloads", {})
            if phase not in {None, "compile"}:
                known = set(mode_spec.get("tier_sequence", [])) | set(workloads)
                if phase not in known:
                    raise ValueError(f"unknown phase {phase}")
            repetitions = int(
                mode_spec.get("warm_repetitions", 2 if mode == "acceptance" else 0)
            )
            labels = ["cold"]
            if repetitions:
                labels.extend(f"warm{index}" for index in range(1, repetitions + 1))
            should_compile = compile_required(contract, mode)

            for label in labels:
                if remaining() <= 0:
                    raise BudgetExhausted("compile")
                label_started = self.clock.monotonic()
                if should_compile and (
                    phase in {None, "compile"}
                    or (
                        phase in workloads
                        or phase in mode_spec.get("tier_sequence", [])
                    )
                ):
                    built = phase_wrap(
                        f"compile_{label}",
                        lambda current=label: self._compile(
                            contract, mode, remaining(), run_dir, current
                        ),
                    )
                    compile_count += int(built.get("compile", 0))
                    link_count += int(built.get("link", 0))
                    cache_hits += int(built.get("cache_hits", 0))
                    cache_misses += int(built.get("cache_misses", 0))
                    durations[f"compile_{label}_s"] = float(built.get("elapsed_s", 0.0))
                    if not built.get("success", True):
                        break
                    if label != "cold" and int(built.get("compile", 0)):
                        success = False
                        result_class = "recompiled_on_warm"
                        message = "warm repetition recompiled unchanged inputs"
                if phase == "compile":
                    continue
                if mode == "release":
                    released = phase_wrap(
                        f"release_{label}",
                        lambda: self._release(contract, remaining(), run_dir),
                    )
                    durations[f"{label}_s"] = float(released.get("elapsed_s", 0.0))
                    continue
                sequence = list(mode_spec.get("tier_sequence", []))
                if phase:
                    sequence = [phase] if phase in sequence else []
                passed = True
                for tier_name in sequence:
                    if not passed:
                        emit(
                            {
                                "phase": f"{tier_name}_{label}",
                                "event": "end",
                                "elapsed_s": 0.0,
                                "success": False,
                                "result_class": "skipped_after_failure",
                            }
                        )
                        continue
                    work = phase_wrap(
                        f"{tier_name}_{label}",
                        lambda name=tier_name: self._workload(
                            contract, mode, name, remaining(), run_dir
                        ),
                    )
                    load_count += int(work.get("load", 0))
                    reference_count += int(work.get("reference", 0))
                    passed = bool(work.get("success", True))
                    if not passed:
                        break
                durations[f"{label}_s"] = self.clock.monotonic() - label_started
        except BudgetExhausted as exc:
            success = False
            result_class = "budget_exhausted"
            message = f"budget exhausted during {exc}"
            self.terminate_owned()
            emit(
                {
                    "phase": str(exc),
                    "event": "end",
                    "elapsed_s": self.clock.monotonic() - started,
                    "success": False,
                    "result_class": "budget_exhausted",
                }
            )
        except ValueError as exc:
            success = False
            message = str(exc)
            result_class = (
                "invalid_phase" if "unknown phase" in message else "invalid_tier"
            )
            emit(
                {
                    "phase": "tier",
                    "event": "end",
                    "elapsed_s": self.clock.monotonic() - started,
                    "success": False,
                    "result_class": "invalid_tier",
                    "message": message,
                }
            )
        except SetupError as exc:
            success = False
            result_class = "setup_failed"
            message = str(exc)
            emit(
                {
                    "phase": "setup",
                    "event": "end",
                    "elapsed_s": self.clock.monotonic() - started,
                    "success": False,
                    "result_class": "setup_failed",
                    "message": message,
                }
            )
        finally:
            if lock is not None:
                lock.release()
            self.terminate_owned()

        result = self._finish(
            task,
            mode,
            run_dir,
            started,
            success,
            result_class,
            message,
            durations,
            compile_count,
            cache_hits,
            cache_misses,
            deadline,
        )
        result["link"] = link_count
        result["load"] = load_count
        result["reference"] = reference_count
        result["phases"] = records
        return result

    def _finish(
        self,
        task: str,
        mode: str,
        run_dir: Path,
        started: float,
        success: bool,
        result_class: str,
        message: str,
        durations: Mapping[str, float],
        compile_count: int,
        cache_hits: int,
        cache_misses: int,
        deadline_s: float = FEEDBACK_BUDGET_S,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task": task,
            "mode": mode,
            "status": "passed" if success else "failed",
            "success": success,
            "result_class": result_class,
            "message": message,
            "elapsed_s": self.clock.monotonic() - started,
            "deadline_s": deadline_s,
            "durations": dict(durations),
            "compile": compile_count,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "claims_throughput": False,
            "run_dir": str(run_dir),
            "owned_children": len(self.owned),
        }

    def _setup(self, contract: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
        if not gpu_setup_required(contract):
            return {
                "success": True,
                "result_class": "ok",
                "lock": None,
                "gpu_setup": False,
            }
        image = str(contract.get("image", IMAGE))
        exists = (
            self.image_exists(image)
            if self.image_exists is not None
            else docker_image_present(image, self.launcher, self.root)
        )
        if not exists:
            raise SetupError(
                f"docker image {image} is missing; build with: {SETUP_IMAGE_COMMAND}"
            )
        model = self.root / str(contract.get("model", ""))
        workloads = contract.get("workloads", {})
        needs_model = any(
            int(workloads[name].get("load_model", 0))
            for name in workloads
            if name != "smoke"
        )
        if needs_model and not model.is_file():
            raise SetupError(
                f"model {model} is missing; place the pinned GGUF at that path"
            )
        apps = (
            self.compute_apps()
            if self.compute_apps is not None
            else list_compute_apps(self.launcher, self.root)
        )
        foreign = [
            app for app in apps if str(app.get("pid", "")) not in {str(os.getpid()), ""}
        ]
        if foreign:
            raise SetupError(
                "foreign GPU compute process present; "
                f"{foreign[0]['pid']} {foreign[0].get('process_name', '')}".strip()
                + "; not killing foreign work"
            )
        lock_path = Path(str(contract.get("gpu_lock", str(run_dir / "gpu.lock"))))
        if not lock_path.is_absolute():
            lock_path = self.root / lock_path
        lock = (self.lock_factory or GpuLock)(lock_path)
        lock.acquire()
        for index, raw in enumerate(contract.get("setup_host_commands", [])):
            command = [
                str(part).format(root=str(self.root), **contract) for part in raw
            ]
            completed = self._launch(
                command,
                600.0,
                run_dir / f"setup-host-{index}",
                None,
            )
            if completed.returncode != 0:
                raise SetupError(
                    f"setup command failed ({completed.returncode}): "
                    + " ".join(command)
                    + (f"\n{completed.stderr}" if completed.stderr else "")
                )
        return {"success": True, "result_class": "ok", "lock": lock}

    def _compile(
        self,
        contract: Mapping[str, Any],
        mode: str,
        timeout_s: float,
        run_dir: Path,
        label: str,
    ) -> dict[str, Any]:
        targets = list(contract["modes"][mode].get("make_targets", []))
        if not targets:
            targets = [str(contract["target"])]
        name = f"qw38-{contract['task'].lower()}-{label}-build"
        command = self._docker_command(
            contract,
            ["make", *targets],
            tier="smoke",
            name=name,
        )
        completed = self._launch(command, timeout_s, run_dir / f"compile-{label}", name)
        compiled = completed.compiled
        return {
            "success": completed.returncode == 0,
            "result_class": "ok" if completed.returncode == 0 else "compile_failed",
            "message": completed.stderr if completed.returncode != 0 else "",
            "compile": 1 if compiled else 0,
            "link": 1 if compiled else 0,
            "cache_hits": 0 if compiled else 1,
            "cache_misses": 1 if compiled else 0,
            "selected_paths": targets,
        }

    def _workload(
        self,
        contract: Mapping[str, Any],
        mode: str,
        name: str,
        timeout_s: float,
        run_dir: Path,
    ) -> dict[str, Any]:
        workload = workload_for_mode(contract["workloads"][name], mode)
        mode_spec = contract.get("modes", {}).get(mode, {})
        tier = validate_tier(str(workload.get("tier", name)))
        target = str(workload.get("target", contract["target"]))
        values = {
            **workload,
            "model": str(contract.get("model", "")),
            "run_dir": str(run_dir),
            "root": str(self.root),
            "mode": mode,
            "phase": name,
            "repetitions": int(
                mode_spec.get("repetitions", workload.get("samples", 1))
            ),
        }
        args = [str(arg).format(**values) for arg in workload.get("args", [])]
        if str(workload.get("runner", "docker")) == "host":
            completed = self._launch(args, timeout_s, run_dir / name, None)
            payload = {
                "success": completed.returncode == 0 and not completed.terminated,
                "result_class": (
                    "timeout"
                    if completed.terminated
                    else "ok"
                    if completed.returncode == 0
                    else "numerical_failure"
                ),
                "message": completed.stderr,
                "tokens": int(workload.get("tokens", 0)),
                "load": int(workload.get("load_model", 0)),
                "reference": int(workload.get("reference", 0)),
                "selected_paths": [workload.get("selector", "ffn_only")],
            }
            admission = validate_performance_admission(
                contract,
                mode=mode,
                workload_name=name,
                workload=workload,
                stdout=completed.stdout,
                success=bool(payload["success"]),
            )
            if not admission["ok"]:
                payload["success"] = False
                payload["result_class"] = admission["result_class"]
                payload["message"] = admission["message"]
            return payload
        docker_name = f"qw38-{contract['task'].lower()}-{name}-{uuid.uuid4().hex[:8]}"
        command = self._docker_command(
            contract,
            [f"./{target}", *args],
            tier=tier,
            name=docker_name,
        )
        if name == "screen" and any(
            oracle in " ".join(command) for oracle in HISTORICAL_ORACLE_TARGETS
        ):
            return {
                "success": False,
                "result_class": "historical_oracle_in_screen",
                "message": "screen must not invoke historical P/D oracles",
            }
        completed = self._launch(command, timeout_s, run_dir / name, docker_name)
        payload = {
            "success": completed.returncode == 0 and not completed.terminated,
            "result_class": (
                "timeout"
                if completed.terminated
                else "ok"
                if completed.returncode == 0
                else "numerical_failure"
            ),
            "message": completed.stderr,
            "tokens": int(workload.get("tokens", 0)),
            "load": int(workload.get("load_model", 0)),
            "reference": int(workload.get("reference", 0)),
            "selected_paths": [workload.get("selector", "ffn_only")],
        }
        admission = validate_performance_admission(
            contract,
            mode=mode,
            workload_name=name,
            workload=workload,
            stdout=completed.stdout,
            success=bool(payload["success"]),
        )
        if not admission["ok"]:
            payload["success"] = False
            payload["result_class"] = admission["result_class"]
            payload["message"] = admission["message"]
        return payload

    def _release(
        self,
        contract: Mapping[str, Any],
        timeout_s: float,
        run_dir: Path,
    ) -> dict[str, Any]:
        started = self.clock.monotonic()
        docker_name = f"qw38-{contract['task'].lower()}-release-{uuid.uuid4().hex[:8]}"
        command = self._docker_command(
            contract,
            ["make", *HISTORICAL_ORACLE_TARGETS],
            tier="acceptance",
            name=docker_name,
        )
        completed = self._launch(command, timeout_s, run_dir / "release", docker_name)
        if completed.returncode != 0:
            return {
                "success": False,
                "result_class": "release_failed",
                "selected_paths": list(HISTORICAL_ORACLE_TARGETS),
            }
        host = list(
            contract.get("modes", {}).get("release", {}).get("release_host_command", [])
        )
        if not host:
            return {
                "success": True,
                "result_class": "ok",
                "selected_paths": list(HISTORICAL_ORACLE_TARGETS),
            }
        remaining = timeout_s - (self.clock.monotonic() - started)
        values = {
            "run_dir": str(run_dir),
            "root": str(self.root),
            "model": str(contract.get("model", "")),
        }
        host_command = [str(part).format(**values) for part in host]
        sitting = self._launch(host_command, remaining, run_dir / "release-host", None)
        return {
            "success": sitting.returncode == 0,
            "result_class": "ok" if sitting.returncode == 0 else "release_failed",
            "message": sitting.stderr if sitting.returncode != 0 else "",
            "selected_paths": list(HISTORICAL_ORACLE_TARGETS),
        }

    def _docker_command(
        self,
        contract: Mapping[str, Any],
        inner: Sequence[str],
        *,
        tier: str,
        name: str,
    ) -> list[str]:
        image = str(contract.get("image", IMAGE))
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--gpus",
            "all",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            f"QW38_CUDA_TEST_TIER={tier}",
            "-v",
            f"{self.root}:/workspace",
            "-w",
            "/workspace",
            image,
            *inner,
        ]

    def _launch(
        self,
        command: Sequence[str],
        timeout_s: float,
        log_dir: Path,
        docker_name: str,
    ) -> Completed:
        log_dir.mkdir(parents=True, exist_ok=True)
        if timeout_s <= 0:
            raise BudgetExhausted("subprocess")
        completed = self.launcher.run(
            command,
            cwd=self.root,
            env=None,
            timeout_s=timeout_s,
            docker_name=docker_name,
        )
        if completed.pgid is not None and completed.pid is not None:
            self.owned.append(
                OwnedChild(
                    pid=completed.pid,
                    pgid=completed.pgid,
                    docker_name=completed.docker_name,
                )
            )
        (log_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (log_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        (log_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        if completed.terminated:
            self.terminate_owned()
            raise BudgetExhausted(log_dir.name)
        return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Ledger task id, e.g. OPT-057")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runner = OptimizationRunner()
    try:
        result = runner.run(
            args.task,
            args.mode,
            phase=args.phase,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (SetupError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if args.dry_run:
        sys.stdout.write(str(result["plan"]))
        return 0
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
