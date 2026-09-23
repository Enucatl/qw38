# /// script
# requires-python = "==3.12.*"
# ///
"""Capture top-20 llama.cpp probabilities for already frozen teacher IDs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

IMAGE = "ghcr.io/ggml-org/llama.cpp:full-cuda13"
MODEL = Path("models/Qwen3.8-27B-Q4_K_M.gguf")
CAP = 24
N_PROBS = 20
EOG_IDS = {248044, 248046, 248063, 248064, 248065}


def sha256(raw: bytes) -> str:
    """Return SHA-256 of a fixture or metadata file."""
    return hashlib.sha256(raw).hexdigest()


def hash_file(path: Path) -> str:
    """Hash a small fixture, reference or metadata file."""
    return sha256(path.read_bytes())


def run(command: list[str], *, combine: bool = False) -> str:
    """Run a container identity command and retain its reported output."""
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed {command!r}: {result.stderr}")
    output = result.stdout + result.stderr if combine else result.stdout
    return output.strip()


def read_u32le(path: Path) -> list[int]:
    """Decode one nonempty little-endian token sequence."""
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"invalid token sequence: {path}")
    return [int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw), 4)]


def request_json(base_url: str, route: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send a local llama.cpp REST request and return its JSON object."""
    request = urllib.request.Request(
        f"{base_url}{route}",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            value = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"llama.cpp {route} request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"llama.cpp {route} returned a non-object")
    return value


def wait_ready(base_url: str, server: subprocess.Popen[bytes], deadline: float) -> None:
    """Retry health checks while the model loads; stop on process exit/deadline."""
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"llama.cpp container exited with {server.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    return
        except Exception as exc:  # noqa: BLE001 - startup is transient until fixed deadline
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"llama.cpp server not ready by deadline: {last_error}")


def select_cases(root: Path) -> list[dict[str, Any]]:
    """Return the frozen ordered 192 P100/C92 teacher cases."""
    rows = [
        json.loads(line)
        for line in (root / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cases = [row for row in rows if row["family"] in {"P100", "C92"}]
    if len(cases) != 192 or len({row["id"] for row in cases}) != 192:
        raise ValueError("expected exactly 192 unique P100/C92 prompts")
    return cases


def main() -> int:
    """Capture exact-reference top-k probabilities without changing teacher IDs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--port", type=int, default=18110)
    parser.add_argument("--startup-timeout", type=int, default=600)
    args = parser.parse_args()

    root, attempt, model = (
        args.fixtures.resolve(),
        args.attempt_dir.resolve(),
        args.model.resolve(),
    )
    if not model.is_file():
        raise SystemExit(f"Q4_K_M GGUF missing: {model}")
    attempt.mkdir(parents=True, exist_ok=True)
    lock_path = root / "source_capture_job" / "llama-probabilities.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another TASK-018 teacher probability capture holds the lock") from exc

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest["reference_capture"]
    teacher = manifest["source_teacher"]
    if capture["status"] != "COMPLETE" or capture["validated_cases"] != 192:
        raise SystemExit("probability replay requires 192 complete frozen teacher refs")
    if model != Path(teacher["path"]).resolve() or model.stat().st_size != teacher["file_size_bytes"]:
        raise SystemExit("probability replay GGUF differs from frozen teacher identity")

    refs_path = root / capture["source_refs_jsonl"]["path"]
    refs = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in refs_path.read_text(encoding="utf-8").splitlines()
        )
    }
    cases = select_cases(root)
    if len(refs) != 192 or set(refs) != {case["id"] for case in cases}:
        raise ValueError("frozen reference inventory is not exactly P100+C92")
    case_tokens = {case["id"]: read_u32le(root / case["prompt_token_file"]) for case in cases}
    target_tokens = {
        case_id: read_u32le(root / refs[case_id]["target_file"])
        for case_id in refs
    }
    maximum_context = max(len(ids) for ids in case_tokens.values()) + CAP
    image_digest = run(["docker", "image", "inspect", IMAGE, "--format", "{{index .RepoDigests 0}}"])
    if image_digest != teacher["image_digest"]:
        raise SystemExit("llama.cpp image digest differs from frozen teacher")
    image_id = run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"])
    version = run(
        ["docker", "run", "--rm", "--entrypoint", "/app/llama-server", IMAGE, "--version"],
        combine=True,
    )
    identity = {
        "image": IMAGE,
        "image_digest": image_digest,
        "image_id": image_id,
        "server_version": version,
        "binary_sha256": teacher["llama_server_binary_sha256"],
        "teacher_identity_sha256": capture["teacher_identity_sha256"],
        "reference_manifest_sha256": hash_file(manifest_path),
        "reference_jsonl_sha256": capture["source_refs_jsonl"]["sha256"],
        "probability_only_replay": True,
        "n_probs": N_PROBS,
        "full_vocabulary_logits_available": False,
        "requested_context": maximum_context,
        "prompt_cache": "default enabled, matching the original reference capture",
    }
    container_name = f"qw38-task018-probs-{os.getpid()}"
    active = run(
        ["docker", "ps", "--filter", "name=qw38-task018-probs-", "--format", "{{.Names}}"],
        combine=True,
    )
    if active:
        raise SystemExit(f"stale or concurrent probability container: {active}")
    server_log = attempt / "llama-server.log"
    log_handle = server_log.open("wb")
    command = [
        "docker", "run", "--rm", "--gpus", "all", "--name", container_name,
        "-p", f"127.0.0.1:{args.port}:8080", "-v", f"{model.parent}:/models:ro",
        "--entrypoint", "/app/llama-server", IMAGE,
        "-m", f"/models/{model.name}", "-ngl", "all", "-c", str(maximum_context),
        "-np", "1", "--host", "0.0.0.0", "--port", "8080",
        "--verbose",
    ]
    server = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
    probability_rows: list[dict[str, Any]] = []
    try:
        base_url = f"http://127.0.0.1:{args.port}"
        wait_ready(base_url, server, time.monotonic() + args.startup_timeout)
        deadline = time.monotonic() + args.startup_timeout
        startup = ""
        while time.monotonic() < deadline:
            status = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
                capture_output=True, text=True, check=False,
            )
            if status.returncode == 0:
                startup_result = subprocess.run(
                    ["docker", "logs", container_name], capture_output=True,
                    text=True, check=False,
                )
                startup = startup_result.stdout + startup_result.stderr
                if "offloaded 65/65 layers to GPU" in startup:
                    break
            time.sleep(2)
        if "offloaded 65/65 layers to GPU" not in startup:
            raise RuntimeError("server startup did not confirm 65/65 GPU layer offload")
        if re.search(r"general\.architecture\s+str\s+= qwen35", startup) is None:
            raise RuntimeError("server startup did not confirm qwen35 architecture")
        (attempt / "startup.log").write_text(startup + "\n", encoding="utf-8")
        progress_path = attempt / "progress.jsonl"
        for index, case in enumerate(cases, start=1):
            case_id = case["id"]
            frozen_ids = case_tokens[case_id]
            ref = refs[case_id]
            targets = target_tokens[case_id]
            target_path = root / ref["target_file"]
            if hash_file(target_path) != ref["target_sha256"]:
                raise ValueError(f"frozen target hash mismatch: {case_id}")
            tokenized = request_json(
                base_url, "/tokenize",
                {"content": case["rendered_prompt"], "add_special": False, "parse_special": True},
            )
            if tokenized.get("tokens") != frozen_ids:
                raise ValueError(f"llama tokenizer changed frozen prompt IDs: {case_id}")
            response = request_json(
                base_url, "/completion",
                {
                    "prompt": case["rendered_prompt"], "n_predict": CAP,
                    "temperature": 0.0, "top_k": 0, "top_p": 1.0,
                    "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0,
                    "repeat_last_n": 0, "presence_penalty": 0.0,
                    "frequency_penalty": 0.0, "dry_multiplier": 0.0,
                    "xtc_probability": 0.0, "seed": 42, "ignore_eos": False,
                    "stream": False, "n_probs": N_PROBS,
                    "samplers": ["temperature"],
                },
            )
            if response.get("prompt") != case["rendered_prompt"] or response.get("truncated"):
                raise ValueError(f"llama changed/truncated probability prompt: {case_id}")
            rows = response.get("completion_probabilities")
            generated_ids = [int(item["id"]) for item in rows] if isinstance(rows, list) else []
            if generated_ids != targets:
                first_diff = next(
                    (
                        (position, generated_ids[position] if position < len(generated_ids) else None,
                         targets[position] if position < len(targets) else None)
                        for position in range(max(len(generated_ids), len(targets)))
                        if position >= len(generated_ids)
                        or position >= len(targets)
                        or generated_ids[position] != targets[position]
                    ),
                    None,
                )
                raise ValueError(
                    f"probability replay changed frozen IDs for {case_id}: "
                    f"first_diff(position,new_id,frozen_id)={first_diff}; "
                    f"new_count={len(generated_ids)} frozen_count={len(targets)}"
                )
            if response.get("stop_type") != ref["stop_reason"]:
                raise ValueError(f"probability replay stop mismatch: {case_id}")
            settings = response.get("generation_settings", {})
            if settings.get("n_probs") != N_PROBS or settings.get("samplers") != ["temperature"] or settings.get("temperature") != 0.0:
                raise ValueError(f"effective top-k/greedy settings differ: {case_id}")
            positions: list[dict[str, Any]] = []
            hit_count = 0
            for token_id, item in zip(targets, rows, strict=True):
                target_logprob = float(item["logprob"])
                top = item.get("top_logprobs")
                if not math.isfinite(target_logprob) or not isinstance(top, list) or len(top) != N_PROBS:
                    raise ValueError(f"invalid top-20 probability row: {case_id}")
                top_ids = [int(candidate["id"]) for candidate in top]
                top_logprobs = [float(candidate["logprob"]) for candidate in top]
                if any(not math.isfinite(value) for value in top_logprobs) or len(set(top_ids)) != N_PROBS:
                    raise ValueError(f"invalid top-20 IDs/probabilities: {case_id}")
                if token_id not in top_ids:
                    raise ValueError(f"greedy target absent from top-20: {case_id}")
                hit_count += 1
                positions.append({
                    "target_id": token_id,
                    "target_logprob": target_logprob,
                    "top20_ids": top_ids,
                    "top20_logprobs": top_logprobs,
                })
            probability_rows.append({
                "id": case_id,
                "target_file_sha256": ref["target_sha256"],
                "loss_mask_sha256": ref["loss_mask_sha256"],
                "teacher_identity_sha256": capture["teacher_identity_sha256"],
                "target_count": len(targets),
                "top20_target_hits": hit_count,
                "stop_reason": ref["stop_reason"],
                "effective_generation_settings": settings,
                "positions": positions,
            })
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write(json.dumps({"case": case_id, "cases_done": index}) + "\n")
            print(f"probabilities {index}/192 {case_id} tokens={len(targets)}", flush=True)
        if len(probability_rows) != 192:
            raise ValueError("probability sidecar case count is incomplete")

        sidecar_dir = root / "teacher_probabilities"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = sidecar_dir / "q4km-top20.jsonl"
        sidecar_raw = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in probability_rows
        ).encode()
        temporary = sidecar_path.with_name(f"{sidecar_path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(sidecar_raw)
        temporary.replace(sidecar_path)
        total_targets = sum(row["target_count"] for row in probability_rows)
        capture_record = {
            "status": "COMPLETE",
            "path": str(sidecar_path.relative_to(root)),
            "bytes": len(sidecar_raw),
            "sha256": sha256(sidecar_raw),
            "cases": len(probability_rows),
            "target_tokens": total_targets,
            "top20_target_hits": sum(row["top20_target_hits"] for row in probability_rows),
            "n_probs": N_PROBS,
            "full_vocabulary_logits_available": False,
            "teacher_identity_sha256": capture["teacher_identity_sha256"],
            "reference_manifest_sha256": identity["reference_manifest_sha256"],
            "reference_jsonl_sha256": capture["source_refs_jsonl"]["sha256"],
            "attempt_dir": str(attempt.relative_to(root)),
        }
        manifest["teacher_probability_capture"] = capture_record
        manifest["coverage"] = "teacher references and top20 probabilities frozen; llama.cpp/V0 comparison not run"
        manifest_raw = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        temp_manifest = manifest_path.with_name(f"manifest.json.{os.getpid()}.tmp")
        temp_manifest.write_bytes(manifest_raw)
        temp_manifest.replace(manifest_path)
        (root / "manifest.sha256").write_text(f"{sha256(manifest_raw)}  manifest.json\n")
        identity["ended_utc"] = datetime.now(UTC).isoformat()
        identity["probability_sidecar_sha256"] = sha256(sidecar_raw)
        (attempt / "probability_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "COMPLETE", "cases": 192,
                          "target_tokens": total_targets,
                          "top20_target_hits": capture_record["top20_target_hits"],
                          "sidecar_sha256": capture_record["sha256"],
                          "fixture_manifest_sha256": sha256(manifest_raw)}, sort_keys=True))
        return 0
    finally:
        if server.poll() is None:
            server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log_handle.close()
        subprocess.run(["docker", "stop", container_name], capture_output=True, check=False)
        subprocess.run(["docker", "rm", container_name], capture_output=True, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
