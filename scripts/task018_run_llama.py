# /// script
# requires-python = "==3.12.*"
# ///
"""Run the routine or manual full core on the pinned Q4_K_M teacher."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .task018_core_selection import CORE_SPEC, select_cases
except ImportError:
    from task018_core_selection import CORE_SPEC, select_cases

IMAGE = "ghcr.io/ggml-org/llama.cpp:full-cuda13"
MODEL = Path("models/Qwen3.8-27B-Q4_K_M.gguf")
REPLAY_IDS = {"case_000", "recNu3MXkvWUzHZr9", "L01", "R-512-s0-d0.1"}
EOG_IDS = {248044, 248046, 248063, 248064, 248065}


def sha256(raw: bytes) -> str:
    """Hash a small fixture, binary or generated output."""
    return hashlib.sha256(raw).hexdigest()


def hash_file(path: Path) -> str:
    """Hash a stored small file."""
    return sha256(path.read_bytes())


def run(command: list[str]) -> str:
    """Run an identity command and return both output streams."""
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed {command!r}: {result.stderr}")
    return (result.stdout + result.stderr).strip()


def post_json(base_url: str, route: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Submit one llama.cpp request and parse its JSON response."""
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
    """Retry health checks while model startup is in progress."""
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"llama.cpp container exited with {server.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    return
        except Exception as exc:  # noqa: BLE001 - transient only until startup deadline
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"llama.cpp did not become ready: {last_error}")


def read_u32le(path: Path) -> list[int]:
    """Read frozen little-endian prompt IDs."""
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"invalid token IDs: {path}")
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def write_u32le(path: Path, values: list[int]) -> None:
    """Write generated token IDs in the suite's little-endian format."""
    path.write_bytes(struct.pack(f"<{len(values)}I", *values))


def request_case(
    base_url: str, case: dict[str, Any], prompt_ids: list[int]
) -> dict[str, Any]:
    """Generate one frozen rendered prompt with deterministic greedy settings."""
    tokenized = post_json(
        base_url,
        "/tokenize",
        {
            "content": case["rendered_prompt"],
            "add_special": False,
            "parse_special": True,
        },
    )
    if tokenized.get("tokens") != prompt_ids:
        raise ValueError(f"llama tokenizer differs from frozen prompt: {case['id']}")
    cap = int(case["details"]["cap"])
    response = post_json(
        base_url,
        "/completion",
        {
            "prompt": case["rendered_prompt"],
            "n_predict": cap,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "xtc_probability": 0.0,
            "seed": 42,
            "ignore_eos": False,
            "stream": False,
            "return_tokens": True,
            "n_probs": 0,
            "samplers": ["temperature"],
        },
    )
    if response.get("prompt") != case["rendered_prompt"] or response.get("truncated"):
        raise ValueError(f"llama changed/truncated frozen prompt: {case['id']}")
    if response.get("tokens_evaluated") != len(prompt_ids):
        raise ValueError(f"llama prompt token count mismatch: {case['id']}")
    raw_tokens = response.get("tokens")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise ValueError(f"llama did not return generated token IDs: {case['id']}")
    token_ids = [
        int(item["id"] if isinstance(item, dict) else item) for item in raw_tokens
    ]
    stop_reason = response.get("stop_type")
    if stop_reason not in {"eos", "limit"} or not 1 <= len(token_ids) <= cap:
        raise ValueError(f"llama stop/count invalid for {case['id']}: {stop_reason}")
    if stop_reason == "eos" and token_ids[-1] not in EOG_IDS:
        raise ValueError(f"llama EOS stop omitted terminal EOG token: {case['id']}")
    settings = response.get("generation_settings", {})
    if (
        settings.get("samplers") != ["temperature"]
        or settings.get("temperature") != 0.0
        or settings.get("n_probs") != 0
        or settings.get("ignore_eos") is not False
        or settings.get("repeat_penalty") != 1.0
        or settings.get("top_k") != 0
        or settings.get("top_p") != 1.0
    ):
        raise ValueError(f"llama effective greedy settings mismatch: {case['id']}")
    return {
        "id": case["id"],
        "family": case["family"],
        "prompt_tokens": len(prompt_ids),
        "target_tokens": 0,
        "generation_tokens": len(token_ids),
        "generation_stop": stop_reason,
        "cap": cap,
        "output_token_ids": token_ids,
        "text": response.get("content", ""),
        "generation_settings": settings,
        "effective_capacity": len(prompt_ids) + cap,
    }


def main() -> int:
    """Run the selected core arm and fresh-request replay, then hash evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--port", type=int, default=18111)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--manual-full", action="store_true")
    args = parser.parse_args()
    if args.manual_full and not sys.stdin.isatty():
        parser.error("full 216-case evaluation requires an interactive manual launch")
    root, output, model = (
        args.fixtures.resolve(),
        args.output.resolve(),
        args.model.resolve(),
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    fixture_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if fixture_manifest["reference_capture"]["status"] != "COMPLETE":
        raise SystemExit("complete Q4_K_M references are required")
    if (
        fixture_manifest.get("teacher_probability_capture", {}).get("status")
        != "COMPLETE"
    ):
        raise SystemExit("complete teacher top-20 probability sidecar is required")
    if model.stat().st_size != fixture_manifest["source_teacher"]["file_size_bytes"]:
        raise SystemExit("llama model size differs from frozen teacher identity")
    lock_path = root / "source_capture_job" / "llama-evaluation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another TASK-018 llama evaluation holds the lock") from exc

    cases, sample = select_cases(root, full=args.manual_full)
    full_cases, _ = select_cases(root, full=True)
    prompts = {
        case["id"]: read_u32le(root / case["prompt_token_file"]) for case in cases
    }
    max_context = max(
        len(prompts[case["id"]]) + case["details"]["cap"] for case in cases
    )
    image_digest = run(
        ["docker", "image", "inspect", IMAGE, "--format", "{{index .RepoDigests 0}}"]
    )
    if image_digest != fixture_manifest["source_teacher"]["image_digest"]:
        raise SystemExit(
            "llama.cpp image digest differs from frozen reference identity"
        )
    image_id = run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"])
    version = run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/app/llama-server",
            IMAGE,
            "--version",
        ]
    )
    binary_sha = fixture_manifest["source_teacher"]["llama_server_binary_sha256"]
    identity = {
        "suite": fixture_manifest["suite"],
        "fixture_manifest_sha256": hash_file(manifest_path),
        "policy_sha256": fixture_manifest["policy_sha256"],
        "prompts_sha256": fixture_manifest["files"]["prompts.jsonl"]["sha256"],
        "teacher_references_sha256": fixture_manifest["reference_capture"][
            "source_refs_jsonl"
        ]["sha256"],
        "teacher_probability_sidecar_sha256": fixture_manifest[
            "teacher_probability_capture"
        ]["sha256"],
        "teacher_identity_sha256": fixture_manifest["reference_capture"][
            "teacher_identity_sha256"
        ],
        "image": IMAGE,
        "image_digest": image_digest,
        "image_id": image_id,
        "server_version": version,
        "server_binary_sha256": binary_sha,
        "model_size_bytes": model.stat().st_size,
        "requested_context": max_context,
        "gpu_layers": "all",
        "prompt_cache": "default enabled, matching the teacher reference capture",
        "cases_expected": len(cases),
        "evaluation_scope": "full-216" if args.manual_full else "core-54",
        "sample_spec_sha256": None if sample is None else hash_file(CORE_SPEC),
        "generation": {
            "temperature": 0.0,
            "samplers": ["temperature"],
            "seed": 42,
            "ignore_eos": False,
        },
    }
    active = run(
        [
            "docker",
            "ps",
            "--filter",
            "name=qw38-task018-eval-",
            "--format",
            "{{.Names}}",
        ]
    )
    if active:
        raise SystemExit(f"stale or concurrent llama evaluation container: {active}")
    container_name = f"qw38-task018-eval-{os.getpid()}"
    server_log = output / "llama-server.log"
    log_handle = server_log.open("wb")
    server_command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--name",
        container_name,
        "-p",
        f"127.0.0.1:{args.port}:8080",
        "-v",
        f"{model.parent}:/models:ro",
        "--entrypoint",
        "/app/llama-server",
        IMAGE,
        "-m",
        f"/models/{model.name}",
        "-ngl",
        "all",
        "-c",
        str(max_context),
        "-np",
        "1",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--verbose",
    ]
    server = subprocess.Popen(
        server_command, stdout=log_handle, stderr=subprocess.STDOUT
    )
    try:
        base_url = f"http://127.0.0.1:{args.port}"
        wait_ready(base_url, server, time.monotonic() + args.startup_timeout)
        startup = run(["docker", "logs", container_name])
        if (
            "offloaded 65/65 layers to GPU" not in startup
            or re.search(r"general\.architecture\s+str\s+= qwen35", startup) is None
        ):
            raise RuntimeError(
                "llama startup lacks expected Q4_K_M/full-offload identity"
            )
        (output / "startup.log").write_text(startup + "\n", encoding="utf-8")
        records: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            record = request_case(base_url, case, prompts[case["id"]])
            write_u32le(
                output / f"{case['id']}.generated.u32le", record["output_token_ids"]
            )
            raw_text = record.pop("text")
            (output / f"{case['id']}.txt").write_text(raw_text, encoding="utf-8")
            record["text_file"] = f"{case['id']}.txt"
            record["text_sha256"] = hash_file(output / record["text_file"])
            record["token_file"] = f"{case['id']}.generated.u32le"
            record["token_sha256"] = hash_file(output / record["token_file"])
            records.append(record)
            print(
                f"llama {index}/{len(cases)} {case['id']} tokens={record['generation_tokens']} stop={record['generation_stop']}",
                flush=True,
            )
        by_id = {case["id"]: case for case in full_cases}
        replay_rows = []
        for case_id in sorted(REPLAY_IDS):
            prompt_ids = prompts.get(case_id) or read_u32le(
                root / by_id[case_id]["prompt_token_file"]
            )
            expected = next((row for row in records if row["id"] == case_id), None)
            first_ids = (
                read_u32le(output / expected["token_file"])
                if expected
                else request_case(base_url, by_id[case_id], prompt_ids)[
                    "output_token_ids"
                ]
            )
            replay = request_case(base_url, by_id[case_id], prompt_ids)
            same = replay["output_token_ids"] == first_ids
            if not same:
                raise RuntimeError(
                    f"fresh llama request replay changed generated IDs: {case_id}"
                )
            replay_rows.append(
                {
                    "id": case_id,
                    "identical_ids": True,
                    "tokens": len(replay["output_token_ids"]),
                    "in_core": expected is not None,
                }
            )
            print(f"llama replay {case_id} identical={same}", flush=True)
        (output / "cases.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in records
            ),
            encoding="utf-8",
        )
        identity.update(
            {
                "status": "COMPLETE",
                "cases_validated": len(records),
                "family_counts": dict(
                    sorted(Counter(row["family"] for row in records).items())
                ),
                "fresh_request_replay": replay_rows,
                "startup_sha256": hash_file(output / "startup.log"),
                "cases_jsonl_sha256": hash_file(output / "cases.jsonl"),
                "cases_jsonl_bytes": (output / "cases.jsonl").stat().st_size,
                "ended_utc": datetime.now(UTC).isoformat(),
            }
        )
        (output / "run_manifest.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "cases": len(records),
                    "run_manifest_sha256": hash_file(output / "run_manifest.json"),
                },
                sort_keys=True,
            )
        )
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
        subprocess.run(
            ["docker", "stop", container_name], capture_output=True, check=False
        )
        subprocess.run(
            ["docker", "rm", container_name], capture_output=True, check=False
        )


if __name__ == "__main__":
    raise SystemExit(main())
