# /// script
# requires-python = "==3.12.*"
# ///
"""Capture EVAL-01 continuations with the local Q4_K_M llama.cpp teacher."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
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
EOG_IDS = {248044, 248046, 248063, 248064, 248065}
CAP = 24


def sha256(raw: bytes) -> str:
    """Hash a small fixture/metadata byte sequence."""
    return hashlib.sha256(raw).hexdigest()


def hash_file(path: Path) -> str:
    """Hash a fixture, source, or executable file, never a model payload."""
    return sha256(path.read_bytes())


def run(command: list[str], *, check: bool = True) -> str:
    """Run a host/container identity command and return stdout."""
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f"command failed: {command!r}: {result.stderr}")
    return result.stdout.strip()


def write_manifest(path: Path, manifest: dict[str, Any]) -> str:
    """Atomically publish the small fixture manifest and its sidecar."""
    raw = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    (path.parent / "manifest.sha256").write_text(f"{sha256(raw)}  manifest.json\n")
    return sha256(raw)


def read_u32le(path: Path) -> list[int]:
    """Read a frozen little-endian token sequence."""
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"invalid token sequence: {path}")
    return [int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw), 4)]


def post_json(base_url: str, route: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST one llama.cpp server request and decode its JSON response."""
    request = urllib.request.Request(
        f"{base_url}{route}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"llama.cpp {route} request failed: {exc}") from exc
    if not isinstance(body, dict):
        raise TypeError(f"llama.cpp {route} returned a non-object response")
    return body


def token_ids(root: Path, case: dict[str, Any]) -> list[int]:
    """Read and bind a fixture's frozen prompt IDs."""
    ids = read_u32le(root / case["prompt_token_file"])
    if (
        len(ids) != case["prompt_tokens"]
        or sha256((root / case["prompt_token_file"]).read_bytes())
        != case["prompt_token_sha256"]
    ):
        raise ValueError(f"frozen prompt token identity mismatch: {case['id']}")
    return ids


def wait_ready(base_url: str, deadline: float) -> None:
    """Wait for the local server to finish loading the model."""
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                body = json.load(response)
            if body.get("status") == "ok":
                return
        except Exception as exc:  # noqa: BLE001 - startup failures are retried to a fixed deadline
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"llama.cpp server did not become ready: {last_error}")


def select_cases(root: Path) -> list[dict[str, Any]]:
    """Load the complete, ordered P100+C92 teacher inventory."""
    rows = [
        json.loads(line)
        for line in (root / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cases = [row for row in rows if row["family"] in {"P100", "C92"}]
    if len(cases) != 192 or len({case["id"] for case in cases}) != 192:
        raise ValueError(
            "source teacher inventory must be exactly 192 unique P100/C92 cases"
        )
    return cases


def main() -> int:
    """Generate, hash, and freeze all teacher continuations before comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--port", type=int, default=18108)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.fixtures.resolve()
    attempt = args.attempt_dir.resolve()
    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"local Q4_K_M GGUF is missing: {model}")
    attempt.mkdir(parents=True, exist_ok=True)
    lock_path = root / "source_capture_job" / "llama-capture.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(
            "another TASK-018 llama.cpp teacher capture holds the lock"
        ) from exc
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["suite"] != "qw38-language-v1":
        raise SystemExit("unexpected fixture suite")
    if manifest["reference_capture"]["status"] == "COMPLETE":
        raise SystemExit("teacher references already frozen; refusing to overwrite")
    teacher = manifest["source_teacher"]
    if (
        model != Path(teacher["path"]).resolve()
        or model.stat().st_size != teacher["file_size_bytes"]
        or teacher["quantization"] != "Q4_K_M"
        or teacher["tokenizer_template_sha256"]
        != manifest["tokenizer"]["files"]["chat_template.jinja"]["sha256"]
    ):
        raise SystemExit(
            "local GGUF size, quantization, or chat-template identity mismatch"
        )

    cases = select_cases(root)
    case_tokens = {case["id"]: token_ids(root, case) for case in cases}
    maximum_context = max(len(ids) for ids in case_tokens.values()) + CAP
    if maximum_context > 131072:
        raise SystemExit(
            f"teacher context request exceeds declared model context: {maximum_context}"
        )

    image_ref = run(
        ["docker", "image", "inspect", IMAGE, "--format", "{{index .RepoDigests 0}}"]
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
    binary_identity = run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            IMAGE,
            "-lc",
            "sha256sum /app/llama-server",
        ]
    )
    teacher.update(
        {
            "image": IMAGE,
            "image_digest": image_ref,
            "image_id": image_id,
            "llama_server_version": version,
            "llama_server_binary_sha256": binary_identity.split()[0],
            "source_capture_script_sha256": hash_file(Path(__file__).resolve()),
            "requested_context": maximum_context,
        }
    )
    teacher_identity_sha = sha256(
        json.dumps(
            teacher, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    started_utc = datetime.now(UTC).isoformat()
    container_name = f"qw38-task018-ref-{os.getpid()}"
    other_containers = run(
        [
            "docker",
            "ps",
            "--filter",
            "name=qw38-task018-ref-",
            "--format",
            "{{.Names}}",
        ],
        check=False,
    )
    if other_containers:
        raise SystemExit(
            f"stale or concurrent teacher container found: {other_containers}"
        )
    host_model = model
    server_command = [
        "docker",
        "run",
        "--gpus",
        "all",
        "--name",
        container_name,
        "-p",
        f"127.0.0.1:{args.port}:8080",
        "-v",
        f"{host_model.parent}:/models:ro",
        "--entrypoint",
        "/app/llama-server",
        IMAGE,
        "-m",
        f"/models/{host_model.name}",
        "-ngl",
        "all",
        "-c",
        str(maximum_context),
        "-np",
        "1",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--verbose",
    ]
    source_dir = root / "source_refs"
    source_dir.mkdir(parents=True, exist_ok=True)
    progress_path = source_dir / "llama_capture.progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    manifest["reference_capture"] = {
        "status": "IN_PROGRESS",
        "required_before_comparison": True,
        "teacher_identity_sha256": teacher_identity_sha,
        "attempt_dir": str(attempt.relative_to(root)),
        "expected_cases": 192,
        "validated_cases": 0,
    }
    write_manifest(manifest_path, manifest)
    started = time.monotonic()
    container_id = ""
    server = subprocess.Popen(
        server_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + args.startup_timeout
        while server.poll() is None and not container_id:
            try:
                container_id = run(
                    ["docker", "inspect", "--format", "{{.Id}}", container_name]
                )
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(2)
        wait_ready(f"http://127.0.0.1:{args.port}", deadline)
        startup_result = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        startup_log = startup_result.stdout + startup_result.stderr
        if (
            re.search(r"arch\s+= qwen35", startup_log) is None
            or re.search(r"file type\s+= Q4_K - Medium", startup_log) is None
            or re.search(r"general\.name\s+= Qwen3\.8-27B", startup_log) is None
            or re.search(r"n_vocab\s+= 248320", startup_log) is None
            or "BOS token             = 248044" not in startup_log
            or "EOS token             = 248046" not in startup_log
            or "offloaded 65/65 layers to GPU" not in startup_log
        ):
            raise ValueError(
                "llama.cpp startup did not confirm Q4_K_M CUDA/65-layer identity"
            )
        (attempt / "startup.log").write_text(startup_log + "\n", encoding="utf-8")

        identity = {
            "teacher": teacher,
            "context": maximum_context,
            "prompt_count": len(cases),
            "generation": teacher["generation"],
        }
        (attempt / "teacher_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        refs: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            case_id = case["id"]
            frozen_ids = case_tokens[case_id]
            rendered = case["rendered_prompt"]
            tokenized = post_json(
                f"http://127.0.0.1:{args.port}",
                "/tokenize",
                {"content": rendered, "add_special": False, "parse_special": True},
            )
            if tokenized.get("tokens") != frozen_ids:
                raise ValueError(
                    f"teacher tokenizer differs from frozen prompt: {case_id}"
                )
            response = post_json(
                f"http://127.0.0.1:{args.port}",
                "/completion",
                {
                    "prompt": rendered,
                    "n_predict": CAP,
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
                    "n_probs": 1,
                    "samplers": ["temperature"],
                },
            )
            if response.get("prompt") != rendered or response.get("truncated"):
                raise ValueError(
                    f"teacher changed or truncated frozen prompt: {case_id}"
                )
            if response.get("tokens_evaluated") != len(frozen_ids):
                raise ValueError(
                    f"teacher evaluated a different prompt length: {case_id}"
                )
            rows = response.get("completion_probabilities")
            predicted = response.get("tokens_predicted")
            if not isinstance(rows, list) or predicted != len(rows):
                raise ValueError(f"teacher omitted generated token IDs: {case_id}")
            settings = response.get("generation_settings", {})
            if (
                settings.get("samplers") != ["temperature"]
                or settings.get("temperature") != 0.0
                or settings.get("top_k") != 0
                or settings.get("top_p") != 1.0
                or settings.get("min_p") != 0.0
                or settings.get("repeat_penalty") != 1.0
                or settings.get("ignore_eos") is not False
            ):
                raise ValueError(f"teacher applied non-greedy processing: {case_id}")
            targets = [int(item["id"]) for item in rows]
            if not 1 <= len(targets) <= CAP:
                raise ValueError(f"teacher target count outside 1..{CAP}: {case_id}")
            stop_reason = response.get("stop_type")
            if stop_reason not in {"eos", "limit"}:
                raise ValueError(
                    f"unexpected teacher stop reason {stop_reason!r}: {case_id}"
                )
            if stop_reason == "eos" and targets[-1] not in EOG_IDS:
                raise ValueError(f"teacher EOG stop omits EOG target: {case_id}")
            target_path = source_dir / f"{case_id}.target.u32le"
            mask_path = source_dir / f"{case_id}.loss-mask.u8"
            target_raw = b"".join(token.to_bytes(4, "little") for token in targets)
            mask_raw = bytes([1]) * len(targets)
            target_path.write_bytes(target_raw)
            mask_path.write_bytes(mask_raw)
            row = {
                "id": case_id,
                "family": case["family"],
                "prompt_token_sha256": case["prompt_token_sha256"],
                "teacher_identity_sha256": teacher_identity_sha,
                "target_file": str(target_path.relative_to(root)),
                "target_count": len(targets),
                "target_bytes": len(target_raw),
                "target_sha256": sha256(target_raw),
                "loss_mask_file": str(mask_path.relative_to(root)),
                "loss_mask_bytes": len(mask_raw),
                "loss_mask_sha256": sha256(mask_raw),
                "stop_reason": stop_reason,
                "stop_token_id": targets[-1] if stop_reason == "eos" else None,
                "tokens_evaluated": response["tokens_evaluated"],
                "tokens_predicted": predicted,
                "llama_stop_type": stop_reason,
                "effective_generation_settings": settings,
            }
            refs.append(row)
            with progress_path.open("a", encoding="utf-8", newline="\n") as progress:
                progress.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
            manifest["reference_capture"]["validated_cases"] = index
            write_manifest(manifest_path, manifest)
            print(
                f"captured {index}/192 {case_id} count={len(targets)} stop={stop_reason}",
                flush=True,
            )

        if len(refs) != 192 or {row["id"] for row in refs} != {
            case["id"] for case in cases
        }:
            raise ValueError("teacher reference inventory is incomplete")
        refs_path = source_dir / "source_refs.jsonl"
        refs_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in refs
            ),
            encoding="utf-8",
        )
        manifest["source_teacher"] = teacher
        manifest["reference_capture"] = {
            "status": "COMPLETE",
            "required_before_comparison": True,
            "teacher_identity_sha256": teacher_identity_sha,
            "source_refs_jsonl": {
                "path": str(refs_path.relative_to(root)),
                "bytes": refs_path.stat().st_size,
                "sha256": hash_file(refs_path),
            },
            "expected_cases": 192,
            "validated_cases": 192,
            "generation_settings": teacher["generation"],
            "container_id": container_id,
            "started_utc": started_utc,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        manifest["coverage"] = "teacher_references_frozen; BF16/V0 comparisons not run"
        (attempt / "teacher_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_hash = write_manifest(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "cases": len(refs),
                    "teacher_identity_sha256": teacher_identity_sha,
                    "manifest_sha256": manifest_hash,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        manifest["reference_capture"].update(
            {
                "status": "INCOMPLETE",
                "failure": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        write_manifest(manifest_path, manifest)
        raise
    finally:
        logs = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        (attempt / "llama-server.log").write_text(
            logs.stdout + logs.stderr, encoding="utf-8"
        )
        subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            ["docker", "rm", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if server.poll() is None:
            server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        lock_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
