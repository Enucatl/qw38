# /// script
# requires-python = "==3.12.*"
# ///
"""Diagnostic fresh Q4_K_M generation for the predeclared P10 speed baseline."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from task018_run_llama import (
    IMAGE,
    core_cases,
    hash_file,
    read_u32le,
    request_case,
    run,
    wait_ready,
    write_u32le,
)

P10_IDS = [
    "case_080",
    "case_082",
    "case_083",
    "case_084",
    "case_087",
    "case_094",
    "case_095",
    "case_096",
    "case_097",
    "case_098",
]


def main() -> int:
    root = Path(".cache/evaluation/qw38-language-v1").resolve()
    output = Path(
        ".cache/evaluation/qw38-language-v1/runs/llama-p10-20260923T1755"
    ).resolve()
    model = Path("models/Qwen3.8-27B-Q4_K_M.gguf").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = {row["id"]: row for row in core_cases(root)}
    selected = [cases[case_id] for case_id in P10_IDS]
    if any(row["family"] != "P100" or row["details"]["cap"] != 256 for row in selected):
        raise ValueError("predeclared P10 fixture selection/caps changed")
    prompts = {
        row["id"]: read_u32le(root / row["prompt_token_file"]) for row in selected
    }
    max_context = max(
        len(prompts[row["id"]]) + row["details"]["cap"] for row in selected
    )
    image_digest = run(
        ["docker", "image", "inspect", IMAGE, "--format", "{{index .RepoDigests 0}}"]
    )
    if image_digest != manifest["source_teacher"]["image_digest"]:
        raise RuntimeError("llama.cpp image differs from frozen Q4_K_M teacher")
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
    name = f"qw38-task018-p10-{__import__('os').getpid()}"
    port = 18112
    server_command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--name",
        name,
        "-p",
        f"127.0.0.1:{port}:8080",
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
    log_handle = (output / "server.log").open("wb")
    start = time.monotonic()
    started_utc = datetime.now(UTC).isoformat()
    server = subprocess.Popen(
        server_command, stdout=log_handle, stderr=subprocess.STDOUT
    )
    case_results = []
    try:
        base_url = f"http://127.0.0.1:{port}"
        wait_ready(base_url, server, time.monotonic() + 600)
        startup = run(["docker", "logs", name])
        if "offloaded 65/65 layers to GPU" not in startup or "qwen35" not in startup:
            raise RuntimeError(
                "llama startup lacks expected architecture/full GPU offload"
            )
        (output / "startup.log").write_text(startup + "\n", encoding="utf-8")
        ready_elapsed = time.monotonic() - start
        requests_start = time.monotonic()
        for row in selected:
            before = time.monotonic()
            result = request_case(base_url, row, prompts[row["id"]])
            elapsed = time.monotonic() - before
            token_path = output / f"{row['id']}.generated.u32le"
            text_path = output / f"{row['id']}.txt"
            write_u32le(token_path, result["output_token_ids"])
            text_path.write_text(result["text"], encoding="utf-8")
            case_results.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "cap": row["details"]["cap"],
                    "prompt_tokens": result["prompt_tokens"],
                    "generated_tokens": result["generation_tokens"],
                    "stop": result["generation_stop"],
                    "request_seconds": elapsed,
                    "token_file": token_path.name,
                    "token_sha256": hash_file(token_path),
                    "text_file": text_path.name,
                    "text_sha256": hash_file(text_path),
                }
            )
            print(
                f"P10 llama {len(case_results)}/10 {row['id']} tokens={result['generation_tokens']} seconds={elapsed:.3f}",
                flush=True,
            )
        requests_elapsed = time.monotonic() - requests_start
        pre_shutdown_elapsed = time.monotonic() - start
        total_tokens = sum(row["generated_tokens"] for row in case_results)
        record = {
            "status": "COMPLETE_DIAGNOSTIC",
            "mode": "language-only",
            "mtp_enabled": False,
            "purpose": "P10 speed comparison only; not TASK-018 quality or acceptance evidence",
            "selection": P10_IDS,
            "caps": {row["id"]: 256 for row in selected},
            "comparator": "Q4_K_M llama.cpp",
            "teacher_image": IMAGE,
            "teacher_image_digest": image_digest,
            "teacher_image_id": image_id,
            "server_version": version,
            "teacher_identity_sha256": manifest["reference_capture"][
                "teacher_identity_sha256"
            ],
            "fixture_manifest_sha256": hash_file(root / "manifest.json"),
            "policy_sha256": manifest["policy_sha256"],
            "max_context": max_context,
            "gpu_layers": "all",
            "generation_settings": {
                "temperature": 0.0,
                "samplers": ["temperature"],
                "seed": 42,
                "ignore_eos": False,
                "top_k": 0,
                "top_p": 1.0,
                "n_probs": 0,
                "prompt_cache": "default enabled",
            },
            "started_utc": started_utc,
            "final_request_utc": datetime.now(UTC).isoformat(),
            "launch_to_final_request_wall_seconds": pre_shutdown_elapsed,
            "startup_to_ready_seconds": ready_elapsed,
            "ten_request_wall_seconds": requests_elapsed,
            "requests_per_second": 10 / requests_elapsed,
            "mean_request_seconds": sum(r["request_seconds"] for r in case_results)
            / 10,
            "generated_tokens": total_tokens,
            "generated_tokens_per_second": total_tokens / requests_elapsed,
            "cases": case_results,
        }
        (output / "speed-baseline.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in (
                        "status",
                        "launch_to_final_request_wall_seconds",
                        "startup_to_ready_seconds",
                        "ten_request_wall_seconds",
                        "requests_per_second",
                        "generated_tokens_per_second",
                    )
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        if server.poll() is None:
            server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log_handle.close()
        subprocess.run(["docker", "stop", name], capture_output=True, check=False)
        subprocess.run(["docker", "rm", name], capture_output=True, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
