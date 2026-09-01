from __future__ import annotations

import hashlib
import json
import os
import re
import select
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
CORE_TEST = ROOT / "build" / "qw38-server-core-test"


def test_server_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "server_core_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "server_core.json").read_text())
    assert contract["model_id"] == "qwen3.8-27b-q4_k_m"
    assert contract["default_bind"] == "127.0.0.1:8080"
    assert contract["maximum_header_bytes"] == 65_536
    assert contract["gpu_sessions"] == 1
    assert contract["public_routes"] == ["GET /health", "GET /v1/models"]
    assert fixture["queue"]["maximum_active"] == 1
    assert fixture["queue"]["cancelled_waiter_removed"] is True
    assert fixture["queue"]["shutdown_wakes_waiters"] is True
    assert fixture["cuda_smoke"]["passed"] is True
    assert fixture["cuda_smoke"]["sessions"] == 1
    assert fixture["cuda_smoke"]["graceful_sigterm_exit_code"] == 0
    assert fixture["routes"]["GET /health"]["status"] == 200
    assert fixture["routes"]["GET /v1/models"]["status"] == 200
    assert fixture["routes"]["malformed"]["status"] == 400
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected

    handbook = (ROOT / "docs" / "58-http-server-core.md").read_text().casefold()
    for term in [
        "tcp socket",
        "http request",
        "http response",
        "loopback",
        "/health",
        "/v1/models",
        "single-flight",
        "fifo",
        "queue depth",
        "queue delay",
        "cancellation",
        "one gpu session",
        "graceful shutdown",
        "proof boundary",
    ]:
        assert term in handbook


def test_single_flight_queue_is_fifo_timed_and_cancellable() -> None:
    result = subprocess.run(
        [str(CORE_TEST)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    fifo = next(line for line in lines if line.startswith("queue_case=fifo"))
    fields = dict(field.split("=", 1) for field in fifo.split())
    assert fields["depth"] == "1"
    assert int(fields["wait_us"]) >= 30_000
    assert fields["max_active"] == "1"
    assert fields["passed"] == "true"
    assert (
        "queue_case=cancelled_waiter status=cancelled removed=true passed=true" in lines
    )
    assert "queue_case=shutdown status=cancelled woke_waiter=true passed=true" in lines
    assert "status=passed" in lines


def _wait_for_server(
    process: subprocess.Popen[bytes], timeout: float
) -> tuple[int, list[str]]:
    assert process.stdout is not None
    deadline = time.monotonic() + timeout
    output = b""
    while time.monotonic() < deadline:
        readable, _, _ = select.select([process.stdout], [], [], 1.0)
        if not readable:
            if process.poll() is not None:
                break
            continue
        chunk = os.read(process.stdout.fileno(), 4096)
        if not chunk:
            break
        output += chunk
        match = re.search(rb"listening=http://127\.0\.0\.1:(\d+)", output)
        if match:
            return int(match.group(1)), output.decode().splitlines()
    raise AssertionError("server did not become ready\n" + output.decode())


def _request(url: str, method: str = "GET") -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.mark.skipif(not MODEL.exists(), reason="the pinned GGUF is required")
def test_cuda_server_owns_one_session_and_serves_control_routes() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    name = f"qw38-server-test-{os.getpid()}"
    common = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--gpus",
        "all",
        "--name",
        name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/workspace",
        IMAGE,
    ]
    build = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{ROOT}:/workspace",
            IMAGE,
            "make",
            "build/cuda/qw38-server",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    process = subprocess.Popen(
        [
            *common,
            "./build/cuda/qw38-server",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        port, startup = _wait_for_server(process, 90)
        assert any("sessions=1" in line for line in startup)
        base = f"http://127.0.0.1:{port}"

        status, health = _request(base + "/health")
        assert status == 200
        assert health == {
            "status": "ok",
            "ready": True,
            "model": "qwen3.8-27b-q4_k_m",
            "single_flight": True,
            "sessions": 1,
            "queue_active": False,
            "queue_depth": 0,
        }
        status, models = _request(base + "/v1/models?ignored=true")
        assert status == 200
        assert models["object"] == "list"
        assert models["data"] == [
            {
                "id": "qwen3.8-27b-q4_k_m",
                "object": "model",
                "created": 0,
                "owned_by": "quartz-watch-38",
            }
        ]
        status, missing = _request(base + "/missing")
        assert status == 404
        assert missing["error"]["code"] == "not_found"
        status, method = _request(base + "/health", method="POST")
        assert status == 405
        assert method["error"]["code"] == "method_not_allowed"

        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            client.sendall(b"not-http\r\n\r\n")
            response = client.recv(4096)
        assert response.startswith(b"HTTP/1.1 400 Bad Request\r\n")

        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            client.sendall(
                b"GET /health HTTP/1.1\r\nX-Oversized: " + (b"a" * 65_536) + b"\r\n\r\n"
            )
            response = client.recv(4096)
        assert response.startswith(b"HTTP/1.1 400 Bad Request\r\n")
    finally:
        subprocess.run(
            ["docker", "stop", "--timeout", "10", name],
            check=False,
            capture_output=True,
            text=True,
        )
        process.wait(timeout=20)
    assert process.returncode == 0
