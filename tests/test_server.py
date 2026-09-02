from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import select
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
CORE_TEST = ROOT / "build" / "qw38-server-core-test"
API_TEST = ROOT / "build" / "qw38-server-api-test"
RESPONSES_API_TEST = ROOT / "build" / "qw38-responses-api-test"


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


def test_chat_api_json_tools_and_output_contract() -> None:
    result = subprocess.run(
        [str(API_TEST)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        "json_case=unicode_canonical_duplicate_depth passed=true",
        "request_case=roles_tools_sampling_stream passed=true",
        "rejection_case=image_structured_output passed=true",
        "output_case=reasoning_tool_stop_utf8 passed=true",
        "status=passed",
    ]


def test_chat_completions_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads(
        (ROOT / "pins" / "chat_completions_contract.json").read_text()
    )
    fixture = json.loads((ROOT / "fixtures" / "chat_completions.json").read_text())
    assert contract["route"] == "POST /v1/chat/completions"
    assert contract["model_id"] == "qwen3.8-27b-q4_k_m"
    assert contract["limits"]["body_bytes"] == 1_048_576
    assert contract["limits"]["json_depth"] == 64
    assert contract["single_flight_sessions"] == 1
    assert fixture["native"]["strict_json"] is True
    assert fixture["native"]["canonical_tool_json"] is True
    assert fixture["cuda_smoke"]["passed"] is True
    assert fixture["cuda_smoke"]["second_request_queue_depth"] == 1
    assert fixture["cuda_smoke"]["active_disconnect_cancelled"] is True
    assert fixture["cuda_smoke"]["healthy_after_cancellation"] is True
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected

    handbook = (ROOT / "docs" / "59-chat-completions.md").read_text().casefold()
    for term in [
        "json",
        "surrogate pair",
        "content-length",
        "chatmessage",
        "reasoning_content",
        "tool_choice",
        "single-session",
        "finish_reason",
        "server-sent events",
        "partial utf-8",
        "cancellation",
        "atomic state",
        "proof boundary",
    ]:
        assert term in handbook


def test_responses_api_mapping_storage_and_handbook_are_connected() -> None:
    result = subprocess.run(
        [str(RESPONSES_API_TEST)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        "request_case=text_tools_reasoning_stream passed=true",
        "continuation_case=function_output_store_false passed=true",
        "rejection_case=media_structured_instruction passed=true",
        "store_case=atomic_missing_corrupt passed=true",
        "status=passed",
    ]

    contract = json.loads((ROOT / "pins" / "responses_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "responses.json").read_text())
    assert contract["route"] == "POST /v1/responses"
    assert contract["continuation"]["stored_identity"] == "exact committed token IDs"
    assert contract["continuation"]["expiry"] == "none"
    assert contract["record_limit_bytes"] == 16_777_216
    assert fixture["native"] == {
        "request_mapping": True,
        "function_output_continuation": True,
        "media_rejected": True,
        "structured_output_rejected": True,
        "instruction_replacement_rejected": True,
        "atomic_record_roundtrip": True,
        "missing_record_rejected": True,
        "corrupt_record_rejected": True,
        "exact_tool_result_suffix": True,
    }
    assert fixture["cuda_smoke"]["passed"] is True
    assert fixture["cuda_smoke"]["non_streaming_response"] is True
    assert fixture["cuda_smoke"]["ordered_stream_events"] is True
    assert fixture["cuda_smoke"]["previous_response_id_exact_prefix"] is True
    assert fixture["cuda_smoke"]["store_false_not_continuable"] is True
    assert fixture["cuda_smoke"]["restart_replay"] is True
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected

    handbook = (
        (ROOT / "docs" / "60-responses-and-continuation.md").read_text().casefold()
    )
    for term in [
        "typed",
        "input item",
        "output item",
        "previous_response_id",
        "exact prefix",
        "token ids",
        "atomic",
        "temporary file",
        "fsync",
        "rename",
        "store:false",
        "server-sent events",
        "sequence_number",
        "restart",
        "proof boundary",
    ]:
        assert term in handbook


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


def _post_json(
    base: str, body: dict[str, object], path: str = "/v1/chat/completions"
) -> tuple[int, dict[str, object], dict[str, str]]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read()), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), dict(error.headers)


@pytest.mark.skipif(not MODEL.exists(), reason="the pinned GGUF is required")
def test_cuda_server_owns_one_session_and_serves_control_routes() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    name = f"qw38-server-test-{os.getpid()}"
    response_directory = ROOT / "checkpoints" / f"responses-test-{os.getpid()}"
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
            "--response-dir",
            str(response_directory.relative_to(ROOT)),
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
            "cancelled_requests": 0,
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

        status, rejected, _ = _post_json(
            base,
            {
                "model": "qwen3.8-27b-q4_k_m",
                "messages": [{"role": "user", "content": "hello"}],
                "response_format": {"type": "json_object"},
            },
        )
        assert status == 400
        assert rejected["error"]["code"] == "unimplemented"

        status, completion, headers = _post_json(
            base,
            {
                "model": "qwen3.8-27b-q4_k_m",
                "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
                "reasoning_effort": "none",
                "temperature": 0,
                "max_completion_tokens": 8,
            },
        )
        assert status == 200
        assert completion["object"] == "chat.completion"
        assert completion["choices"][0]["message"] == {
            "role": "assistant",
            "content": "hello",
        }
        assert completion["choices"][0]["finish_reason"] == "stop"
        assert completion["usage"]["prompt_tokens"] > 0
        assert completion["usage"]["completion_tokens"] > 0
        assert int(headers["X-QW38-Queue-Depth"]) == 0
        assert int(headers["X-QW38-Queue-Us"]) >= 0

        status, tool_prompt, _ = _post_json(
            base,
            {
                "model": "qwen3.8-27b-q4_k_m",
                "messages": [
                    {"role": "developer", "content": "Use tools when useful."},
                    {"role": "user", "content": "What is the weather in Bern?"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "reasoning_effort": "none",
                "temperature": 0,
                "max_tokens": 1,
            },
        )
        assert status == 200
        assert (
            tool_prompt["usage"]["prompt_tokens"] > completion["usage"]["prompt_tokens"]
        )
        assert tool_prompt["choices"][0]["finish_reason"] == "length"

        status, stopped, _ = _post_json(
            base,
            {
                "model": "qwen3.8-27b-q4_k_m",
                "messages": [{"role": "user", "content": "Reply with exactly: hello!"}],
                "reasoning_effort": "none",
                "temperature": 0,
                "max_tokens": 8,
                "stop": "!",
            },
        )
        assert status == 200
        assert stopped["choices"][0]["message"]["content"] == "hello"
        assert stopped["choices"][0]["finish_reason"] == "stop"

        status, first_response, _ = _post_json(
            base,
            {
                "model": "qwen3.8-27b-q4_k_m",
                "input": "Reply with exactly: first",
                "reasoning": {"effort": "none"},
                "temperature": 0,
                "max_output_tokens": 8,
            },
            "/v1/responses",
        )
        assert status == 200
        assert first_response["object"] == "response"
        assert first_response["status"] == "completed"
        assert first_response["store"] is True
        assert first_response["output"][0]["content"][0]["text"] == "first"
        first_id = first_response["id"]
        first_record = json.loads((response_directory / f"{first_id}.json").read_text())

        status, continued, _ = _post_json(
            base,
            {
                "model": "qwen3.8-27b-q4_k_m",
                "previous_response_id": first_id,
                "input": "Reply with exactly: second",
                "reasoning": {"effort": "none"},
                "temperature": 0,
                "max_output_tokens": 8,
            },
            "/v1/responses",
        )
        assert status == 200
        assert continued["previous_response_id"] == first_id
        assert continued["output"][0]["content"][0]["text"] == "second"
        continued_record = json.loads(
            (response_directory / f"{continued['id']}.json").read_text()
        )
        assert (
            continued_record["tokens"][: len(first_record["tokens"])]
            == first_record["tokens"]
        )

        status, transient, _ = _post_json(
            base,
            {
                "model": "qwen3.8-27b-q4_k_m",
                "input": "Reply with exactly: transient",
                "reasoning": {"effort": "none"},
                "temperature": 0,
                "max_output_tokens": 8,
                "store": False,
            },
            "/v1/responses",
        )
        assert status == 200
        assert transient["store"] is False
        assert not (response_directory / f"{transient['id']}.json").exists()
        status, unavailable, _ = _post_json(
            base,
            {
                "model": "qwen3.8-27b-q4_k_m",
                "previous_response_id": transient["id"],
                "input": "continue",
            },
            "/v1/responses",
        )
        assert status == 400
        assert "not found or was not stored" in unavailable["error"]["message"]

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        connection.request(
            "POST",
            "/v1/responses",
            body=json.dumps(
                {
                    "model": "qwen3.8-27b-q4_k_m",
                    "input": "Reply with exactly: events",
                    "reasoning": {"effort": "none"},
                    "temperature": 0,
                    "max_output_tokens": 8,
                    "stream": True,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 200
        lines = [line.decode().strip() for line in response.readlines()]
        connection.close()
        event_names = [line[7:] for line in lines if line.startswith("event: ")]
        event_data = [
            json.loads(line[6:]) for line in lines if line.startswith("data: ")
        ]
        assert event_names[:2] == ["response.created", "response.in_progress"]
        assert event_names[-1] == "response.completed"
        assert [event["sequence_number"] for event in event_data] == list(
            range(len(event_data))
        )
        streamed_text = "".join(
            event.get("delta", "")
            for event in event_data
            if event["type"] == "response.output_text.delta"
        )
        assert streamed_text == "events"
        assert event_data[-1]["response"]["output"][0]["content"][0]["text"] == (
            "events"
        )

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(
                {
                    "model": "qwen3.8-27b-q4_k_m",
                    "messages": [
                        {"role": "user", "content": "Reply with exactly: stream"}
                    ],
                    "reasoning_effort": "none",
                    "temperature": 0,
                    "max_tokens": 8,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"
        event_lines = [
            line.decode().strip()
            for line in response.readlines()
            if line.startswith(b"data: ")
        ]
        connection.close()
        assert event_lines[-1] == "data: [DONE]"
        events = [json.loads(line[6:]) for line in event_lines[:-1]]
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"
        streamed = "".join(
            event["choices"][0]["delta"].get("content", "")
            for event in events
            if event["choices"]
        )
        assert streamed == "stream"
        assert events[-1]["choices"] == []
        assert events[-1]["usage"]["completion_tokens"] > 0

        long_body: dict[str, object] = {
            "model": "qwen3.8-27b-q4_k_m",
            "messages": [
                {
                    "role": "user",
                    "content": "List twenty distinct Swiss municipalities, one per line.",
                }
            ],
            "reasoning_effort": "none",
            "temperature": 0,
            "max_tokens": 64,
        }
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(_post_json, base, long_body)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                _, busy = _request(base + "/health")
                if busy["queue_active"]:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("first generation never acquired the queue")
            second_status, second, second_headers = _post_json(
                base,
                {
                    "model": "qwen3.8-27b-q4_k_m",
                    "messages": [
                        {"role": "user", "content": "Reply with exactly: queued"}
                    ],
                    "reasoning_effort": "none",
                    "temperature": 0,
                    "max_tokens": 8,
                },
            )
            first_status, _, _ = first.result(timeout=30)
        assert first_status == 200
        assert second_status == 200
        assert second["choices"][0]["message"]["content"] == "queued"
        assert int(second_headers["X-QW38-Queue-Depth"]) == 1
        assert int(second_headers["X-QW38-Queue-Us"]) > 0

        cancelled_body = json.dumps(
            {
                "model": "qwen3.8-27b-q4_k_m",
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a long essay that must not complete.",
                    }
                ],
                "reasoning_effort": "none",
                "temperature": 0,
                "max_tokens": 128,
            }
        ).encode()
        client = socket.create_connection(("127.0.0.1", port), timeout=5)
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(cancelled_body)}\r\n\r\n".encode()
            + cancelled_body
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            _, before_cancel = _request(base + "/health")
            if before_cancel["queue_active"]:
                break
            time.sleep(0.01)
        else:
            client.close()
            raise AssertionError("request never became active before cancellation")
        client.close()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            _, after_cancel = _request(base + "/health")
            if after_cancel["cancelled_requests"] >= 1:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("client disconnect was not observed as cancellation")
        assert after_cancel["queue_active"] is False
        assert after_cancel["queue_depth"] == 0

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

    restart_name = name + "-restart"
    restart_common = list(common)
    restart_common[restart_common.index(name)] = restart_name
    restart_process = subprocess.Popen(
        [
            *restart_common,
            "./build/cuda/qw38-server",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--response-dir",
            str(response_directory.relative_to(ROOT)),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        restart_port, _ = _wait_for_server(restart_process, 90)
        restart_base = f"http://127.0.0.1:{restart_port}"
        status, restarted, _ = _post_json(
            restart_base,
            {
                "model": "qwen3.8-27b-q4_k_m",
                "previous_response_id": first_id,
                "input": "Reply with exactly: restarted",
                "reasoning": {"effort": "none"},
                "temperature": 0,
                "max_output_tokens": 8,
            },
            "/v1/responses",
        )
        assert status == 200
        assert restarted["previous_response_id"] == first_id
        assert restarted["output"][0]["content"][0]["text"] == "restarted"
        restart_record = json.loads(
            (response_directory / f"{restarted['id']}.json").read_text()
        )
        assert (
            restart_record["tokens"][: len(first_record["tokens"])]
            == first_record["tokens"]
        )
    finally:
        subprocess.run(
            ["docker", "stop", "--timeout", "10", restart_name],
            check=False,
            capture_output=True,
            text=True,
        )
        restart_process.wait(timeout=20)
        shutil.rmtree(response_directory, ignore_errors=True)
    assert restart_process.returncode == 0
