"""Host tests for the OPT-057 bounded optimization iteration loop."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from cuda_test_support import cuda_test_tier
from tools.run_optimization_task import (
    FEEDBACK_BUDGET_S,
    Completed,
    HISTORICAL_ORACLE_TARGETS,
    OptimizationRunner,
    OwnedChild,
    ProcessLauncher,
    describe_plan,
    load_contract,
    loop_product,
    validate_tier,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt057_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt057_iteration_loop.json"
REPORT = ROOT / "evidence/optimization/opt057-iteration-loop/REPORT.md"
HISTORICAL_FIXTURE = ROOT / "fixtures/opt056_performance_gate.json"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"
TIER_HEADER = ROOT / "cuda/test_tier.h"
MAKEFILE = ROOT / "Makefile"


class FakeClock:
    def __init__(self) -> None:
        self.mono = 10.0
        self.wall = 1_700_000_000.0

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += seconds


@dataclass
class Scripted:
    duration: float = 0.05
    returncode: int = 0
    stdout: str = "status=passed\n"
    stderr: str = ""
    compiled: bool = False


class FakeLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.held = False

    def acquire(self) -> None:
        self.held = True

    def release(self) -> None:
        self.held = False


class FakeLauncher(ProcessLauncher):
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.commands: list[list[str]] = []
        self.terminated: list[OwnedChild] = []
        self.by_substring: dict[str, Scripted] = {}
        self.make_calls = 0
        self.pid = 5000

    def script(self, substring: str, **kwargs: Any) -> None:
        self.by_substring[substring] = Scripted(**kwargs)

    def run(
        self,
        command: list[str] | tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str] | None,
        timeout_s: float | None,
        docker_name: str | None = None,
    ) -> Completed:
        listed = list(command)
        self.commands.append(listed)
        joined = " ".join(listed)
        spec: Scripted | None = None
        for key, value in self.by_substring.items():
            if key in joined:
                spec = value
                break
        if "make" in listed:
            self.make_calls += 1
            if spec is None:
                spec = Scripted(
                    duration=0.05,
                    compiled=self.make_calls == 1,
                    stdout=(
                        "nvcc -c cuda/optimization_engine_probe.cu\n"
                        if self.make_calls == 1
                        else "make: Nothing to be done for "
                        "'build/qw38-cuda-optimization-engine-probe'.\n"
                    ),
                )
        if spec is None:
            spec = Scripted()
        self.clock.advance(spec.duration)
        pid = self.pid
        self.pid += 1
        terminated = False
        returncode = spec.returncode
        if timeout_s is not None and spec.duration > timeout_s:
            child = OwnedChild(pid=pid, pgid=pid, docker_name=docker_name)
            self.terminate(child)
            terminated = True
            returncode = 1
        return Completed(
            command=listed,
            returncode=returncode,
            stdout=spec.stdout,
            stderr=spec.stderr,
            compiled=spec.compiled or ("nvcc" in spec.stdout),
            pid=pid,
            pgid=pid,
            docker_name=docker_name,
            terminated=terminated,
        )

    def terminate(self, child: OwnedChild) -> None:
        self.terminated.append(child)


def _contract(model: Path) -> dict[str, Any]:
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data["model"] = str(model)
    return data


def _runner(
    tmp_path: Path, clock: FakeClock | None = None
) -> tuple[OptimizationRunner, FakeLauncher, FakeClock]:
    clock = clock or FakeClock()
    launcher = FakeLauncher(clock)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    runner = OptimizationRunner(
        root=ROOT,
        clock=clock,
        launcher=launcher,
        image_exists=lambda _image: True,
        compute_apps=lambda: [],
        lock_factory=FakeLock,
    )
    return runner, launcher, clock


def test_contract_and_makefile_declare_the_iteration_loop() -> None:
    contract = load_contract("OPT-057")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    header = TIER_HEADER.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    assert contract["aggregate_deadline_s"] == 300
    assert contract["claims_throughput"] is False
    assert contract["modes"]["feedback"]["tier_sequence"] == [
        "smoke",
        "correctness",
        "screen",
    ]
    assert loop_product(contract["workloads"]["correctness"]) == 4
    assert "-MMD" in makefile and "-MP" in makefile
    assert "CUDA_STRICT_DIR" in makefile
    assert "CUDA_EXPERIMENTAL_DIR" in makefile
    assert "nvccflags.stamp" in makefile
    assert "cuda-opt057-diagnostics" in makefile
    assert "qw38-cuda-optimization-engine-probe" in makefile
    assert "kScreen" in header
    assert "screen is not implemented by this binary" in header
    assert "--workload" in probe
    assert "tiny" in probe


def test_validate_tier_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QW38_CUDA_TEST_TIER", raising=False)
    with pytest.raises(ValueError, match="QW38_CUDA_TEST_TIER"):
        validate_tier(None)
    with pytest.raises(ValueError, match="fast-but-unsafe"):
        validate_tier("fast-but-unsafe")
    monkeypatch.setenv("QW38_CUDA_TEST_TIER", "screen")
    assert cuda_test_tier() == "screen"
    monkeypatch.setenv("QW38_CUDA_TEST_TIER", "")
    with pytest.raises(ValueError, match="QW38_CUDA_TEST_TIER"):
        cuda_test_tier()


def test_dry_run_prints_target_tier_products_and_deadline(
    tmp_path: Path,
) -> None:
    runner, launcher, _clock = _runner(tmp_path)
    model = tmp_path / "model.gguf"
    result = runner.run(
        "OPT-057",
        "feedback",
        dry_run=True,
        contract_data=_contract(model),
    )
    plan = result["plan"]
    assert "task=OPT-057" in plan
    assert "mode=feedback" in plan
    assert "target=build/qw38-cuda-optimization-engine-probe" in plan
    assert "deadline_s=300" in plan
    assert "product=" in plan
    assert "historical_oracles=none" in plan
    assert launcher.commands == []


def test_feedback_builds_once_then_smoke_correctness_screen(
    tmp_path: Path,
) -> None:
    runner, launcher, _clock = _runner(tmp_path)
    model = tmp_path / "model.gguf"
    result = runner.run(
        "OPT-057",
        "feedback",
        output_dir=tmp_path / "out",
        contract_data=_contract(model),
    )
    assert result["success"] is True
    assert result["elapsed_s"] < FEEDBACK_BUDGET_S
    joined = [" ".join(command) for command in launcher.commands]
    assert any("make" in command for command in joined)
    assert any("--workload tiny" in command for command in joined)
    assert any("--workload tokens" in command for command in joined)
    assert any("--workload decode" in command for command in joined)
    assert all(
        oracle not in command
        for command in joined
        for oracle in HISTORICAL_ORACLE_TARGETS
    )
    assert result["result_class"] == "ok"


def test_acceptance_reports_cold_and_two_warm_durations(
    tmp_path: Path,
) -> None:
    runner, launcher, _clock = _runner(tmp_path)
    model = tmp_path / "model.gguf"
    result = runner.run(
        "OPT-057",
        "acceptance",
        output_dir=tmp_path / "out",
        contract_data=_contract(model),
    )
    assert result["success"] is True
    assert "cold_s" in result["durations"]
    assert "warm1_s" in result["durations"]
    assert "warm2_s" in result["durations"]
    assert result["cache_hits"] >= 1
    assert result["compile"] == 1
    assert launcher.make_calls == 3
    joined = [" ".join(command) for command in launcher.commands]
    assert all(
        oracle not in command
        for command in joined
        for oracle in HISTORICAL_ORACLE_TARGETS
    )


def test_release_invokes_historical_oracles_only_in_release_mode(
    tmp_path: Path,
) -> None:
    runner, launcher, _clock = _runner(tmp_path)
    model = tmp_path / "model.gguf"
    result = runner.run(
        "OPT-057",
        "release",
        output_dir=tmp_path / "out",
        contract_data=_contract(model),
    )
    assert result["success"] is True
    joined = "\n".join(" ".join(command) for command in launcher.commands)
    for oracle in HISTORICAL_ORACLE_TARGETS:
        assert oracle in joined


def test_missing_image_is_setup_failure_with_concrete_command(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    launcher = FakeLauncher(clock)
    runner = OptimizationRunner(
        root=ROOT,
        clock=clock,
        launcher=launcher,
        image_exists=lambda _image: False,
        compute_apps=lambda: [],
        lock_factory=FakeLock,
    )
    result = runner.run(
        "OPT-057",
        "feedback",
        output_dir=tmp_path / "out",
        contract_data=_contract(tmp_path / "model.gguf"),
    )
    assert result["result_class"] == "setup_failed"
    assert "docker build -f docker/cuda.Dockerfile" in result["message"]
    assert not any(
        "optimization-engine-probe" in " ".join(c) for c in launcher.commands
    )


def test_foreign_compute_fails_setup_without_killing(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    launcher = FakeLauncher(clock)
    runner = OptimizationRunner(
        root=ROOT,
        clock=clock,
        launcher=launcher,
        image_exists=lambda _image: True,
        compute_apps=lambda: [{"pid": "9", "process_name": "python"}],
        lock_factory=FakeLock,
    )
    (tmp_path / "model.gguf").write_bytes(b"gguf")
    result = runner.run(
        "OPT-057",
        "feedback",
        output_dir=tmp_path / "out",
        contract_data=_contract(tmp_path / "model.gguf"),
    )
    assert result["result_class"] == "setup_failed"
    assert "not killing foreign work" in result["message"]
    assert launcher.terminated == []


def test_malformed_tier_fails_before_gpu_probe(tmp_path: Path) -> None:
    runner, launcher, _clock = _runner(tmp_path)
    model = tmp_path / "model.gguf"
    contract = _contract(model)
    contract["workloads"]["smoke"]["tier"] = "fast-but-unsafe"
    result = runner.run(
        "OPT-057",
        "feedback",
        phase="smoke",
        output_dir=tmp_path / "out",
        contract_data=contract,
    )
    assert result["result_class"] == "invalid_tier"
    assert not any(
        "./build/qw38-cuda-optimization-engine-probe" in " ".join(command)
        for command in launcher.commands
    )


def test_over_budget_subprocess_is_torn_down(tmp_path: Path) -> None:
    runner, launcher, _clock = _runner(tmp_path)
    launcher.script("make", duration=400.0, returncode=0, compiled=True)
    model = tmp_path / "model.gguf"
    result = runner.run(
        "OPT-057",
        "feedback",
        output_dir=tmp_path / "out",
        contract_data=_contract(model),
    )
    assert result["result_class"] == "budget_exhausted"
    assert launcher.terminated
    assert result["owned_children"] == 0
    compile_log = next((tmp_path / "out").rglob("stderr.txt"))
    assert compile_log.is_file()


def test_screen_failure_does_not_run_release_oracles(tmp_path: Path) -> None:
    runner, launcher, _clock = _runner(tmp_path)
    launcher.script("tiny", returncode=1, stdout="", stderr="numerical")
    model = tmp_path / "model.gguf"
    result = runner.run(
        "OPT-057",
        "feedback",
        output_dir=tmp_path / "out",
        contract_data=_contract(model),
    )
    joined = "\n".join(" ".join(command) for command in launcher.commands)
    assert result["success"] is False
    for oracle in HISTORICAL_ORACLE_TARGETS:
        assert oracle not in joined
    assert "--workload decode" not in joined


def test_runner_does_not_mutate_historical_evidence(tmp_path: Path) -> None:
    before = HISTORICAL_FIXTURE.read_bytes() if HISTORICAL_FIXTURE.is_file() else b""
    report = ROOT / "evidence/optimization/opt056-performance-gate/REPORT.md"
    report_before = report.read_bytes() if report.is_file() else b""
    runner, _launcher, _clock = _runner(tmp_path)
    model = tmp_path / "model.gguf"
    runner.run(
        "OPT-057",
        "feedback",
        output_dir=tmp_path / "out",
        contract_data=_contract(model),
    )
    if HISTORICAL_FIXTURE.is_file():
        assert HISTORICAL_FIXTURE.read_bytes() == before
    if report.is_file():
        assert report.read_bytes() == report_before


def test_make_fixture_invalidates_on_flags_and_headers(tmp_path: Path) -> None:
    source = tmp_path / "probe.c"
    header = tmp_path / "header.h"
    makefile = tmp_path / "Makefile"
    source.write_text('#include "header.h"\nint main(void) { return VALUE; }\n')
    header.write_text("#define VALUE 0\n")
    makefile.write_text(
        """
BUILD := build
STRICT := $(BUILD)/cuda/strict
EXPERIMENTAL := $(BUILD)/cuda/experimental
STAMP := $(STRICT)/nvccflags.stamp
EXPERIMENTAL_STAMP := $(EXPERIMENTAL)/nvccflags.stamp
FLAGS ?= -O2
CC ?= cc

$(BUILD) $(STRICT) $(EXPERIMENTAL):
	mkdir -p $@

.PHONY: FORCE all
FORCE:

$(STAMP): FORCE | $(STRICT)
	@printf '%s\\n' '$(FLAGS)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(EXPERIMENTAL_STAMP): FORCE | $(EXPERIMENTAL)
	@printf '%s\\n' '$(FLAGS) experimental' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(BUILD)/probe.o: probe.c header.h $(STAMP) | $(BUILD)
	$(CC) -MMD -MP -MF $(@:.o=.d) -MT $@ -c probe.c -o $@

$(BUILD)/probe: $(BUILD)/probe.o
	$(CC) $^ -o $@

all: $(BUILD)/probe $(EXPERIMENTAL_STAMP)

-include $(BUILD)/probe.d
"""
    )

    def make(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["make", "-C", str(tmp_path), *args],
            check=False,
            capture_output=True,
            text=True,
        )

    first = make("all")
    assert first.returncode == 0, first.stdout + first.stderr
    probe_o = tmp_path / "build/probe.o"
    stamp = tmp_path / "build/cuda/strict/nvccflags.stamp"
    experimental = tmp_path / "build/cuda/experimental/nvccflags.stamp"
    assert probe_o.is_file()
    assert stamp.is_file()
    assert experimental.is_file()
    first_mtime = probe_o.stat().st_mtime_ns
    first_stamp = stamp.read_text()
    second = make("all")
    assert second.returncode == 0, second.stdout + second.stderr
    assert probe_o.stat().st_mtime_ns == first_mtime
    assert stamp.read_text() == first_stamp
    third = make("all", "FLAGS=-O3")
    assert third.returncode == 0, third.stdout + third.stderr
    assert probe_o.stat().st_mtime_ns != first_mtime
    rebuilt = probe_o.stat().st_mtime_ns
    header.write_text("#define VALUE 1\n")
    fourth = make("all", "FLAGS=-O3")
    assert fourth.returncode == 0, fourth.stdout + fourth.stderr
    assert probe_o.stat().st_mtime_ns != rebuilt
    assert (tmp_path / "build/probe.d").is_file()
    assert stamp.read_text() != first_stamp
    assert experimental.is_file()
    assert experimental.read_text() != stamp.read_text()


def test_checked_in_fixture_is_read_only_and_complete() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for key in contract["required_fixture_keys"]:
        assert key in fixture, key
    assert fixture["claims_throughput"] is False
    assert fixture["warm2_recompiled"] is False
    assert fixture["aggregate_deadline_s"] == 300
    assert REPORT.is_file()
    text = REPORT.read_text(encoding="utf-8")
    for item in contract["proof_limit"]:
        assert item in text
    assert "tok/s acceptance" not in text.lower() or "no throughput" in text.lower()


def test_describe_plan_separates_screen_from_release() -> None:
    contract = load_contract("OPT-057")
    feedback = describe_plan("OPT-057", "feedback", contract, None)
    release = describe_plan("OPT-057", "release", contract, None)
    assert "historical_oracles=none" in feedback
    assert "build/qw38-cuda-decode-oracle-test" in release
    assert "screen:" in feedback
    assert "deadline_s=300" in feedback
