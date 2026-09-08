from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
PREFIX = "QW38_PROMPT_MMQ_RESULT="
CONTRACT = ROOT / "pins/cuda_prompt_mmq_contract.json"
FIXTURE = ROOT / "fixtures/cuda_prompt_mmq.json"
RAW = ROOT / "evidence/profiling/opt009-mmq-tile-sweep-raw.txt"
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 256, 4096)
TILES = (1, 2, 4, 8, 16, 32, 64)


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def _str_keys(mapping: dict[str, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in mapping.items()}


def sweep_winners(raw_text: str) -> dict[str, dict[str, int]]:
    samples: dict[tuple[str, int, int, int, int, int], list[float]] = defaultdict(list)
    summaries: dict[tuple[str, int, int, int, int, int], dict[str, int]] = {}
    for line in raw_text.splitlines():
        if line.startswith("tune=mmq "):
            fields = dict(part.split("=", 1) for part in line.split())
            key = (
                fields["kind"],
                int(fields["output_rows"]),
                int(fields["columns"]),
                int(fields["prompt_rows"]),
                int(fields["tile"]),
                int(fields["replicate"]),
            )
            if int(fields["launched"]) == 1:
                samples[key].append(float(fields["milliseconds"]))
        elif line.startswith("tune=summary "):
            fields = dict(part.split("=", 1) for part in line.split())
            key = (
                fields["kind"],
                int(fields["output_rows"]),
                int(fields["columns"]),
                int(fields["prompt_rows"]),
                int(fields["tile"]),
                int(fields["replicate"]),
            )
            summaries[key] = {
                "occupancy": int(fields["occupancy"]),
                "nonzero": int(fields["nonzero"]),
                "eligible": int(fields["eligible"]),
            }

    def eligible(
        kind: str, rows: int, columns: int, prompt_rows: int, tile: int
    ) -> bool:
        for replicate in (1, 2, 3):
            key = (kind, rows, columns, prompt_rows, tile, replicate)
            if summaries.get(key, {}).get("eligible") != 1:
                return False
            if len(samples.get(key, [])) != 30:
                return False
        return prompt_rows < 2 or tile >= 2

    def replicate_mean(
        kind: str, rows: int, columns: int, prompt_rows: int, tile: int
    ) -> float:
        means = []
        for replicate in (1, 2, 3):
            values = samples[(kind, rows, columns, prompt_rows, tile, replicate)]
            means.append(sum(values) / len(values))
        return sum(means) / len(means)

    def pick(kind: str, shapes: list[tuple[int, int]]) -> dict[str, int]:
        winners: dict[str, int] = {}
        for prompt_rows in BUCKETS:
            best: tuple[float, int] | None = None
            for tile in TILES:
                if any(
                    not eligible(kind, rows, columns, prompt_rows, tile)
                    for rows, columns in shapes
                ):
                    continue
                score = sum(
                    replicate_mean(kind, rows, columns, prompt_rows, tile)
                    for rows, columns in shapes
                )
                if (
                    best is None
                    or score < best[0]
                    or (score == best[0] and tile < best[1])
                ):
                    best = (score, tile)
            assert best is not None, (
                f"no eligible tile for {kind} prompt_rows={prompt_rows}"
            )
            winners[str(prompt_rows)] = best[1]
        return winners

    return {
        "q8_0": pick("q8_0", [(12288, 5120)]),
        "q4_k": pick("q4_k", [(17408, 5120), (5120, 17408)]),
        "q6_k": pick("q6_k", [(5120, 6144)]),
    }


def candidate_means(raw_text: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int, int, int], list[float]] = defaultdict(list)
    occupancy: dict[tuple[str, int, int, int, int], int] = {}
    eligible: dict[tuple[str, int, int, int, int, int], int] = {}
    for line in raw_text.splitlines():
        if line.startswith("tune=mmq "):
            fields = dict(part.split("=", 1) for part in line.split())
            key = (
                fields["kind"],
                int(fields["output_rows"]),
                int(fields["columns"]),
                int(fields["prompt_rows"]),
                int(fields["tile"]),
                int(fields["replicate"]),
            )
            occupancy[key[:-1]] = int(fields["occupancy"])
            if int(fields["launched"]) == 1:
                grouped[key].append(float(fields["milliseconds"]))
        elif line.startswith("tune=summary "):
            fields = dict(part.split("=", 1) for part in line.split())
            key = (
                fields["kind"],
                int(fields["output_rows"]),
                int(fields["columns"]),
                int(fields["prompt_rows"]),
                int(fields["tile"]),
                int(fields["replicate"]),
            )
            eligible[key] = int(fields["eligible"])
    records = []
    for key, values in sorted(grouped.items()):
        kind, rows, columns, prompt_rows, tile, replicate = key
        records.append(
            {
                "replicate": replicate,
                "kind": kind,
                "rows": rows,
                "columns": columns,
                "prompt_rows": prompt_rows,
                "tile": tile,
                "mean_ms": sum(values) / len(values),
                "occupancy": occupancy[key[:-1]],
                "eligible": bool(eligible.get(key, 0)),
            }
        )
    return records


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict)
    for key in (
        "schema_version",
        "task",
        "status",
        "device",
        "compute_capability",
        "driver",
        "runtime",
        "toolkit",
        "pinned_image",
        "measurement_utc",
        "threads",
        "output_rows_per_block",
        "semantic",
        "selected_tiles",
        "launches",
        "kernel_attributes",
        "timing",
        "proof_limit",
    ):
        assert key in result
    assert result["schema_version"] == 1 and result["task"] == "OPT-009"
    assert result["status"] == "measured"
    for key in contract["metadata_required"]:
        assert isinstance(result[key], str) and result[key].strip()
        assert "unknown" not in result[key].lower()
        assert "placeholder" not in result[key].lower()
    assert result["compute_capability"] == "12.0"
    assert result["toolkit"] == contract["toolkit"]
    assert result["pinned_image"] == contract["pinned_image"]
    assert result["threads"] == contract["threads"]
    assert result["output_rows_per_block"] == contract["output_rows_per_block"]
    assert "component-only" in result["proof_limit"]
    assert "no end-to-end" in result["proof_limit"]
    assert set(result["semantic"]) == set(contract["semantic_predicates"])
    assert all(type(value) is bool and value for value in result["semantic"].values())
    for kind, table in contract["selection"].items():
        if kind == "rule":
            continue
        assert _str_keys(result["selected_tiles"][kind]) == _str_keys(table)
    assert len(result["launches"]) == 8
    for launch in result["launches"]:
        assert launch["block"] == [256, 1, 1]
        assert launch["grid"][0] > 0 and launch["grid"][2] == 1
        if launch["kind"].endswith("reference"):
            assert launch["kernel_nodes"] == 1
            assert launch["grid"][1] == launch["prompt_rows"]
        elif launch["kind"] == "q8_0":
            tile = launch["selected_tile"]
            assert tile >= 2 and launch["prompt_rows"] >= 2
            assert launch["kernel_nodes"] == 1
            assert launch["grid"][1] == (launch["prompt_rows"] + tile - 1) // tile
        else:
            tile = launch["selected_tile"]
            if launch["kernel_nodes"] == 1:
                mma_tile = 128
                assert launch["grid"][1] == (
                    launch["prompt_rows"] + mma_tile - 1
                ) // mma_tile
            else:
                assert launch["kernel_nodes"] == 2
                assert launch["grid"][1] == (
                    launch["prompt_rows"] + tile - 1
                ) // tile
    for kind in ("q8_0", "q4_k", "q6_k"):
        attributes = result["kernel_attributes"][kind]
        assert attributes["registers"] > 0
        assert attributes["active_blocks_per_sm"] >= 1
        assert (
            attributes["local_bytes_per_thread"]
            <= contract["maximum_local_bytes_per_thread"]
        )
    timing = result["timing"]
    assert (
        timing["q8_0_12288x5120_64"]["production_ms"]
        < timing["q8_0_12288x5120_64"]["reference_ms"]
    )
    assert (
        timing["q8_0_12288x5120_256"]["production_ms"]
        < timing["q8_0_12288x5120_256"]["reference_ms"]
    )
    assert (
        timing["q4_k_joint_4096"]["selected_ms"]
        <= timing["q4_k_joint_4096"]["tile8_ms"]
    )


def _common() -> list[str]:
    return [
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
    ]


def _build_and_run() -> dict[str, Any]:
    commands = [
        [
            *_common(),
            "make",
            "build/quant_mmv.cuda.o",
            "build/quant.o",
            "build/status.o",
        ],
        [
            *_common(),
            "nvcc",
            "-std=c++17",
            "-O2",
            "-arch=sm_120",
            "--expt-relaxed-constexpr",
            "--fmad=false",
            "-Xcompiler=-Wall,-Wextra,-Werror,-fno-exceptions,-fno-rtti,-ffp-contract=off,-pthread",
            "-Iinclude",
            "-Isrc",
            "-Ithird_party/utf8proc",
            "-Icuda",
            "cuda/prompt_mmq_test.cu",
            "build/quant_mmv.cuda.o",
            "build/quant.o",
            "build/status.o",
            "-o",
            "build/qw38-cuda-prompt-mmq-test",
        ],
        [*_common(), "./build/qw38-cuda-prompt-mmq-test"],
    ]
    outputs = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        outputs.append(completed.stdout)
    records = [
        json.loads(line.removeprefix(PREFIX))
        for line in outputs[-1].splitlines()
        if line.startswith(PREFIX)
    ]
    assert len(records) == 1
    validate_result(records[0])
    return records[0]


def test_prompt_mmq_contract_fixture_and_handbook_are_connected() -> None:
    contract = _contract()
    fixture = json.loads(FIXTURE.read_text())
    validate_result(fixture)
    assert fixture["warmups_per_candidate"] == 3
    assert fixture["samples_per_candidate"] == 30
    assert fixture["replicates"] == 3
    assert fixture["raw_run"] == "evidence/profiling/opt009-mmq-tile-sweep-raw.txt"
    assert len(fixture["candidates"]) == 756
    winners = sweep_winners(RAW.read_text())
    for kind, table in contract["selection"].items():
        if kind == "rule":
            continue
        assert winners[kind] == _str_keys(table)
        assert winners[kind] == _str_keys(fixture["selected_tiles"][kind])
    chapter = (ROOT / "docs" / "40-cuda-prompt-mmq.md").read_text().casefold()
    for term in ["prompt row", "tile", "weight reuse", "token-major", "tail"]:
        assert term in chapter
    assert RAW.is_file()
    assert (
        sum(line.startswith("tune=mmq ") for line in RAW.read_text().splitlines())
        == 22680
    )


def test_prompt_mmq_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []
    for mutate in (
        lambda x: x["semantic"].pop("q8_production_reference_exact"),
        lambda x: x["semantic"].__setitem__("occupancy_admitted", False),
        lambda x: x["launches"][0].__setitem__("kernel_nodes", 2),
        lambda x: x["launches"][2].__setitem__("grid", [1536, 8, 1]),
        lambda x: x["kernel_attributes"]["q8_0"].__setitem__("active_blocks_per_sm", 0),
        lambda x: x["kernel_attributes"]["q4_k"].__setitem__(
            "local_bytes_per_thread", 2048
        ),
        lambda x: x.__setitem__("driver", "placeholder"),
        lambda x: x["timing"]["q8_0_12288x5120_64"].__setitem__(
            "production_ms", x["timing"]["q8_0_12288x5120_64"]["reference_ms"]
        ),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)


def test_real_rtx5090_prompt_mmq_diagnostic() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    _build_and_run()


def _regenerate() -> None:
    result = _build_and_run()
    result["warmups_per_candidate"] = 3
    result["samples_per_candidate"] = 30
    result["replicates"] = 3
    result["raw_run"] = "evidence/profiling/opt009-mmq-tile-sweep-raw.txt"
    result["candidates"] = candidate_means(RAW.read_text())
    fd, temporary = tempfile.mkstemp(
        dir=FIXTURE.parent, prefix=f".{FIXTURE.name}.", text=True
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, FIXTURE)
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["--regenerate-fixture"]:
        raise SystemExit("usage: test_cuda_prompt_mmq.py --regenerate-fixture")
    _regenerate()
