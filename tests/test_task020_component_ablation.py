"""Independent checks for the activation quantizer used in TASK-020 probes."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import task020_component_ablation as component
from scripts.task019_fp4_reference import quantize_pack, reconstruct
from scripts.task020_component_ablation import nvfp4_activation
from scripts.task020_selected_ablation import quantize_grouped


@pytest.mark.parametrize("include_head", [False, True])
def test_projection_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_head: bool
) -> None:
    """Replay both phases of all projections even when a BF16 head is present."""
    monkeypatch.chdir(tmp_path)
    index = (
        tmp_path
        / ".cache/authorities/qwen3.8-27b-transformers/model.safetensors.index.json"
    )
    index.parent.mkdir(parents=True)
    index.write_text("{}")
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    names = sorted(
        f"blk.{layer}.{family}.weight"
        for layer in (0, 31, 63)
        for family in (
            ("attn_qkv", "ssm_out", "ffn_gate", "ffn_down")
            if layer == 0
            else ("attn_q", "attn_output", "ffn_gate", "ffn_down")
        )
    )
    trace_names = names + (["output.weight"] if include_head else [])
    for name in trace_names:
        for phase in ("prefill", "decode"):
            path = trace_dir / f"{name}.{phase}.bin"
            np.ones((1, 16), dtype="<f4").tofile(path)
            path.with_suffix(".bin.txt").write_text(
                f"{name}\t{phase}\tf32\t16\t1\t64\n"
            )
    bf16 = [
        SimpleNamespace(
            name=name,
            tensor_type=component.gguf.GGMLQuantizationType.BF16,
            data=np.ones((2, 16), dtype=np.float32),
        )
        for name in trace_names
    ]
    fp4 = [
        SimpleNamespace(
            name=t.name,
            tensor_type=component.gguf.GGMLQuantizationType.NVFP4,
            data=t.data.copy(),
        )
        for t in bf16
        if t.name != "output.weight"
    ]
    if include_head:
        fp4.append(bf16[-1])
    monkeypatch.setattr(
        component.gguf,
        "GGUFReader",
        lambda path: SimpleNamespace(tensors=bf16 if path == "bf16.gguf" else fp4),
    )
    monkeypatch.setattr(
        component.gguf.quants, "dequantize", lambda data, tensor_type: data
    )
    verify = Mock()
    monkeypatch.setattr(component, "verify_source", verify)
    monkeypatch.setattr(component.torch, "set_num_threads", lambda count: None)
    output = tmp_path / "results.json"
    monkeypatch.setattr(
        sys, "argv", ["component", "bf16.gguf", "fp4.gguf", str(trace_dir), str(output)]
    )

    component.main()

    result = json.loads(output.read_text())
    assert result["schema"] == "qw38-task020-component-ablation-v1"
    assert [(row["tensor"], row["phase"]) for row in result["results"]] == [
        (name, phase) for name in names for phase in ("prefill", "decode")
    ]
    assert [call.args[0] for call in verify.call_args_list] == names
    assert all(row["weight_only"]["relative_l2"] == 0.0 for row in result["results"])


class Nvfp4ActivationTest(unittest.TestCase):
    def test_matches_independent_block_reference(self) -> None:
        rng = np.random.default_rng(20)
        inputs = rng.normal(size=(7, 64)).astype(np.float32)
        inputs[0] = 0
        inputs[1, 0] = 100
        expected = np.array(
            reconstruct(quantize_pack(inputs.tolist(), "nvfp4")), dtype=np.float32
        )
        actual = nvfp4_activation(inputs)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            actual, np.vstack([nvfp4_activation(row[None, :]) for row in inputs])
        )

    def test_rejects_nonfinite_or_partial_block(self) -> None:
        with self.assertRaises(ValueError):
            nvfp4_activation(np.array([[float("nan")] + [0.0] * 15], dtype=np.float32))
        with self.assertRaises(ValueError):
            nvfp4_activation(np.zeros((1, 17), dtype=np.float32))

    def test_pinned_gguf_scale_boundaries(self) -> None:
        peaks = [0.0, 6.375, 6 * 7 / 512, 6 * 7.5 / 512, 6 / 64, 1536.0, 2688.0]
        inputs = np.zeros((len(peaks), 16), dtype=np.float32)
        inputs[:, 0] = peaks
        actual = nvfp4_activation(inputs)
        np.testing.assert_array_equal(
            actual[:, 0],
            [0.0, 6.75, 6 * 7 / 512, 6 * 7 / 512, 6 / 64, 1344.0, 2688.0],
        )
        np.testing.assert_array_equal(actual[:, 1:], 0.0)
        native = np.array(
            reconstruct(quantize_pack(inputs.tolist(), "nvfp4")), dtype=np.float32
        )
        self.assertEqual(native[1, 0], 6.0)
        self.assertNotEqual(actual[1, 0], native[1, 0])


class SelectedLogicalQuantizerTest(unittest.TestCase):
    def test_q4_and_q8_zero_and_ties(self) -> None:
        q4 = np.zeros((2, 64), dtype=np.float32)
        q4[1, :4] = [7.0, 3.5, -3.5, -7.0]
        out4 = quantize_grouped(q4, 64)
        np.testing.assert_array_equal(out4[0], np.zeros(64, dtype=np.float32))
        np.testing.assert_array_equal(out4[1, :4], [7.0, 4.0, -4.0, -7.0])

        q8 = np.zeros((2, 32), dtype=np.float32)
        q8[1, :4] = [127.0, 0.5, -0.5, -127.0]
        out8 = quantize_grouped(q8, 32)
        np.testing.assert_array_equal(out8[0], np.zeros(32, dtype=np.float32))
        np.testing.assert_array_equal(out8[1, :4], [127.0, 0.0, 0.0, -127.0])

    def test_scale_ceiling_and_nonfinite_rejection(self) -> None:
        data = np.zeros((1, 64), dtype=np.float32)
        data[0, :2] = [1.0, 0.5]
        result = quantize_grouped(data, 64)
        self.assertGreaterEqual(result[0, 0], 1.0)
        self.assertLessEqual(result[0, 0], 1.015625)
        data[0, 2] = np.nan
        with self.assertRaises(ValueError):
            quantize_grouped(data, 64)


if __name__ == "__main__":
    unittest.main()
