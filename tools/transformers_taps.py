"""Diagnostic tensor taps for the pinned Transformers semantic authority."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


GDN_LAYERS = (0, 62)
ATTENTION_LAYERS = (3, 7, 63)
VISIBLE_LAYERS = frozenset((*GDN_LAYERS, *ATTENTION_LAYERS))


class TapCapture:
    """Collect selected upstream tensors without changing model arithmetic."""

    def __init__(self) -> None:
        self.position = -1
        self.active_gdn: int | None = None
        self.active_attention: int | None = None
        self._blob = bytearray()
        self.records: list[dict[str, Any]] = []
        self._names: set[str] = set()
        self._handles: list[Any] = []
        self._restores: list[tuple[Any, str, Callable[..., Any]]] = []

    def set_position(self, position: int) -> None:
        self.position = position

    def add(self, name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return
        qualified = f"position.{self.position}.{name}"
        if qualified in self._names:
            raise ValueError(f"duplicate Transformers tap: {qualified}")
        self._names.add(qualified)
        source_dtype = str(tensor.dtype).removeprefix("torch.")
        array = (
            tensor.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
        )
        raw = array.tobytes()
        flat = array.reshape(-1).astype(np.float64, copy=False)
        finite = np.isfinite(flat)
        finite_values = flat[finite]
        summary: dict[str, int | float | None] = {
            "count": int(flat.size),
            "finite_count": int(finite.sum()),
            "nan_count": int(np.isnan(flat).sum()),
            "positive_infinity_count": int(np.isposinf(flat).sum()),
            "negative_infinity_count": int(np.isneginf(flat).sum()),
            "minimum": None,
            "maximum": None,
            "mean": None,
            "rms": None,
        }
        if finite_values.size:
            summary.update(
                {
                    "minimum": float(finite_values.min()),
                    "maximum": float(finite_values.max()),
                    "mean": float(finite_values.mean()),
                    "rms": float(
                        math.sqrt(float(np.mean(finite_values * finite_values)))
                    ),
                }
            )
        offset = len(self._blob)
        self._blob.extend(raw)
        self.records.append(
            {
                "name": qualified,
                "shape": list(array.shape),
                "source_dtype": source_dtype,
                "storage_dtype": "float32_le",
                "offset": offset,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "summary": summary,
            }
        )

    def _output_tensor(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("tap output is not a tensor")
        return output

    def _capture_output(self, name: str) -> Callable[..., None]:
        def hook(module: Any, args: Any, output: Any) -> None:
            del module, args
            self.add(name, self._output_tensor(output))

        return hook

    def _capture_input(self, name: str) -> Callable[..., None]:
        def hook(module: Any, args: Any) -> None:
            del module
            self.add(name, args[0])

        return hook

    def _activate(self, kind: str, layer: int) -> Callable[..., None]:
        def hook(module: Any, args: Any) -> None:
            del module, args
            setattr(self, f"active_{kind}", layer)

        return hook

    def _deactivate(self, kind: str) -> Callable[..., None]:
        def hook(module: Any, args: Any, output: Any) -> None:
            del module, args, output
            setattr(self, f"active_{kind}", None)

        return hook

    def _hook_output(self, module: Any, name: str) -> None:
        self._handles.append(module.register_forward_hook(self._capture_output(name)))

    def install(self, model: Any, modeling: Any) -> None:
        language = model.model.language_model
        self._hook_output(language.embed_tokens, "embedding.output")
        self._hook_output(language.norm, "final_norm.output")

        for layer_index in sorted(VISIBLE_LAYERS):
            layer = language.layers[layer_index]
            prefix = f"layer.{layer_index}"
            self._handles.append(
                layer.register_forward_pre_hook(
                    self._capture_input(f"{prefix}.residual.input")
                )
            )
            self._hook_output(layer.input_layernorm, f"{prefix}.input_norm.output")
            self._handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    self._capture_input(f"{prefix}.mixer_residual.output")
                )
            )
            self._hook_output(
                layer.post_attention_layernorm, f"{prefix}.post_mixer_norm.output"
            )
            self._hook_output(layer.mlp.gate_proj, f"{prefix}.ffn.gate_projection")
            self._hook_output(layer.mlp.up_proj, f"{prefix}.ffn.up_projection")
            self._hook_output(layer.mlp.act_fn, f"{prefix}.ffn.silu_gate")
            self._handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    self._capture_input(f"{prefix}.ffn.activated")
                )
            )
            self._hook_output(layer.mlp.down_proj, f"{prefix}.ffn.output")
            self._hook_output(layer, f"{prefix}.residual.output")

            if layer_index in GDN_LAYERS:
                mixer = layer.linear_attn
                self._handles.append(
                    mixer.register_forward_pre_hook(self._activate("gdn", layer_index))
                )
                self._handles.append(
                    mixer.register_forward_hook(self._deactivate("gdn"))
                )
                self._hook_output(mixer.in_proj_qkv, f"{prefix}.gdn.qkv_projection")
                self._hook_output(mixer.in_proj_z, f"{prefix}.gdn.gate_projection")
                self._hook_output(mixer.in_proj_a, f"{prefix}.gdn.decay_projection")
                self._hook_output(mixer.in_proj_b, f"{prefix}.gdn.beta_projection")
                self._hook_output(mixer.norm, f"{prefix}.gdn.gated_norm")
                self._hook_output(mixer.out_proj, f"{prefix}.gdn.output")
            else:
                mixer = layer.self_attn
                self._handles.append(
                    mixer.register_forward_pre_hook(
                        self._activate("attention", layer_index)
                    )
                )
                self._handles.append(
                    mixer.register_forward_hook(self._deactivate("attention"))
                )
                self._hook_output(
                    mixer.q_proj, f"{prefix}.attention.query_gate_projection"
                )
                self._hook_output(mixer.k_proj, f"{prefix}.attention.key_projection")
                self._hook_output(mixer.v_proj, f"{prefix}.attention.value_projection")
                self._hook_output(mixer.q_norm, f"{prefix}.attention.query_norm")
                self._hook_output(mixer.k_norm, f"{prefix}.attention.key_norm")
                self._handles.append(
                    mixer.o_proj.register_forward_pre_hook(
                        self._capture_input(f"{prefix}.attention.gated_context")
                    )
                )
                self._hook_output(mixer.o_proj, f"{prefix}.attention.output")

        self._wrap_functions(modeling)

    def _replace(self, module: Any, name: str, replacement: Callable[..., Any]) -> None:
        original = getattr(module, name)
        self._restores.append((module, name, original))
        setattr(module, name, replacement)

    def _wrap_functions(self, modeling: Any) -> None:
        for function_name in ("causal_conv1d_fn", "causal_conv1d_update"):
            original_convolution = getattr(modeling, function_name)

            def convolution(
                *args: Any,
                _original: Callable[..., Any] = original_convolution,
                **kwargs: Any,
            ) -> Any:
                result = _original(*args, **kwargs)
                layer = self.active_gdn
                if layer is not None:
                    self.add(f"layer.{layer}.gdn.convolution", result)
                    current = result[..., -1] if result.ndim == 3 else result
                    self.add(f"layer.{layer}.gdn.convolution_current", current)
                return result

            self._replace(modeling, function_name, convolution)

        for function_name in (
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        ):
            original = getattr(modeling, function_name)

            def gated(
                *args: Any, _original: Callable[..., Any] = original, **kwargs: Any
            ) -> Any:
                names = ("query", "key", "value", "g", "beta")
                values = {name: kwargs.get(name) for name in names}
                for index, name in enumerate(names):
                    if values[name] is None and index < len(args):
                        values[name] = args[index]
                result = _original(*args, **kwargs)
                layer = self.active_gdn
                if layer is not None:
                    for name in names:
                        self.add(f"layer.{layer}.gdn.{name}", values[name])
                    self.add(
                        f"layer.{layer}.gdn.query_unique",
                        values["query"][:, :, ::3, :],
                    )
                    self.add(
                        f"layer.{layer}.gdn.key_unique",
                        values["key"][:, :, ::3, :],
                    )
                    self.add(f"layer.{layer}.gdn.core_output", result[0])
                    self.add(f"layer.{layer}.gdn.recurrent_state", result[1])
                return result

            self._replace(modeling, function_name, gated)

        original_rope = modeling.apply_rotary_pos_emb

        def rope(q: torch.Tensor, k: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            result = original_rope(q, k, *args, **kwargs)
            layer = self.active_attention
            if layer is not None:
                self.add(f"layer.{layer}.attention.query_before_rope", q)
                self.add(f"layer.{layer}.attention.key_before_rope", k)
                self.add(f"layer.{layer}.attention.query_after_rope", result[0])
                self.add(f"layer.{layer}.attention.key_after_rope", result[1])
            return result

        self._replace(modeling, "apply_rotary_pos_emb", rope)

        original_eager = modeling.eager_attention_forward

        def eager(
            module: Any,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            result = original_eager(module, query, key, value, *args, **kwargs)
            layer = self.active_attention
            if layer is not None:
                self.add(f"layer.{layer}.attention.query", query)
                self.add(f"layer.{layer}.attention.key_cache", key)
                self.add(f"layer.{layer}.attention.value_cache", value)
                self.add(f"layer.{layer}.attention.key_row", key[:, :, -1:, :])
                self.add(f"layer.{layer}.attention.value_row", value[:, :, -1:, :])
                self.add(f"layer.{layer}.attention.core_output", result[0])
                self.add(f"layer.{layer}.attention.weights", result[1])
            return result

        self._replace(modeling, "eager_attention_forward", eager)

    def add_logits(self, logits: torch.Tensor) -> None:
        self.add("logits", logits)

    def write(self, path: Path) -> dict[str, Any]:
        raw = bytes(self._blob)
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temporary.write_bytes(raw)
        temporary.replace(path)
        return {
            "schema_version": 1,
            "storage": "concatenated_float32_little_endian",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "records": self.records,
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        for module, name, original in reversed(self._restores):
            setattr(module, name, original)
        self._handles.clear()
        self._restores.clear()
