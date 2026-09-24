#!/usr/bin/env python3
"""Estimate language execution VRAM, including retained inactive MTP payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INVENTORY = Path("docs/architecture/model-inventory.md")
MIB = 1 << 20
PROJECTION_PREFIXES = ("linear_attn.in_proj_", "linear_attn.out_proj",
                       "self_attn.", "mlp.")
BF16_SMALL = ("linear_attn.in_proj_a", "linear_attn.in_proj_b",
              "self_attn.q_norm", "self_attn.k_norm")


def rounded(value: int, alignment: int) -> int:
    """Round a positive count to its storage alignment."""
    return (value + alignment - 1) // alignment * alignment


def inventory() -> dict[str, object]:
    """Read the machine-checkable inventory embedded in the model document."""
    source = INVENTORY.read_text()
    marker = "## Machine-checkable totals"
    document = source[source.index(marker):]
    payload = document.split("```json\n", 1)[1].split("\n```", 1)[0]
    return json.loads(payload)


def bytes_for(family: dict[str, object], policy: str) -> int:
    """Count payload and scale bytes for a complete family under a policy."""
    shape = family["shapes"][0]
    count = family["n_tensors"]
    if len(shape) != 2:
        return family["n_bytes"]
    n, k = shape
    if policy == "bf16":
        return count * n * k * 2
    if policy == "fp8":
        return count * (n * k + 4)
    if policy == "q4":
        return count * (n * k // 2 + n * rounded(k, 64) // 64 * 2)
    block = 16 if policy == "nvfp4" else 32
    padded_n = rounded(n, 128)
    padded_k = rounded(k, 128)
    return count * (padded_n * padded_k // 2 +
                    padded_n * padded_k // block +
                    (4 if policy == "nvfp4" else 0))


def main() -> None:
    """Print per-policy resident and transient budget with explicit assumptions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--free-mib", type=int, required=True,
                        help="cudaMemGetInfo free MiB before model allocations")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--scratch-mib", type=int, default=1024)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--graphs-mib", type=int, default=256)
    parser.add_argument("--metadata-mib", type=int, default=128)
    parser.add_argument("--reserve-mib", type=int, default=2048)
    args = parser.parse_args()
    families = inventory()["families"]
    state = 153_944_064 + 65_536 * args.context
    transient = (args.scratch_mib + args.workspace_mib + args.graphs_mib +
                 args.metadata_mib) * MIB
    available = (args.free_mib - args.reserve_mib) * MIB
    retained_mtp = sum(
        bytes_for(family, "q4") if name == "mtp.fc" or
        name.startswith(("mtp.self_attn.", "mtp.mlp.")) and
        name not in ("mtp.self_attn.q_norm", "mtp.self_attn.k_norm")
        else family["n_bytes"]
        for name, family in families.items() if family["level1"] == "mtp"
    )
    result: dict[str, object] = {
        "source": str(INVENTORY),
        "basis": "active language execution; full embedding, Q8 head, projection weights/scales and retained inactive MTP payloads",
        "free_before_alloc_bytes": args.free_mib * MIB,
        "reserve_bytes": args.reserve_mib * MIB,
        "context_tokens": args.context,
        "persistent_state_bytes": state,
        "activation_output_and_scratch_bytes": args.scratch_mib * MIB,
        "library_workspace_bytes": args.workspace_mib * MIB,
        "graph_allowance_bytes": args.graphs_mib * MIB,
        "metadata_alignment_allowance_bytes": args.metadata_mib * MIB,
        "retained_mtp_artifact_bytes": retained_mtp,
        "candidate_views": {},
    }
    active_q4_projection = sum(
        bytes_for(family, "q4") for name, family in families.items()
        if name.startswith(PROJECTION_PREFIXES) and name not in BF16_SMALL
    )
    result["optional_full_q4_projection_view_bytes"] = active_q4_projection
    head = families["lm_head"]
    q8_head = head["n_parameters"] * 17 // 16
    result["head_options_bytes"] = {
        "q8g32_base": q8_head,
        "nvfp4_estimate": bytes_for(head, "nvfp4"),
        "mxfp4_estimate": bytes_for(head, "mxfp4"),
        "fp8_estimate": bytes_for(head, "fp8"),
        "bf16": bytes_for(head, "bf16"),
    }
    for policy in ("q4", "nvfp4", "mxfp4"):
        total = retained_mtp
        for name, family in families.items():
            if name.startswith(PROJECTION_PREFIXES) and name not in BF16_SMALL:
                total += bytes_for(family, policy)
            elif name == "lm_head":
                total += family["n_parameters"] * 17 // 16
            elif family["level1"] not in ("mtp", "vision"):
                total += family["n_bytes"]
        peak = total + state + transient
        result["candidate_views"][policy] = {
            "resident_weights_scales_and_retained_payload_bytes": total,
            "transient_and_state_peak_bytes": peak,
            "headroom_after_reserve_bytes": available - peak,
            "fits": peak <= available,
            "second_q4_projection_view_peak_bytes": peak +
                active_q4_projection,
            "second_q4_projection_view_fits": peak +
                active_q4_projection <= available,
            "head_option_headroom_after_reserve_bytes": {
                option: available - peak - (head_bytes - q8_head)
                for option, head_bytes in result["head_options_bytes"].items()
            },
        }
    deltas = {}
    for policy in ("q4", "nvfp4", "mxfp4"):
        deltas[policy] = {}
        headroom = result["candidate_views"][policy]["headroom_after_reserve_bytes"]
        for name, family in families.items():
            if name.startswith(PROJECTION_PREFIXES) and name not in BF16_SMALL and \
                    len(family["shapes"][0]) == 2:
                deltas[policy][name] = {}
                for exception in ("fp8", "bf16"):
                    delta = bytes_for(family, exception) - bytes_for(family, policy)
                    deltas[policy][name][exception] = {
                        "delta_bytes": delta,
                        "single_family_headroom_after_reserve_bytes": headroom - delta,
                        "single_family_fits": delta <= headroom,
                    }
    result["exception_deltas"] = deltas
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
