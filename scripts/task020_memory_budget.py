#!/usr/bin/env python3
"""Estimate the selected Q4 MLP / Q8 attention-GDN one-view policy."""

import json
from pathlib import Path

from task019_memory_budget import (
    BF16_SMALL,
    MIB,
    PROJECTION_PREFIXES,
    bytes_for,
    inventory,
)


def main() -> None:
    families = inventory()["families"]
    context = 32768
    free = 31566 * MIB
    reserve = 2048 * MIB
    state = 153_944_064 + 65_536 * context
    transients = (1024 + 512 + 256 + 128) * MIB
    mtp = sum(
        bytes_for(family, "q4")
        if name == "mtp.fc"
        or name.startswith(("mtp.self_attn.", "mtp.mlp."))
        and name not in ("mtp.self_attn.q_norm", "mtp.self_attn.k_norm")
        else family["n_bytes"]
        for name, family in families.items()
        if family["level1"] == "mtp"
    )
    resident = mtp
    by_policy = {
        "q4g64_mlp": 0,
        "q8g32_attention_gdn": 0,
        "bf16_controls": 0,
        "q8g32_head": 0,
        "retained_mtp": mtp,
    }
    for name, family in families.items():
        if name.startswith(PROJECTION_PREFIXES) and name not in BF16_SMALL:
            if name.startswith(("linear_attn.", "self_attn.")):
                # Q8G32: 32 signed bytes plus one FP16 scale per group.
                amount = family["n_parameters"] * 17 // 16
                bucket = "q8g32_attention_gdn"
            else:
                amount = bytes_for(family, "q4")
                bucket = "q4g64_mlp"
        elif name == "lm_head":
            amount = family["n_parameters"] * 17 // 16
            bucket = "q8g32_head"
        elif family["level1"] not in ("mtp", "vision"):
            amount = family["n_bytes"]
            bucket = "bf16_controls"
        else:
            continue
        resident += amount
        by_policy[bucket] += amount
    peak = resident + state + transients
    result = {
        "schema": "qw38-task020-selected-memory-budget-v1",
        "basis": "TASK-019 inventory arithmetic; one resident projection view; Q8G32 sensitive projection exceptions",
        "free_before_alloc_bytes": free,
        "reserve_bytes": reserve,
        "context_tokens": context,
        "resident_weights_scales_and_retained_payload_bytes": resident,
        "persistent_state_bytes": state,
        "activation_output_and_scratch_bytes": 1024 * MIB,
        "library_workspace_bytes": 512 * MIB,
        "graph_allowance_bytes": 256 * MIB,
        "metadata_alignment_allowance_bytes": 128 * MIB,
        "estimated_peak_bytes": peak,
        "headroom_after_reserve_bytes": free - reserve - peak,
        "fits": peak <= free - reserve,
        "family_bytes": by_policy,
        "measured_integrated_peak": False,
    }
    Path("docs/implementation/task020-memory-budget.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                key: round(result[key] / MIB, 2)
                for key in (
                    "resident_weights_scales_and_retained_payload_bytes",
                    "estimated_peak_bytes",
                    "headroom_after_reserve_bytes",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
