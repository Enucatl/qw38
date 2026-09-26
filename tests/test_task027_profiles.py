"""Check transient allocation accounting and exclusive kernel attribution."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from task027_profiles import allocation_peak, category


def test_transient_peak_and_invalid_lifetime() -> None:
    """A temporary allocation affects the peak even when freed before readout."""
    events = [
        (0, 1, 100, 50, 2, 0),
        (1, 1, 200, 80, 2, 0),
        (2, 1, 200, 80, 2, 1),
        (3, 1, 100, 50, 2, 1),
    ]
    assert allocation_peak(events) == {
        "peak_tracked_device_bytes": 130,
        "retained_at_process_exit_bytes": 0,
    }
    with pytest.raises(ValueError, match="unmatched"):
        allocation_peak(events[1:])
    with pytest.raises(ValueError, match="duplicate"):
        allocation_peak(events[:1] * 2)
    assert (
        category("unpack_tile_kernel", 5, "qw38", False)
        == "weight_unpack_or_activation_packing"
    )
    assert category("unpack_tile_kernel", 5, "qw38", True) == "head"
    assert category("attention_prefill_scan_kernel", 5, "qw38", False) == "attention"
