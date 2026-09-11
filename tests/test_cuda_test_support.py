from __future__ import annotations

import pytest

from cuda_test_support import cuda_test_tier


def test_cuda_test_tier_accepts_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QW38_CUDA_TEST_TIER", "screen")
    assert cuda_test_tier() == "screen"


def test_cuda_test_tier_requires_explicit_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QW38_CUDA_TEST_TIER", raising=False)
    with pytest.raises(ValueError, match="QW38_CUDA_TEST_TIER"):
        cuda_test_tier()


def test_cuda_test_tier_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QW38_CUDA_TEST_TIER", "fast-but-unsafe")
    with pytest.raises(ValueError, match="QW38_CUDA_TEST_TIER"):
        cuda_test_tier()
