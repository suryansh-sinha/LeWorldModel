"""Minimal CUDA / BF16 smoke tests for the target GPU (RTX 5070 Ti, sm_120)."""

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_target_gpu_visible():
    name = torch.cuda.get_device_name(0)
    assert "5070 Ti" in name, name
    assert torch.cuda.get_device_capability(0) == (12, 0)
    assert torch.cuda.is_bf16_supported()


def test_bf16_autocast_forward_backward():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    x = torch.randn(64, 128, device=dev)  # (B, D_in)
    lin = nn.Linear(128, 32).to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = lin(x)  # (B, D_out)
        assert y.dtype == torch.bfloat16
        loss = y.float().pow(2).mean()
    loss.backward()
    assert lin.weight.grad is not None
    assert lin.weight.grad.dtype == torch.float32  # master params stay FP32
    assert torch.isfinite(y).all()
    assert torch.isfinite(lin.weight.grad).all()
