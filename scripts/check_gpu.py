"""Report the CUDA/PyTorch setup and run a minimal BF16 autocast smoke test.

Usage: uv run python scripts/check_gpu.py
"""

import sys

import torch
from torch import nn


def main() -> int:
    print(
        f"torch {torch.__version__} | CUDA build {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}"
    )
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 1

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    cap = torch.cuda.get_device_capability(dev)
    print(
        f"device: {props.name} | capability sm_{cap[0]}{cap[1]} | memory {props.total_memory / 2**30:.2f} GiB"
    )
    print(f"arch list in build: {torch.cuda.get_arch_list()}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device=dev)
    b = torch.randn(1024, 1024, device=dev)
    lin = nn.Linear(1024, 256).to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        c = a @ b  # (1024, 1024)
        y = lin(c)  # (1024, 256)
        loss = y.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()

    ok = (
        c.dtype == torch.bfloat16
        and y.dtype == torch.bfloat16
        and torch.isfinite(c).all().item()
        and torch.isfinite(y).all().item()
        and torch.isfinite(lin.weight.grad).all().item()
    )
    print(
        f"autocast matmul dtype {c.dtype}, linear out dtype {y.dtype}, grad dtype {lin.weight.grad.dtype}"
    )
    print(f"loss {loss.item():.4f} | all finite: {ok}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
