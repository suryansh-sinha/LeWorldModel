"""LeWM model: ViT-Tiny encoder, BN projectors, action encoder, adaLN-Zero causal predictor.

Usage (prints parameter counts):
    uv run python -m lewm.model
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 192  # embedding width shared by encoder, action encoder and predictor


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, causal: bool = False, dropout: float = 0.0):
        super().__init__()
        self.heads, self.head_dim, self.causal = heads, head_dim, causal
        self.qkv = nn.Linear(dim, 3 * heads * head_dim)
        self.out = nn.Linear(heads * head_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.drop(self.out(y.transpose(1, 2).reshape(B, N, -1)))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Projector(nn.Module):
    """Linear -> BatchNorm1d -> GELU -> Linear, applied to the last dim of any (..., D) tensor."""

    def __init__(self, dim: int = D, hidden: int = 2048):
        super().__init__()
        self.fc1, self.bn, self.fc2 = nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        return self.fc2(F.gelu(self.bn(self.fc1(x)))).reshape(shape)


# ---------------------------------------------------------------- encoder

class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, heads: int, mlp: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dim // heads)
        self.mlp = MLP(dim, mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    """ViT-Tiny/14 on 224x224: 256 patches + CLS, 12 layers, 3 heads, width 192; CLS -> projector."""

    def __init__(self, img: int = 224, patch: int = 14, dim: int = D, depth: int = 12, heads: int = 3, mlp: int = 768):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, (img // patch) ** 2 + 1, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, mlp) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.proj = Projector(dim)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: float images (N, 3, H, W), already normalized -> z (N, D)."""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.proj(self.norm(x)[:, 0])


# ---------------------------------------------------------------- actions + predictor

class ActionEncoder(nn.Module):
    """One action block (5 steps x 2 dims = 10) -> D."""

    def __init__(self, in_dim: int = 10, dim: int = D):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(in_dim, dim), nn.Linear(dim, dim)
        self.apply(_init_weights)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(a)))


class AdaLNBlock(nn.Module):
    """DiT-style adaLN-Zero block; each token is modulated by its own condition vector."""

    def __init__(self, dim: int, heads: int, head_dim: int, mlp: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads, head_dim, causal=True, dropout=dropout)
        self.mlp = MLP(dim, mlp, dropout=dropout)
        self.ada = nn.Linear(dim, 6 * dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(c)).chunk(6, dim=-1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + sc1) + sh1)
        return x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + sh2)


class Predictor(nn.Module):
    """6 layers, 16 heads x 64, width 192, MLP 2048, dropout 0.1, causal over <= 3 positions."""

    def __init__(self, dim: int = D, depth: int = 6, heads: int = 16, head_dim: int = 64, mlp: int = 2048,
                 dropout: float = 0.1, max_len: int = 3):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList(AdaLNBlock(dim, heads, head_dim, mlp, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.proj = Projector(dim)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.apply(_init_weights)
        for blk in self.blocks:  # adaLN-Zero: every block starts as the identity
            nn.init.zeros_(blk.ada.weight)
            nn.init.zeros_(blk.ada.bias)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) embeddings, c: (B, T, D) encoded actions, T <= max_len -> (B, T, D)."""
        x = z + self.pos[:, : z.shape[1]]
        for blk in self.blocks:
            x = blk(x, c)
        return self.proj(self.norm(x))


# ---------------------------------------------------------------- full model

class LeWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.action_encoder = ActionEncoder()
        self.predictor = Predictor()

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: uint8 (B, T, 3, H, W) -> emb (B, T, D)."""
        B, T = frames.shape[:2]
        x = (frames.flatten(0, 1).float() / 255.0 - 0.5) / 0.5
        return self.encoder(x).view(B, T, -1)

    def predict(self, emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """emb: (B, T, D), actions: (B, T, 10) -> predicted next embeddings (B, T, D)."""
        return self.predictor(emb, self.action_encoder(actions))

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """frames (B, 4, ...), actions (B, 4, 10) -> emb (B, 4, D), pred (B, 3, D). Block a3 is unused."""
        emb = self.encode(frames)
        pred = self.predict(emb[:, :-1], actions[:, :-1])
        return emb, pred


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    m = LeWM()
    for name in ("encoder", "action_encoder", "predictor"):
        sub = getattr(m, name)
        proj = f" (projector {count_params(sub.proj) / 1e6:.2f}M)" if hasattr(sub, "proj") else ""
        print(f"{name:15s} {count_params(sub) / 1e6:6.2f}M{proj}")
    print(f"{'total':15s} {count_params(m) / 1e6:6.2f}M")


if __name__ == "__main__":
    main()
