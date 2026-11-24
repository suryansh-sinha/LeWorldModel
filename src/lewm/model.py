"""LeWM model: ViT-Tiny encoder, BN projectors, action encoder, adaLN-Zero causal predictor.

Usage (prints parameter counts):
    uv run python -m lewm.model
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

D = 192  # embedding width shared by encoder, action encoder and predictor


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        causal: bool = True,
        dropout: float = 0.0,
        inner_norm: bool = False,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.heads, self.head_dim, self.drop, self.causal = heads, head_dim, dropout, causal
        self.qkv = nn.Linear(dim, 3 * heads * head_dim, bias=qkv_bias)
        self.out = nn.Sequential(
            nn.Linear(heads * head_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim) if inner_norm else nn.Identity()  # on top of the block's own pre-norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim)   N = tokens (257 in the encoder, T <= 3 in the predictor)
        B, N, _ = x.shape
        x = self.norm(x)
        qkv = self.qkv(x)  # (B, N, 3*H*hd)
        qkv = qkv.view(B, N, 3, self.heads, self.head_dim)  # (B, N, 3, H, hd)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, H, N, hd)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop if self.training else 0.0, is_causal=self.causal  # SDPA ignores eval mode
        )  # (B, H, N, hd)
        y = y.transpose(1, 2).reshape(B, N, -1)  # (B, N, H*hd)
        return self.out(y)  # (B, N, dim)


class MLP(nn.Module):
    """MLP used after attention in transformers"""
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0, inner_norm: bool = False):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim) if inner_norm else nn.Identity()  # on top of the block's own pre-norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., dim) -> fc1 -> (..., hidden) -> fc2 -> (..., dim)
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(self.norm(x))))))


class Projector(nn.Module):
    """Linear -> BatchNorm1d -> GELU -> Linear, applied to the last dim of any (..., D) tensor."""

    def __init__(self, dim: int = D, hidden: int = 2048):
        super().__init__()
        self.fc1, self.bn, self.fc2 = (
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape  # (..., 192), e.g. (B*T, 192) or (B, T, 192)
        x = x.reshape(-1, shape[-1])  # (M, 192)   BatchNorm1d needs 2-D input
        x = F.gelu(self.bn(self.fc1(x)))  # (M, 2048)  BN statistics over all M rows
        return self.fc2(x).reshape(shape)  # (M, 192) -> (..., 192)


# ---------------------------------------------------------------- encoder


class Block(nn.Module):
    """Pre-norm transformer block (for use in Encoder)"""

    def __init__(self, dim: int, heads: int, mlp: int, dropout=0.0, inner_norm: bool = False, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-12)
        self.norm2 = nn.LayerNorm(dim, eps=1e-12)
        # causal=False: the encoder is a ViT; CLS (position 0) must see every patch
        self.attn = Attention(dim, heads, dim // heads, causal=False, dropout=dropout, inner_norm=inner_norm,
                              qkv_bias=qkv_bias)
        self.mlp = MLP(dim, mlp, dropout=dropout, inner_norm=inner_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 257, 192) throughout; both residual branches keep the shape
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    """ViT-Tiny/14 on 224x224: 256 patches + CLS, 12 layers, 3 heads, width 192; CLS -> projector."""

    def __init__(
        self,
        img: int = 224,
        patch: int = 14,
        dim: int = D,
        depth: int = 12,
        heads: int = 3,
        mlp: int = 768,
    ):
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
        # N = number of images (B*T when called from LeWM.encode)
        x = self.patch_embed(x)  # (N, 3, 224, 224) -> (N, 192, 16, 16)
        x = x.flatten(2).transpose(1, 2)  # (N, 256, 192)   one token per 14x14 patch
        cls = self.cls.expand(len(x), -1, -1)  # (N, 1, 192)
        x = torch.cat([cls, x], dim=1)  # (N, 257, 192)   CLS first
        x = x + self.pos  # (N, 257, 192)   + (1, 257, 192)
        for blk in self.blocks:
            x = blk(x)  # (N, 257, 192)
        x = self.norm(x)[:, 0]  # (N, 192)        CLS token only
        return self.proj(x)  # (N, 192)


# ---------------------------------------------------------------- actions + predictor


class ActionEncoder(nn.Module):
    """One action block (5 steps x 2 dims = 10) -> D."""

    def __init__(
        self, in_dim: int = 10, dim: int = D, smoothed: int = 10, scale: int = 4
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_dim, smoothed, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed, scale * dim), nn.SiLU(), nn.Linear(scale * dim, dim)
        )
        self.apply(_init_weights)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        a = a.float().permute(0, 2, 1)  # (B, T, 10) -> (B, 10, T)
        a = self.patch_embed(a)  # (B, 10, T) -> (B, smoothed_dim, T)
        # a: (B, T, smoothed) -> fc1 -> (B, T, scale*dim) -> SiLU + fc2 -> (B, T, dim)
        return self.embed(a.permute(0, 2, 1))


class AdaLNBlock(nn.Module):
    """Transformer block with AdaLN support (for use in Predictor)"""

    def __init__(self, dim: int, heads: int, head_dim: int, mlp: int, dropout: float, inner_norm: bool = True,
                 qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # inner_norm=True: a second LN after the adaLN modulation (owner's choice, see PROGRESS.md)
        self.attn = Attention(dim, heads, head_dim, causal=True, dropout=dropout, inner_norm=inner_norm,
                              qkv_bias=qkv_bias)
        self.mlp = MLP(dim, mlp, dropout=dropout, inner_norm=inner_norm)
        self.ada = nn.Linear(dim, 6 * dim, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 192) tokens, c: (B, T, 192) encoded action for each position
        mod = self.ada(F.silu(c))  # (B, T, 6*192)
        # g -> gate, sh -> shift, sc -> scale, 1 represents after self attn block, 2 for after mlp
        sh1, sc1, g1, sh2, sc2, g2 = mod.chunk(6, dim=-1)  # each (B, T, 192)
        h = self.norm1(x) * (1 + sc1) + sh1  # (B, T, 192)
        x = x + g1 * self.attn(h)  # (B, T, 192)  attn inner width 16*64 = 1024
        h = self.norm2(x) * (1 + sc2) + sh2  # (B, T, 192)
        return x + g2 * self.mlp(h)  # (B, T, 192)  MLP hidden 2048


class Predictor(nn.Module):
    """6 layers, 16 heads x 64, width 192, MLP 2048, dropout 0.1, causal over <= 3 positions."""

    def __init__(
        self,
        dim: int = D,
        depth: int = 6,
        heads: int = 16,
        head_dim: int = 64,
        mlp: int = 2048,
        dropout: float = 0.1,
        max_len: int = 3,
    ):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList(
            AdaLNBlock(dim, heads, head_dim, mlp, dropout, qkv_bias=False) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.proj = Projector(dim)
        nn.init.normal_(self.pos, std=1.0)
        self.apply(_init_weights)
        for blk in self.blocks:  # adaLN-Zero: every block starts as the identity
            nn.init.zeros_(blk.ada.weight)
            nn.init.zeros_(blk.ada.bias)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) embeddings, c: (B, T, D) encoded actions, T <= max_len -> (B, T, D)."""
        x = z + self.pos[:, : z.shape[1]]  # (B, T, 192) + (1, T, 192)
        for blk in self.blocks:
            x = blk(x, c)  # (B, T, 192)
        return self.proj(
            self.norm(x)
        )  # (B, T, 192)  output t predicts the embedding at t+1


# ---------------------------------------------------------------- full model


class LeWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.action_encoder = ActionEncoder()
        self.predictor = Predictor()

        # ImageNet normalization
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: uint8 (B, T, 3, H, W) -> emb (B, T, D)."""
        B, T = frames.shape[:2]
        x = frames.flatten(0, 1).float() / 255.0  # (B*T, 3, 224, 224) float in [0, 1]
        x = (x - self.image_mean) / self.image_std  # (B*T, 3, 224, 224) ImageNet-normalized, roughly [-2.1, 2.6]
        z = self.encoder(x)  # (B*T, 192)
        return z.view(B, T, -1)  # (B, T, 192)

    def predict(self, emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """emb: (B, T, D), actions: (B, T, 10) -> predicted next embeddings (B, T, D)."""
        c = self.action_encoder(actions)  # (B, T, 10) -> (B, T, 192)
        return self.predictor(emb, c)  # (B, T, 192)

    def forward(
        self, frames: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """frames (B, 4, ...), actions (B, 4, 10) -> emb (B, 4, D), pred (B, 3, D). Block a3 is unused."""
        emb = self.encode(frames)  # (B, 4, 192)
        pred = self.predict(
            emb[:, :-1], actions[:, :-1]
        )  # (B, 3, 192) from frames 0-2 + blocks a0-a2
        return emb, pred  # pred[:, t] targets emb[:, t+1]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    m = LeWM()
    for name in ("encoder", "action_encoder", "predictor"):
        sub = getattr(m, name)
        proj = (
            f" (projector {count_params(sub.proj) / 1e6:.2f}M)"
            if hasattr(sub, "proj")
            else ""
        )
        print(f"{name:15s} {count_params(sub) / 1e6:6.2f}M{proj}")
    print(f"{'total':15s} {count_params(m) / 1e6:6.2f}M")


if __name__ == "__main__":
    main()
