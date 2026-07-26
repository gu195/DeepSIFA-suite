"""MonoXtract three-SAC plus Transformer-Cell architecture."""

from __future__ import annotations

import math
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Apply stochastic depth to a residual branch."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device
    )
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PRM(nn.Module):
    """Pyramidal reduction module based on dilated one-dimensional convolutions."""

    def __init__(
        self,
        img_size: int = 224,
        kernel_size: int = 4,
        downsample_ratio: int = 4,
        dilations=(1, 6, 12),
        in_chans: int = 3,
        embed_dim: int = 64,
        share_weights: bool = False,
        op: str = "cat",
    ):
        super().__init__()
        self.dilations = list(dilations)
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.op = op
        self.kernel_size = kernel_size
        self.stride = downsample_ratio
        self.share_weights = share_weights
        self.outSize = img_size // downsample_ratio

        if share_weights:
            self.convolution = nn.Conv1d(
                in_channels=in_chans,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=3 * self.dilations[0] // 2,
                dilation=self.dilations[0],
            )
        else:
            self.convs = nn.ModuleList()
            for dilation in self.dilations:
                padding = math.ceil(
                    ((kernel_size - 1) * dilation + 1 - self.stride) / 2
                )
                self.convs.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_chans,
                            out_channels=embed_dim,
                            kernel_size=kernel_size,
                            stride=self.stride,
                            padding=padding,
                            dilation=dilation,
                        ),
                        nn.GELU(),
                    )
                )

        if op == "sum":
            self.out_chans = embed_dim
        elif op == "cat":
            self.out_chans = embed_dim * len(self.dilations)
        else:
            raise ValueError(f"Unsupported PRM operation: {op}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.share_weights:
            outputs = []
            for dilation in self.dilations:
                padding = math.ceil(
                    ((self.kernel_size - 1) * dilation + 1 - self.stride) / 2
                )
                outputs.append(
                    nn.functional.conv1d(
                        x,
                        weight=self.convolution.weight,
                        bias=self.convolution.bias,
                        stride=self.downsample_ratio,
                        padding=padding,
                        dilation=dilation,
                    ).unsqueeze(dim=-1)
                )
        else:
            outputs = [conv(x).unsqueeze(dim=-1) for conv in self.convs]

        y = torch.cat(outputs, dim=-1)
        batch_size, channels, width, levels = y.shape
        if self.op == "sum":
            return y.sum(dim=-1).flatten(2).permute(0, 2, 1).contiguous()
        return (
            y.permute(0, 3, 1, 2)
            .flatten(3)
            .reshape(batch_size, levels * channels, width)
            .permute(0, 2, 1)
            .contiguous()
        )


class TokenPerformer(nn.Module):
    """Performer-style attention used inside each SAC."""

    def __init__(
        self,
        dim: int,
        in_dim: int,
        head_cnt: int = 1,
        kernel_ratio: float = 0.5,
        dp1: float = 0.1,
        dp2: float = 0.1,
    ):
        super().__init__()
        self.emb = in_dim * head_cnt
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8
        self.drop_path = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(self.emb, self.emb),
            nn.GELU(),
            nn.Linear(self.emb, self.emb),
            nn.Dropout(dp2),
        )
        self.m = int(self.emb * kernel_ratio)
        self.w = nn.Parameter(
            nn.init.orthogonal_(torch.randn(self.m, self.emb)) * math.sqrt(self.m),
            requires_grad=False,
        )

    def prm_exp(self, x: torch.Tensor) -> torch.Tensor:
        squared_norm = ((x * x).sum(dim=-1, keepdim=True)).repeat(
            1, 1, self.m
        ) / 2
        projection = torch.einsum("bti,mi->btm", x.float(), self.w)
        return torch.exp(projection - squared_norm) / math.sqrt(self.m)

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)
        denominator = torch.einsum(
            "bti,bi->bt", qp, kp.sum(dim=1)
        ).unsqueeze(dim=2)
        kptv = torch.einsum("bin,bim->bnm", v.float(), kp)
        y = torch.einsum("bti,bni->btn", qp, kptv) / (
            denominator.repeat(1, 1, self.emb) + self.epsilon
        )
        return v + self.dp(self.proj(y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# The legacy checkpoint uses this class name in the module hierarchy.
Token_performer = TokenPerformer


class SELayer(nn.Module):
    """Checkpoint-compatible squeeze-and-excitation layer."""

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _ = x.size()
        weights = self.avg_pool(x).view(batch_size, channels)
        weights = self.fc(weights).view(batch_size, channels, 1)
        return x * weights.expand_as(x)


class SAC1(nn.Module):
    """First Spatial Aware Cell: 1,024 input points to 256 tokens."""

    def __init__(self):
        super().__init__()
        self.PRM = PRM(
            img_size=1024,
            kernel_size=7,
            downsample_ratio=4,
            dilations=(1, 2, 3, 4),
            in_chans=1,
            embed_dim=64,
        )
        self.se_layer = SELayer(channel=256)
        self.attn = TokenPerformer(dim=256, in_dim=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn.attn(self.attn.norm1(self.PRM(x)))
        return x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))


class SAC2(nn.Module):
    """Second Spatial Aware Cell: 256 tokens to 128 tokens."""

    def __init__(self):
        super().__init__()
        self.PRM = PRM(
            img_size=1024,
            kernel_size=3,
            downsample_ratio=2,
            dilations=(1, 2, 3),
            in_chans=64,
            embed_dim=64,
        )
        self.attn = TokenPerformer(dim=192, in_dim=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn.attn(self.attn.norm1(self.PRM(x.permute(0, 2, 1))))
        return x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))


class SAC3(nn.Module):
    """Third Spatial Aware Cell: 128 tokens to 64 tokens."""

    def __init__(self):
        super().__init__()
        self.PCM = None
        self.PRM = PRM(
            img_size=1024,
            kernel_size=3,
            downsample_ratio=2,
            dilations=(1, 2),
            in_chans=64,
            embed_dim=160,
        )
        self.attn = TokenPerformer(dim=320, in_dim=320)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn.attn(self.attn.norm1(self.PRM(x.permute(0, 2, 1))))
        return x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))


# Legacy names keep released checkpoint keys unchanged.
BlockRC1 = SAC1
BlockRC2 = SAC2
BlockRC3 = SAC3


class Mlp(nn.Module):
    """Feed-forward sublayer used in each Transformer Cell."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    """Multi-head self-attention for one-dimensional trace tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop_ratio: float = 0.0,
        proj_drop_ratio: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch_size,
                token_count,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ v).transpose(1, 2).reshape(
            batch_size, token_count, channels
        )
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    """Transformer Cell."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio,
        )
        self.drop_path = (
            DropPath(drop_path_ratio)
            if drop_path_ratio > 0.0
            else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """Legacy checkpoint-compatible patch-embedding module."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 33,
        in_c: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = 255
        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=768,
            kernel_size=patch_size,
            stride=4,
            padding=patch_size // 2,
        )
        self.conv1d = nn.Conv1d(257, 255, kernel_size=3, padding=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).transpose(1, 2)
        return self.norm(self.conv1d(x))


class MonoXtract(nn.Module):
    """MonoXtract classifier with three SACs and configurable TC depth."""

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 320,
        tc_depth: int = 7,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        representation_size: Optional[int] = None,
    ):
        super().__init__()
        if tc_depth < 1:
            raise ValueError("tc_depth must be at least 1")
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.tc_depth = tc_depth
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # These checkpoint-compatible parameters are retained even though the
        # released forward path begins directly with the three SACs.
        self.patch_embed = PatchEmbed(
            img_size=224,
            patch_size=16,
            in_c=1,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        decay = torch.linspace(0, drop_path_ratio, tc_depth + 1).tolist()
        transformer_cells = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                drop_path_ratio=decay[index + 1],
                norm_layer=norm_layer,
            )
            for index in range(tc_depth)
        ]
        self.blocks = nn.Sequential(
            SAC1(), SAC2(), SAC3(), *transformer_cells
        )
        self.norm = norm_layer(embed_dim)

        if representation_size is not None:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.blocks(x))
        return self.pre_logits(x[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def build_model(
    num_classes: int = 2,
    tc_depth: int = 7,
    drop_ratio: float = 0.0,
) -> MonoXtract:
    """Construct the released MonoXtract architecture."""
    return MonoXtract(
        num_classes=num_classes,
        tc_depth=tc_depth,
        drop_ratio=drop_ratio,
    )


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Extract a state dictionary from common checkpoint formats."""
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                checkpoint = candidate
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The checkpoint does not contain a state dictionary")

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        clean_key = str(key)
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        state_dict[clean_key] = value
    if not state_dict:
        raise ValueError("No tensor parameters were found in the checkpoint")
    return state_dict


def infer_tc_depth(state_dict: Mapping[str, torch.Tensor]) -> int:
    """Infer the number of trained Transformer Cells from checkpoint keys."""
    block_indices = []
    for key in state_dict:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.append(int(parts[1]))
    if not block_indices:
        raise ValueError("Unable to infer Transformer depth from checkpoint")
    total_blocks = max(block_indices) + 1
    tc_depth = total_blocks - 3
    if tc_depth < 1:
        raise ValueError(f"Invalid checkpoint block count: {total_blocks}")
    return tc_depth


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    strict: bool = True,
) -> MonoXtract:
    """Construct the matching architecture and load a released checkpoint."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=False
    )
    state_dict = extract_state_dict(checkpoint)
    model = build_model(tc_depth=infer_tc_depth(state_dict))
    model.load_state_dict(state_dict, strict=strict)
    return model.to(device)
