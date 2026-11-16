"""TFA-PM architecture extracted from the Paper 66 research snapshot."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
    MAMBA_IMPORT_ERROR = None
except Exception as exc:
    MAMBA_AVAILABLE = False
    MAMBA_IMPORT_ERROR = exc


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MambaWrapper(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float = 0.1) -> None:
        super().__init__()
        if MAMBA_AVAILABLE:
            self.block = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.use_mamba = True
            self.norm = nn.LayerNorm(d_model)
        else:
            raise RuntimeError(
                "mamba_ssm is required for the official Mamba block, but it is not available in "
                "the current Python environment. Install mamba-ssm for this interpreter before "
                "running the paper experiments. Original import error: "
                f"{MAMBA_IMPORT_ERROR}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(self.norm(x))


class AdaptivePatchSelector(nn.Module):
    def __init__(self, c_in: int, patch_candidates: Sequence[int]) -> None:
        super().__init__()
        self.patch_candidates = list(patch_candidates)
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, len(self.patch_candidates)),
        )

    def _stats(self, x: torch.Tensor) -> torch.Tensor:
        # SAPM 频谱统计: 高频能量占比、频谱熵、主导频率、频谱偏度（高低频比）
        # x: [B, L, C]
        fft = torch.fft.rfft(x, dim=1)
        mag = torch.abs(fft) + 1e-8  # [B, F, C]
        energy = mag.pow(2).mean(dim=2)  # [B, F]
        Fbins = energy.size(1)
        if Fbins < 4:
            # 极短序列兜底
            e = energy
            high_ratio = torch.zeros(e.size(0), 1, device=x.device)
            p = e / (e.sum(dim=1, keepdim=True) + 1e-8)
            entropy = -(p * torch.log(p + 1e-8)).sum(dim=1, keepdim=True) / math.log(max(2, Fbins))
            dom_idx = torch.argmax(e, dim=1, keepdim=True).float() / max(1, Fbins - 1)
            skew_ratio = torch.ones(e.size(0), 1, device=x.device)
            return torch.cat([high_ratio, entropy, dom_idx, skew_ratio], dim=1)

        split = max(1, int(0.25 * (Fbins - 1)))
        low_e = energy[:, :split].sum(dim=1, keepdim=True)
        high_e = energy[:, split:].sum(dim=1, keepdim=True)
        high_ratio = high_e / (low_e + high_e + 1e-8)

        p = energy / (energy.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1, keepdim=True) / math.log(Fbins)

        dom_idx = torch.argmax(energy, dim=1, keepdim=True).float() / (Fbins - 1)
        skew_ratio = high_e / (low_e + 1e-8)
        return torch.cat([high_ratio, entropy, dom_idx, skew_ratio], dim=1)

    def forward(self, x: torch.Tensor, train_mode: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self._stats(x)
        logits = self.mlp(s)
        if train_mode:
            weights = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)
        else:
            idx = logits.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(logits).scatter_(1, idx, 1.0)
        return logits, weights


def patchify_3d(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    # x: [B, L, C] -> [B, Np, P, C]
    B, L, C = x.shape
    if L < patch_len:
        pad = patch_len - L
        x = F.pad(x.transpose(1, 2), (0, pad)).transpose(1, 2)
        L = x.shape[1]
    patches = x.unfold(dimension=1, size=patch_len, step=stride)  # [B, Np, C, P]
    patches = patches.permute(0, 1, 3, 2).contiguous()
    return patches


class TFA_PM(nn.Module):
    """Time-Frequency Adaptive Patch-Mamba with multi-task heads.

    Supports:
    - Optional frequency branch (`use_freq_branch`)
    - Optional adaptive patch selection (`use_adaptive_patch`)
    - Directional prediction head (`use_direction_head`)
    - Multi-quantile risk head (`use_multi_quantile_risk`)
    - Decoupled risk features (`decoupled_risk_head`)
    """

    def __init__(self, c_in: int, horizon: int, cfg: ModelConfig, fusion_type: Optional[str] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.horizon = horizon
        _fusion = fusion_type
        if _fusion is None:
            _fusion = getattr(cfg, "fusion_mode", getattr(cfg, "fusion_type", "concat"))
        if _fusion == "cross_attn":
            _fusion = "cross_attention"
        self.fusion_type = _fusion
        self.use_freq_branch = cfg.use_freq_branch
        self.use_adaptive_patch = cfg.use_adaptive_patch
        self.use_direction_head = cfg.use_direction_head
        self.use_multi_quantile_risk = cfg.use_multi_quantile_risk
        self.decoupled_risk_head = cfg.decoupled_risk_head
        self.risk_quantiles = tuple(cfg.risk_quantiles)

        self.patch_candidates = (
            tuple(cfg.patch_candidates) if self.use_adaptive_patch else (int(cfg.fixed_patch_len),)
        )
        self.selector = AdaptivePatchSelector(c_in=c_in, patch_candidates=self.patch_candidates)

        self.time_proj = nn.ModuleDict()
        self.freq_proj = nn.ModuleDict()
        for p in self.patch_candidates:
            k = math.ceil(p / 2) + 1
            self.time_proj[str(p)] = nn.Linear(p * c_in, cfg.d_model)
            self.freq_proj[str(p)] = nn.Linear(k * c_in, cfg.d_model)

        self.pos_t = LearnablePositionalEncoding(cfg.token_count, cfg.d_model)
        self.pos_f = LearnablePositionalEncoding(cfg.token_count, cfg.d_model)

        self.time_blocks = nn.ModuleList(
            [
                MambaWrapper(cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )
        self.freq_blocks = (
            nn.ModuleList(
                [
                    MambaWrapper(cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand, cfg.dropout)
                    for _ in range(cfg.n_layers)
                ]
            )
            if self.use_freq_branch
            else nn.ModuleList([])
        )

        self.cross_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.concat_proj = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.time_only_fuse = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.token_count * cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, horizon),
        )
        self.direction_head = (
            nn.Sequential(
                nn.Linear(cfg.token_count * cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, max(1, horizon - 1)),
            )
            if self.use_direction_head
            else None
        )
        risk_in_dim = cfg.d_model * 2 if self.decoupled_risk_head else cfg.token_count * cfg.d_model
        risk_out_dim = len(self.risk_quantiles) if self.use_multi_quantile_risk else 2
        risk_hidden = int(getattr(cfg, "risk_mlp_hidden", 64))
        self.risk_head = nn.Sequential(
            nn.Linear(risk_in_dim, risk_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(risk_hidden, risk_out_dim),
        )


    @staticmethod
    def _monotone_quantiles(raw: torch.Tensor) -> torch.Tensor:
        """保证 q_0.05 < q_0.25 < ...，避免插值 VaR 失真。"""
        q0 = F.softplus(raw[:, :1]) + 1e-8
        deltas = F.softplus(raw[:, 1:]) + 1e-8
        if deltas.size(1) == 0:
            return q0
        return torch.cat([q0, q0 + torch.cumsum(deltas, dim=1)], dim=1)

    def _resample_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # [B, Np, d] -> [B, token_count, d]
        if x.size(1) == self.cfg.token_count:
            return x
        x_t = x.transpose(1, 2)
        x_t = F.interpolate(x_t, size=self.cfg.token_count, mode="linear", align_corners=False)
        return x_t.transpose(1, 2)

    def _encode_candidate(self, x: torch.Tensor, p: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        stride = max(1, p // 2)
        patches = patchify_3d(x, patch_len=p, stride=stride)  # [B, Np, P, C]
        B, Np, P, C = patches.shape

        time_in = patches.reshape(B, Np, P * C)
        t = self.time_proj[str(p)](time_in)

        f: Optional[torch.Tensor] = None
        if self.use_freq_branch:
            fft = torch.fft.rfft(patches, dim=2)
            mag = torch.abs(fft)
            freq_in = mag.reshape(B, Np, -1)
            f = self.freq_proj[str(p)](freq_in)

        t = self._resample_tokens(t)
        if f is not None:
            f = self._resample_tokens(f)
        return t, f

    def forward(self, x: torch.Tensor, return_aux: bool = False, train_mode: bool = True) -> Dict[str, torch.Tensor]:
        logits, weights = self.selector(x, train_mode=train_mode)
        t_list, f_list = [], []
        for p in self.patch_candidates:
            t, f = self._encode_candidate(x, p)
            t_list.append(t)
            if f is not None:
                f_list.append(f)
        t_stack = torch.stack(t_list, dim=1)  # [B, K, Tk, d]
        f_stack = torch.stack(f_list, dim=1) if self.use_freq_branch else None
        w = weights.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        t = (t_stack * w).sum(dim=1)
        f = (f_stack * w).sum(dim=1) if f_stack is not None else None

        t = self.pos_t(t)
        for blk in self.time_blocks:
            t = blk(t)
        if self.use_freq_branch and f is not None:
            f = self.pos_f(f)
            for blk in self.freq_blocks:
                f = blk(f)

        if not self.use_freq_branch or f is None:
            z = self.norm(self.time_only_fuse(t))
            attn_w = torch.empty(0, device=x.device)
        else:
            if self.fusion_type == "concat":
                z = self.norm(self.concat_proj(torch.cat([t, f], dim=-1)))
                attn_w = torch.empty(0, device=x.device)
            else:
                z_attn, attn_w = self.cross_attn(query=t, key=f, value=f, need_weights=True)
                z = self.norm(t + z_attn)
        z_flat = z.reshape(z.size(0), -1)
        pred_resid = self.decoder(z_flat)

        if self.decoupled_risk_head:
            t_pool = t.mean(dim=1)
            f_pool = f.mean(dim=1) if f is not None else torch.zeros_like(t_pool)
            risk_feat = torch.cat([t_pool, f_pool], dim=-1)
        else:
            risk_feat = z_flat
        risk_out = self.risk_head(risk_feat)
        if self.use_multi_quantile_risk:
            risk_out = self._monotone_quantiles(risk_out)

        result: Dict[str, torch.Tensor] = {"pred_resid": pred_resid}
        if self.use_multi_quantile_risk:
            result["risk_quantiles"] = risk_out
        else:
            result["var_pred"] = risk_out[:, :1]
            result["es_pred"] = risk_out[:, 1:2]
        if self.use_direction_head and self.direction_head is not None:
            direction_logits = self.direction_head(z_flat)
            result["direction_logits"] = direction_logits
            result["direction_prob"] = torch.sigmoid(direction_logits)
        if return_aux:
            result.update({"patch_logits": logits, "patch_weights": weights, "attn": attn_w, "z": z})
        return result
