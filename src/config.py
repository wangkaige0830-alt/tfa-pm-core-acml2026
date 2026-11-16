"""Architecture settings extracted from the Paper 66 research snapshot."""

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass
class ModelConfig:
    """Defaults retained from the historical model configuration."""

    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 2
    n_heads: int = 4
    patch_candidates: Sequence[int] = (4, 6, 8, 12)
    fixed_patch_len: int = 6
    token_count: int = 24
    dropout: float = 0.1
    use_freq_branch: bool = True
    use_adaptive_patch: bool = True
    use_multi_quantile_risk: bool = True
    decoupled_risk_head: bool = True
    risk_mlp_hidden: int = 64
    fusion_mode: str = "concat"
    fusion_type: str = "concat"  # 兼容旧字段，与 fusion_mode 同步
    use_direction_head: bool = True
    risk_quantiles: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
