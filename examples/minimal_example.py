"""Inspect TFA-PM components or run a synthetic Mamba forward pass."""

import argparse

import torch

from src.config import ModelConfig
from src.tfa_pm import AdaptivePatchSelector, MAMBA_AVAILABLE, MAMBA_IMPORT_ERROR, TFA_PM, patchify_3d


def check_components() -> None:
    features = torch.linspace(-1, 1, 2 * 96 * 8).reshape(2, 96, 8)
    selector = AdaptivePatchSelector(c_in=8, patch_candidates=ModelConfig().patch_candidates).eval()
    logits, weights = selector(features, train_mode=False)
    assert logits.shape == weights.shape == (2, 4)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    assert ((weights == 0) | (weights == 1)).all()

    patches = patchify_3d(features, patch_len=6, stride=3)
    assert patches.shape == (2, 31, 6, 8)
    torch.testing.assert_close(patches[:, 1], features[:, 3:9])

    raw_scores = torch.tensor([[-2.0, 0.0, 1.0, -1.0, 0.5], [0.0] * 5])
    quantiles = TFA_PM._monotone_quantiles(raw_scores)
    assert quantiles.shape == (2, 5) and torch.isfinite(quantiles).all()
    assert (quantiles.diff(dim=1) >= 0).all()
    print("CPU component checks passed: spectral selection, patch extraction, and noncrossing quantiles.")


def check_forward() -> None:
    if not MAMBA_AVAILABLE:
        raise SystemExit(f"Full forward requires a compatible mamba-ssm build: {MAMBA_IMPORT_ERROR}")
    if not torch.cuda.is_available():
        raise SystemExit("This full-forward example requires CUDA.")

    features = torch.randn(2, 96, 8, device="cuda")
    model = TFA_PM(c_in=8, horizon=24, cfg=ModelConfig()).to("cuda").eval()
    outputs = model(features, train_mode=False)
    residuals = outputs["pred_resid"]
    quantiles = outputs["risk_quantiles"]
    assert residuals.shape == (2, 24) and torch.isfinite(residuals).all()
    assert quantiles.shape == (2, 5) and torch.isfinite(quantiles).all()
    assert (quantiles.diff(dim=1) >= 0).all()
    print(f"Full forward passed: residuals {tuple(residuals.shape)}, quantiles {tuple(quantiles.shape)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", action="store_true", help="Also run the full Mamba model on CUDA.")
    args = parser.parse_args()
    torch.manual_seed(66)
    with torch.inference_mode():
        check_components()
        if args.forward:
            check_forward()


if __name__ == "__main__":
    main()
