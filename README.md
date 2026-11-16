# TFA-PM core model

**A Time-Frequency Adaptive Patch-Mamba Framework for Accurate and Risk-Aware Cryptocurrency Volatility Forecasting**  
Kaige Wang · ACML 2026 Conference Track · Paper 66  
[Paper PDF on Google Drive](https://drive.google.com/file/d/178iM0bTSnsXvaTkHD14W5AfrnuXYzivZ/view?usp=sharing)

[The core model](src/tfa_pm.py) computes local spectral statistics and uses them to select a patch scale. Parallel Mamba encoders process temporal patches and their Fourier magnitudes. A fusion layer feeds the multistep realised-volatility (RV) residual decoder. A separate head uses pooled encoded states to produce noncrossing quantiles of the future RV path. These quantiles describe realised volatility and carry no conventional financial VaR or ES interpretation.

## Try the components

From the repository root, in a Python environment with PyTorch installed:

```sh
python -m pip install -r requirements.txt
python -B -m examples.minimal_example
```

The example checks spectral patch selection, patch extraction and the ordered quantile transform on synthetic CPU inputs. These checks were run with Python 3.12.10 and PyTorch 2.5.1+cu121. The installation command has not been tested in a clean environment.

The complete network calls [Mamba](https://github.com/state-spaces/mamba) through an external `mamba-ssm` dependency. For the optional CUDA example, first install a compatible build following the [upstream installation instructions](https://github.com/state-spaces/mamba/tree/v2.2.2#installation), then run:

```sh
python -B -m examples.minimal_example --forward
```

The complete Mamba forward pass remains unverified in the checked environment. The example uses untrained weights and arbitrary synthetic inputs. Its expected shapes are `[batch, horizon]` for `pred_resid` and `[batch, quantile levels]` for `risk_quantiles`. The latter has five columns under the [default configuration](src/config.py). The model expects a prepared feature tensor of shape `[batch, lookback, channels]`. Reconstructing an absolute RV forecast requires an independently prepared reference and the matching residual scale.

## Source and scope

The architecture was extracted from Paper 66 research snapshot `3f751bb8f98348a6650055181e5cc92abc470974`. [The model file](src/tfa_pm.py) preserves the original forward methods and layer definitions. The training-phase helper and train-only loss-balancer registration were omitted. No upstream Mamba implementation is included in this extraction.

This architecture snapshot is for inspection. It excludes the historical HAR estimator and evaluation pipeline. The known HAR time-alignment and look-ahead issues remain unresolved for that historical evaluation. The code does not reproduce the paper's experimental results.
