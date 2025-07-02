# REBEL

# Sparse Identification of Evolution Equations via Bayesian Model Selection

> **Tim W. Kroll, Oliver Kamps**\
> arXiv:2501.01476 [physics.data-an] 

## Overview

This repository implements the `` class—a self-contained Python module for sparse identification of nonlinear dynamical systems from data. It combines:

- thresholded least-squares (SINDy)
- dual-error criteria (integral‐error + Wasserstein)
- Bayesian hyperparameter search (Hyperopt)
- equation-specific regularization

to infer interpretable differential equations and validate them via numerical integration.

## Features

- **Automatic derivative estimation** (via finite differences)
- **Polynomial candidate library** up to configurable order
- **Mixed error metric**:
  - Mean-squared on time‐derivative (SINDy)
  - Wasserstein distance on trajectory forecasts
- **Model selection** via BIC or RMS criteria
- **Bayesian optimization** of sparsity thresholds + error weights
- **Built-in plotting** of original vs. estimated trajectories

## Repository Structure

```text
.
├── data/                    # sample datasets (e.g. lorenz_data.npy)
├── rebel.py                 # single-file implementation of the rebel class
├── requirements.txt         # Python dependencies
├── examples/                # demonstration notebooks
│   ├── notebook_lorenz.ipynb # Lorenz system
│   └── notebook_cylinder.ipynb # Wake flow behind cylinder
└── README.md                # this file
```

## Installation

```bash
# 1. Clone
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# 2. Create virtual env & install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Quickstart with Lorenz data

```bash
# ensure data file is present
ls data/lorenz_data.npy
```

```python
# run_rebel.py
import numpy as np
import matplotlib.pyplot as plt
from rebel import rebel

# 1) Load data (shape: [n_vars, n_timesteps])
data = np.load("data/lorenz_data.npy")

# 2) Instantiate with your settings
model = rebel(
    data=data,
    stepsize=0.01,    # time‐step for finite differences & solver
    evals=20,         # max Hyperopt evaluations
    error="L2",       # "mixed", "L2", or "W"–controls loss blend
    order=3           # polynomial library up to x^3
)

# 3) Preprocess (compute derivatives, build library)
model.preprocess()

# 4) (Optional) inspect norms
model.calc_norms()

# 5) Run optimization (returns best hyperparams, loss history, errors)
optimum, losses, errors = model.optimize()
print("Best params:", optimum)
print("Loss evolution:", losses)
print("Error breakdown:", errors)

# 6) Plot original vs. estimated trajectories
model.plot_comparison()
plt.show()
```

Execute:

```bash
python run_rebel.py
```

### 2. Example Notebooks

Two interactive Jupyter notebooks demonstrate common use cases:

- **examples/notebook_lorenz.ipynb** — Identification of the Lorenz system
- **examples/notebook_cylinder.ipynb** — Explainable fMRI workflows via differential‐equation extraction

Open them with:

```bash
jupyter notebook examples/
```

### 3. Embedding in Your Workflow

Import and call `rebel` in scripts or notebooks:

```python
from rebel import rebel

# … load or stream your time‐series …
model = rebel(data=my_data, error="mixed", evals=100)
model.preprocess()
opt, _, _ = model.optimize()
estimate = model.estimate   # shape: [n_vars, n_timesteps]
```

## License

MIT License — see `LICENSE` file.

## Citation

If you use this code, please cite:

> Kroll, T. W., & Kamps, O. (2025). *Sparse identification of evolution equations via Bayesian model selection*. arXiv:2501.01476
