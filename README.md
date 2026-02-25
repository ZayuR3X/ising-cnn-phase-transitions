# Ising-CNN Phase Transitions

A computational study of phase transitions in the two-dimensional Ising model using
Cellular Automaton simulations and Convolutional Neural Networks, with a transfer
learning protocol for critical temperature prediction under modified coupling constants.

---

## Overview

This repository implements a full machine learning pipeline for detecting and
characterizing phase transitions in the 2D Ising model. The project is organized
around three interconnected components: (1) a high-fidelity Monte Carlo dataset
generated via a Metropolis-Hastings Cellular Automaton, (2) a physics-informed CNN
trained to classify ferromagnetic and paramagnetic phases from raw spin configurations,
and (3) a transfer learning protocol based on the *learning by confusion* method
(van Nieuwenburg et al., 2017) that predicts the critical temperature T_c for a new
coupling constant J=2 without retraining the backbone network.

The study reproduces the Onsager exact result T_c(J=1) = 2J / ln(1 + sqrt(2)) ~ 2.269
and demonstrates generalization to T_c(J=2) ~ 4.538, leveraging the universality of
critical phenomena across the 2D Ising universality class.

---

## Table of Contents

- [Theoretical Background](#theoretical-background)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Model Architecture](#model-architecture)
- [Transfer Learning Protocol](#transfer-learning-protocol)
- [Reproducing Results](#reproducing-results)
- [References](#references)
- [License](#license)

---

## Theoretical Background

### The 2D Ising Model

The Hamiltonian of the system on an L x L square lattice with periodic boundary
conditions is:

```
H = -J * sum_{<i,j>} s_i * s_j
```

where s_i in {-1, +1} are spin variables, J is the exchange coupling constant, and
the sum runs over nearest-neighbor pairs. In the absence of an external field (h = 0),
the model undergoes a continuous second-order phase transition at the Onsager critical
temperature:

```
k_B * T_c = 2J / ln(1 + sqrt(2))
```

For J=1 this gives T_c ~ 2.269; for J=2, T_c ~ 4.538. The two systems belong to the
same universality class — they share identical critical exponents — and differ only
in energy scale. This property is the theoretical foundation for the transfer learning
approach used in this work.

### Critical Exponents (2D Ising Universality Class)

| Exponent       | Symbol | Value    | Observable                         |
|----------------|--------|----------|------------------------------------|
| Magnetization  | beta   | 1/8      | m ~ |t|^beta                       |
| Correlation length | nu | 1        | xi ~ |t|^{-nu}                     |
| Susceptibility | gamma  | 7/4      | chi ~ |t|^{-gamma}                 |
| Specific heat  | alpha  | 0 (log)  | C ~ ln|t|                          |
| Anomalous dim. | eta    | 1/4      | G(r) ~ r^{-(d-2+eta)}              |

where the reduced temperature t = (T - T_c) / T_c.

### Finite-Size Scaling

For a finite lattice of linear size L, the pseudo-critical temperature shifts as:

```
T_c(L) = T_c(inf) + a * L^{-1/nu}
```

All CNN phase labels are assigned using FSS-corrected boundaries. Critical temperature
estimates from multiple lattice sizes are extrapolated to the thermodynamic limit via
this ansatz with nu = 1.

### Cellular Automaton as a Monte Carlo Engine

The canonical ensemble is realized through the Metropolis-Hastings algorithm expressed
as a CA update rule. For each spin s_i, the local energy change upon flipping is:

```
Delta_E = 2J * s_i * sum_{nn} s_j
```

The flip is accepted with probability min(1, exp(-Delta_E / k_B T)). A checkerboard
(bipartite sublattice) update schedule is used to satisfy detailed balance while
maintaining the CA structure.

Near T_c, the system exhibits critical slowing down characterized by a diverging
integrated autocorrelation time:

```
tau_int ~ xi^z,    z ~ 2.17   (Metropolis dynamical exponent)
```

Configurations are sampled every 2 * tau_int sweeps after thermalization to ensure
statistical independence. The Madras-Sokal windowing estimator is used to compute
tau_int from the magnetization time series.

---

## Repository Structure

```
ising-cnn-phase-transitions/
|
|-- README.md
|-- THEORY.md                  # Extended derivations and scaling law discussion
|-- requirements.txt
|-- setup.py
|-- .gitignore
|
|-- configs/
|   |-- simulation.yaml        # Lattice size, temperature range, MC parameters
|   |-- model.yaml             # CNN architecture and training hyperparameters
|   `-- transfer.yaml          # Learning-by-confusion sweep configuration
|
|-- ising/                     # Main Python package
|   |-- simulation/
|   |   |-- ca_ising.py        # Metropolis CA core (Numba JIT)
|   |   |-- thermalization.py  # Burn-in detection and convergence monitoring
|   |   |-- autocorrelation.py # Integrated autocorrelation time estimation
|   |   `-- observables.py     # m, chi, E, C, U4, xi computation
|   |
|   |-- data/
|   |   |-- generator.py       # Simulation-to-HDF5 pipeline orchestrator
|   |   |-- augmentation.py    # Symmetry-based data augmentation
|   |   |-- dataset.py         # PyTorch Dataset and DataLoader factories
|   |   `-- labels.py          # FSS-corrected phase label assignment
|   |
|   |-- models/
|   |   |-- cnn.py             # IsingCNN architecture with circular padding
|   |   |-- heads.py           # Phase classifier and temperature regressor heads
|   |   `-- losses.py          # Composite loss with physics regularization term
|   |
|   |-- training/
|   |   |-- trainer.py         # Training loop, checkpointing, early stopping
|   |   |-- metrics.py         # Phase accuracy, T_pred RMSE, Binder cross-val
|   |   `-- callbacks.py       # LR scheduling and logging hooks
|   |
|   |-- transfer/
|   |   |-- confusion.py       # Learning-by-confusion W-sweep
|   |   |-- mc_dropout.py      # Predictive entropy via MC Dropout
|   |   `-- fss.py             # Finite-size scaling extrapolation to L -> inf
|   |
|   `-- visualization/
|       |-- spin_configs.py    # Spin configuration rendering
|       |-- phase_diagram.py   # Order parameter and susceptibility curves
|       |-- confusion_plot.py  # W vs. accuracy confusion curve
|       |-- gradcam.py         # Grad-CAM spatial attention maps
|       `-- scaling_plots.py   # FSS data collapse and exponent fits
|
|-- scripts/
|   |-- generate_dataset.py    # Entry point: dataset generation
|   |-- train_model.py         # Entry point: CNN training
|   |-- predict_tc.py          # Entry point: T_c prediction for J=2
|   `-- visualize_results.py   # Entry point: figure generation
|
|-- notebooks/
|   |-- 01_simulation_sanity_check.ipynb
|   |-- 02_dataset_exploration.ipynb
|   |-- 03_model_training_analysis.ipynb
|   |-- 04_gradcam_interpretability.ipynb
|   `-- 05_transfer_learning_tc.ipynb
|
|-- tests/
|   |-- test_simulation.py
|   |-- test_dataset.py
|   |-- test_model.py
|   `-- test_transfer.py
|
`-- data/                      # Git-ignored; populated by generate_dataset.py
    |-- raw/
    `-- processed/
```

---

## Installation

**Requirements:** Python >= 3.10, CUDA-capable GPU recommended for training.

```bash
git clone https://github.com/ZayuR3X/ising-cnn-phase-transitions
cd ising-cnn-phase-transitions

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

Core dependencies (see `requirements.txt` for pinned versions):

- numpy, scipy, numba — simulation and numerical methods
- torch, torchvision — model training
- h5py — HDF5 dataset storage
- matplotlib, seaborn — visualization
- pyyaml — configuration management
- pytest — test suite

---

## Quick Start

The full pipeline runs in three steps:

**1. Generate dataset (J=1, L=64)**
```bash
python scripts/generate_dataset.py --config configs/simulation.yaml
```

**2. Train the CNN**
```bash
python scripts/train_model.py --config configs/model.yaml
```

**3. Predict T_c for J=2 via learning by confusion**
```bash
python scripts/predict_tc.py --J 2 --method confusion --config configs/transfer.yaml
```

All figures are generated with:
```bash
python scripts/visualize_results.py --fig all
```

---

## Pipeline Overview

### Stage 1 — Dataset Generation

Spin configurations are generated by the Metropolis CA for temperatures
T in [1.0, 3.5] (coarse grid, step 0.05) and T in [2.0, 2.6] (dense grid,
step 0.01) for lattice sizes L in {32, 64, 128}.

Each configuration is stored alongside a label vector containing: temperature T,
absolute magnetization |m|, magnetic susceptibility chi, internal energy per spin E,
Binder cumulant U4, and an FSS-corrected binary phase label.

Near T_c (|T - T_c| < 0.3), thermalization is extended to 50,000 sweeps and
configurations are spaced by 2 * tau_int sweeps to avoid autocorrelation bias.
Symmetry augmentation (rotations, reflections, global spin flip) expands the
effective dataset by up to 8x.

### Stage 2 — CNN Training

The IsingCNN takes a 4-channel L x L tensor as input (raw spins, local energy
density, coarse-grained magnetization, and a Fourier amplitude map). Circular
(toroidal) padding is used in all convolutional layers to respect periodic boundary
conditions.

The network is trained with a composite loss combining cross-entropy on the binary
phase label, mean-squared error on temperature regression, and a physics regularization
term that penalizes nonzero predicted magnetization above T_c.

### Stage 3 — Transfer Learning

The trained backbone is frozen and applied to a small J=2 pilot dataset (~50
configurations per temperature point). The learning-by-confusion method sweeps a
candidate critical temperature W and re-trains only the classification head for each
value of W. The accuracy curve A(W) peaks at the true T_c, providing a data-efficient
estimate without full retraining.

As a complementary zero-shot method, MC Dropout predictive entropy is computed by
running 100 stochastic forward passes per configuration. The entropy H[p] peaks
sharply at T_c with no additional training required.

FSS extrapolation across L in {32, 64, 128} then yields T_c(inf) for comparison
against the Onsager exact value.

---

## Model Architecture

```
Input: (B, 4, L, L)  [raw spins, local energy, coarse mag, Fourier map]

Block 1 — Local pattern detection
  Conv2d(4 -> 32, 3x3, circular padding) -> BN -> GELU
  Conv2d(32 -> 32, 3x3, circular padding) -> BN -> GELU -> MaxPool

Block 2 — Mesoscale cluster geometry
  Conv2d(32 -> 64, 3x3) -> BN -> GELU
  Conv2d(64 -> 64, 5x5, circular padding) -> BN -> GELU -> MaxPool

Block 3 — Long-range correlations
  Conv2d(64 -> 128, 3x3) -> BN -> GELU
  Conv2d(128 -> 128, 3x3) -> BN -> GELU -> AdaptiveAvgPool(4x4)

Global feature injection
  Flatten -> Concat([features, m, chi, E, U4])

Dual output head
  Linear(2052 -> 256) -> GELU -> Dropout(0.3)
  Linear(256 -> 64)   -> GELU -> Dropout(0.2)
  Linear(64 -> 2)     -> [phase_logit, T_pred]

Total parameters: ~1.2M
```

Grad-CAM analysis of the trained network shows that attention near T_c concentrates
on percolating cluster boundaries — the fractal geometric structures that define
criticality — validating that the model has encoded physically meaningful
representations rather than spurious correlations.

---

## Transfer Learning Protocol

The *learning by confusion* method (van Nieuwenburg et al., 2017) identifies T_c
without prior knowledge of the phase boundary by exploiting the following property:
a binary classifier trained with an artificially imposed boundary at W achieves
maximum accuracy when W coincides with the true T_c. At any other value of W,
mislabeled configurations near the true boundary degrade performance.

The confusion curve A(W) exhibits a characteristic shape with a single sharp maximum
at W = T_c. The width of this peak provides an uncertainty estimate on the T_c
prediction.

| Method                   | J=2 data required        | Expected error  |
|--------------------------|--------------------------|-----------------|
| Learning by confusion    | ~50 configs/T, head only | +/- 0.05        |
| MC Dropout entropy       | 0 (zero-shot)            | +/- 0.10        |
| Binder cumulant (physics)| ~500 configs/T, 3 sizes  | +/- 0.01        |
| Full retraining          | ~500 configs/T           | +/- 0.02        |

The transfer learning approach achieves near-benchmark accuracy at a fraction of the
simulation cost, demonstrating that universality-class knowledge encoded in the CNN
backbone is transferable across coupling constants.

---

## Reproducing Results

All experiments were run on a single NVIDIA GPU (8 GB VRAM). Approximate runtimes:

| Stage                        | Time        |
|------------------------------|-------------|
| Dataset generation (J=1, L=64) | ~2 hours  |
| CNN training (100 epochs)    | ~45 minutes |
| Confusion sweep (J=2)        | ~20 minutes |
| All figures                  | ~5 minutes  |

Random seeds are fixed in all configuration files. To reproduce the main result:

```bash
python scripts/generate_dataset.py --config configs/simulation.yaml --seed 42
python scripts/train_model.py --config configs/model.yaml --seed 42
python scripts/predict_tc.py --J 2 --method confusion --config configs/transfer.yaml
python scripts/predict_tc.py --J 2 --method entropy --config configs/transfer.yaml
python scripts/visualize_results.py --fig all
```

Expected output: T_c(J=2) prediction in [4.50, 4.58] (Onsager exact: 4.5138...).

---

## References

1. Onsager, L. (1944). Crystal statistics. I. A two-dimensional model with an
   order-disorder transition. *Physical Review*, 65(3-4), 117-149.

2. Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E.
   (1953). Equation of state calculations by fast computing machines.
   *Journal of Chemical Physics*, 21(6), 1087-1092.

3. Carrasquilla, J., & Melko, R. G. (2017). Machine learning phases of matter.
   *Nature Physics*, 13(5), 431-434.

4. van Nieuwenburg, E. P. L., Liu, Y. H., & Huber, S. D. (2017). Learning phase
   transitions by confusion. *Nature Physics*, 13(5), 435-439.

5. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing
   model uncertainty in deep learning. *Proceedings of ICML 2016*.

6. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks
   via gradient-based localization. *Proceedings of ICCV 2017*.

7. Binder, K. (1981). Finite size scaling analysis of Ising model block distribution
   functions. *Zeitschrift fur Physik B*, 43(2), 119-140.

8. Madras, N., & Sokal, A. D. (1988). The pivot algorithm: A highly efficient Monte
   Carlo method for the self-avoiding walk. *Journal of Statistical Physics*, 50, 109-186.

---

## License

MIT License. See `LICENSE` for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ising-cnn-phase-transitions,
  author       = {Ahmet KILIÇ},
  title        = {Phase Transitions in the 2D Ising Model via CNN and Transfer Learning},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/ZayuR3X/ising-cnn-phase-transitions}
}
```
