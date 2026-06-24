# CVAE-kMC: Trajectory Generation and Scaling

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation accompanying the paper *"Accelerating and Scaling On-Lattice Solid-on-Solid Kinetic Monte Carlo Simulations Using an Autoregressive Conditional Variational Autoencoder"*. The project investigates the generation of surface growth trajectories using a Conditional Variational Autoencoder (CVAE) and compares its performance against conventional kinetic Monte Carlo (kMC) simulations.

The proposed framework aims to generate physically consistent trajectories while significantly reducing computational costs relative to traditional kMC approaches.

## Dependencies

The project requires the following Python packages:

* accelerate
* matplotlib
* numba
* numpy
* scipy
* torch
* tqdm

All dependencies can be installed automatically using the provided Python package configuration.

## Repository Structure

The repository is organized as follows:

```text
cvae/          CVAE architecture, data loaders, trained models and training scripts 
python_kmc/    Ground Truth kinetic Monte Carlo model
examples/      Example scripts with parameters used for data generation or training
data/          Scripts for data generation and pathway for datasets
metrics/       Scripts to calculate metrics from lattices
```

## Installation

The project uses a `pyproject.toml` based setup and supports installation via `uv` or `pip`.

### Clone the repository

```bash
git clone https://gitlab.com/MBM-UBT/cvae-kmc.git
cd cvae-kmc
```

### Create the environment and install dependencies

Using `uv` (recommended):

```bash
uv venv
uv pip install -e .
```

Using `pip`:

```bash
pip install -e .
```

## Usage

The most important scripts are listed below:

| Script                                    | Description                                                          |
| ----------------------------------------- | -------------------------------------------------------------------- |
| `cvae/training/training.py`               | Train the CVAE model                                                 |
| `cvae/model/architecture.py`              | CVAE Architecture                                                    |
| `data/data_generation/training_data.py`   | Script for creating training dataset                                 |
| `data/data_generation/kmc_trajectories.py`| Script for creating validation data and for ground truth comparison  |
| `python_kmc/dendrite_models.py`           | kMC Model used as ground truth                                       |

In the examples folder, you can see how to use these scripts.

## Reproducibility

To reproduce the results reported in the paper:

1. Generate the training dataset.
2. Train the CVAE model using the provided training scripts.
3. Generate trajectories from the trained model (ml_trajectories.py).
4. Compare generated trajectories against kMC reference simulations.

You can also find the trained models from the paper inside `cvae/trained_models`.

## Support and Contact
If you encounter any issues with the code or want to report a bug, please open an issue on GitLab. For other inquiries, contact Luis Henkelmann at luis.henkelmann@uni-bayreuth.de.

[comment]: ## Citation

[comment]: If you use this code or the generated models in your research, please cite:

[comment]:```bibtex
@article{henkelmann2026cvae,
  title   = {Your Paper Title},
  author  = {Henkelmann, Luis and and Margraf, Johannes and Röder, Fridolin},
  journal = {Journal Name},
  year    = {2026}
}
```