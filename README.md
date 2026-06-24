# CVAE-kMC: Trajectory Generation and Scaling

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains the official implementation for the manuscript *"Accelerating and Scaling On-Lattice Solid-on-Solid Kinetic Monte Carlo Simulations Using an Autoregressive Conditional Variational Autoencoder"*. The project provides a framework to generate stochastic surface growth trajectories using a Conditional Variational Autoencoder (CVAE). This machine learning surrogate aims to maintain physical consistency while significantly accelerating the computational performance compared to the traditional kinetic Monte Carlo (kMC) ground truth.

## Dependencies
The computational environment requires Python 3.12 or higher. The primary dependencies include `accelerate`, `matplotlib`, `numba`, `numpy`, `scipy`, `torch`, and `tqdm`. These packages can be installed automatically via the provided project configuration.

## Repository Structure
The codebase is logically partitioned into several distinct directories to separate the generative model from the physical simulations. The `cvae/` directory contains the neural network architecture, data loaders, training scripts, and the pre-trained models. The physical ground truth 1D+1 Solid-on-Solid kMC model is housed in `python_kmc/`. Scripts for dataset generation and the pathways for data storage are located in `data/`, while the `metrics/` directory provides utilities to calculate physical observables from the generated lattice states. Finally, the `examples/` directory offers reference scripts detailing the parameters utilized for data generation and model training.

## Installation
The project utilizes a `pyproject.toml` configuration. Installation is supported via `uv` (recommended) or the standard `pip` package manager.

Clone the repository to your local machine:
```bash
git clone [https://gitlab.com/MBM-UBT/cvae-kmc.git](https://gitlab.com/MBM-UBT/cvae-kmc.git)
cd cvae-kmc
```

Create the virtual environment and install the dependencies using `uv`:
```bash
uv venv
uv pip install -e .
```

Alternatively, proceed using `pip`:
```bash
pip install -e .
```

## Usage
The execution of the framework relies on several core scripts. The table below outlines the primary modules required for training and simulation:

| Script | Description |
| :--- | :--- |
| `cvae/model/architecture.py` | Defines the fully convolutional CVAE architecture. |
| `cvae/training/training.py` | Executes the model training process. |
| `data/data_generation/training_data.py` | Generates the temporal lattice transitions for the training dataset. |
| `data/data_generation/kmc_trajectories.py` | Generates the validation data and ground truth reference trajectories. |
| `python_kmc/dendrite_models.py` | Executes the underlying kMC model simulations. |

Reference configurations demonstrating the application of these scripts are provided within the `examples/` directory.

## Data
The comprehensive training and validation datasets required to execute the models are publicly accessible at [https://doi.org/10.5281/zenodo.20826635](https://doi.org/10.5281/zenodo.20826635). 

Pre-trained CVAE models corresponding to the physical results discussed in the manuscript are provided directly within the `cvae/trained_models` directory. To generate new morphological trajectories using these models, execute the `ml_trajectories.py` script. The structural outputs can subsequently be quantitatively compared against the reference simulations generated via the `kmc_trajectories.py` script.

## Support and Contact
For technical issues or bug reports, please open an issue directly on the GitLab repository. For academic inquiries or further project-related questions, contact Luis Henkelmann at luis.henkelmann@uni-bayreuth.de.

## Citation
If you utilize this framework or the associated datasets in your research, please cite the software using the following metadata:

```bibtex
@software{luis_henkelmann_2026_20828357,
  author       = {Luis Henkelmann and
                  Margraf, Johannes T. and
                  Röder, Fridolin},
  title        = {MBM-UBT/cvae\_kmc: Software for: Accelerating and
                   Scaling On-Lattice Solid-on-Solid kMC Simulations
                   using an Autoregressive CVAE
                  },
  month        = jun,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.2},
  doi          = {10.5281/zenodo.20828357},
  url          = {https://doi.org/10.5281/zenodo.20828357},
}
```