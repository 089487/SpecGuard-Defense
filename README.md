# SpecGuard-Defense

This repository contains the official PyTorch implementation of **SpecGuard** and **SpecGuard2**, novel defense mechanisms against model poisoning attacks in Federated Learning.

## Overview

**PoisonedFL** is a state-of-the-art model poisoning attack that leverages multi-round consistency to bypass existing defenses. This repository provides the implementation of our proposed defenses, **SpecGuard** and **SpecGuard2**, which are designed to robustly defend against various attacks, **including PoisonedFL**.

The code is adapted from the original [PoisonedFL](https://github.com/xyq7/PoisonedFL/tree/main) repository (originally in MXNet), ported to PyTorch, and extended with our defense algorithms.

Key files include:
- `test_agr_pytorch.py`: Main script for running experiments.
- `byzantine_pytorch.py`: Implementation of byzantine attacks.
- `nd_aggregation_pytorch.py`: Implementation of aggregation rules/defenses.
- `utils_pytorch.py`: Utility functions.

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r pytorch_requirements.txt
```

## Usage

To run an experiment, use the `test_agr_pytorch.py` script.

### Example

```bash
python test_agr_pytorch.py --dataset FashionMNIST --gpu 0 --net cnn --niter 200 --nworkers 1200 --nfake 100 --aggregation fltrust --byz_type no
```

### Arguments

- `--dataset`: Dataset to use (e.g., FashionMNIST).
- `--net`: Network architecture (e.g., cnn).
- `--nworkers`: Total number of workers.
- `--nfake`: Number of malicious workers (fake clients).
- `--byz_type`: Type of byzantine attack (e.g., no, poisonedfl).
- `--aggregation`: Aggregation rule to use (e.g., fltrust, median).
- `--gpu`: GPU index to use.

## Results

We evaluate our defenses on the MNIST dataset against various attacks. The table below shows the test accuracy (%) on **MNIST**.

| Defense | No Attack | Fang | Opt. Fang | LIE | Min-Max | Min-Sum | Random | MPAF | PoisonedFL |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | **98.12** | 94.24 | 96.01 | 64.57 | 70.58 | 70.74 | **97.98** | 9.80 | 9.80 |
| Multi-Krum | 97.85 | 97.86 | 97.88 | 98.10 | 94.66 | 95.13 | 97.73 | 90.80 | 96.77 |
| Median | 94.67 | 94.00 | 94.11 | 95.49 | 93.64 | 93.79 | 96.82 | 90.19 | 10.47 |
| TrMean | 97.12 | 92.81 | 94.34 | 90.35 | 93.58 | 93.24 | 97.27 | 87.07 | 10.46 |
| Norm Bound | 97.85 | 96.21 | 96.07 | 95.70 | 94.49 | 94.88 | 97.81 | 41.20 | 92.19 |
| FLTrust | 97.40 | 93.53 | 96.11 | 97.49 | 83.54 | 82.92 | 97.43 | 97.24 | 10.15 |
| **SpecGuard** | 97.67 | 97.23 | 97.36 | 98.02 | 95.72 | 94.74 | 97.88 | 97.87 | **97.72** |
| **SpecGuard2** | 97.85 | **98.06** | **98.22** | **98.13** | **98.02** | **98.09** | 97.69 | **97.90** | 96.26 |

## Acknowledgements

This code is based on the implementation from [PoisonedFL](https://github.com/xyq7/PoisonedFL/tree/main). We thank the original authors for their work.
