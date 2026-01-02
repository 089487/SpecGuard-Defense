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

## Acknowledgements

This code is based on the implementation from [PoisonedFL](https://github.com/xyq7/PoisonedFL/tree/main). We thank the original authors for their work.
