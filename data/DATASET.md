# Dataset: CICIoT2023

This document describes the dataset setup used in the DPF-IDS experiments. The experiments are based on CICIoT2023 and are organized around two collaborator scales: 200 collaborators and 500 collaborators.

## Overview

CICIoT2023 is a network security dataset for IoT environments. It contains benign traffic and multiple attack categories. In this framework, the dataset is used for distributed intrusion detection, where each collaborator/client owns a local data partition and trains an FFNN model.

Each sample contains:

- Numeric network-flow features.
- A `Label` column representing the benign class or an attack class.

## Preprocessing

The preprocessing pipeline includes:

- Removing or handling rows with `NaN`, `Inf`, or `-Inf` values.
- Scaling numeric features with Min-Max normalization.
- Splitting the dataset into train and test sets.
- Converting `.csv` files into `.pt` files for efficient PyTorch loading.
- Partitioning the training set into multiple collaborator/client files.

The related modules are:

- `src/data/preprocessing.py`
- `src/data/partition.py`
- `src/data/dataset.py`

## 200-Collaborator Setup

In the 200-collaborator setup, the training data is divided into 200 local partitions. Each collaborator corresponds to one client in the distributed training process.

This setup is used to evaluate the framework at a medium collaboration scale, including:

- Prototype representation learning across clients.
- Stability of topology clustering.
- Effectiveness of poisoned-client detection.
- The impact of attacks such as Label Flipping, PGA Poisoning, and BoTPA Poisoning.

Expected data format:

```text
client_001.pt
client_002.pt
...
client_200.pt
test.pt
```

## 500-Collaborator Setup

In the 500-collaborator setup, the training data is divided into 500 local partitions. This larger-scale setting is used to evaluate the scalability of the framework.

This setup focuses on:

- Scalability as the number of clients increases.
- Training and evaluation cost across communication rounds.
- Robustness of prototype-only detection.
- The effect of Non-IID data distributions across many collaborators.

Expected data format:

```text
client_001.pt
client_002.pt
...
client_500.pt
test.pt
```

## Data Distribution

The framework supports two main data partitioning strategies:

- IID/random split: the data is shuffled and split approximately evenly across collaborators.
- Non-IID split: the data is partitioned with a Dirichlet distribution to simulate realistic settings where each collaborator has a different label distribution.

The Non-IID setup is especially important for distributed IDS experiments because network traffic observed by different devices, gateways, or organizations is usually heterogeneous.

## Role In The Framework

After partitioning, CICIoT2023 is used throughout the DPF-IDS pipeline:

- Local model training at each collaborator.
- Local prototype extraction from hidden representations.
- Prototype-based topology optimization among collaborators.
- Poisoned-client detection using prototype-only signals.
- Global prototype aggregation from benign clusters.
- Model evaluation on a shared test set.

## Notes

The `data/` directory in this repository stores documentation and placeholders only. The actual CICIoT2023 data files are large and are not committed to the repository. To run experiments, configure the dataset paths through `FrameworkConfig` in `src/main.py`.
