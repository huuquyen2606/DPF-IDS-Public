# DPF-IDS

**DPF-IDS: A Robust Decentralized Prototype Federated Learning-Based Intrusion Detection with Poisoning Attack Resilience**

Nguyen Huu Quyen, Hoang Ngoc Khanh, Nguyen Tran Minh Khoi, Lu Le Huong Giang, Van-Hau Pham

DPF-IDS is a decentralized prototype-federated IDS framework that exchanges compact penultimate-layer prototypes instead of model parameters or gradients, reducing inference exposure while enabling adaptive collaboration and poisoning-resilient aggregation of benign prototypes.

This repository accompanies the research paper and is intended to support research reproducibility.

## Paper Highlights

- DPF-IDS enables inference-resilient decentralized prototype-only federated intrusion detection.
- Poisoning-resilient aggregation filters malicious collaborators before global prototype aggregation.
- Dynamic temporary-server selection avoids relying on a fixed central aggregation server.
- DPF-IDS reduces communication overhead by `334.2x` to `6370.7x` compared with evaluated baselines.
- DPF-IDS achieves the top Macro-F1-score under PGA and BoTPA poisoning attacks among the evaluated methods.

## Paper Overview

The paper studies robust decentralized intrusion detection under clean and poisoning-attack settings. DPF-IDS uses local IDS models to extract compact class-agnostic prototype vectors. These prototypes are then used for:

- Adaptive topology construction among collaborators.
- Poisoned-collaborator detection using prototype-only signals.
- Benign-only prototype aggregation.
- Prototype-guided local alignment.
- Next-round temporary-server selection.

The framework is evaluated on CICIoT2023 under two deployment scales:

- 200 collaborators.
- 500 collaborators.

The evaluated data and attack settings are:

- Clean Non-IID data with Dirichlet `alpha = 0.5`.
- Label-flipping data poisoning.
- PGA-based untargeted model poisoning.
- BoTPA targeted poisoning.

## Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- data/
|   |-- DATASET.md
|   `-- .gitkeep
|-- checkpoints/
|-- experiments/
`-- src/
    |-- main.py
    |-- train.py
    |-- evaluate.py
    |-- scripts/
    |   |-- run_clean_non_iid.py
    |   |-- run_label_flipping.py
    |   |-- run_pga.py
    |   `-- run_botpa.py
    |-- data/
    |   |-- preprocessing.py
    |   |-- partition.py
    |   `-- dataset.py
    |-- models/
    |   `-- ffnn.py
    |-- dpf_ids/
    |   |-- prototype.py
    |   |-- topology.py
    |   |-- detection.py
    |   |-- aggregation.py
    |   `-- alignment.py
    |-- attacks/
    |   |-- label_flipping.py
    |   |-- pga.py
    |   `-- botpa.py
    `-- utils/
        |-- metrics.py
        |-- logging.py
        |-- seed.py
        `-- io.py
```

## Installation

Create a Python 3 environment before running experiments. For TensorFlow 2.11, use a compatible Python 3.x version supported by that TensorFlow release.

Example command:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the core scientific dependencies:

```bash
pip install numpy pandas scikit-learn tqdm
```

The paper reports experiments with **TensorFlow 2.11**:

```bash
pip install "tensorflow==2.11.*"
```

The provided refactored runner scripts in this repository use `.pt` tensor files and Python modules under `src/`. If you run this code path as-is, install the required backend used by the implementation:

```bash
pip install torch
```

Exact package versions should be pinned in your own environment file when preparing camera-ready reproduction artifacts.

## Dataset Preparation

DPF-IDS is evaluated on CICIoT2023. The dataset must be obtained according to the dataset provider's terms and conditions. This repository does not redistribute CICIoT2023.

If you need the processed dataset used for these experiments, access it through the following Google Drive folder:

[https://drive.google.com/drive/folders/1NO26qAGugxhaCr-ZZjevqk11QoHNVGg1?usp=drive_link](https://drive.google.com/drive/folders/1NO26qAGugxhaCr-ZZjevqk11QoHNVGg1?usp=drive_link)

Please ensure that any dataset access and use comply with the CICIoT2023 provider's terms and any applicable institutional or project requirements.

Use a local dataset path such as:

```text
/path/to/CICIoT2023
```

A typical preparation workflow is:

1. Place the original CICIoT2023 files outside the repository or under a local ignored data directory.
2. Clean invalid values such as `NaN`, `Inf`, and `-Inf`.
3. Normalize numeric features with Min-Max scaling.
4. Split the dataset into training and test sets.
5. Partition the training set into collaborator files.
6. Convert collaborator files and the shared test set to `.pt` tensors.

Expected collaborator layouts:

```text
/path/to/processed/CICIoT2023/200c/
|-- client_001.pt
|-- client_002.pt
|-- ...
`-- client_200.pt
```

```text
/path/to/processed/CICIoT2023/500c/
|-- client_001.pt
|-- client_002.pt
|-- ...
`-- client_500.pt
```

Shared test file:

```text
/path/to/processed/CICIoT2023/test.pt
```

See [data/DATASET.md](data/DATASET.md) for additional dataset notes.

## Configuration

The main configuration object is `FrameworkConfig` in `src/main.py`.

Important configuration fields include:

- `data_train`: directory containing collaborator `client_*.pt` files.
- `data_test`: path to the shared `test.pt` file.
- `num_clients`: number of collaborators, usually `200` or `500`.
- `batch_size`: local training batch size, default `1024`.
- `num_epochs`: local epochs per communication round, default `5`.
- `learning_rate`: SGD learning rate, default `1e-2`.
- `checkpoint_dir`: output directory for checkpoints and logs.
- `data_mode`: either `benign` or `data_poison`.
- `true_poison_index_path`: optional text file containing poisoned collaborator IDs for attack evaluation.
- `enable_pga_attack`: enables PGA-based model poisoning.
- `enable_botpa_attack`: enables BoTPA targeted poisoning.

Main model and training settings used in the paper:

- Local IDS model: fully connected neural network.
- Input features: `39`.
- Hidden layers: `64 -> 32 -> 16`.
- Activation: ReLU.
- Dropout: `0.1`.
- Output classes: `8`.
- Prototype dimension: `16`.
- Optimizer: SGD.
- Learning rate: `1e-2`.
- Momentum: `0.9`.
- Weight decay: `1e-4`.
- Batch size: `1024`.
- Local epochs: `5`.
- Communication rounds: `11` scheduled rounds.

Note: although the scripts are configured with `--num-rounds 11`, the current pipeline performs 10 effective local-training/communication rounds because the final scheduled round stops before starting another local-training step.
- Gradient clipping max norm: `5.0`.

## Running Experiments

The scripts under `src/scripts/` are provided as reproducibility-oriented runners. Replace all placeholder paths with local paths prepared on your machine.

Use `--num-collaborators 200` or `--num-collaborators 500` depending on the experiment scale.

### Clean Non-IID Setting

Example command:

```bash
python -m src.scripts.run_clean_non_iid \
  --train-data-dir /path/to/processed/CICIoT2023/200c \
  --test-data-path /path/to/processed/CICIoT2023/test.pt \
  --checkpoint-dir checkpoints/clean_noniid_200c \
  --num-collaborators 200 \
  --num-rounds 11
```

### Label-Flipping Attack

This setting assumes that label-flipped collaborator files have already been generated.

Example command:

```bash
python -m src.scripts.run_label_flipping \
  --train-data-dir /path/to/processed/CICIoT2023/label_flipping_200c \
  --test-data-path /path/to/processed/CICIoT2023/test.pt \
  --checkpoint-dir checkpoints/label_flipping_200c \
  --num-collaborators 200 \
  --num-rounds 11 \
  --true-poison-index-path /path/to/poisoned_collaborators_200c.txt
```

### PGA-Based Untargeted Model Poisoning

Example command:

```bash
python -m src.scripts.run_pga \
  --train-data-dir /path/to/processed/CICIoT2023/200c \
  --test-data-path /path/to/processed/CICIoT2023/test.pt \
  --checkpoint-dir checkpoints/pga_200c \
  --num-collaborators 200 \
  --num-rounds 11 \
  --true-poison-index-path /path/to/poisoned_collaborators_200c.txt
```

### BoTPA Targeted Poisoning

Example command:

```bash
python -m src.scripts.run_botpa \
  --train-data-dir /path/to/processed/CICIoT2023/200c \
  --test-data-path /path/to/processed/CICIoT2023/test.pt \
  --checkpoint-dir checkpoints/botpa_200c \
  --num-collaborators 200 \
  --num-rounds 11 \
  --true-poison-index-path /path/to/poisoned_collaborators_200c.txt \
  --botpa-source-class 2 \
  --botpa-target-class 0
```

For 500-collaborator experiments, replace the data directory, checkpoint directory, poison-index file, and `--num-collaborators` value accordingly.

## Reproducing Paper Tables And Figures

The training pipeline writes checkpoints, detection reports, mapping summaries, and evaluation metrics to the selected `checkpoint_dir`.

Common output locations include:

```text
checkpoints/<experiment_name>/
|-- *_checkpoint_round*.pt
|-- detection_logs/
|-- poison_index_mapping_logs/
`-- eval_logs/
```

To reproduce paper tables and figures:

1. Run each experiment setting for both 200 and 500 collaborators.
2. Collect the generated CSV files from `detection_logs/`, `poison_index_mapping_logs/`, and `eval_logs/`.
3. Aggregate metrics across runs using the same seeds and settings reported in the paper.
4. Use the aggregated CSV files to regenerate the tables and plots.

## Main Results Summary

The following summary should be interpreted as a high-level guide to the paper results, not as a substitute for the full experimental tables.

- In the clean Non-IID setting, DPF-IDS does not always achieve the highest Accuracy or Macro-F1, but it obtains strong weighted precision and competitive Weighted-F1 while using much lower communication cost.
- Under PGA and BoTPA poisoning, DPF-IDS achieves the highest Macro-F1 across all evaluated settings.
- DPF-IDS achieves poisoned-collaborator detection F1 up to `0.9950` under PGA and `0.9529` under BoTPA.
- DPF-IDS communicates only `0.0125 KB` per collaborator per round, while compared baselines require hundreds to thousands of times higher communication overhead.
- Label-flipping remains the most challenging setting because poisoned prototypes may remain close to benign Non-IID prototype variations.

## Authors / Maintainers

- Nguyen Huu Quyen, University of Information Technology, VNU-HCM.
- Hoang Ngoc Khanh, University of Information Technology, VNU-HCM.
- Nguyen Tran Minh Khoi, University of Information Technology, VNU-HCM.
- Lu Le Huong Giang, University of Information Technology, VNU-HCM.
- Van-Hau Pham, University of Information Technology, VNU-HCM.

Corresponding author: **Van-Hau Pham**.

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.


## Research Reproducibility Notice

This repository is provided for research reproducibility. Results may vary with hardware, software versions, random seeds, preprocessing choices, and dataset partitioning. CICIoT2023 must be obtained and used according to the dataset provider's terms.
