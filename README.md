# HUDS Active Learning App

A file-based command-line tool for iterative surrogate model training using Hybrid Uncertainty-driven Data Selection (HUDS). The app manages the complete active learning lifecycle: pool generation, simulation request export, label import, model training, and uncertainty-based sample selection.

## Overview

The HUDS Active Learning App orchestrates a human-in-the-loop workflow for building high-fidelity surrogate models. It generates stratified sampling pools from user-defined parameter ranges, exports batches of configurations for external simulation, imports labeled results, trains neural network surrogates with Monte Carlo dropout uncertainty estimation, and selects the most informative new training samples using hybrid uncertainty-driven data selection (HUDS).

The workflow is file-based with persistent state tracking. Each command reads/writes CSV files and updates `state.json` to track progress through the active learning cycle. This design enables seamless integration with external simulation tools while maintaining checkpointing at every step for resumable workflows.

## Important Note

This application does not perform real simulations. The app generates parameter configurations (request files) that you must evaluate using your own simulator or experimental setup externally. Once your simulator produces output, you import the labeled results back into the app for model training.

Additionally, HUDS uncertainty estimation requires a trained model with dropout enabled. Without dropout layers, the variance component of hybrid uncertainty will be zero, and sampling will degenerate to entropy-only selection. Make sure your configuration includes `model.dropout > 0`.

## Installation

```bash
cd huds-active-learning-app
pip install -e .
```

For optional FAISS-accelerated nearest-neighbor search during sampling:

```bash
pip install -e ".[fast]"
```

## Quick Start

The following walkthrough demonstrates a complete active learning cycle. Replace file paths and sample sizes with your own values.

### Step 1: Initialize the Run

Generate stratified candidate pools from your configuration:

```bash
huds-app init --config examples/config.example.json --out runs/demo
```

This creates `runs/demo/` containing `candidate_pool.csv`, `train_pool.csv`, `validation_pool.csv`, and a copy of your configuration.

### Step 2: Export and Label Validation Set

Export validation samples for simulation:

```bash
huds-app export-validation --run runs/demo
```

The app writes `runs/demo/requests/validation_request.csv`. Run this file through your external simulator, then import the labeled output:

```bash
huds-app import-labels --run runs/demo --kind validation --input sim_val_output.csv
```

### Step 3: Export and Label Initial Training Set

Export the first batch of training samples:

```bash
huds-app export-initial-train --run runs/demo
```

This writes `runs/demo/requests/train_step_000_request.csv`. Run your simulator on these configurations, then import results:

```bash
huds-app import-labels --run runs/demo --kind train --step 0 --input sim_train_output.csv
```

### Step 4: Train the Surrogate Model

Train a neural network surrogate on all labeled training data:

```bash
huds-app train --run runs/demo
```

The trained checkpoint is saved to `runs/demo/checkpoints/model_latest.pt`. Training metrics are written to `runs/demo/metrics/`.

### Step 5: Active Learning Sampling Loop

Select new samples using HUDS hybrid uncertainty, label them, retrain, and repeat:

```bash
# Sample step 1
huds-app sample --run runs/demo --step 1
# Export the request file from runs/demo/requests/train_step_001_request.csv
# Run simulator externally
huds-app import-labels --run runs/demo --kind train --step 1 --input sim_train_output.csv

# Retrain with expanded dataset
huds-app train --run runs/demo

# Evaluate current model quality
huds-app evaluate --run runs/demo

# Continue sampling next step
huds-app sample --run runs/demo --step 2
```

Repeat the import-labels, train, and sample cycle until you reach `training.max_steps` or are satisfied with model performance. Monitor progress at any time:

```bash
huds-app status --run runs/demo
```

## Configuration Format

The JSON configuration defines variables, pool sizes, model architecture, training hyperparameters, and HUDS sampling settings:

```json
{
  "project_name": "my_project",
  "random_seed": 42,
  "variables": [
    {"name": "lx", "min": 0.01, "max": 0.08, "sample_points": 12},
    {"name": "ly", "min": 0.01, "max": 0.10, "sample_points": 12}
  ],
  "candidate_pool": {
    "total_samples": 12000,
    "train_ratio": 0.8,
    "validation_ratio": 0.2
  },
  "model": {
    "output_names": ["fx", "fy"],
    "hidden_dim": 128,
    "residual_blocks": 4,
    "dropout": 0.1
  },
  "validation": {
    "default_size": 1000
  },
  "training": {
    "initial_train_size": 256,
    "sample_per_step": 64,
    "max_steps": 10,
    "epochs_per_step": 300,
    "batch_size": 256,
    "learning_rate": 0.001,
    "patience": 30,
    "device": "cuda"
  },
  "huds": {
    "pre_n": 0,
    "repeat_times": 30,
    "topk_ratio": 0.6,
    "batch_size": 256,
    "use_faiss": true
  }
}
```

### Configuration Sections

| Section | Purpose |
|---------|---------|
| `variables` | Parameter names, ranges, and level counts for stratified sampling |
| `candidate_pool` | Total pool size and train/validation split ratios |
| `model` | Output variable names, network architecture, dropout rate |
| `validation` | Default number of validation samples to export |
| `training` | Batch sizes, learning rate, epochs, early stopping patience, device |
| `huds` | Monte Carlo repeat count, top-k selection ratio, FAISS acceleration |

## Data File Formats

All data files are CSV. The app uses the following schemas:

### candidate_pool.csv

Contains all generated samples with metadata columns and variable columns defined in your configuration:

| Column | Description |
|--------|-------------|
| `sample_id` | Unique integer identifier for each sample |
| `split` | Assignment: `train_pool` or `validation_pool` |
| `status` | State: `unlabeled`, `selected`, `labeled`, `used` |
| `<variable_name>` | Parameter values (one column per variable in config) |

### train_pool.csv / validation_pool.csv

Same columns as candidate_pool.csv, filtered by split assignment.

### Request Files

Written to `runs/<run_dir>/requests/`. Format:

```
sample_id,<var1>,<var2>,...
0,0.045,0.067,...
```

Only contains sample_id and variable columns (no metadata). Use these files as input to your external simulator.

### Labeled Datasets

Stored in `runs/<run_dir>/datasets/`. After importing labels, the app creates:

- `train_labeled.csv` -- training samples with output values appended
- `validation_labeled.csv` -- validation samples with output values appended

Your simulator output CSV must contain at minimum a `sample_id` column matching the request file, plus columns for each output variable defined in `model.output_names`.

## Full Command Reference

### init

Initialize a new active learning run from configuration.

```bash
huds-app init --config <path> --out <dir> [--snap-to-levels]
```

| Argument | Description |
|----------|-------------|
| `--config` | Path to JSON configuration file (required) |
| `--out` | Output run directory path (required) |
| `--snap-to-levels` | Snap continuous samples to discrete variable levels instead of stratified random values |

### export-validation

Export validation pool samples for external simulation.

```bash
huds-app export-validation --run <dir> [--size N]
```

| Argument | Description |
|----------|-------------|
| `--run` | Run directory (required) |
| `--size` | Number of validation samples to export (defaults to config.validation.default_size) |

### export-initial-train

Export the initial training batch for external simulation.

```bash
huds-app export-initial-train --run <dir>
```

Uses `training.initial_train_size` from configuration to determine batch size.

### import-labels

Import labeled results from your simulator into the run datasets.

```bash
huds-app import-labels --run <dir> --kind validation|train [--step N] --input <path>
```

| Argument | Description |
|----------|-------------|
| `--run` | Run directory (required) |
| `--kind` | Destination: `validation` or `train` (required) |
| `--step` | Training step number, required when kind is `train` |
| `--input` | Path to simulator output CSV (required) |

### train

Train the surrogate model on all available labeled training data.

```bash
huds-app train --run <dir>
```

Saves checkpoint to `checkpoints/model_latest.pt`. Writes per-epoch metrics to `metrics/`.

### sample

Run HUDS uncertainty-based sampling for the next active learning step.

```bash
huds-app sample --run <dir> --step N
```

| Argument | Description |
|----------|-------------|
| `--run` | Run directory (required) |
| `--step` | Active learning step number (required, must be sequential: 1, 2, 3...) |

Writes request file to `requests/train_step_<N>_request.csv`.

### status

Display run progress and next recommended action.

```bash
huds-app status --run <dir>
```

Shows pool sizes, labeled counts, checkpoint status, remaining unlabeled samples, and current step.

### validate-files

Verify that all expected files exist in the run directory.

```bash
huds-app validate-files --run <dir>
```

Returns a report of present and missing files with their schemas.

### inspect-config

Validate and summarize a JSON configuration file.

```bash
huds-app inspect-config --config <path>
```

Checks config structure, prints summary statistics (pool sizes, variable ranges).

### predict

Run inference on arbitrary candidate rows using the trained model.

```bash
huds-app predict --run <dir> --input <path> --output <path>
```

| Argument | Description |
|----------|-------------|
| `--run` | Run directory containing trained model (required) |
| `--input` | Input CSV with variable columns (required) |
| `--output` | Output CSV path for predictions (required) |

### evaluate

Evaluate the current model on labeled validation data.

```bash
huds-app evaluate --run <dir>
```

Computes regression metrics (MSE, MAE, R2) per output variable and prints a summary table.

## Troubleshooting

**Error: Run state file not found.** The run directory was not initialized with `huds-app init`. Run the init command before any other operation.

**Error: train_step required for kind train.** When importing training labels, you must specify which step the labels belong to using `--step N`. Validation imports do not require a step number.

**Error: No labeled training data found.** You must complete at least one full cycle of export-initial-train, simulate externally, and import-labels before running `huds-app train`.

**Sampling returns identical results every run.** Check that your configuration has `model.dropout > 0`. HUDS uses MC Dropout variance for uncertainty estimation—without dropout layers, the model produces deterministic predictions with zero variance, and sampling degrades to pure diversity-based selection (KMeans clustering only).

**CUDA out of memory during training or sampling.** Reduce `training.batch_size` and `huds.batch_size` in your configuration. For large models, consider reducing `model.hidden_dim` or `model.residual_blocks`. Set `training.device` to `"cpu"` as a fallback.

**FAISS import error during sampling.** FAISS is an optional dependency. Either install it with `pip install faiss-cpu` or set `huds.use_faiss` to `false` in your configuration.

**Imported labels do not match any request samples.** The app validates that every `sample_id` in your simulator output CSV matches a sample from the corresponding request file. If IDs don't match, import fails with an error listing unknown sample IDs. To allow partial imports (when some simulations fail), use `--allow-partial` flag:

```bash
huds-app import-labels --run <dir> --kind train --step N --input sim_output.csv --allow-partial
```

Without this flag, the app enforces complete label coverage to prevent silent data corruption from missing simulator results.
