# HUDS Active Learning Core

Core Python implementation of a file-based Hybrid Uncertainty-driven Data
Selection (HUDS) active-learning workflow for surrogate model training.

This repository intentionally contains only the core engine:

- candidate pool generation
- simulation request export
- labeled result import
- residual MLP model training
- MC-dropout uncertainty estimation
- uncertainty/diversity active sampling
- validation, prediction, evaluation, and CLI orchestration

Desktop GUI, visualization panels, executable packaging, and simulator-specific
conversion helpers have been removed.

## Install

```powershell
cd E:\tt\huds-active-learning-app
pip install -e .
```

Optional FAISS acceleration for clustering:

```powershell
pip install -e ".[fast]"
```

## Test

```powershell
python -m pytest -q
```

## Basic Workflow

Initialize a run:

```powershell
huds-app init --config examples\config.example.json --out runs\demo
```

Export validation samples:

```powershell
huds-app export-validation --run runs\demo
```

After external simulation, import validation labels:

```powershell
huds-app import-labels --run runs\demo --kind validation --input sim_val_output.csv
```

Export the initial training batch:

```powershell
huds-app export-initial-train --run runs\demo
```

Import training labels:

```powershell
huds-app import-labels --run runs\demo --kind train --step 0 --input sim_train_output.csv
```

Train:

```powershell
huds-app train --run runs\demo
```

Select the next active-learning batch:

```powershell
huds-app sample --run runs\demo --step 1
```

Check status:

```powershell
huds-app status --run runs\demo
```

## Key Modules

- `huds_app/config.py` - configuration dataclasses and validation
- `huds_app/sampling.py` - candidate pool generation and splitting
- `huds_app/validation.py` - request export and label import
- `huds_app/model.py` - residual MLP model
- `huds_app/train.py` - training, checkpoints, metrics, normalization
- `huds_app/huds.py` - MC-dropout uncertainty and HUDS sample selection
- `huds_app/workflow.py` - high-level workflow/status/predict/evaluate API
- `huds_app/cli.py` - command-line interface

## Data Contract

Request files contain:

```text
sample_id,<variable_1>,<variable_2>,...
```

Simulator output files must contain:

```text
sample_id,<output_1>,<output_2>,...
```

Output names are defined by `model.output_names` in the JSON config.

