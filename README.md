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
cd D:\TempData\huds
pip install -e .
```

Optional FAISS acceleration for clustering:

```powershell
pip install -e ".[fast]"
```

If you prefer `requirements.txt`, it contains only the core runtime
dependencies. `faiss-cpu` remains optional because the code automatically
falls back to scikit-learn KMeans when FAISS is unavailable.

## Test

```powershell
python -m pytest -q
```

The current repository test suite passes with `73 passed`.

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

Sampling is stateful. The CLI rejects:

- non-positive step numbers
- sampling the same step twice
- skipping ahead to a later step
- sampling beyond `training.max_steps`
- sampling while a prior training request is still unlabeled or partially labeled

Check status:

```powershell
huds-app status --run runs\demo
```

Run predictions on arbitrary candidate rows after training:

```powershell
huds-app predict --run runs\demo --input candidates.csv --output predictions.csv
```

Evaluate the current checkpoint on validation data:

```powershell
huds-app evaluate --run runs\demo
```

Convert a HUDS request CSV into a Maxwell parametric table:

```powershell
huds-app export-maxwell --run runs\demo --input runs\demo\requests\train_step_000_request.csv --output ParametricSetup1_Table.csv
```

The Maxwell export writes a CSV shaped like:

```text
*,gap,v
1,5mm,0m_per_sec
2,8mm,1m_per_sec
```

Variable units come from `variables[].unit` in the JSON config. You can also
override them from the CLI:

```powershell
huds-app export-maxwell --input request.csv --output ParametricSetup1_Table.csv --unit gap=mm --unit v=m_per_sec
```

Imports are strict by default. If a simulator run only returns part of a
requested batch, the import fails unless you explicitly allow cumulative
partial imports:

```powershell
huds-app import-labels --run runs\demo --kind train --step 1 --input sim_out.csv --allow-partial
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

Maxwell parametric export files contain:

```text
*,<variable_1>,<variable_2>,...
```

with each variable value serialized as `number + unit`, for example `5mm` or
`0m_per_sec`.

Simulator output files must contain:

```text
sample_id,<variable_1>,<variable_2>,...,<output_1>,<output_2>,...
```

Output names are defined by `model.output_names` in the JSON config.

The import path validates `sample_id` coverage, duplicate IDs, missing numeric
values, and missing output columns before it writes anything to disk.

