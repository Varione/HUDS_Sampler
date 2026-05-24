## Task Plan

Task: create `huds_app/workflow.py` orchestration and status reporting module.

Phases:

1. Inspect existing module contracts for config, storage, sampling, schema, model, train, validation, and HUDS.
2. Implement the requested workflow API without training, HUDS sampling, or CLI parsing logic.
3. Validate changed file with diagnostics, imports, initialization, file validation, and status surface calls.

Key decisions:

- Use `RunState` as the single persisted workflow state source.
- Use root pool files because `sampling.save_pool_files` and `validation.py` read them from `run_dir`.
- Reuse `train.load_normalization` and `train.apply_normalization` to keep prediction and evaluation consistent with training.
