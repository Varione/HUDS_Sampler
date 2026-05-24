## Findings

- `sampling.create_candidate_pool` already creates the split labels, and `sampling.split_pool` returns train and validation pool dataframes.
- `sampling.save_pool_files` writes `candidate_pool.csv`, `train_pool.csv`, and `validation_pool.csv` in the run directory root.
- `RunState` stores `validation_request_created`, `validation_labeled`, `train_requests`, `latest_checkpoint`, `best_checkpoint`, `used_sample_ids`, and `pending_sample_ids`.
- `validation.py` exports validation and initial train requests from root pool CSV files, then writes labels to `datasets/*_labeled.csv`.
- `train.py` writes checkpoints under `checkpoints/` and normalization statistics to `artifacts/normalization.json`.
- Model checkpoints contain `model_state_dict` when saved by `train.py`.
