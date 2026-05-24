# HUDS Active Learning App - Fix Tracking

## Must Fix Items (Critical)
### 1. pending_sample_ids state machine bug ✅ FIXED
- validation.py: Don't add to pending_sample_ids when exporting validation request
- import_labels: Clear pending IDs from that step after successful label import
- workflow.py: Update _next_command logic to check train_requests status properly

### 2. Checkpoint loading in training ✅ FIXED
- train.py: Load existing model_latest.pt checkpoint before training if it exists
- Add config option for retrain_from_scratch (default false)

### 3. Label import completeness check ✅ FIXED  
- validation.py: Verify all request IDs were covered in simulator output
- Add --allow-partial flag to permit partial imports

### 4. UTF-8 BOM support ✅ FIXED
- config.py and storage.py: Use utf-8-sig encoding when reading JSON files

## Should Fix Items (Important)
### 5. Training metrics normalization ⚠️ PARTIAL
- Evaluate model uses normalized predictions, but logs show normalized RMSE
- TODO: Add denormalization before computing metrics in _evaluate_model

### 6. sample_id type consistency ✅ FIXED  
- huds.py: Force int conversion for integer IDs in _sample_id function
- Ensure all state/diagnostics use consistent ID types

### 7. CLI overwrite/partial flags ✅ FIXED
- cli.py: Add --overwrite and --allow-partial to import-labels command

### 8. README documentation updates ⚠️ PARTIAL
- Update to match actual implementation (stateful, not stateless)
- Clarify HUDS uses MC dropout variance only, not entropy + variance
- Document error handling behavior accurately

## Nice to Have
### 9. Git repository initialization ✅ DONE
### 10. End-to-end workflow test ⚠️ TODO
- Add tests/test_cli_workflow.py covering full active learning cycle
