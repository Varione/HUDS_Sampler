# Task Plan

Task: simplify the repository to the core HUDS active-learning engine.

## Scope Kept

- Candidate pool generation and train/validation split
- File-based request export and label import
- Residual MLP surrogate model training
- MC-dropout uncertainty estimation
- HUDS sample selection with uncertainty and diversity
- CLI workflow orchestration and tests

## Scope Removed

- Desktop GUI package and widgets
- Desktop packaging files
- GUI-specific service tests
- Maxwell/AEDT conversion helper
- Temporary smoke/review runs
- Build and distribution artifacts

## Current Verification Target

Run:

```powershell
python -m pytest -q
```

Expected: core tests pass without desktop or GUI dependencies.
