"""Integration test: full HUDS workflow end-to-end."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from huds_app.core.config import load_config, validate_config
from huds_app.data.schema import SAMPLE_ID_COLUMN
from huds_app.data.pool import create_candidate_pool
from huds_app.data.validation import (
    export_initial_train_request,
    export_validation_request,
    import_labels,
)
from huds_app.interface.workflow import init_run
from huds_app.model.architecture import build_model
from huds_app.model.train import train_model
from huds_app.sampling.huds import run_huds_sampling
from huds_app.core.storage import RunState

# Test directory
TEST_DIR = Path(__file__).parent.parent / "runs" / "integration_test"
CONFIG_PATH = Path(__file__).parent / "integration_test_config.json"


def generate_simulator_output(df, output_names):
    """Generate fake simulator output based on input variables."""
    result = df.copy()
    for name in output_names:
        # Simple synthetic function with some noise
        x = df["x"].values
        y = df["y"].values
        if name == "z":
            values = 0.5 * x + 0.3 * y + 0.2 * x * y + np.random.normal(0, 0.01, len(df))
        else:
            values = np.random.randn(len(df)) * 0.1
        result[name] = values
    return result


def main():
    print("=" * 60)
    print("HUDS Integration Test - Full Workflow")
    print("=" * 60)

    # Clean up previous run
    if TEST_DIR.exists():
        import shutil
        shutil.rmtree(TEST_DIR)

    # Step 1: Load and validate config
    print("\n[Step 1] Loading configuration...")
    config = load_config(str(CONFIG_PATH))
    print(f"  Project: {config.project_name}")
    print(f"  Variables: {[v.name for v in config.variables]}")
    print(f"  Outputs: {config.model.output_names}")
    print("  OK")

    # Step 2: Initialize run (creates pool, state, directory structure)
    print("\n[Step 2] Initializing run...")
    if TEST_DIR.exists():
        import shutil
        shutil.rmtree(TEST_DIR)

    result = init_run(str(CONFIG_PATH), str(TEST_DIR))
    print(f"  Run dir: {result['run_dir']}")
    print(f"  Total candidates: {result['total_candidates']}")
    print(f"  Train pool: {result['train_pool_size']}")
    print(f"  Validation pool: {result['validation_pool_size']}")
    print("  OK")

    # Load pool for later use
    pool_df = pd.read_csv(TEST_DIR / "candidate_pool.csv")

    # Step 4: Export validation request
    print("\n[Step 4] Exporting validation request...")
    val_path = export_validation_request(TEST_DIR, config, size=config.validation.default_size)
    val_df = pd.read_csv(val_path)
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Output: {val_path}")
    print("  OK")

    # Step 5: Generate fake validation labels (only for requested samples)
    print("\n[Step 5] Generating fake validation labels...")
    val_labels = generate_simulator_output(val_df, config.model.output_names)
    val_labels_path = TEST_DIR / "datasets" / "validation_labeled.csv"
    val_labels.to_csv(val_labels_path, index=False)
    print(f"  Labels for {len(val_labels)} validation samples")
    print("  OK")

    # Step 6: Import validation labels (use overwrite to avoid duplicate check)
    print("\n[Step 6] Importing validation labels...")
    import_labels(str(TEST_DIR), "validation", None, str(val_labels_path), overwrite=True)
    state = RunState.load(str(TEST_DIR))
    print(f"  Validation labeled: {state.validation_labeled}")
    print("  OK")

    # Step 7: Export initial training request
    print("\n[Step 7] Exporting initial training request...")
    train_path = export_initial_train_request(TEST_DIR, config)
    train_df = pd.read_csv(train_path)
    print(f"  Initial train samples: {len(train_df)}")
    print(f"  Output: {train_path}")
    print("  OK")

    # Step 8: Generate fake training labels
    print("\n[Step 8] Generating fake training labels...")
    train_labels = generate_simulator_output(train_df, config.model.output_names)
    train_labels_path = TEST_DIR / "datasets" / "train_step_000_labeled.csv"
    train_labels.to_csv(train_labels_path, index=False)
    print("  OK")

    # Step 9: Import training labels (step 0)
    print("\n[Step 9] Importing training labels (step 0)...")
    import_labels(str(TEST_DIR), "train", 0, str(train_labels_path), overwrite=True)
    state = RunState.load(str(TEST_DIR))
    print(f"  Train requests: {list(state.train_requests.keys())}")
    print("  OK")

    # Step 10: Train initial model
    print("\n[Step 10] Training initial model...")
    config.current_step = 0
    metrics = train_model(str(TEST_DIR), config)
    print(f"  Final train loss: {metrics.get('train_loss', 'N/A'):.6f}")
    print(f"  Final val loss: {metrics.get('val_loss', 'N/A'):.6f}")
    print(f"  R2 avg: {metrics.get('val_r2_avg', 'N/A'):.4f}")
    print("  OK")

    # Step 11: Run HUDS sampling (Step 1)
    print("\n[Step 11] Running HUDS sampling (step 1)...")
    result = run_huds_sampling(TEST_DIR, config, step=1)
    selected_ids = result["selected_ids"]
    print(f"  Selected samples: {len(selected_ids)}")
    print(f"  Top-K size: {result['topk_size']}")
    print(f"  Clusters: {result['n_clusters']}")
    print("  OK")

    # Step 12: Export step 1 request and generate labels
    print("\n[Step 12] Processing step 1 results...")
    state = RunState.load(str(TEST_DIR))
    print(f"  Current step: {state.current_step}")
    print(f"  Train requests: {list(state.train_requests.keys())}")

    # Generate labels for step 1 samples
    step1_df = pool_df[pool_df[SAMPLE_ID_COLUMN].isin(selected_ids)].copy()
    step1_labels = generate_simulator_output(step1_df, config.model.output_names)
    step1_labels_path = TEST_DIR / "datasets" / "train_step_001_labeled.csv"
    step1_labels.to_csv(step1_labels_path, index=False)
    print("  OK")

    # Step 13: Import step 1 labels
    print("\n[Step 13] Importing training labels (step 1)...")
    import_labels(str(TEST_DIR), "train", 1, str(step1_labels_path), overwrite=True)
    state = RunState.load(str(TEST_DIR))
    print("  OK")

    # Step 14: Train with updated data
    print("\n[Step 14] Training with step 1 data...")
    config.current_step = 1
    metrics = train_model(str(TEST_DIR), config)
    print(f"  Final train loss: {metrics.get('train_loss', 'N/A'):.6f}")
    print(f"  Final val loss: {metrics.get('val_loss', 'N/A'):.6f}")
    print(f"  R2 avg: {metrics.get('val_r2_avg', 'N/A'):.4f}")
    print("  OK")

    # Step 15: Run HUDS sampling (Step 2)
    print("\n[Step 15] Running HUDS sampling (step 2)...")
    result = run_huds_sampling(TEST_DIR, config, step=2)
    print(f"  Selected samples: {len(result['selected_ids'])}")
    print("  OK")

    # Summary
    print("\n" + "=" * 60)
    print("Integration Test PASSED")
    print("=" * 60)
    print(f"\nRun directory: {TEST_DIR}")
    print(f"Files created:")
    for f in sorted(TEST_DIR.rglob("*")):
        if f.is_file():
            rel = f.relative_to(TEST_DIR)
            print(f"  {rel}")


if __name__ == "__main__":
    main()
