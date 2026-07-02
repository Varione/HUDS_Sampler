# HUDS vs Random Sampling Benchmark

## Goal
Compare HUDS active learning against uniform random sampling baseline across 9 scenarios (3 model types x 3 conditions each).

## Model Types

| Type | Input Shape | Output Shape | Ground Truth Function |
|------|------------|-------------|----------------------|
| **V2V** | `(B, D_in)` | `(B, D_out)` | Multi-output regression |
| **V2TS** | `(B, D_in)` | `(B, seq_len, D_out)` | Time-series prediction |
| **V2I** | `(B, D_in)` | `(B, ch, H, W)` | Image generation |

## Conditions (3 per model type)

### Condition A: Low-Dimensional / Simple
- Input dims: 2-3
- Output complexity: minimal
- Pool size: 500
- Initial train: 50
- Steps: 5

### Condition B: Medium-Dimensional / Moderate
- Input dims: 5-8
- Output complexity: moderate
- Pool size: 1000
- Initial train: 100
- Steps: 8

### Condition C: High-Dimensional / Complex
- Input dims: 10-15
- Output complexity: high (long seq / large image)
- Pool size: 2000
- Initial train: 200
- Steps: 10

## Metrics

| Metric | Description |
|--------|-------------|
| **Final Val R2** | Validation R2 after all steps |
| **Sample Efficiency** | R2 vs labeled count curve |
| **Convergence Step** | Step where R2 first exceeds threshold (e.g. 0.9) |
| **Wall Time** | Total training + sampling time |

## Experiment Script Structure

```
experiments/
├── benchmark.py          # Main orchestrator
├── ground_truth.py       # Synthetic data generators for V2V/V2TS/V2I
├── scenarios.yaml        # All 9 scenario configs
└── results/              # Output directory (CSV + plots)
```

## Phases

- [ ] Phase 1: Create `experiments/ground_truth.py` with synthetic data generators
- [ ] Phase 2: Create `experiments/scenarios.yaml` with 9 scenario definitions
- [ ] Phase 3: Create `experiments/benchmark.py` orchestrator (HUDS vs Random loop)
- [ ] Phase 4: Run benchmark, collect results to CSV
- [ ] Phase 5: Analyze results, summarize comparison
