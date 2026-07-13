# HUDS GUI Design

## Overview

Build three independent GUI implementations for the HUDS active learning workflow, each in its own folder with no cross-references. All three call into `huds_app/` core layer without modifying it.

## Architecture

```
HUDS/
├── huds_app/              # Core logic (unchanged)
├── launch_huds.py         # CLI entry (preserved)
├── gui_wizard/            # A: PyQt5 QWizard step-by-step
│   ├── main.py
│   ├── wizard.py
│   │   └── pages/
│   │       ├── config_page.py
│   │       ├── aedt_page.py
│   │       ├── monitor_page.py
│   │       └── result_page.py
│   └── requirements.txt
├── gui_tabs/              # B: PyQt5 QTabWidget multi-panel
│   ├── main.py
│   ├── window.py
│   │   └── panels/
│   │       ├── config_panel.py
│   │       ├── aedt_panel.py
│   │       ├── monitor_panel.py
│   │       └── result_panel.py
│   └── requirements.txt
├── gui_modern/            # C: PySide6 + QtMaterial
│   ├── main.py
│   ├── window.py
│   │   └── pages/
│   │       ├── config_page.py
│   │       ├── aedt_page.py
│   │       ├── monitor_page.py
│   │       └── result_page.py
│   └── requirements.txt
└── launch_gui.py          # Unified launcher menu (A/B/C selector)
```

## Shared Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Charting | pyqtgraph | Real-time performance for R2 curves and data distribution |
| Concurrency | QThread | Background thread for Maxwell simulation and training, GUI stays responsive |
| Core access | Import `huds_app.*` only | No modification to core layer |
| Isolation | Each GUI has own requirements.txt | No cross-dependency between A/B/C |

## Functional Flow

1. **Configuration** - AEDT project path (relative or absolute), variable names/ranges, output names, training parameters
2. **AEDT Connection** - Auto-detect version via COM, select project and design
3. **Simulation Monitoring** - Maxwell subprocess progress bar, log output, label extraction status
4. **HUDS Loop** - Iteration display: sampling → simulation → training → metrics
5. **Results** - R2/RMSE curves (pyqtgraph), data distribution, CSV export

## Thread Architecture

```
Main Thread (GUI)          Worker Thread (QThread)
─────────────────          ───────────────────────
Display config form        Connect to AEDT
Show progress bar          Run maxwell_sweep subprocess
Update charts              Execute HUDS sampling
Handle button clicks       Train model
Export results             Extract labels

Signals:  progress → QProgressBar
          log_msg → QTextBrowser append
          r2_update → pyqtgraph plot update
          step_done → advance wizard / enable next tab
```

## Dependencies

| Solution | Dependencies |
|----------|-------------|
| A (Wizard) | PyQt5>=5.15, pyqtgraph>=0.13 |
| B (Tabs) | PyQt5>=5.15, pyqtgraph>=0.13 |
| C (Modern) | PySide6>=6.7, QtMaterial>=2.14, pyqtgraph>=0.13 |

## Launch Flow

`launch_gui.py` presents a simple menu:
```
HUDS GUI Launcher
[1] Wizard (PyQt5 step-by-step)
[2] Tabs   (PyQt5 multi-panel)
[3] Modern (PySide6 + Material)
Choice:
```

Each option imports and runs the corresponding `main.py`.

## Error Handling

- AEDT connection failure: display error dialog, allow retry or manual path input
- Simulation timeout: show elapsed time, offer abort via worker thread signal
- Training failure: log traceback, allow re-run from last checkpoint
- GUI crash: save config state to disk on each step completion for recovery
