# HUDS Sampler

Hybrid Uncertainty-driven Data Selection (HUDS) active-learning toolkit for surrogate model training, with integrated Ansys Electronics Desktop (AEDT) automation.

## Features

Three desktop GUIs:
- **Wizard** — step-by-step configuration workflow
- **Tabs** — tabbed multi-panel interface
- **Modern** — PySide6-based modern UI

Core capabilities:
- Direct COM connection to running AEDT instances or direct `.aedt` file browsing
- Automatic variable and output detection from `.aedt` design files
- Candidate pool generation with configurable sampling strategies
- Seamless Maxwell parametric sweep automation (import CSV, run sweep, export results)
- Residual MLP surrogate model training with MC-dropout uncertainty estimation
- Active learning loop: uncertainty/diversity sampling → simulation → retraining

## Quick Start

### Source

```powershell
cd E:\大型数据库构建\HUDS
D:\miniconda\envs\agents\python.exe gui_wizard\main.py
```

Or for the tabbed interface:

```powershell
D:\miniconda\envs\agents\python.exe gui_tabs\main.py
```

### Packaged Build

Run PyInstaller with the provided spec file:

```powershell
cd E:\大型数据库构建\HUDS
D:\miniconda\envs\agents\python.exe -m PyInstaller HUDS_Wizard.spec
```

Output is in `dist\HUDS_Wizard\`. Use `--workpath` and `--distpath` to control build artifacts location:

```powershell
D:\miniconda\envs\agents\python.exe -m PyInstaller --workpath "D:\HUDS_Builds\build" --distpath "D:\HUDS_Builds" HUDS_Wizard.spec
```

## Project Structure

```
huds_app/
  core/         Configuration, data models
  interface/    AEDT COM bridge, Maxwell sweep automation, workflow orchestration
  utils/        .aedt file parser (variables, outputs, designs)
gui_wizard/     Step-by-step wizard GUI (PyQt5)
gui_tabs/       Tabbed panel GUI (PyQt5)
gui_modern/     Modern GUI (PySide6)
```

## AEDT Integration

The toolkit connects to AEDT v2021.1+ via COM automation:

1. Connect to a running AEDT instance or browse to an `.aedt` file directly
2. Select project and design from dropdowns
3. Variables and outputs are auto-detected by parsing the `.aedt` file
4. During training, the toolkit automates Maxwell parametric sweeps: export request CSV → import into AEDT → run sweep → collect results

When multiple `.aedt` files exist in the same directory, the toolkit matches by project name to avoid loading the wrong design.

## Dependencies

Python 3.10+ with:
- PyQt5 or PySide6 (depending on GUI)
- pywin32 (COM automation for AEDT)
- torch, numpy, pandas, scikit-learn (ML pipeline)
- scipy, matplotlib (analysis and visualization)

Optional: `faiss-cpu` for accelerated clustering (falls back to scikit-learn KMeans).

## License

Proprietary