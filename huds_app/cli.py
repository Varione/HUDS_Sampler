"""Command line interface for the HUDS active learning app."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .config import load_config


def _run_config_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "config.json"


def _load_run_config(run_dir: str | Path) -> Any:
    return load_config(str(_run_config_path(run_dir)))


def _print_result(result: Any, success_message: str | None = None) -> None:
    if success_message:
        print(success_message)

    if result is None:
        return
    if isinstance(result, Path):
        print(f"Output: {result}")
        return
    if isinstance(result, (str, int, float, bool)):
        print(result)
        return
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, default=str))
        return

    print(result)


def _load_module(name: str) -> Any:
    return importlib.import_module(f"huds_app.{name}")


def _run_with_error_handling(handler: Callable[[argparse.Namespace], Any], args: argparse.Namespace) -> int:
    try:
        handler(args)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _handle_init(args: argparse.Namespace) -> None:
    workflow = _load_module("workflow")
    load_config(args.config)  # Validate config exists and is valid before proceeding
    result = workflow.init_run(args.config, args.out, snap_to_levels=args.snap_to_levels)
    _print_result(result, f"Initialized run: {args.out}")


def _handle_export_validation(args: argparse.Namespace) -> None:
    validation = _load_module("validation")
    config = _load_run_config(args.run)
    output_path = validation.export_validation_request(args.run, config, size=args.size)
    _print_result(output_path, "Exported validation request.")


def _handle_export_initial_train(args: argparse.Namespace) -> None:
    validation = _load_module("validation")
    config = _load_run_config(args.run)
    output_path = validation.export_initial_train_request(args.run, config)
    _print_result(output_path, "Exported initial training request.")


def _handle_import_labels(args: argparse.Namespace) -> None:
    """Handle import-labels command with overwrite and allow_partial flags.
    
    FIX 7: Add CLI support for --overwrite and --allow-partial options.
    """
    validation = _load_module("validation")
    imported_count = validation.import_labels(
        args.run, 
        args.kind, 
        args.step, 
        args.input,
        overwrite=getattr(args, 'overwrite', False),  # FIX 7: Add overwrite flag support
        allow_partial=getattr(args, 'allow_partial', False)  # FIX 7: Add allow-partial flag support
    )
    
    if getattr(args, 'allow_partial', False):
        print(f"Imported {imported_count} {args.kind} label row(s). Note: Partial import was allowed.")
    else:
        print(f"Imported {imported_count} {args.kind} label row(s).")


def _handle_train(args: argparse.Namespace) -> None:
    train_model = _load_module("train").train_model
    config = _load_run_config(args.run)
    metrics = train_model(args.run, config)
    _print_result(metrics, "Training completed.")


def _handle_sample(args: argparse.Namespace) -> None:
    run_huds_sampling = _load_module("huds").run_huds_sampling
    config = _load_run_config(args.run)
    result = run_huds_sampling(args.run, config, args.step)
    _print_result(result, f"Sampled active learning step {args.step}.")


def _handle_status(args: argparse.Namespace) -> None:
    workflow = _load_module("workflow")
    _load_run_config(args.run)  # Validate config exists
    result = workflow.show_status(args.run)
    _print_result(result)


def _handle_validate_files(args: argparse.Namespace) -> None:
    workflow = _load_module("workflow")
    _load_run_config(args.run)  # Validate config exists
    result = workflow.validate_files(args.run)
    _print_result(result, "Validated run files.")


def _handle_inspect_config(args: argparse.Namespace) -> None:
    workflow = _load_module("workflow")
    load_config(args.config)  # Validate and load config for inspection
    result = workflow.inspect_config(args.config)
    _print_result(result)


def _handle_predict(args: argparse.Namespace) -> None:
    workflow = _load_module("workflow")
    _load_run_config(args.run)  # Validate config exists
    result = workflow.predict(args.run, args.input, args.output)
    _print_result(result, f"Wrote predictions: {args.output}")


def _handle_evaluate(args: argparse.Namespace) -> None:
    workflow = _load_module("workflow")
    _load_run_config(args.run)  # Validate config exists
    result = workflow.evaluate(args.run)
    _print_result(result, "Evaluation completed.")


def _add_run_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run directory, for example runs/demo.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="huds-app",
        description="File-based HUDS active learning workflow CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a new HUDS run.")
    init_parser.add_argument("--config", required=True, help="Path to the JSON configuration file.")
    init_parser.add_argument("--out", required=True, help="Output run directory.")
    init_parser.add_argument("--snap-to-levels", action="store_true", help="Snap generated samples to variable levels.")
    init_parser.set_defaults(handler=_handle_init)

    export_validation_parser = subparsers.add_parser("export-validation", help="Export validation samples for simulation.")
    _add_run_argument(export_validation_parser)
    export_validation_parser.add_argument("--size", type=int, help="Number of validation samples to export.")
    export_validation_parser.set_defaults(handler=_handle_export_validation)

    export_initial_train_parser = subparsers.add_parser("export-initial-train", help="Export initial training samples.")
    _add_run_argument(export_initial_train_parser)
    export_initial_train_parser.set_defaults(handler=_handle_export_initial_train)

    import_labels_parser = subparsers.add_parser("import-labels", help="Import simulator labels into the run datasets.")
    _add_run_argument(import_labels_parser)
    import_labels_parser.add_argument("--kind", required=True, choices=("validation", "train"), help="Label destination kind.")
    import_labels_parser.add_argument("--step", type=int, help="Training request step, required for --kind train.")
    import_labels_parser.add_argument("--input", required=True, help="Simulator output CSV path.")
    # FIX 7: Add CLI flags for overwrite and allow-partial options
    import_labels_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labeled data.")
    import_labels_parser.add_argument("--allow-partial", action="store_true", help="Allow partial label imports when not all request IDs are covered.")
    import_labels_parser.set_defaults(handler=_handle_import_labels)

    train_parser = subparsers.add_parser("train", help="Train the surrogate model.")
    _add_run_argument(train_parser)
    train_parser.set_defaults(handler=_handle_train)

    sample_parser = subparsers.add_parser("sample", help="Run HUDS sampling for the next training request.")
    _add_run_argument(sample_parser)
    sample_parser.add_argument("--step", required=True, type=int, help="Active learning step number.")
    sample_parser.set_defaults(handler=_handle_sample)

    status_parser = subparsers.add_parser("status", help="Show run status and next recommended action.")
    _add_run_argument(status_parser)
    status_parser.set_defaults(handler=_handle_status)

    validate_files_parser = subparsers.add_parser("validate-files", help="Validate expected run files.")
    _add_run_argument(validate_files_parser)
    validate_files_parser.set_defaults(handler=_handle_validate_files)

    inspect_config_parser = subparsers.add_parser("inspect-config", help="Inspect a JSON configuration file.")
    inspect_config_parser.add_argument("--config", required=True, help="Path to the JSON configuration file.")
    inspect_config_parser.set_defaults(handler=_handle_inspect_config)

    predict_parser = subparsers.add_parser("predict", help="Run model predictions for candidate rows.")
    _add_run_argument(predict_parser)
    predict_parser.add_argument("--input", required=True, help="Input candidates CSV path.")
    predict_parser.add_argument("--output", required=True, help="Output predictions CSV path.")
    predict_parser.set_defaults(handler=_handle_predict)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the current model on validation data.")
    _add_run_argument(evaluate_parser)
    evaluate_parser.set_defaults(handler=_handle_evaluate)

    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'handler'):
        return _run_with_error_handling(args.handler, args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
