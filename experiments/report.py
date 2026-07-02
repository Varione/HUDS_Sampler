"""HUDS vs Random Benchmark Report Generator.

Aggregates all benchmark CSV files and generates:
1. Combined CSV with all results
2. Markdown summary table
3. Per-scenario comparison tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all bench_*.csv files from results directory."""
    csv_files = sorted(results_dir.glob("bench_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No bench_*.csv files found in {results_dir}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def generate_summary(df: pd.DataFrame) -> str:
    """Generate markdown summary report."""
    lines = []
    lines.append("# HUDS vs Random Sampling Benchmark Report\n")

    # Overall summary
    scenarios = sorted(df["scenario_id"].unique())
    lines.append("## Overview\n")
    lines.append(f"- **Scenarios:** {len(scenarios)}")
    lines.append(f"- **Strategies:** {', '.join(sorted(df['strategy'].unique()))}")
    lines.append(f"- **Total data points:** {len(df)}\n")

    # Per-scenario comparison tables
    for scenario in scenarios:
        sdf = df[df["scenario_id"] == scenario]
        model_type = sdf["model_type"].iloc[0]
        lines.append(f"## {scenario} ({model_type})\n")

        strategies = sorted(sdf["strategy"].unique())
        if len(strategies) < 2:
            lines.append(f"> Only strategy available: {', '.join(strategies)}\n")

        # Build comparison table
        steps = sorted(sdf["step"].unique())
        header = "| Step | labeled |"
        for s in strategies:
            header += f" {s.upper()} R2 |"
        lines.append(header)
        lines.append("|------|---------|" + "|------|" * len(strategies))

        for step in steps:
            sdf_step = sdf[sdf["step"] == step]
            row = f"| {int(step)} | {int(sdf_step['labeled_count'].iloc[0])} |"
            for s in strategies:
                r = sdf_step[sdf_step["strategy"] == s]
                if len(r) > 0:
                    row += f" {r['val_r2_avg'].iloc[0]:.4f} |"
                else:
                    row += " N/A |"
            lines.append(row)

        # Final comparison (last step)
        lines.append("\n**Final step comparison:**\n")
        last_step = int(max(sdf["step"]))
        sdf_last = sdf[sdf["step"] == last_step]
        for s in strategies:
            r = sdf_last[sdf_last["strategy"] == s]
            if len(r) > 0:
                lines.append(
                    f"- **{s.upper()}**: R2={r['val_r2_avg'].iloc[0]:.4f}, "
                    f"labeled={int(r['labeled_count'].iloc[0])}, "
                    f"time={r['elapsed_s'].iloc[0]:.1f}s"
                )

        lines.append("")

    # Sample efficiency analysis
    lines.append("## Sample Efficiency Analysis\n")
    lines.append("Steps needed to reach R2 >= 0.5:\n")
    lines.append("| Scenario | Random | HUDS |\n")
    lines.append("|----------|--------|------|\n")

    for scenario in scenarios:
        sdf = df[df["scenario_id"] == scenario]
        strategies = sorted(sdf["strategy"].unique())

        random_step = "N/A"
        huds_step = "N/A"

        for s in strategies:
            r = sdf[sdf["strategy"] == s].sort_values("step")
            reached = r[r["val_r2_avg"] >= 0.5]
            if len(reached) > 0:
                val = f"{int(reached['step'].iloc[0])} ({int(reached['labeled_count'].iloc[0])} samples)"
                if s == "random":
                    random_step = val
                else:
                    huds_step = val

        lines.append(f"| {scenario} | {random_step} | {huds_step} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate HUDS benchmark report")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory (default: experiments/results)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output combined CSV path",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Output Markdown report path",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path("experiments/results")
    output_csv = Path(args.output_csv) if args.output_csv else results_dir / "benchmark_combined.csv"
    output_md = Path(args.output_md) if args.output_md else results_dir / "benchmark_report.md"

    print(f"Loading results from: {results_dir}")
    df = load_all_results(results_dir)

    # Save combined CSV
    df.to_csv(output_csv, index=False)
    print(f"Combined CSV saved to: {output_csv}")

    # Generate Markdown report
    report = generate_summary(df)
    output_md.write_text(report, encoding="utf-8")
    print(f"Markdown report saved to: {output_md}")

    print(f"\nTotal records: {len(df)}")
    print(f"Scenarios: {', '.join(sorted(df['scenario_id'].unique()))}")


if __name__ == "__main__":
    main()
