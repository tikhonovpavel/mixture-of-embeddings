import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot residual similarity vs mixing cosine from CSV")
    p.add_argument("csv", type=str, nargs="+", help="One or more CSV files produced by run_similarity_compute.py")
    p.add_argument("--output", type=str, default=None, help="Optional path to save the figure (PNG)")
    p.add_argument("--bin-lines", action="store_true", help="Overlay binned mean curves")
    return p.parse_args()


def render_scatter(df: pd.DataFrame, models: List[str], output: str = None, bin_lines: bool = False) -> None:
    # Choose up to 4 models for a 2x2 grid; if fewer, still render available ones
    unique_models = models or sorted(df["model"].unique().tolist())
    show_models = unique_models[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten().tolist()

    for i, model in enumerate(show_models):
        ax = axes[i]
        sub = df[df["model"] == model]
        if sub.empty:
            ax.set_title(f"{model} (no data)")
            ax.axis("off")
            continue
        x = sub["similarity"].to_numpy()
        y = sub["cosine_pre_norm_mix_sum"].to_numpy()
        ax.scatter(x, y, s=6, alpha=0.35)
        ax.set_title(model)
        ax.set_xlabel("Residual similarity (A vs B) at chosen depth")
        ax.set_ylabel("Cosine (pre-norm mix vs A+B)")
        ax.grid(True, linestyle="--", alpha=0.4)

        if bin_lines:
            bins = np.linspace(-1.0, 1.0, 21)
            inds = np.digitize(x, bins)
            centers = 0.5 * (bins[:-1] + bins[1:])
            means = [np.mean(y[inds == b]) if np.any(inds == b) else np.nan for b in range(1, len(bins))]
            mask = ~np.isnan(means)
            if np.any(mask):
                ax.plot(centers[mask], np.array(means)[mask], color="black", linewidth=2)

    fig.tight_layout()
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    # Merge multiple CSVs if provided
    frames = [pd.read_csv(path) for path in args.csv]
    df = pd.concat(frames, ignore_index=True)

    # Preserve provided model ordering if possible
    models = []
    if "model" in df.columns:
        models = sorted(df["model"].unique().tolist())

    output = args.output
    if output is None:
        # Build a default output name based on the first CSV
        first = os.path.abspath(args.csv[0])
        base, _ = os.path.splitext(first)
        output = base + "_scatter.png"

    render_scatter(df, models, output=output, bin_lines=args.bin_lines)


if __name__ == "__main__":
    main()



