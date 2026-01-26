import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TVD temperature optimization results from batch CSV output.")
    parser.add_argument("csv", type=str, help="Path to the results CSV file (e.g., batch_tvd_...csv)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the figure (PNG). If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--prob-bins",
        type=int,
        default=20,
        help="Number of bins for aggregating over token probabilities.",
    )
    parser.add_argument(
        "--pos-block",
        type=int,
        default=20,
        help="Block size (in tokens) for aggregating over token positions.",
    )
    return parser.parse_args()


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {
        "model",
        "token_position",
        "sequence_length",
        "rank_top1_a_in_mix_half",
        "rank_top1_b_in_mix_half",
        "prob_top1_a",
        "prob_top1_b",
        "tvd_optimal",
    }
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Treat A/B samples as IID by stacking them.
    stacked_rows: List[pd.DataFrame] = []
    for suffix in ("a", "b"):
        cols = [
            "model",
            "token_position",
            "sequence_length",
            f"rank_top1_{suffix}_in_mix_half",
            f"prob_top1_{suffix}",
            "tvd_optimal",
        ]
        frame = df[cols]
        renames = {
            f"rank_top1_{suffix}_in_mix_half": "rank_half",
            f"prob_top1_{suffix}": "prob_top1",
        }
        stacked_rows.append(
            frame.rename(columns=renames).assign(sample=f"top1_{suffix}")
        )

    stacked = pd.concat(stacked_rows, ignore_index=True)
    return stacked


def compute_position_block_aggregates(df: pd.DataFrame, block: int) -> pd.DataFrame:
    """Aggregate by position blocks for rank and TVD."""
    df = df.copy()
    max_pos = int(df["token_position"].max())
    # Create bin edges every `block` tokens, inclusive of the last token
    edges = np.arange(0, max_pos + block + 1, block)
    # Midpoints for labeling
    mids = 0.5 * (edges[:-1] + edges[1:])
    df["pos_bin"] = pd.cut(df["token_position"], bins=edges, labels=mids, include_lowest=True, right=False)
    grouped = (
        df.groupby(["model", "pos_bin"], as_index=False, observed=True)
        .agg(
            median_rank=("rank_half", "median"),
            mean_tvd=("tvd_optimal", "mean"),
        )
        .dropna(subset=["pos_bin"]) 
        .sort_values(["model", "pos_bin"])
    )
    grouped["pos_bin_mid"] = grouped["pos_bin"].astype(float)
    return grouped


def compute_probability_aggregates(df: pd.DataFrame, bins: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """Aggregate by probability bins for rank and TVD."""
    eps = 1e-8
    df = df.copy()
    df["prob_top1"] = df["prob_top1"].clip(eps, 1.0)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_labels = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    df["prob_bin"] = pd.cut(df["prob_top1"], bins=bin_edges, labels=bin_labels, include_lowest=True)

    grouped = (
        df.groupby(["model", "prob_bin"], as_index=False, observed=True)
        .agg(
            median_rank=("rank_half", "median"),
            mean_tvd=("tvd_optimal", "mean"),
        )
        .dropna(subset=["prob_bin"])
        .sort_values(["model", "prob_bin"])
    )
    grouped["prob_bin"] = grouped["prob_bin"].astype(float)
    return grouped, bin_edges


def setup_axes() -> Tuple[plt.Figure, List[List[plt.Axes]]]:
    """Create 2x2 figure with 4 subplots."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    
    ax_rank_pos = fig.add_subplot(gs[0, 0])
    ax_rank_prob = fig.add_subplot(gs[0, 1])
    ax_tvd_pos = fig.add_subplot(gs[1, 0])
    ax_tvd_prob = fig.add_subplot(gs[1, 1])
    
    ax_rank_pos.set_title("Median rank (position blocks)")
    ax_rank_prob.set_title("Median rank (probability)")
    ax_tvd_pos.set_title("Optimal TVD (position blocks)")
    ax_tvd_prob.set_title("Optimal TVD (probability)")

    ax_rank_pos.set_xlabel("Token position (block midpoint)")
    ax_rank_pos.set_ylabel("Median rank / 2")
    ax_rank_prob.set_xlabel("Top-1 probability")
    ax_rank_prob.set_ylabel("Median rank / 2")
    ax_tvd_pos.set_xlabel("Token position (block midpoint)")
    ax_tvd_pos.set_ylabel("Mean optimal TVD")
    ax_tvd_prob.set_xlabel("Top-1 probability")
    ax_tvd_prob.set_ylabel("Mean optimal TVD")

    axes = [[ax_rank_pos, ax_rank_prob], [ax_tvd_pos, ax_tvd_prob]]
    return fig, axes


def plot_lines(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str):
    ax.plot(x, y, marker="o", label=label)


def render_plots(
    pos_block_stats: pd.DataFrame,
    prob_stats: pd.DataFrame,
    fig: plt.Figure,
    axes: List[List[plt.Axes]],
):
    models = sorted(set(pos_block_stats["model"]) | set(prob_stats["model"]))
    
    for model in models:
        pos_block_subset = pos_block_stats[pos_block_stats["model"] == model]
        prob_subset = prob_stats[prob_stats["model"] == model]

        # Top-left: median rank vs position
        if not pos_block_subset.empty:
            plot_lines(
                axes[0][0],
                pos_block_subset["pos_bin_mid"].to_numpy(),
                pos_block_subset["median_rank"].to_numpy(),
                model,
            )

        # Top-right: median rank vs probability
        if not prob_subset.empty:
            plot_lines(
                axes[0][1],
                prob_subset["prob_bin"].to_numpy(),
                prob_subset["median_rank"].to_numpy(),
                model,
            )

        # Bottom-left: TVD vs position
        if not pos_block_subset.empty:
            plot_lines(
                axes[1][0],
                pos_block_subset["pos_bin_mid"].to_numpy(),
                pos_block_subset["mean_tvd"].to_numpy(),
                model,
            )

        # Bottom-right: TVD vs probability
        if not prob_subset.empty:
            plot_lines(
                axes[1][1],
                prob_subset["prob_bin"].to_numpy(),
                prob_subset["mean_tvd"].to_numpy(),
                model,
            )

    # Add grids and legend
    for row in axes:
        for ax in row:
            if ax is not None:
                ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(
            handles,
            labels,
            title="Model",
            loc="center left",
            bbox_to_anchor=(0.02, 0.5),
            frameon=True,
        )

    fig.tight_layout(rect=(0.12, 0.02, 1, 1))


def main() -> None:
    args = parse_args()

    # Load and prepare data
    stacked = load_and_prepare(args.csv)
    pos_block_stats = compute_position_block_aggregates(stacked, args.pos_block)
    prob_stats, _ = compute_probability_aggregates(stacked, args.prob_bins)

    # Create figure
    fig, axes = setup_axes()
    render_plots(pos_block_stats, prob_stats, fig, axes)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

