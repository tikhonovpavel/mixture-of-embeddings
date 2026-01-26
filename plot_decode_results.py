import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot decode-with-small-model experiment results")
    parser.add_argument("csv", type=str, help="Path to the results CSV file")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the figure (PNG)")
    parser.add_argument("--prob-bins", type=int, default=20, help="Number of bins for probability aggregation")
    parser.add_argument("--pos-block", type=int, default=20, help="Token position block size")
    return parser.parse_args()


def load_and_stack(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Accept presence or absence of B columns
    stacked: List[pd.DataFrame] = []
    # A-side
    if {"accuracy_a", "rank_clean_top1_a_in_filtered", "prob_top1_a_big"}.issubset(df.columns):
        stacked.append(
            df[[
                "model_big",
                "token_position",
                "sequence_length",
                "prob_top1_a_big",
                "accuracy_a",
                "rank_clean_top1_a_in_filtered",
            ]]
            .rename(
                columns={
                    "prob_top1_a_big": "prob_top1",
                    "rank_clean_top1_a_in_filtered": "rank",
                    "accuracy_a": "accuracy",
                }
            )
            .assign(side="A")
        )
    # B-side (optional)
    if {"accuracy_b", "rank_clean_top1_b_in_filtered", "prob_top1_b_big"}.issubset(df.columns):
        stacked.append(
            df[[
                "model_big",
                "token_position",
                "sequence_length",
                "prob_top1_b_big",
                "accuracy_b",
                "rank_clean_top1_b_in_filtered",
            ]]
            .rename(
                columns={
                    "prob_top1_b_big": "prob_top1",
                    "rank_clean_top1_b_in_filtered": "rank",
                    "accuracy_b": "accuracy",
                }
            )
            .assign(side="B")
        )

    if not stacked:
        raise ValueError("CSV missing required columns for plotting.")

    return pd.concat(stacked, ignore_index=True)


def compute_position_block_aggregates(df: pd.DataFrame, block: int) -> pd.DataFrame:
    df = df.copy()
    max_pos = int(df["token_position"].max())
    edges = np.arange(0, max_pos + block + 1, block)
    mids = 0.5 * (edges[:-1] + edges[1:])
    df["pos_bin"] = pd.cut(df["token_position"], bins=edges, labels=mids, include_lowest=True, right=False)
    grouped = (
        df.groupby(["model_big", "pos_bin"], as_index=False, observed=True)
        .agg(
            mean_acc=("accuracy", "mean"),
            median_rank=("rank", "median"),
        )
        .dropna(subset=["pos_bin"]) 
        .sort_values(["model_big", "pos_bin"]) 
    )
    grouped["pos_bin_mid"] = grouped["pos_bin"].astype(float)
    return grouped


def compute_probability_aggregates(df: pd.DataFrame, bins: int) -> Tuple[pd.DataFrame, np.ndarray]:
    eps = 1e-8
    df = df.copy()
    df["prob_top1"] = df["prob_top1"].clip(eps, 1.0)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_labels = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    df["prob_bin"] = pd.cut(df["prob_top1"], bins=bin_edges, labels=bin_labels, include_lowest=True)
    grouped = (
        df.groupby(["model_big", "prob_bin"], as_index=False, observed=True)
        .agg(
            mean_acc=("accuracy", "mean"),
            median_rank=("rank", "median"),
        )
        .dropna(subset=["prob_bin"]) 
        .sort_values(["model_big", "prob_bin"]) 
    )
    grouped["prob_bin"] = grouped["prob_bin"].astype(float)
    return grouped, bin_edges


def setup_axes() -> Tuple[plt.Figure, List[List[plt.Axes]]]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.tolist()
    axes[0][0].set_title("Mean accuracy (by position)")
    axes[0][1].set_title("Mean accuracy (by prob)")
    axes[1][0].set_title("Median rank (by position)")
    axes[1][1].set_title("Median rank (by prob)")

    axes[0][0].set_xlabel("Token position (block midpoint)")
    axes[0][0].set_ylabel("Mean accuracy")
    axes[0][1].set_xlabel("Top-1 probability")
    axes[0][1].set_ylabel("Mean accuracy")

    axes[1][0].set_xlabel("Token position (block midpoint)")
    axes[1][0].set_ylabel("Median rank")
    axes[1][1].set_xlabel("Top-1 probability")
    axes[1][1].set_ylabel("Median rank")

    return fig, axes


def plot_lines(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str):
    ax.plot(x, y, marker="o", label=label)


def render_plots(
    pos_stats: pd.DataFrame,
    prob_stats: pd.DataFrame,
    fig: plt.Figure,
    axes: List[List[plt.Axes]],
):
    models = sorted(set(pos_stats["model_big"]) | set(prob_stats["model_big"]))
    for model in models:
        pos_sub = pos_stats[pos_stats["model_big"] == model]
        prob_sub = prob_stats[prob_stats["model_big"] == model]

        if not pos_sub.empty:
            plot_lines(axes[0][0], pos_sub["pos_bin_mid"].to_numpy(), pos_sub["mean_acc"].to_numpy(), model)
            plot_lines(axes[1][0], pos_sub["pos_bin_mid"].to_numpy(), pos_sub["median_rank"].to_numpy(), model)

        if not prob_sub.empty:
            plot_lines(axes[0][1], prob_sub["prob_bin"].to_numpy(), prob_sub["mean_acc"].to_numpy(), model)
            plot_lines(axes[1][1], prob_sub["prob_bin"].to_numpy(), prob_sub["median_rank"].to_numpy(), model)

    for row in axes:
        for ax in row:
            ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, title="Big model", loc="center left", bbox_to_anchor=(0.02, 0.5), frameon=True)

    fig.tight_layout(rect=(0.12, 0.02, 1, 1))


def main() -> None:
    args = parse_args()
    df = load_and_stack(args.csv)
    pos_stats = compute_position_block_aggregates(df, args.pos_block)
    prob_stats, _ = compute_probability_aggregates(df, args.prob_bins)

    fig, axes = setup_axes()
    render_plots(pos_stats, prob_stats, fig, axes)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()


