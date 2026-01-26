import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment results from batch CSV output.")
    parser.add_argument("csv", type=str, help="Path to the results CSV file (e.g., batch_...csv)")
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
    parser.add_argument(
        "--no-plot-proj-cos",
        action="store_true",
        help="If set, do not include the projection-cosine bottom row, even if present in CSV.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "ev"],
        help="Metric to plot for layerwise curve and aggregates: 'cosine' or 'ev' (explained variance).",
    )
    parser.add_argument(
        "--ev-layers",
        type=str,
        default=None,
        help="Optional path to EV-by-layer CSV (<base>_ev_layers.csv) to plot EV on the second row instead of cosine.",
    )
    return parser.parse_args()


def load_and_prepare(csv_path: str, metric: str = "cosine") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    metric_col = "cosine_pre_norm_mix_sum" if metric == "cosine" else "ev_pre_norm_mix_sum"
    required_cols = {
        "model",
        "token_position",
        "sequence_length",
        "rank_top1_a_in_mix_half",
        "rank_top1_b_in_mix_half",
        "prob_top1_a",
        "prob_top1_b",
    }
    # Only require the metric_col if it exists in the CSV; otherwise, we can still plot ranks
    if metric_col in df.columns:
        required_cols = set(required_cols) | {metric_col}
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
        ]
        if metric_col in df.columns:
            cols.append(metric_col)
        frame = df[cols]
        renames = {
            f"rank_top1_{suffix}_in_mix_half": "rank_half",
            f"prob_top1_{suffix}": "prob_top1",
        }
        if metric_col in df.columns:
            renames[metric_col] = "metric_value"
        stacked_rows.append(
            frame.rename(columns=renames).assign(sample=f"top1_{suffix}")
        )

    stacked = pd.concat(stacked_rows, ignore_index=True)
    return stacked


def compute_position_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "token_position"], as_index=False)
        .agg(
            median_rank=("rank_half", "median"),
            mean_metric=("metric_value", "mean"),
        )
        .sort_values(["model", "token_position"])
    )
    return grouped


def compute_probability_aggregates(df: pd.DataFrame, bins: int) -> Tuple[pd.DataFrame, np.ndarray]:
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
            mean_metric=("metric_value", "mean"),
        )
        .dropna(subset=["prob_bin"])
        .sort_values(["model", "prob_bin"])
    )
    grouped["prob_bin"] = grouped["prob_bin"].astype(float)
    return grouped, bin_edges


def setup_axes(include_proj: bool) -> Tuple[plt.Figure, List[List[plt.Axes]]]:
    if include_proj:
        fig = plt.figure(figsize=(14, 14))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0])
        ax_rank_pos = fig.add_subplot(gs[0, 0])
        ax_rank_prob = fig.add_subplot(gs[0, 1])
        ax_layer_cos = fig.add_subplot(gs[1, :])
        ax_layer_proj = fig.add_subplot(gs[2, :])
        axes = [[ax_rank_pos, ax_rank_prob], [ax_layer_cos, ax_layer_proj]]
    else:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
        ax_rank_pos = fig.add_subplot(gs[0, 0])
        ax_rank_prob = fig.add_subplot(gs[0, 1])
        ax_layer_cos = fig.add_subplot(gs[1, :])
        axes = [[ax_rank_pos, ax_rank_prob], [ax_layer_cos, None]]

    ax_rank_pos.set_title("Median rank (position blocks)")
    ax_rank_prob.set_title("Median rank (probability)")

    ax_rank_pos.set_xlabel("Token position (block midpoint)")
    ax_rank_pos.set_ylabel("Median rank / 2")
    ax_rank_prob.set_xlabel("Top-1 probability")
    ax_rank_prob.set_ylabel("Median rank / 2")

    return fig, axes


def plot_lines(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str):
    ax.plot(x, y, marker="o", label=label)


def render_plots(
    pos_block_stats: pd.DataFrame,
    prob_stats: pd.DataFrame,
    fig: plt.Figure,
    axes: List[List[plt.Axes]],
    df_full: pd.DataFrame,
    metric: str,
):
    models = sorted(set(pos_block_stats["model"]) | set(prob_stats["model"]))
    for model in models:
        pos_block_subset = pos_block_stats[pos_block_stats["model"] == model]
        prob_subset = prob_stats[prob_stats["model"] == model]

        if not pos_block_subset.empty:
            plot_lines(
                axes[0][0],
                pos_block_subset["pos_bin_mid"].to_numpy(),
                pos_block_subset["median_rank"].to_numpy(),
                model,
            )

        if not prob_subset.empty:
            plot_lines(
                axes[0][1],
                prob_subset["prob_bin"].to_numpy(),
                prob_subset["median_rank"].to_numpy(),
                model,
            )

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

    # Render layerwise curve on second row
    ax_layer = axes[1][0]
    if metric == "ev":
        # Expect a companion EV file to be provided; fall back quietly if not
        # We will read it in main and pass via df_full when metric == ev
        if "layer_percent" in df_full.columns and "ev" in df_full.columns:
            for model in models:
                sub = df_full[df_full["model"] == model]
                if sub.empty:
                    continue
                xs = sub.sort_values("layer_percent")["layer_percent"].to_numpy()
                ys = sub.sort_values("layer_percent")["ev"].to_numpy()
                ax_layer.plot(xs, ys, marker="o", label=model)
            ax_layer.set_title("Explained variance vs layer depth")
            ax_layer.set_xlabel("Layer percent")
            ax_layer.set_ylabel("Mean explained variance")
            if not df_full.empty:
                ax_layer.set_xticks(sorted(df_full["layer_percent"].unique().tolist()))
            ax_layer.grid(True, linestyle="--", alpha=0.4)
    else:
        prefix = "cosine_residual_pct"
        layer_cols = sorted(
            [c for c in df_full.columns if c.startswith(prefix)],
            key=lambda c: int(c[len(prefix):])
        )
        if layer_cols:
            percents = [int(c[len(prefix):]) for c in layer_cols]
            for model in models:
                sub = df_full[df_full["model"] == model]
                means = [float(sub[c].mean()) for c in layer_cols]
                ax_layer.plot(percents, means, marker="o", label=model)
            ax_layer.set_title("Cosine similarity vs layer depth")
            ax_layer.set_xlabel("Layer percent")
            ax_layer.set_ylabel("Mean cosine similarity")
            ax_layer.set_xticks(percents)
            ax_layer.grid(True, linestyle="--", alpha=0.4)

    # Render projection-based cosine (row 3 wide axis) only if axis exists
    if axes[1][1] is not None:
        prefix_proj = "cosine_projspan_pct"
        proj_cols = sorted(
            [c for c in df_full.columns if c.startswith(prefix_proj)],
            key=lambda c: int(c[len(prefix_proj):])
        )
        if proj_cols:
            percents_proj = [int(c[len(prefix_proj):]) for c in proj_cols]
            ax_layer_proj = axes[1][1]
            for model in models:
                sub = df_full[df_full["model"] == model]
                means = [float(sub[c].mean()) for c in proj_cols]
                ax_layer_proj.plot(percents_proj, means, marker="o", label=model)
            ax_layer_proj.set_title("Projection cosine (z vs Proj_{span(a,b)} z) vs layer depth")
            ax_layer_proj.set_xlabel("Layer percent")
            ax_layer_proj.set_ylabel("Mean projection cosine")
            ax_layer_proj.set_xticks(percents_proj)
            ax_layer_proj.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=(0.12, 0.02, 1, 1))


def build_layer_cosine_figure(df: pd.DataFrame, output: str = None) -> None:
    """
    If layerwise residual cosine columns are present (cosine_residual_pct10..100),
    render a figure with x = percent of layers, y = mean cosine across tokens, one line per model.
    """
    prefix = "cosine_residual_pct"
    layer_cols = sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(c[len(prefix):])
    )
    if not layer_cols:
        return

    percents = [int(c[len(prefix):]) for c in layer_cols]
    models = sorted(df["model"].unique())

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for model in models:
        sub = df[df["model"] == model]
        # Mean over token rows for each percent
        means = [float(sub[c].mean()) for c in layer_cols]
        ax.plot(percents, means, marker="o", label=model)

    ax.set_title("Layerwise residual cosine (mix vs A+B)")
    ax.set_xlabel("Layer percent")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xticks(percents)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Model")

    if output:
        base, ext = os.path.splitext(os.path.abspath(output))
        out_path = f"{base}_layers{ext or '.png'}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved layerwise plot to {out_path}")
    else:
        plt.show()

def compute_position_block_aggregates(df: pd.DataFrame, block: int) -> pd.DataFrame:
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
        )
        .dropna(subset=["pos_bin"]) 
        .sort_values(["model", "pos_bin"])
    )
    grouped["pos_bin_mid"] = grouped["pos_bin"].astype(float)
    return grouped


def main() -> None:
    args = parse_args()

    # For top-row plots we don't need EV per-token; keep metric arg only for layer curve fallback
    stacked = load_and_prepare(args.csv, metric="cosine")
    pos_block_stats = compute_position_block_aggregates(stacked, args.pos_block)
    prob_stats, _ = compute_probability_aggregates(stacked, args.prob_bins)

    if args.metric == "ev" and args.ev_layers and os.path.exists(args.ev_layers):
        df_layer = pd.read_csv(args.ev_layers)
    else:
        df_layer = pd.read_csv(args.csv)

    # Decide whether to include projection-cosine row
    include_proj = not args.no_plot_proj_cos and any(col.startswith("cosine_projspan_pct") for col in pd.read_csv(args.csv).columns)
    fig, axes = setup_axes(include_proj)
    render_plots(pos_block_stats, prob_stats, fig, axes, df_layer, args.metric)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

