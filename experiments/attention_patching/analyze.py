"""Aggregate patching-experiment results and produce the final artifacts.

Given one or more ``*.parquet`` files from ``run_patching.py`` (one per
model), produce:

    * Rank-CDF plot (baseline vs A2) -- styled after fig:rank_curve in the
      main paper.
    * Entropy histogram of P_{A1} vs P_{A2} (sanity check that A2 isn't just
      collapsing to uniform).
    * Summary table (CSV + markdown) in the format of tab:dist_metrics, with
      KL/JS/Wasserstein and the corresponding Superposition Approximation
      Ratios.
    * Markdown report (``report.md``) with key findings.

Example:
    python -m experiments.attention_patching.analyze \
        --inputs experiments/attention_patching/results/*.parquet \
        --output_dir experiments/attention_patching/results/analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more parquet files (globs expanded) produced by run_patching.py.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="experiments/attention_patching/results/analysis",
    )
    return p.parse_args()


def _expand_globs(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        matched = sorted(glob.glob(p))
        if not matched and os.path.exists(p):
            matched = [p]
        out.extend(matched)
    return out


def _load_summary_json(parquet_path: str) -> Dict:
    json_path = parquet_path.replace(".parquet", ".json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _rank_cdf(ranks: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return np.array([np.mean(ranks <= x) for x in xs])


def plot_rank_cdf(
    dfs: Dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    xs = np.logspace(0, 3, 50)  # 1..1000

    # Colorblind-friendly styles
    linestyles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5))]
    markers = ["o", "s", "^", "D", "v"]
    # Explicitly pick high-contrast colors
    colors = ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3", "#ff7f00"]

    # Sort labels so that those containing 'permutation' appear last in the legend
    sorted_labels = sorted(dfs.keys(), key=lambda x: ("permutation" in x.lower(), x))

    for i, label in enumerate(sorted_labels):
        df = dfs[label]
        if "rank_top1_a1_in_a2" not in df.columns:
            continue
        ranks = df["rank_top1_a1_in_a2"].to_numpy()
        ys = _rank_cdf(ranks, xs)

        ls = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]

        # Plot line with markers at intervals to ensure distinguishability
        ax.plot(
            xs, ys,
            ls=ls,
            marker=marker,
            markevery=max(1, len(xs) // 10),
            markersize=6,
            lw=2,
            label=f"{label}: $A_2$ (patched)",
            color=color
        )

        # Baseline sanity check: rank of A1's own argmax in A1 is always 1
        ax.axhline(1.0, color="grey", lw=0.5, ls=":")

    ax.set_xscale("log")
    ax.set_xlabel(r"rank $i$ (top-$i$)")
    ax.set_ylabel(r"$P(\mathrm{rank}_{P_{A2}}(\hat{t}_{A1}) \leq i)$")
    ax.set_title(
        "Rank of top-1 A1 token under patched attention (A2)\n"
        "High curve => model is robust to donor-attention substitution"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_entropy_hist(
    dfs: Dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    n = len(dfs)
    fig, axes = plt.subplots(1, max(1, n), figsize=(5 * max(1, n), 4), squeeze=False)
    for i, (label, df) in enumerate(dfs.items()):
        ax = axes[0, i]
        if "entropy_a1" in df.columns:
            ax.hist(
                df["entropy_a1"].to_numpy(), bins=50, alpha=0.5,
                label=r"$H(P_{A1})$", color="C0",
            )
        if "entropy_a2" in df.columns:
            ax.hist(
                df["entropy_a2"].to_numpy(), bins=50, alpha=0.5,
                label=r"$H(P_{A2})$", color="C3",
            )
        ax.set_title(label)
        ax.set_xlabel("entropy (nats)")
        ax.set_ylabel("count")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_summary_table(
    dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for label, df in dfs.items():
        base = {"model": label, "n_rows": len(df)}

        def _mean(col):
            return float(np.mean(df[col].to_numpy())) if col in df.columns else np.nan

        num_kl = _mean("kl_a1_a2")
        num_js = _mean("js_a1_a2")
        num_ws = _mean("wasserstein_a1_a2")
        den_kl = _mean("kl_a1_aprime")
        den_js = _mean("js_a1_aprime")
        den_ws = _mean("wasserstein_a1_aprime")

        def _ratio(num, den):
            if np.isnan(num) or np.isnan(den) or den == 0:
                return np.nan
            return num / den

        # A1 vs A2 row (our patched condition)
        rows.append({
            **base,
            "condition": "A2 (patched)",
            "KL": num_kl, "JS": num_js, "Wasserstein": num_ws,
            "R_KL": _ratio(num_kl, den_kl),
            "R_JS": _ratio(num_js, den_js),
            "R_W":  _ratio(num_ws, den_ws),
        })
        # Baseline A1 vs A' row (denominator)
        rows.append({
            **base,
            "condition": "A1 vs A' (random baseline)",
            "KL": den_kl, "JS": den_js, "Wasserstein": den_ws,
            "R_KL": 1.0, "R_JS": 1.0, "R_W": 1.0,
        })
    return pd.DataFrame(rows)


def summary_to_markdown(summary: pd.DataFrame) -> str:
    cols = ["model", "condition", "KL", "JS", "Wasserstein", "R_KL", "R_JS", "R_W", "n_rows"]
    cols = [c for c in cols if c in summary.columns]
    df = summary[cols].copy()
    float_cols = [c for c in cols if df[c].dtype.kind == "f"]
    for c in float_cols:
        df[c] = df[c].apply(lambda v: f"{v:.4f}" if not np.isnan(v) else "—")
    return df.to_markdown(index=False)


def build_report(
    dfs: Dict[str, pd.DataFrame],
    summaries: Dict[str, Dict],
    output_dir: str,
) -> str:
    lines: List[str] = []
    lines.append("# Attention-Patching Experiment: Report\n")
    lines.append(
        "Setup: for each FineWeb text $A$ we replace every post-softmax "
        "attention map at every layer with the corresponding map from an "
        "unrelated donor text $C$. We keep $Q, K, V$ projections of $A$ "
        "intact; only the mixing weights change. No finetuning, no "
        "prompt manipulation.\n"
    )
    lines.append(
        "If the model is largely robust to this substitution, then "
        "'mixed embeddings don't break attention' cannot be what makes "
        "superposition work in the main paper -- because attention was "
        "wholesale replaced with unrelated content here, and the model "
        "still tracked the right tokens.\n"
    )

    for label, df in dfs.items():
        lines.append(f"## {label}\n")
        ranks = df["rank_top1_a1_in_a2"].to_numpy() if "rank_top1_a1_in_a2" in df.columns else None
        if ranks is not None:
            frac_top1 = float(np.mean(ranks <= 1))
            frac_top5 = float(np.mean(ranks <= 5))
            frac_top10 = float(np.mean(ranks <= 10))
            frac_top100 = float(np.mean(ranks <= 100))
            median_rank = float(np.median(ranks))
            lines.append(
                f"- rank(top-1 of A1) under A2: "
                f"P(rank=1)={frac_top1:.3f}, P(rank<=5)={frac_top5:.3f}, "
                f"P(rank<=10)={frac_top10:.3f}, P(rank<=100)={frac_top100:.3f}, "
                f"median rank={median_rank:.1f}.\n"
            )

        summ = summaries.get(label, {})
        r_kl = summ.get("R_kl")
        r_js = summ.get("R_js")
        r_w = summ.get("R_wasserstein")
        parts = []
        if r_kl is not None:
            parts.append(f"R_KL={r_kl:.3f}")
        if r_js is not None:
            parts.append(f"R_JS={r_js:.3f}")
        if r_w is not None:
            parts.append(f"R_Wasserstein={r_w:.3f}")
        if parts:
            lines.append(
                f"- Superposition Approximation Ratio "
                f"(patched vs. unrelated-text baseline): "
                + ", ".join(parts) + ".\n"
            )

        if "entropy_a1" in df.columns and "entropy_a2" in df.columns:
            ea1 = float(np.mean(df["entropy_a1"].to_numpy()))
            ea2 = float(np.mean(df["entropy_a2"].to_numpy()))
            lines.append(
                f"- Entropy sanity: mean H(P_A1)={ea1:.3f} nats, "
                f"mean H(P_A2)={ea2:.3f} nats "
                f"(delta={ea2 - ea1:+.3f}; should not blow up if the "
                f"patched distribution stays informative).\n"
            )

    lines.append("## Summary table\n")
    lines.append(summary_to_markdown(build_summary_table(dfs)))
    lines.append("\n")

    lines.append("## Interpretation\n")
    lines.append(
        "- If rank-CDF(A2) sits far above chance and R_* << 1, the output "
        "distribution remains close to the baseline even with fully "
        "unrelated attention maps. This empirically contradicts the folk "
        "explanation that superposition works because mixed embeddings "
        "induce 'almost-correct' attention; since here attention is "
        "explicitly wrong, yet predictions survive, the mechanism must "
        "live elsewhere (residual stream + MLP, plausibly also OV-circuits "
        "that don't care which particular query-key interaction fired).\n"
    )
    lines.append(
        "- If R_* approaches 1 and the rank CDF hugs the diagonal, the "
        "model relies heavily on content-specific attention and the "
        "superposition phenomenon cannot be explained without invoking "
        "it.\n"
    )

    path = os.path.join(output_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    inputs = _expand_globs(args.inputs)
    if not inputs:
        raise FileNotFoundError(f"No files matched: {args.inputs}")

    dfs: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, Dict] = {}
    for path in inputs:
        df = pd.read_parquet(path)
        summary = _load_summary_json(path)
        model_name = summary.get("model_name") or os.path.splitext(os.path.basename(path))[0]
        mode = summary.get("patch_mode", "donor")
        label = f"{model_name} [{mode}]"
        # If there's a duplicate label, append a short hash of the filename
        # so the two entries stay distinguishable in the legend.
        if label in dfs:
            label = f"{label} ({os.path.basename(path)[:10]})"
        dfs[label] = df
        summaries[label] = summary

    plot_rank_cdf(dfs, os.path.join(args.output_dir, "rank_cdf.png"))
    plot_entropy_hist(dfs, os.path.join(args.output_dir, "entropy_hist.png"))

    summary_df = build_summary_table(dfs)
    summary_df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    with open(os.path.join(args.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_to_markdown(summary_df))

    report_path = build_report(dfs, summaries, args.output_dir)
    print(f"[analyze] wrote plots and summary to {args.output_dir}")
    print(f"[analyze] wrote report to {report_path}")


if __name__ == "__main__":
    main()
