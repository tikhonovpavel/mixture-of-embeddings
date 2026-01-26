import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset

from metrics import total_variation_distance
from model_utils import (
    get_probs_from_ids,
    get_probs_from_embeds,
    get_probs_with_layer_override,
    get_logits_from_embeds,
    get_logits_with_layer_override,
    get_probs_from_ids_with_layer_inputs,
    find_transformer_blocks,
    layer_indices_for_percentages,
    load_model_and_tokenizer,
    tokenize_and_align_pair,
    find_all_norm_layers,
    compute_adaptive_norm_scale_factors,
    temporarily_scale_norm_outputs_adaptively,
)


DEFAULT_MODEL_LIST = [
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen2.5-7B",
    "google/gemma-2-9b",
    "EleutherAI/pythia-6.9b",
    "Qwen/Qwen2.5-3B",
    "google/gemma-2-2b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-410m"
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch run with TVD temperature optimization")
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of HF model identifiers to evaluate (defaults to Meta-Llama-3, Qwen2.5, and Gemma-2).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="(Deprecated) Single model identifier. Use --models instead.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="Skylion007/openwebtext",
        help="Dataset to use (e.g. 'Skylion007/openwebtext')",
    )
    p.add_argument("--split", type=str, default="train", help="HF datasets split to sample from (use 'train' for streaming, 'train[:2000]' for non-streaming)")
    p.add_argument("--streaming", action="store_true", help="Use streaming mode to avoid downloading full dataset")
    p.add_argument("--num_sentences", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--truncate", type=str, default="min", choices=["min", "a", "b"])
    p.add_argument("--sqrt2-norm", action="store_true")
    p.add_argument(
        "--mix-depth-percent",
        type=int,
        default=0,
        help="Depth (in percent of layers) at which to mix residuals. 0 mixes input embeddings; e.g., 20 mixes at 20%% layer input.",
    )
    p.add_argument(
        "--mix-mode",
        type=str,
        default="sum",
        choices=["sum", "avg"],
        help="Combine mode for A and B residuals at mix depth: sum or avg (0.5*(A+B)).",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Truncate paired sequences to this many tokens and skip samples shorter than this length.",
    )
    p.add_argument(
        "--adaptive-norm-scaling",
        action="store_true",
        help="Enable adaptive normalization scaling based on activation norm differences between original and mixed passes.",
    )
    p.add_argument("--save_dir", type=str, default="./results_batch_tvd")
    args = p.parse_args()
    if args.models is None and args.model is None:
        args.models = DEFAULT_MODEL_LIST.copy()
    return args


def extract_short_sentences(ds, num: int, seed: int, tokenizer=None, min_tokens: int = None, streaming: bool = False) -> List[str]:
    import random
    rnd = random.Random(seed)
    candidates: List[str] = []
    
    # For streaming datasets, we need to limit how many rows we iterate through
    max_rows_to_check = num * 10 if streaming else None
    rows_checked = 0
    
    for row in ds:
        if max_rows_to_check and rows_checked >= max_rows_to_check:
            break
        rows_checked += 1
        
        text = row.get("text") or row.get("story") or ""
        if not text:
            continue
        
        # If tokenizer provided, treat each dataset sample as a complete unit (may contain many sentences)
        # Filter by token length on the full sample text
        if tokenizer is not None and min_tokens is not None:
            enc = tokenizer(text, return_tensors="pt")
            token_length = int(enc["input_ids"].shape[1])
            if token_length >= min_tokens:
                candidates.append(text)
                if streaming and len(candidates) >= num:
                    # Early exit in streaming mode once we have enough candidates
                    break
        else:
            # Original character-length filter (for backward compatibility with TinyStories)
            parts = [s.strip() for s in text.replace("\n", " ").split(".")]
            for s in parts:
                if len(s) >= 15 and len(s) <= 180:
                    candidates.append(s + ".")
                    if streaming and len(candidates) >= num:
                        break
            if streaming and len(candidates) >= num:
                break
    
    rnd.shuffle(candidates)
    return candidates[:num]


def filter_sentences_by_token_length(
    tokenizer,
    sentences: List[str],
    min_length_tokens: Optional[int],
) -> List[str]:
    if min_length_tokens is None:
        return sentences

    filtered: List[str] = []
    for s in sentences:
        enc = tokenizer(s, return_tensors="pt")
        length = int(enc["input_ids"].shape[1])
        if length >= min_length_tokens:
            filtered.append(s)
    return filtered


def rank_of_token_in_distribution(prob_dist: torch.Tensor, token_id: int) -> int:
    token_prob = prob_dist[token_id]
    return int((prob_dist > token_prob).sum().item()) + 1


def find_optimal_temperature(
    logits_mix: torch.Tensor,
    probs_avg: torch.Tensor,
    temperatures: np.ndarray,
) -> Tuple[float, float]:
    """
    Find optimal temperature that minimizes TVD between temperature-scaled probs_mix and probs_avg.
    
    Args:
        logits_mix: Logits from mix distribution [V] (vocab size)
        probs_avg: Average probabilities [V] (vocab size)
        temperatures: Array of temperatures to try
        
    Returns:
        Tuple of (optimal_temperature, optimal_tvd)
    """
    device = logits_mix.device
    best_temp = temperatures[0]
    best_tvd = float('inf')
    
    for temp in temperatures:
        temp_tensor = torch.tensor(temp, device=device, dtype=logits_mix.dtype)
        # Scale logits by temperature
        scaled_logits = logits_mix / temp_tensor
        # Convert to probabilities
        probs_mix_temp = torch.softmax(scaled_logits, dim=-1)
        # Compute TVD
        tvd = total_variation_distance(probs_avg, probs_mix_temp)
        tvd_val = float(tvd.detach().cpu().item())
        
        if tvd_val < best_tvd:
            best_tvd = tvd_val
            best_temp = temp
    
    return best_temp, best_tvd


def run_pair(
    model,
    tokenizer,
    a: str,
    b: str,
    truncate: str,
    sqrt2_norm: bool,
    mix_depth_percent: int,
    mix_mode: str,
    max_length: Optional[int],
    temperatures: np.ndarray,
    adaptive_norm_scaling: bool = False,
) -> List[Dict[str, float]]:
    pair = tokenize_and_align_pair(
        tokenizer,
        a,
        b,
        truncate=truncate,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    input_ids_a = pair.input_ids_a.to(device)
    input_ids_b = pair.input_ids_b.to(device)
    attn_mask_a = pair.attn_mask_a.to(device)
    attn_mask_b = pair.attn_mask_b.to(device)

    probs_a, _ = get_probs_from_ids(
        model,
        input_ids_a,
        attn_mask_a,
        return_pre_norm=False,
    )
    probs_b, _ = get_probs_from_ids(
        model,
        input_ids_b,
        attn_mask_b,
        return_pre_norm=False,
    )
    probs_avg = 0.5 * (probs_a + probs_b)

    # Decide how to compute mixed distribution: input-level or mid-layer override
    T = pair.T
    attn_mask = torch.ones((1, T), dtype=torch.long, device=device)

    # Prepare for adaptive normalization scaling if enabled
    scale_factors = None
    norm_modules = None
    if adaptive_norm_scaling:
        norm_modules = find_all_norm_layers(model)

    if mix_depth_percent <= 0:
        # Mix at input embeddings level
        emb = model.get_input_embeddings()
        emb_a = emb(input_ids_a)
        emb_b = emb(input_ids_b)
        if mix_mode == "avg":
            inputs_embeds = 0.5 * (emb_a + emb_b)
        else:
            inputs_embeds = emb_a + emb_b
        
        # Compute scale factors if adaptive scaling is enabled
        if adaptive_norm_scaling and norm_modules:
            scale_factors = compute_adaptive_norm_scale_factors(
                model,
                input_ids_a,
                input_ids_b,
                attn_mask_a,
                attn_mask_b,
                inputs_embeds_mix=inputs_embeds,
                attn_mask_mix=attn_mask,
                norm_modules=norm_modules,
            )
        
        # Apply adaptive scaling during mixed forward pass
        if adaptive_norm_scaling and scale_factors and norm_modules:
            with temporarily_scale_norm_outputs_adaptively(norm_modules, scale_factors):
                probs_mix, _ = get_probs_from_embeds(
                    model,
                    inputs_embeds,
                    attn_mask,
                    sqrt2_rmsnorm=sqrt2_norm,
                )
                # Get logits for temperature scaling
                logits_mix = get_logits_from_embeds(
                    model,
                    inputs_embeds,
                    attn_mask,
                    sqrt2_rmsnorm=sqrt2_norm,
                )
        else:
            probs_mix, _ = get_probs_from_embeds(
                model,
                inputs_embeds,
                attn_mask,
                sqrt2_rmsnorm=sqrt2_norm,
            )
            # Get logits for temperature scaling
            logits_mix = get_logits_from_embeds(
                model,
                inputs_embeds,
                attn_mask,
                sqrt2_rmsnorm=sqrt2_norm,
            )
    else:
        # Mix at internal layer input
        blocks = find_transformer_blocks(model)
        num_layers = len(blocks)
        target_idx = layer_indices_for_percentages(num_layers, [mix_depth_percent])[0]
        # Get target layer inputs for A and B
        _, layer_inputs_a_target = get_probs_from_ids_with_layer_inputs(
            model,
            input_ids_a,
            attn_mask_a,
            layer_indices=[target_idx],
        )
        _, layer_inputs_b_target = get_probs_from_ids_with_layer_inputs(
            model,
            input_ids_b,
            attn_mask_b,
            layer_indices=[target_idx],
        )
        a_l = layer_inputs_a_target[target_idx]
        b_l = layer_inputs_b_target[target_idx]
        override = 0.5 * (a_l + b_l) if mix_mode == "avg" else (a_l + b_l)
        
        # Compute scale factors if adaptive scaling is enabled
        if adaptive_norm_scaling and norm_modules:
            scale_factors = compute_adaptive_norm_scale_factors(
                model,
                input_ids_a,
                input_ids_b,
                attn_mask_a,
                attn_mask_b,
                inputs_embeds_mix=None,
                attn_mask_mix=attn_mask,
                norm_modules=norm_modules,
                layer_override_index=target_idx,
                layer_override_hidden=override,
            )
        
        # Apply adaptive scaling during mixed forward pass
        if adaptive_norm_scaling and scale_factors and norm_modules:
            with temporarily_scale_norm_outputs_adaptively(norm_modules, scale_factors):
                # Get probabilities and logits with override
                probs_mix, _ = get_probs_with_layer_override(
                    model,
                    base_input_ids=input_ids_a,
                    attention_mask=attn_mask,
                    layer_index=target_idx,
                    override_hidden=override,
                )
                logits_mix = get_logits_with_layer_override(
                    model,
                    base_input_ids=input_ids_a,
                    attention_mask=attn_mask,
                    layer_index=target_idx,
                    override_hidden=override,
                )
        else:
            # Get probabilities and logits with override
            probs_mix, _ = get_probs_with_layer_override(
                model,
                base_input_ids=input_ids_a,
                attention_mask=attn_mask,
                layer_index=target_idx,
                override_hidden=override,
            )
            logits_mix = get_logits_with_layer_override(
                model,
                base_input_ids=input_ids_a,
                attention_mask=attn_mask,
                layer_index=target_idx,
                override_hidden=override,
            )

    if max_length is not None and T < max_length:
        return []

    positions = range(T)
    rows: List[Dict[str, float]] = []
    
    for t in positions:
        p_avg = probs_avg[0, t, :]
        p_mix = probs_mix[0, t, :]
        logits_mix_t = logits_mix[0, t, :]

        prob_top1_a, top1_token_a = torch.max(probs_a[0, t, :], dim=0)
        prob_top1_b, top1_token_b = torch.max(probs_b[0, t, :], dim=0)

        rank_a = rank_of_token_in_distribution(p_mix, int(top1_token_a.item())) / 2.0
        rank_b = rank_of_token_in_distribution(p_mix, int(top1_token_b.item())) / 2.0

        # Find optimal temperature for this token position
        optimal_temp, optimal_tvd = find_optimal_temperature(
            logits_mix_t,
            p_avg,
            temperatures,
        )

        row: Dict[str, float] = {
            "token_position": int(t),
            "prob_top1_a": float(prob_top1_a.detach().cpu().item()),
            "prob_top1_b": float(prob_top1_b.detach().cpu().item()),
            "rank_top1_a_in_mix_half": float(rank_a),
            "rank_top1_b_in_mix_half": float(rank_b),
            "tvd_optimal": float(optimal_tvd),
            "temperature_optimal": float(optimal_temp),
            "top1_token_id_a": int(top1_token_a.detach().cpu().item()),
            "top1_token_id_b": int(top1_token_b.detach().cpu().item()),
            "sequence_length": int(T),
        }
        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    model_names: List[str] = []
    if args.models:
        model_names.extend(args.models)
    if args.model and not args.models:
        model_names = [args.model]

    if not model_names:
        raise ValueError("No models specified. Provide --models or --model.")

    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = args.dataset.replace("/", "_")
    base = os.path.join(args.save_dir, f"batch_tvd_{dataset_tag}_{timestamp}")

    # Temperature grid search space
    temperatures = np.linspace(0.1, 5.0, 20)

    all_rows: List[Dict] = []
    summaries: List[Dict[str, object]] = []

    for model_name in model_names:
        print(f"Evaluating model {model_name}...")
        tokenizer, model = load_model_and_tokenizer(model_name, dtype=args.dtype, device_map=args.device_map)

        sents = extract_short_sentences(
            ds,
            args.num_sentences,
            args.seed,
            tokenizer=tokenizer,
            min_tokens=args.max_length,
            streaming=args.streaming,
        )
        sents = filter_sentences_by_token_length(tokenizer, sents, args.max_length)

        if len(sents) < 2:
            print(
                f"Warning: Not enough sentences meeting the length requirement for model {model_name}.",
            )
            summaries.append(
                {
                    "model": model_name,
                    "num_sentences": len(sents),
                    "num_pairs": 0,
                    "num_pairs_evaluated": 0,
                    "num_token_rows": 0,
                    "sequence_length": args.max_length,
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        pairs: List[Tuple[str, str]] = []
        for i in range(0, len(sents) - 1, 2):
            pairs.append((sents[i], sents[i + 1]))

        if not pairs:
            print(f"Warning: Not enough sentence pairs for model {model_name}.")
            summaries.append(
                {
                    "model": model_name,
                    "num_sentences": len(sents),
                    "num_pairs": 0,
                    "num_pairs_evaluated": 0,
                    "num_token_rows": 0,
                    "sequence_length": args.max_length,
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        model_rows: List[Dict] = []
        evaluated_pairs = 0
        for idx, (a, b) in enumerate(pairs):
            token_rows = run_pair(
                model,
                tokenizer,
                a,
                b,
                truncate=args.truncate,
                sqrt2_norm=args.sqrt2_norm,
                mix_depth_percent=args.mix_depth_percent,
                mix_mode=args.mix_mode,
                max_length=args.max_length,
                temperatures=temperatures,
                adaptive_norm_scaling=args.adaptive_norm_scaling,
            )
            if not token_rows:
                continue
            evaluated_pairs += 1
            for row in token_rows:
                row.update(
                    {
                        "model": model_name,
                        "pair_idx": idx,
                        "a": a,
                        "b": b,
                    }
                )
            model_rows.extend(token_rows)

        all_rows.extend(model_rows)

        summaries.append(
            {
                "model": model_name,
                "num_sentences": len(sents),
                "num_pairs": len(pairs),
                "num_pairs_evaluated": evaluated_pairs,
                "num_token_rows": len(model_rows),
                "sequence_length": args.max_length,
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_rows:
        print("Warning: No token-level results were generated across all models.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = base + ".csv"
    df.to_csv(csv_path, index=False)

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    exclude_from_median = {"pair_idx", "top1_token_id_a", "top1_token_id_b"}
    metric_cols = [col for col in numeric_cols if col not in exclude_from_median]

    overall_medians: Dict[str, float] = {}
    if metric_cols:
        med = df[metric_cols].median(numeric_only=True).to_dict()
        overall_medians = {k: float(v) for k, v in med.items() if pd.notna(v)}

    aggregate_by_model: Dict[str, Dict[str, float]] = {}
    for model_name, group in df.groupby("model"):
        if metric_cols:
            med = group[metric_cols].median(numeric_only=True).to_dict()
            aggregate_by_model[model_name] = {k: float(v) for k, v in med.items() if pd.notna(v)}
        else:
            aggregate_by_model[model_name] = {}

    out_json = {
        "dataset": args.dataset,
        "models": model_names,
        "num_models": len(model_names),
        "max_length": args.max_length,
        "truncate": args.truncate,
        "sqrt2_norm": bool(args.sqrt2_norm),
        "mix_depth_percent": args.mix_depth_percent,
        "mix_mode": args.mix_mode,
        "summary_by_model": summaries,
        "total_pairs": int(sum(info["num_pairs"] for info in summaries)),
        "total_pairs_evaluated": int(sum(info["num_pairs_evaluated"] for info in summaries)),
        "total_token_rows": int(len(df)),
        "aggregate_medians_overall": overall_medians,
        "aggregate_medians_by_model": aggregate_by_model,
    }

    json_path = base + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print(f"Saved batch results to: {json_path} and {csv_path}")
    if overall_medians:
        print("Overall medians:")
        for k, v in overall_medians.items():
            print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()

