"""End-to-end attention-patching experiment.

For each FineWeb text ``A`` we sample an unrelated donor text ``C`` from the
same stream and run three forward passes on ``A``:

    * A1 (baseline): unmodified forward on X_A.
    * A2 (patched) : forward on X_A with post-softmax attention at every layer
                     replaced by the attention map of the donor text X_C.
    * A'           : baseline forward on a second unrelated text A' (used to
                     build the denominator of the Superposition Approximation
                     Ratio R_D).

Per-position metrics collected:

    * rank of argmax(P_{A1}) inside P_{A2}
    * KL( P_{A1} || P_{A2} ),  JS( P_{A1}, P_{A2} ),  TVD
    * Wasserstein top-256 with cosine ground metric between P_{A1}, P_{A2}
    * entropies of P_{A1}, P_{A2}
    * KL/JS/Wasserstein between P_{A1} and P_{A'} (denominator of R_D)

Results are written to ``<save_dir>/<run_name>.parquet`` (one row per token)
and ``<save_dir>/<run_name>.json`` (aggregate summary).

Example:
    python -m experiments.attention_patching.run_patching \
        --model_name EleutherAI/pythia-410m \
        --num_pairs 50 \
        --max_length 128 \
        --save_dir experiments/attention_patching/results

Scale up once debugged:
    python -m experiments.attention_patching.run_patching \
        --model_name EleutherAI/pythia-2.8b \
        --num_pairs 1000 \
        --max_length 128 \
        --save_dir experiments/attention_patching/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

# Make the repo root importable so we can reuse existing utilities.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics import (  # noqa: E402  (repo root imports)
    kl_divergence,
    js_divergence,
    total_variation_distance,
)
from model_utils import load_model_and_tokenizer  # noqa: E402
from run_batch_tvd_temperature import extract_short_sentences  # noqa: E402

from experiments.attention_patching.metrics_extra import (  # noqa: E402
    wasserstein_topk_cosine,
    entropy,
)
from experiments.attention_patching.patching import (  # noqa: E402
    capture_donor_attentions,
    patched_forward_probs,
    baseline_forward_probs,
    supports_patching,
    permute_attentions_causal,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-410m",
        help="HF model id. Default matches debug preset.",
    )
    p.add_argument("--num_pairs", type=int, default=50)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--patch_mode",
        type=str,
        default="donor",
        choices=["donor", "permutation"],
        help=(
            "Where attention weights in A2 come from. "
            "'donor': attention of an unrelated text C (default, original "
            "experiment). "
            "'permutation': per-row random permutation of A's own attention "
            "within the causal prefix (control to check if the model is robust "
            "to any valid attention, not just donor attention from another "
            "text)."
        ),
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HF dataset id to sample texts from.",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="sample-10BT",
        help="Dataset configuration name (e.g. sample-10BT for fineweb-edu).",
    )
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--wasserstein_k", type=int, default=256)
    p.add_argument(
        "--skip_wasserstein",
        action="store_true",
        help="Skip Wasserstein (expensive, ~O(|S|^3)). Useful during debug.",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="experiments/attention_patching/results",
    )
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def _load_texts(
    tokenizer,
    dataset_id: str,
    dataset_config: Optional[str],
    split: str,
    seed: int,
    min_tokens: int,
    num_needed: int,
) -> List[str]:
    """Stream ``dataset_id`` and return texts that tokenize to ``>=min_tokens``.

    Reuses the helper from ``run_batch_tvd_temperature.extract_short_sentences``
    (despite the misleading name, in streaming+tokenizer+min_tokens mode it
    collects full documents of the required length -- see its source).
    """
    ds_kwargs = {"split": split, "streaming": True}
    if dataset_config:
        ds_kwargs["name"] = dataset_config
    ds = load_dataset(dataset_id, **ds_kwargs)
    texts = extract_short_sentences(
        ds,
        num=num_needed,
        seed=seed,
        tokenizer=tokenizer,
        min_tokens=min_tokens,
        streaming=True,
    )
    return texts


def _tokenize_fixed_length(
    tokenizer,
    texts: List[str],
    T: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Tokenize every text and truncate to exactly T tokens. Drops any text
    that ends up shorter than T (shouldn't happen because ``_load_texts``
    already filtered by token length, but double-check to be safe)."""
    result: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for txt in texts:
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=T)
        ids = enc["input_ids"]
        if ids.shape[1] < T:
            continue
        ids = ids[:, :T].to(device)
        mask = torch.ones_like(ids, device=device)
        result.append((ids, mask))
    return result


def _get_input_embedding_weight(model) -> torch.Tensor:
    emb = model.get_input_embeddings()
    return emb.weight.detach()


def _safe_rank(dist: torch.Tensor, token_id: int) -> int:
    """1-based rank of ``token_id`` in ``dist`` (``[V]``), ties broken by
    counting strictly-greater probabilities (same semantics as
    ``run_batch_tvd_temperature.rank_of_token_in_distribution``)."""
    p = dist[token_id]
    return int((dist > p).sum().item()) + 1


def _row_for_position(
    t: int,
    probs_a1: torch.Tensor,  # [V]
    probs_a2: torch.Tensor,  # [V]
    probs_aprime: torch.Tensor,  # [V]  (baseline on unrelated text)
    token_emb: Optional[torch.Tensor],
    wasserstein_k: int,
    skip_wasserstein: bool,
) -> Dict[str, float]:
    top1_a1 = int(torch.argmax(probs_a1).item())
    top1_prob_a1 = float(probs_a1[top1_a1].item())
    rank_top1_a1_in_a2 = _safe_rank(probs_a2, top1_a1)
    rank_top1_a1_in_a1 = _safe_rank(probs_a1, top1_a1)  # sanity: must be 1

    # Distributional distances: A1 vs A2 (numerator of R_D)
    kl_a1_a2 = float(kl_divergence(probs_a1, probs_a2).item())
    js_a1_a2 = float(js_divergence(probs_a1, probs_a2).item())
    tvd_a1_a2 = float(total_variation_distance(probs_a1, probs_a2).item())
    ent_a1 = float(entropy(probs_a1).item())
    ent_a2 = float(entropy(probs_a2).item())

    # Denominator of R_D: baseline A1 vs baseline A' (unrelated)
    kl_a1_aprime = float(kl_divergence(probs_a1, probs_aprime).item())
    js_a1_aprime = float(js_divergence(probs_a1, probs_aprime).item())
    tvd_a1_aprime = float(
        total_variation_distance(probs_a1, probs_aprime).item()
    )

    row = {
        "position": t,
        "top1_token_a1": top1_a1,
        "top1_prob_a1": top1_prob_a1,
        "rank_top1_a1_in_a1": rank_top1_a1_in_a1,
        "rank_top1_a1_in_a2": rank_top1_a1_in_a2,
        "kl_a1_a2": kl_a1_a2,
        "js_a1_a2": js_a1_a2,
        "tvd_a1_a2": tvd_a1_a2,
        "entropy_a1": ent_a1,
        "entropy_a2": ent_a2,
        "kl_a1_aprime": kl_a1_aprime,
        "js_a1_aprime": js_a1_aprime,
        "tvd_a1_aprime": tvd_a1_aprime,
    }

    if not skip_wasserstein and token_emb is not None:
        row["wasserstein_a1_a2"] = wasserstein_topk_cosine(
            probs_a1, probs_a2, token_emb, k=wasserstein_k,
        )
        row["wasserstein_a1_aprime"] = wasserstein_topk_cosine(
            probs_a1, probs_aprime, token_emb, k=wasserstein_k,
        )
    return row


def _process_pair(
    pair_idx: int,
    model,
    input_ids_a: torch.Tensor,
    attn_mask_a: torch.Tensor,
    input_ids_c: Optional[torch.Tensor],
    attn_mask_c: Optional[torch.Tensor],
    probs_aprime: torch.Tensor,  # precomputed baseline on A' [1, T, V] cpu
    token_emb: Optional[torch.Tensor],
    wasserstein_k: int,
    skip_wasserstein: bool,
    patch_mode: str = "donor",
    perm_generator: Optional[torch.Generator] = None,
) -> List[Dict]:
    # Baseline forward on A (A1)
    probs_a1 = baseline_forward_probs(model, input_ids_a, attn_mask_a)  # [1,T,V] cpu

    if patch_mode == "donor":
        # Attention weights come from an unrelated text C.
        assert input_ids_c is not None and attn_mask_c is not None
        attn_source = capture_donor_attentions(model, input_ids_c, attn_mask_c)
    elif patch_mode == "permutation":
        # Capture A's OWN attentions, then randomly permute each row within
        # its causal prefix. This preserves the multiset of weights per row,
        # causal mask, and row-sum=1, but destroys positional/content
        # structure. Control for the donor experiment.
        own_attn = capture_donor_attentions(model, input_ids_a, attn_mask_a)
        attn_source = permute_attentions_causal(own_attn, generator=perm_generator)
        del own_attn
    else:
        raise ValueError(f"Unknown patch_mode={patch_mode!r}")

    probs_a2 = patched_forward_probs(model, input_ids_a, attn_mask_a, attn_source)
    # Free patched-attention tensors early (they are large: L * [1, H, T, T]).
    del attn_source

    T = probs_a1.shape[1]
    rows: List[Dict] = []
    # Skip the very first position because causal attention there is just
    # attending to the BOS/first token itself -- patching is a no-op.
    for t in range(1, T):
        row = _row_for_position(
            t=t,
            probs_a1=probs_a1[0, t],
            probs_a2=probs_a2[0, t],
            probs_aprime=probs_aprime[0, t],
            token_emb=token_emb,
            wasserstein_k=wasserstein_k,
            skip_wasserstein=skip_wasserstein,
        )
        row["pair_idx"] = pair_idx
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"patch_{args.patch_mode}_{args.model_name.replace('/', '_')}_"
        f"N{args.num_pairs}_T{args.max_length}_{timestamp}"
    )
    base_path = os.path.join(args.save_dir, run_name)

    print(f"[run] model={args.model_name}")
    print(f"[run] patch_mode={args.patch_mode}")
    print(f"[run] num_pairs={args.num_pairs}  T={args.max_length}")
    print(f"[run] saving to {base_path}.{{parquet,json}}")

    tokenizer, model = load_model_and_tokenizer(
        args.model_name, dtype=args.dtype, device_map=args.device_map,
    )
    if not supports_patching(model):
        raise RuntimeError(
            "Model does not expose integer layer_idx on attention modules; "
            "patching infrastructure cannot route donor weights. This script "
            "targets transformers v5+ decoder-only architectures."
        )
    device = next(model.parameters()).device

    # We need per 'pair': A, A' (unrelated for R_D denominator), plus C
    # (donor) only in patch_mode='donor'. Using a uniform slot count makes
    # downstream indexing simpler, so we always ask for 3 distinct texts
    # per pair and simply ignore the donor slot in permutation mode.
    texts_per_pair = 3 if args.patch_mode == "donor" else 2
    num_texts_needed = texts_per_pair * args.num_pairs
    # extract_short_sentences has a * 10 cap internally, but in streaming+
    # min_tokens mode it breaks early when enough are found. Ask for a
    # comfortable safety margin.
    print(f"[data] streaming {args.dataset}/{args.dataset_name} ...")
    texts = _load_texts(
        tokenizer=tokenizer,
        dataset_id=args.dataset,
        dataset_config=args.dataset_name,
        split=args.split,
        seed=args.seed,
        min_tokens=args.max_length,
        num_needed=int(num_texts_needed * 1.5),
    )
    print(f"[data] got {len(texts)} raw texts; tokenizing to T={args.max_length}")

    tokenized = _tokenize_fixed_length(
        tokenizer, texts, T=args.max_length, device=device,
    )
    if len(tokenized) < num_texts_needed:
        raise RuntimeError(
            f"Only {len(tokenized)} texts meet length={args.max_length}, "
            f"need {num_texts_needed}. Raise num_needed or lower max_length."
        )

    # Deterministically split texts into triples (A, C, A') or pairs
    # (A, A'), depending on patch_mode. ``C`` is unused in permutation
    # mode; we still advance indices by a fixed stride so seeds give
    # comparable A/A' assignments across modes.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tokenized))
    triples: List[Tuple[int, Optional[int], int]] = []
    for p in range(args.num_pairs):
        base = texts_per_pair * p
        i_a = perm[base]
        if args.patch_mode == "donor":
            i_c: Optional[int] = int(perm[base + 1])
            i_aprime = int(perm[base + 2])
        else:
            i_c = None
            i_aprime = int(perm[base + 1])
        triples.append((int(i_a), i_c, i_aprime))

    token_emb = None if args.skip_wasserstein else _get_input_embedding_weight(model)

    # Independent generator for the row-wise permutation in patch_mode=
    # permutation, so the result is reproducible from --seed alone.
    perm_generator: Optional[torch.Generator] = None
    if args.patch_mode == "permutation":
        perm_generator = torch.Generator(device=device)
        perm_generator.manual_seed(int(args.seed))

    all_rows: List[Dict] = []
    t0 = time.time()
    for p, (i_a, i_c, i_aprime) in enumerate(triples):
        ids_a, mask_a = tokenized[i_a]
        ids_aprime, mask_aprime = tokenized[i_aprime]
        if i_c is not None:
            ids_c, mask_c = tokenized[i_c]
        else:
            ids_c = mask_c = None  # type: ignore[assignment]

        # Pre-compute baseline on A' once per pair (not per position).
        probs_aprime = baseline_forward_probs(model, ids_aprime, mask_aprime)

        rows = _process_pair(
            pair_idx=p,
            model=model,
            input_ids_a=ids_a,
            attn_mask_a=mask_a,
            input_ids_c=ids_c,
            attn_mask_c=mask_c,
            probs_aprime=probs_aprime,
            token_emb=token_emb,
            wasserstein_k=args.wasserstein_k,
            skip_wasserstein=args.skip_wasserstein,
            patch_mode=args.patch_mode,
            perm_generator=perm_generator,
        )
        all_rows.extend(rows)

        if (p + 1) % max(1, args.num_pairs // 20) == 0 or p == args.num_pairs - 1:
            elapsed = time.time() - t0
            eta = elapsed / (p + 1) * (args.num_pairs - p - 1)
            print(
                f"[run] pair {p+1}/{args.num_pairs} "
                f"({100*(p+1)/args.num_pairs:.1f}%)  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s"
            )

    df = pd.DataFrame(all_rows)
    parquet_path = base_path + ".parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"[run] wrote {parquet_path}  ({len(df)} rows)")

    # Aggregate summary (median + mean for robustness), plus R_D ratios.
    def _agg(col: str) -> Dict[str, float]:
        if col not in df.columns:
            return {}
        x = df[col].to_numpy()
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "std": float(np.std(x)),
        }

    summary = {
        "model_name": args.model_name,
        "patch_mode": args.patch_mode,
        "num_pairs": args.num_pairs,
        "max_length": args.max_length,
        "seed": args.seed,
        "wasserstein_k": args.wasserstein_k,
        "skip_wasserstein": bool(args.skip_wasserstein),
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "n_rows": int(len(df)),
        "rank_top1_a1_in_a2": _agg("rank_top1_a1_in_a2"),
        "kl_a1_a2": _agg("kl_a1_a2"),
        "js_a1_a2": _agg("js_a1_a2"),
        "tvd_a1_a2": _agg("tvd_a1_a2"),
        "kl_a1_aprime": _agg("kl_a1_aprime"),
        "js_a1_aprime": _agg("js_a1_aprime"),
        "tvd_a1_aprime": _agg("tvd_a1_aprime"),
        "entropy_a1": _agg("entropy_a1"),
        "entropy_a2": _agg("entropy_a2"),
    }
    if not args.skip_wasserstein:
        summary["wasserstein_a1_a2"] = _agg("wasserstein_a1_a2")
        summary["wasserstein_a1_aprime"] = _agg("wasserstein_a1_aprime")

    # Rank CDF (probability that rank <= i) for a few canonical i values.
    if "rank_top1_a1_in_a2" in df.columns:
        ranks = df["rank_top1_a1_in_a2"].to_numpy()
        summary["rank_cdf"] = {
            str(i): float(np.mean(ranks <= i)) for i in [1, 5, 10, 50, 100]
        }

    # Superposition Approximation Ratio (patched attention version).
    #   R_D = mean[D(P_A1 || P_A2)] / mean[D(P_A1 || P_A')]
    for dname in ("kl", "js", "tvd", "wasserstein"):
        num_key = f"{dname}_a1_a2"
        den_key = f"{dname}_a1_aprime"
        if num_key in df.columns and den_key in df.columns:
            num = float(np.mean(df[num_key].to_numpy()))
            den = float(np.mean(df[den_key].to_numpy()))
            summary[f"R_{dname}"] = num / den if den > 0 else float("nan")

    json_path = base_path + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] wrote {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
