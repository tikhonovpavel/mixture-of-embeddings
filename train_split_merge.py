#!/usr/bin/env python3
"""
Full-model training with split/merge architecture.

Architecture:
  Stream A: embed(A) → layers[0..K-1] → h_A
  Stream B: embed(B) → layers[0..K-1] → h_B
      ↓ mark_A(h_A), mark_B(h_B)  →  merge: 0.5*(marked_A + marked_B)
  Shared:  layers[K..N-M-1]  →  h_mix
      ↓ split_A(h_mix), split_B(h_mix)
  Stream A: layers[N-M..N-1] → norm → lm_head → logits_A
  Stream B: layers[N-M..N-1] → norm → lm_head → logits_B

Everything is trained end-to-end (backbone + marking + split projections).
"""

import json
import math
import os
import sys
import random
from datetime import datetime
from typing import List, Dict, Tuple, Iterator, Optional

import torch
import torch.nn as nn
import numpy as np
import fire
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter

from metrics import compute_all_metrics
from model_utils import (
    load_model_and_tokenizer,
    find_transformer_blocks,
    get_probs_from_ids,
)
from run_batch_tvd_temperature import rank_of_token_in_distribution


# ---------------------------------------------------------------------------
# Trainable modules (reused from train_split_projections.py)
# ---------------------------------------------------------------------------

def _haar_orthogonal(d: int, device) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(d, d, device=device, dtype=torch.float32))
    q *= torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q


class MarkProjections(nn.Module):
    """Trainable stream-specific projections applied before merging."""

    def __init__(self, hidden_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.mark_A = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
        self.mark_B = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
        with torch.no_grad():
            self.mark_A.weight.copy_(_haar_orthogonal(hidden_dim, self.mark_A.weight.device).to(dtype))
            self.mark_B.weight.copy_(_haar_orthogonal(hidden_dim, self.mark_B.weight.device).to(dtype))

    def forward(self, h_a, h_b):
        return self.mark_A(h_a), self.mark_B(h_b)


class SplitLinear(nn.Module):
    """Two learnable d×d linear projections initialised near identity."""

    def __init__(self, hidden_dim: int, init_noise_std: float = 0.001, dtype=torch.bfloat16):
        super().__init__()
        self.proj_A = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
        self.proj_B = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
        with torch.no_grad():
            eye = torch.eye(hidden_dim, dtype=dtype)
            self.proj_A.weight.copy_(eye + torch.randn_like(eye) * init_noise_std)
            self.proj_B.weight.copy_(eye + torch.randn_like(eye) * init_noise_std)

    def forward(self, h_mix):
        return self.proj_A(h_mix), self.proj_B(h_mix)


# ---------------------------------------------------------------------------
# Layer-range forward (architecture-agnostic)
# ---------------------------------------------------------------------------

def _get_model_internals(model):
    """Extract embedding, blocks, norm, lm_head, rotary_emb from a CausalLM model."""
    blocks = find_transformer_blocks(model)

    candidate_embed = [
        ("model", "embed_tokens"),
        ("transformer", "wte"),
        ("gpt_neox", "embed_in"),
    ]
    embed_fn = None
    for path in candidate_embed:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            embed_fn = obj
            break
    if embed_fn is None:
        embed_fn = model.get_input_embeddings()

    candidate_norm = [
        ("model", "norm"),
        ("model", "final_layer_norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
    ]
    norm_fn = None
    for path in candidate_norm:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            norm_fn = obj
            break

    candidate_rotary = [
        ("model", "rotary_emb"),
        ("transformer", "rotary_emb"),
        ("gpt_neox", "rotary_emb"),
    ]
    rotary_emb = None
    for path in candidate_rotary:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            rotary_emb = obj
            break

    lm_head = getattr(model, "lm_head", None)

    return embed_fn, blocks, norm_fn, lm_head, rotary_emb


def _forward_layers(blocks, hidden_states, start, end, position_embeddings=None):
    """Run transformer blocks[start:end] sequentially.

    Args:
        position_embeddings: (cos, sin) tuple from rotary_emb.
            With SDPA and is_causal=True, no explicit attention mask is needed
            for non-padded inputs.
    """
    for block in blocks[start:end]:
        out = block(
            hidden_states,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        hidden_states = out if isinstance(out, torch.Tensor) else out[0]
    return hidden_states


# ---------------------------------------------------------------------------
# Data helpers (same as train_split_projections)
# ---------------------------------------------------------------------------

def _tokenized_pair_iterator(
    dataset, tokenizer, max_length: int, batch_size: int = 1,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    buf: List[torch.Tensor] = []
    for row in dataset:
        text = row.get("text", "")
        if not text:
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"]
        if ids.shape[1] < max_length:
            continue
        ids = ids[:, :max_length]
        buf.append(ids)
        if len(buf) == 2 * batch_size:
            ids_a = torch.cat(buf[::2], dim=0)
            ids_b = torch.cat(buf[1::2], dim=0)
            yield ids_a, ids_b
            buf = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    model_name: str = "Qwen/Qwen2.5-3B",
    merge_after: int = 8,
    split_before_end: int = 8,
    # data
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    dataset_split: str = "train",
    streaming: bool = True,
    max_length: int = 128,
    batch_size: int = 4,
    # training
    num_train_steps: int = 1_000_000,
    lr_backbone: float = 3e-5,
    lr_projections: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    grad_accum: int = 1,
    max_grad_norm: float = 1.0,
    init_noise_std: float = 0.001,
    # eval / logging
    eval_interval: int = 500,
    num_eval_pairs: int = 30,
    log_interval: int = 50,
    lambada_interval: int = 10_000,
    num_lambada_pairs: int = 500,
    # scheduler
    scheduler_type: str = "plateau",  # "cosine" or "plateau"
    plateau_patience: int = 5,        # in units of lambada_interval
    plateau_factor: float = 0.5,
    plateau_threshold: float = 0.005,
    # resume
    resume_from: str = "",            # path to checkpoint or run dir
    # misc
    seed: int = 42,
    dtype: str = "auto",
    device_map: str = "auto",
    save_dir: str = "./results_split_merge",
):
    """Train split/merge architecture end-to-end.

    Args:
        merge_after:      K — first K layers run separately per stream.
        split_before_end: M — last M layers run separately per stream.
        lr_backbone:      LR for pretrained backbone parameters.
        lr_projections:   LR for marking + split projection parameters.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {model_name}")
    tokenizer, model = load_model_and_tokenizer(model_name, dtype=dtype, device_map=device_map)
    embed_fn, blocks, norm_fn, lm_head, rotary_emb = _get_model_internals(model)
    if rotary_emb is None:
        raise RuntimeError("Could not find rotary_emb in model — unsupported architecture")

    num_layers = len(blocks)
    K = merge_after
    split_layer = num_layers - split_before_end

    if K <= 0 or K >= num_layers:
        raise ValueError(f"merge_after={K} out of range for {num_layers} layers")
    if split_layer <= K or split_layer >= num_layers:
        raise ValueError(
            f"split_before_end={split_before_end} gives split_layer={split_layer}, "
            f"must be in ({K}, {num_layers})"
        )

    hidden_dim = model.config.hidden_size
    model_dtype = next(model.parameters()).dtype
    device = next(embed_fn.parameters()).device

    print(f"Layers: {num_layers},  K={K} (separate first),  "
          f"split_layer={split_layer} (separate last {split_before_end})")
    print(f"Hidden dim: {hidden_dim},  dtype: {model_dtype}")

    # Gradient checkpointing: recompute activations to save memory
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing: enabled")

    # Unfreeze backbone
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    # ------------------------------------------------------------------
    # Create marking + split projections
    # ------------------------------------------------------------------
    merge_device = next(blocks[K].parameters()).device
    marking = MarkProjections(hidden_dim, dtype=model_dtype).to(merge_device)

    split_device = next(blocks[split_layer].parameters()).device
    split_proj = SplitLinear(hidden_dim, init_noise_std=init_noise_std, dtype=model_dtype).to(split_device)

    mark_params = sum(p.numel() for p in marking.parameters())
    split_params = sum(p.numel() for p in split_proj.parameters())
    backbone_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Marking params: {mark_params:,},  Split params: {split_params:,}")
    print(f"Backbone trainable params: {backbone_params:,}")

    # ------------------------------------------------------------------
    # Optimizer: two param groups with different LRs
    # ------------------------------------------------------------------
    projection_param_ids = set(id(p) for p in marking.parameters()) | set(id(p) for p in split_proj.parameters())
    backbone_group = [p for p in model.parameters() if p.requires_grad and id(p) not in projection_param_ids]
    proj_group = list(marking.parameters()) + list(split_proj.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_group, "lr": lr_backbone},
        {"params": proj_group, "lr": lr_projections},
    ], weight_decay=weight_decay)

    if scheduler_type == "cosine":
        def _lr_schedule(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(num_train_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_schedule)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=plateau_factor,
            patience=plateau_patience, threshold=plateau_threshold,
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    resume_run_dir = ""
    if resume_from:
        ckpt_path = resume_from
        if os.path.isdir(resume_from):
            # It's a run directory — find the best checkpoint to resume from
            best_pt = os.path.join(resume_from, "checkpoint_best.pt")
            last_pt = os.path.join(resume_from, "checkpoint_last.pt")
            ckpt_path = last_pt if os.path.exists(last_pt) else best_pt
            resume_run_dir = resume_from
        elif os.path.isfile(resume_from):
            resume_run_dir = os.path.dirname(resume_from)

        print(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        marking.load_state_dict(ckpt["marking"])
        split_proj.load_state_dict(ckpt["split_proj"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # Update optimizer LRs to new values (checkpoint may have decayed LRs)
        for pg in optimizer.param_groups:
            if pg is optimizer.param_groups[0]:
                pg["lr"] = lr_backbone
            else:
                pg["lr"] = lr_projections
        if scheduler_type == "cosine" and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        print(f"Resumed at step {start_step}, LRs reset to backbone={lr_backbone}, proj={lr_projections}")
        del ckpt

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"Loading dataset: {dataset_name} ({dataset_config})")
    print(f"Batch size: {batch_size},  max_length: {max_length},  grad_accum: {grad_accum}")
    ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=streaming)
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    train_iter = _tokenized_pair_iterator(ds, tokenizer, max_length, batch_size=batch_size)

    print("Preparing evaluation set...")
    eval_ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=streaming)
    eval_ds = eval_ds.shuffle(seed=seed + 1000, buffer_size=5000)
    eval_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for ids_a, ids_b in _tokenized_pair_iterator(eval_ds, tokenizer, max_length, batch_size=1):
        eval_pairs.append((ids_a, ids_b))
        if len(eval_pairs) >= num_eval_pairs:
            break
    del eval_ds
    print(f"Eval set: {len(eval_pairs)} pairs")

    # ------------------------------------------------------------------
    # LAMBADA pairs (for periodic downstream eval)
    # ------------------------------------------------------------------
    print("Loading LAMBADA dataset...")
    _lambada_ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    _lambada_data = []
    for ex in _lambada_ds:
        text = ex["text"]
        last_space = text.rfind(" ")
        _lambada_data.append({"prompt": text[:last_space], "completion": text[last_space + 1:]})
    _lambada_data.sort(key=lambda x: len(tokenizer.encode(x["prompt"])))
    _lambada_pairs = [(_lambada_data[i], _lambada_data[i + 1])
                      for i in range(0, len(_lambada_data) - 1, 2)]
    _rng = random.Random(seed)
    _rng.shuffle(_lambada_pairs)
    lambada_pairs = _lambada_pairs[:num_lambada_pairs]
    del _lambada_ds, _lambada_data, _lambada_pairs
    print(f"LAMBADA eval: {len(lambada_pairs)} pairs, interval={lambada_interval} steps")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    if resume_run_dir and os.path.isdir(resume_run_dir):
        run_dir = resume_run_dir
        print(f"Continuing in existing run dir: {run_dir}")
    else:
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = model_name.replace("/", "_")
        run_name = f"sm_K{K}_M{split_before_end}_{model_tag}_{ts}"
        run_dir = os.path.join(save_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    config = dict(
        model_name=model_name, merge_after=K, split_before_end=split_before_end,
        split_layer=split_layer, num_layers=num_layers, hidden_dim=hidden_dim,
        dataset_name=dataset_name, dataset_config=dataset_config,
        batch_size=batch_size, max_length=max_length, grad_accum=grad_accum,
        num_train_steps=num_train_steps,
        lr_backbone=lr_backbone, lr_projections=lr_projections,
        weight_decay=weight_decay, warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm, init_noise_std=init_noise_std,
        seed=seed, mark_params=mark_params, split_params=split_params,
        lambada_interval=lambada_interval, num_lambada_pairs=num_lambada_pairs,
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    loss_fn = nn.CrossEntropyLoss()

    def _get_pos_emb(hidden_states: torch.Tensor) -> Tuple:
        """Compute (cos, sin) position embeddings for the given hidden states."""
        B, T, _ = hidden_states.shape
        dev = hidden_states.device
        pos_ids = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
        return rotary_emb(hidden_states, pos_ids)

    def _forward_split_merge(ids_a: torch.Tensor, ids_b: torch.Tensor):
        """Full differentiable forward through the split/merge architecture.

        Returns logits_a (B,T,V), logits_b (B,T,V).
        """
        # --- Embed ---
        emb_a = embed_fn(ids_a)
        emb_b = embed_fn(ids_b)

        # Pre-compute position embeddings (shared, same seq length everywhere)
        pos_emb = _get_pos_emb(emb_a)

        # --- First K layers: separate per stream ---
        h_a = _forward_layers(blocks, emb_a, 0, K, position_embeddings=pos_emb)
        h_b = _forward_layers(blocks, emb_b, 0, K, position_embeddings=pos_emb)

        # --- Mark + merge ---
        h_a = h_a.to(merge_device)
        h_b = h_b.to(merge_device)
        marked_a, marked_b = marking(h_a, h_b)
        h_mix = 0.5 * (marked_a + marked_b)

        # --- Shared middle layers ---
        h_mix = _forward_layers(blocks, h_mix, K, split_layer, position_embeddings=pos_emb)

        # --- Split ---
        h_mix = h_mix.to(split_device)
        h_split_a, h_split_b = split_proj(h_mix)

        # --- Last M layers: separate per stream ---
        h_out_a = _forward_layers(blocks, h_split_a, split_layer, num_layers, position_embeddings=pos_emb)
        h_out_b = _forward_layers(blocks, h_split_b, split_layer, num_layers, position_embeddings=pos_emb)

        # --- Norm + LM head ---
        norm_device = next(norm_fn.parameters()).device
        h_out_a = norm_fn(h_out_a.to(norm_device))
        h_out_b = norm_fn(h_out_b.to(norm_device))

        head_device = next(lm_head.parameters()).device
        logits_a = lm_head(h_out_a.to(head_device))
        logits_b = lm_head(h_out_b.to(head_device))

        return logits_a, logits_b

    def _compute_loss(ids_a, ids_b):
        logits_a, logits_b = _forward_split_merge(ids_a, ids_b)
        la = loss_fn(logits_a[:, :-1].contiguous().view(-1, logits_a.size(-1)),
                     ids_a[:, 1:].contiguous().view(-1).to(logits_a.device))
        lb = loss_fn(logits_b[:, :-1].contiguous().view(-1, logits_b.size(-1)),
                     ids_b[:, 1:].contiguous().view(-1).to(logits_b.device))
        return 0.5 * (la + lb), la.item(), lb.item()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    eval_loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _evaluate():
        model.eval()
        marking.eval()
        split_proj.eval()
        records: List[Dict[str, float]] = []

        for ids_a, ids_b in eval_pairs:
            ids_a = ids_a.to(device)
            ids_b = ids_b.to(device)
            mask = torch.ones_like(ids_a)
            T = ids_a.shape[1]

            # Oracle = standard (non-split) forward of the *current* model
            probs_a, _ = get_probs_from_ids(model, ids_a, mask)
            probs_b, _ = get_probs_from_ids(model, ids_b, mask)

            logits_sa, logits_sb = _forward_split_merge(ids_a, ids_b)
            probs_sa = torch.softmax(logits_sa, dim=-1)
            probs_sb = torch.softmax(logits_sb, dim=-1)

            # CE loss: how well split/merge predicts next tokens
            ce_a = eval_loss_fn(
                logits_sa[:, :-1].contiguous().view(-1, logits_sa.size(-1)),
                ids_a[:, 1:].contiguous().view(-1).to(logits_sa.device),
            ).item()
            ce_b = eval_loss_fn(
                logits_sb[:, :-1].contiguous().view(-1, logits_sb.size(-1)),
                ids_b[:, 1:].contiguous().view(-1).to(logits_sb.device),
            ).item()

            # Divergence from oracle (standard forward)
            da = compute_all_metrics(probs_sa[0], probs_a[0])
            db = compute_all_metrics(probs_sb[0], probs_b[0])

            ranks_a, ranks_b = [], []
            t5a = t5b = t10a = t10b = 0
            for t in range(T):
                tok_a = int(torch.argmax(probs_a[0, t]).item())
                tok_b = int(torch.argmax(probs_b[0, t]).item())
                ra = rank_of_token_in_distribution(probs_sa[0, t], tok_a)
                rb = rank_of_token_in_distribution(probs_sb[0, t], tok_b)
                ranks_a.append(ra)
                ranks_b.append(rb)
                if ra <= 5: t5a += 1
                if rb <= 5: t5b += 1
                if ra <= 10: t10a += 1
                if rb <= 10: t10b += 1

            records.append({
                "ce_a": ce_a, "ce_b": ce_b,
                "kl_a": float(da["kl"].mean()), "kl_b": float(db["kl"].mean()),
                "js_a": float(da["js"].mean()), "js_b": float(db["js"].mean()),
                "tvd_a": float(da["tvd"].mean()), "tvd_b": float(db["tvd"].mean()),
                "med_rk_a": float(np.median(ranks_a)),
                "med_rk_b": float(np.median(ranks_b)),
                "top5_a": t5a / T, "top5_b": t5b / T,
                "top10_a": t10a / T, "top10_b": t10b / T,
            })

        model.train()
        marking.train()
        split_proj.train()
        agg = {k: float(np.mean([r[k] for r in records])) for k in records[0]}
        return agg

    def _log_eval(step, ev):
        print(
            f"[Step {step:>5d}] EVAL  "
            f"CE_A={ev['ce_a']:.3f}  CE_B={ev['ce_b']:.3f}  "
            f"KL_A={ev['kl_a']:.3f}  KL_B={ev['kl_b']:.3f}  "
            f"TVD_A={ev['tvd_a']:.3f}  TVD_B={ev['tvd_b']:.3f}  "
            f"med_rk_A={ev['med_rk_a']:.1f}  med_rk_B={ev['med_rk_b']:.1f}  "
            f"top5_A={ev['top5_a']:.3f}  top5_B={ev['top5_b']:.3f}"
        )
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # LAMBADA eval
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_lambada(step: int):
        model.eval()
        marking.eval()
        split_proj.eval()

        correct_a = correct_b = total = 0
        ce_sum_a = ce_sum_b = 0.0
        lambada_loss_fn = nn.CrossEntropyLoss()

        mark_dev = next(marking.parameters()).device
        split_dev = next(split_proj.parameters()).device
        norm_dev = next(norm_fn.parameters()).device
        head_dev = next(lm_head.parameters()).device

        for item_a, item_b in lambada_pairs:
            tok_a = tokenizer(item_a["prompt"], return_tensors="pt",
                              truncation=True, max_length=max_length)["input_ids"].to(device)
            tok_b = tokenizer(item_b["prompt"], return_tensors="pt",
                              truncation=True, max_length=max_length)["input_ids"].to(device)

            min_len = min(tok_a.shape[1], tok_b.shape[1])
            inp_a = tok_a[:, -min_len:]
            inp_b = tok_b[:, -min_len:]

            # Split-merge forward
            emb_a = embed_fn(inp_a)
            emb_b = embed_fn(inp_b)
            pos_emb = _get_pos_emb(emb_a)

            h_a = _forward_layers(blocks, emb_a, 0, K, position_embeddings=pos_emb)
            h_b = _forward_layers(blocks, emb_b, 0, K, position_embeddings=pos_emb)

            m_a, m_b = marking(h_a.to(mark_dev), h_b.to(mark_dev))
            h_mix = 0.5 * (m_a + m_b)

            h_mix = _forward_layers(blocks, h_mix, K, split_layer, position_embeddings=pos_emb)
            h_sa, h_sb = split_proj(h_mix.to(split_dev))

            h_oa = _forward_layers(blocks, h_sa, split_layer, num_layers, position_embeddings=pos_emb)
            h_ob = _forward_layers(blocks, h_sb, split_layer, num_layers, position_embeddings=pos_emb)

            h_oa = norm_fn(h_oa.to(norm_dev))
            h_ob = norm_fn(h_ob.to(norm_dev))

            logits_a = lm_head(h_oa.to(head_dev))
            logits_b = lm_head(h_ob.to(head_dev))

            # Accuracy
            exp_a = tokenizer.encode(" " + item_a["completion"], add_special_tokens=False)
            exp_b = tokenizer.encode(" " + item_b["completion"], add_special_tokens=False)
            correct_a += (logits_a[0, -1].argmax().item() == exp_a[0])
            correct_b += (logits_b[0, -1].argmax().item() == exp_b[0])
            total += 1

            # CE on last token
            tgt_a = torch.tensor([exp_a[0]], device=logits_a.device)
            tgt_b = torch.tensor([exp_b[0]], device=logits_b.device)
            ce_sum_a += lambada_loss_fn(logits_a[0, -1:], tgt_a).item()
            ce_sum_b += lambada_loss_fn(logits_b[0, -1:], tgt_b).item()

        acc_a = correct_a / total
        acc_b = correct_b / total
        acc_avg = (acc_a + acc_b) / 2
        ce_a = ce_sum_a / total
        ce_b = ce_sum_b / total

        print(f"[Step {step:>5d}] LAMBADA  "
              f"AccA={acc_a:.4f}  AccB={acc_b:.4f}  Avg={acc_avg:.4f}  "
              f"CE_A={ce_a:.3f}  CE_B={ce_b:.3f}")
        sys.stdout.flush()

        writer.add_scalar("lambada/acc_a", acc_a, step)
        writer.add_scalar("lambada/acc_b", acc_b, step)
        writer.add_scalar("lambada/acc_avg", acc_avg, step)
        writer.add_scalar("lambada/ce_a", ce_a, step)
        writer.add_scalar("lambada/ce_b", ce_b, step)

        model.train()
        marking.train()
        split_proj.train()
        return {"acc_a": acc_a, "acc_b": acc_b, "acc_avg": acc_avg, "ce_a": ce_a, "ce_b": ce_b}

    # ------------------------------------------------------------------
    # Checkpoint helper
    # ------------------------------------------------------------------
    def _save_checkpoint(path: str, step: int):
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "marking": marking.state_dict(),
            "split_proj": split_proj.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scheduler_type": scheduler_type,
        }, path)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_losses: List[Dict] = []
    eval_history: List[Dict] = []
    best_eval_loss = float("inf")
    best_step = -1

    print(f"\n{'='*90}")
    print(f"Training split/merge  K={K}, M={split_before_end}  "
          f"for {num_train_steps} steps (starting from {start_step})")
    print(f"LR backbone={lr_backbone},  LR projections={lr_projections}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Batch size: {batch_size},  grad_accum: {grad_accum},  "
          f"effective batch: {batch_size * grad_accum}")
    print(f"{'='*90}")

    if start_step == 0:
        ev = _evaluate()
        eval_history.append({"step": 0, **ev})
        _log_eval(0, ev)
        for k, v in ev.items():
            writer.add_scalar(f"eval/{k}", v, 0)

    model.train()
    marking.train()
    split_proj.train()
    optimizer.zero_grad()
    running_loss = 0.0
    accum_count = 0

    for step in range(start_step + 1, start_step + num_train_steps + 1):
        # --- get batch ---
        try:
            ids_a, ids_b = next(train_iter)
        except StopIteration:
            ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=streaming)
            ds = ds.shuffle(seed=seed + step, buffer_size=5000)
            train_iter = _tokenized_pair_iterator(ds, tokenizer, max_length, batch_size=batch_size)
            ids_a, ids_b = next(train_iter)

        ids_a = ids_a.to(device)
        ids_b = ids_b.to(device)

        loss, la, lb = _compute_loss(ids_a, ids_b)
        (loss / grad_accum).backward()
        accum_count += 1

        if accum_count >= grad_accum:
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(marking.parameters()) + list(split_proj.parameters()),
                max_grad_norm,
            )
            optimizer.step()
            if scheduler_type == "cosine":
                scheduler.step()
            optimizer.zero_grad()
            accum_count = 0

        running_loss += loss.item()

        if step % log_interval == 0:
            avg = running_loss / log_interval
            cur_lr_bb = optimizer.param_groups[0]["lr"]
            cur_lr_proj = optimizer.param_groups[1]["lr"]
            print(f"[Step {step:>5d}] loss={avg:.4f}  (A={la:.4f} B={lb:.4f})  "
                  f"lr_bb={cur_lr_bb:.2e}  lr_proj={cur_lr_proj:.2e}")
            sys.stdout.flush()
            train_losses.append({"step": step, "loss": avg, "loss_a": la, "loss_b": lb})
            writer.add_scalar("train/loss", avg, step)
            writer.add_scalar("train/loss_a", la, step)
            writer.add_scalar("train/loss_b", lb, step)
            writer.add_scalar("lr/backbone", cur_lr_bb, step)
            writer.add_scalar("lr/projections", cur_lr_proj, step)
            running_loss = 0.0

        if step % eval_interval == 0:
            ev = _evaluate()
            eval_history.append({"step": step, **ev})
            _log_eval(step, ev)
            for k, v in ev.items():
                writer.add_scalar(f"eval/{k}", v, step)
            cur_eval_loss = 0.5 * (ev["ce_a"] + ev["ce_b"])
            if cur_eval_loss < best_eval_loss:
                best_eval_loss = cur_eval_loss
                best_step = step
                _save_checkpoint(os.path.join(run_dir, "checkpoint_best.pt"), step)
                print(f"  -> New best checkpoint (CE={best_eval_loss:.4f})")
            model.train()
            marking.train()
            split_proj.train()

        if step % lambada_interval == 0:
            lambada_res = _eval_lambada(step)
            if scheduler_type == "plateau":
                old_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(lambada_res["acc_avg"])
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr < old_lr:
                    print(f"  -> ReduceLROnPlateau: LR reduced {old_lr:.2e} -> {new_lr:.2e}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {"config": config, "train_losses": train_losses, "eval_history": eval_history}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    last_step = start_step + num_train_steps
    _save_checkpoint(os.path.join(run_dir, "checkpoint_last.pt"), last_step)

    # Remove best checkpoint if it's the same step as last (no point keeping both)
    best_path = os.path.join(run_dir, "checkpoint_best.pt")
    if best_step == last_step and os.path.exists(best_path):
        os.remove(best_path)
        print(f"Best checkpoint == last checkpoint (step {last_step}), keeping only last.")

    writer.close()
    print(f"\nDone! Results saved to: {run_dir}")


if __name__ == "__main__":
    fire.Fire(train)
