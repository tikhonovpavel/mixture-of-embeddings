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
import queue
import threading
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

def _prefetch_iterator(iterator: Iterator, maxsize: int = 4) -> Iterator:
    """Wraps an iterator with a background thread that prefetches batches.

    Allows data loading / tokenization to overlap with GPU compute.
    """
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    _sentinel = object()

    def _worker():
        try:
            for item in iterator:
                q.put(item)
        finally:
            q.put(_sentinel)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is _sentinel:
            break
        yield item


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
    scheduler_type: str = "plateau_windowed",  # "cosine", "plateau", or "plateau_windowed"
    plateau_patience: int = 5,        # for "plateau": in units of lambada_interval
    plateau_factor: float = 0.5,
    plateau_threshold: float = 0.005, # LEGACY: used only for plateau_signal="lambada"
    plateau_window: int = 5,          # LEGACY: used only for plateau_signal="lambada"
    # scheduler — new eval_ce-based plateau
    plateau_signal: str = "eval_ce",  # "eval_ce" (recommended) or "lambada"
    plateau_window_evals: int = 20,   # window size in units of eval_interval (500 steps)
    plateau_threshold_ce_abs: float = 0.003,  # absolute CE drop required per window
    plateau_cooldown_evals: int = 40, # after a drop, skip decisions for N eval_interval's
    # loss balancing
    use_loss_balancing: bool = True,
    balance_signal: str = "eval_ce",  # "eval_ce" (recommended) or "train_loss"
    loss_balance_strength: float = 5.0,
    loss_balance_ema_momentum: float = 0.99,      # for train_loss signal (per-step EMA)
    loss_balance_ema_momentum_eval: float = 0.9,  # for eval_ce signal (per-eval EMA)
    loss_balance_warmup_steps: int = 100,         # train_loss-only: skip balance for first N steps
    loss_balance_warmup_evals: int = 2,           # eval_ce-only: skip balance for first N evals
    loss_balance_clip_min: float = 0.1,
    loss_balance_clip_max: float = 10.0,
    # resume
    resume_from: str = "",               # path to checkpoint or run dir
    resume_warmup_steps: int = 2000,     # linear LR warmup after resume
    resume_reset_optimizer: bool = False, # drop Adam moments on resume (recommended if LR was reset from ~0)
    # manual override + safety
    override_check_interval: int = 100,  # poll run_dir/override.json every N steps
    checkpoint_last_interval: int = 10_000,  # save checkpoint_last.pt every N steps (0 = only at end)
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

    Manual override mechanism:
        Write a JSON file to {run_dir}/override.json. Every
        `override_check_interval` steps the training loop reads it, applies
        the fields, appends a record to {run_dir}/override_applied.jsonl, and
        renames the file to override.applied_<ts>.json. Supported keys:

          "lr_backbone":              float   — set optimizer param_group[0] lr directly
          "lr_projections":           float   — set optimizer param_group[1] lr directly
          "balance_alpha":            float   — set current balance_alpha directly
          "balance_beta":             float   — set current balance_beta directly
          "plateau_enabled":          bool    — turn eval_ce plateau detector on/off
          "use_loss_balancing":       bool    — turn auto-balance on/off
          "plateau_factor":           float   — change next drop magnitude
          "plateau_threshold_ce_abs": float   — change plateau threshold
          "plateau_window_evals":     int     — change window length
          "plateau_cooldown_evals":   int     — change cooldown length (for future drops)
          "plateau_cooldown_until":   int     — set absolute step until which plateau is frozen
          "num_train_steps":          int     — new total training steps (recomputes target)
          "target_last_step":         int     — absolute step to train until (overrides num_train_steps)

        Overrides affect ONLY the listed fields. EMA, eval history, optimizer
        moments, etc. are untouched. If you override LR and want it to stick,
        also set `"plateau_enabled": false` or a large `plateau_cooldown_until`,
        otherwise the plateau detector may reduce it on the next eval.
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

    print("Gradient checkpointing: disabled (sufficient VRAM available)")

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
    elif scheduler_type == "plateau_windowed":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    # Windowed plateau state (legacy: LAMBADA-based)
    lambada_acc_history: List[float] = []
    # Windowed plateau state (new: eval_ce-based)
    eval_ce_history: List[float] = []
    plateau_cooldown_until: int = 0
    plateau_enabled: bool = True      # can be flipped via override file

    # Loss balancing state
    ema_loss_a: Optional[float] = None
    ema_loss_b: Optional[float] = None
    balance_alpha = 1.0
    balance_beta = 1.0
    balance_eval_count: int = 0       # how many eval_ce samples seen (for eval_ce warmup)

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    resume_run_dir = ""
    if resume_from:
        ckpt_path = resume_from
        if os.path.isdir(resume_from):
            # REFUSE to pick silently — this has already cost a 420k-step run.
            best_pt = os.path.join(resume_from, "checkpoint_best.pt")
            last_pt = os.path.join(resume_from, "checkpoint_last.pt")
            candidates = [p for p in (best_pt, last_pt) if os.path.exists(p)]
            raise ValueError(
                "resume_from is a directory; I won't guess which checkpoint to load. "
                "Pass an explicit file path. Available in this dir:\n  "
                + "\n  ".join(candidates)
            )
        elif os.path.isfile(resume_from):
            resume_run_dir = os.path.dirname(resume_from)

        print(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        marking.load_state_dict(ckpt["marking"])
        split_proj.load_state_dict(ckpt["split_proj"])
        if resume_reset_optimizer:
            print("  resume_reset_optimizer=True: fresh AdamW (discarding old Adam moments)")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
        for pg in optimizer.param_groups:
            if pg is optimizer.param_groups[0]:
                pg["lr"] = lr_backbone
            else:
                pg["lr"] = lr_projections
        if scheduler_type == "cosine" and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        # Restore loss-balance EMA only if signal matches old run (same scale).
        # If balance_signal="eval_ce" now but old ckpt was train_loss-based, the
        # EMA magnitudes are compatible (both are CE) but we still prefer to
        # re-init from the first eval to avoid carrying over a stale EMA across
        # warmup; tweak here if you want to preserve.
        ckpt_balance_signal = ckpt.get("balance_signal", "train_loss")
        if "ema_loss_a" in ckpt and ckpt["ema_loss_a"] is not None \
                and ckpt_balance_signal == balance_signal:
            ema_loss_a = ckpt["ema_loss_a"]
            ema_loss_b = ckpt["ema_loss_b"]
            balance_alpha = ckpt.get("balance_alpha", 1.0)
            balance_beta = ckpt.get("balance_beta", 1.0)
            balance_eval_count = ckpt.get("balance_eval_count", 0)
            print(f"  Restored loss balancing ({balance_signal}): "
                  f"ema_A={ema_loss_a:.4f}, ema_B={ema_loss_b:.4f}, "
                  f"alpha={balance_alpha:.4f}, beta={balance_beta:.4f}")
        else:
            print(f"  Loss balancing reset (ckpt signal={ckpt_balance_signal!r}, new={balance_signal!r})")
        if "lambada_acc_history" in ckpt and plateau_signal == "lambada":
            lambada_acc_history = ckpt["lambada_acc_history"]
            print(f"  Restored lambada_acc_history: {len(lambada_acc_history)} entries")
        if "eval_ce_history" in ckpt and plateau_signal == "eval_ce":
            eval_ce_history = ckpt["eval_ce_history"]
            plateau_cooldown_until = ckpt.get("plateau_cooldown_until", 0)
            plateau_enabled = ckpt.get("plateau_enabled", True)
            print(f"  Restored eval_ce_history: {len(eval_ce_history)} entries, "
                  f"cooldown_until={plateau_cooldown_until}, enabled={plateau_enabled}")
        start_step = ckpt["step"]
        print(f"Resumed at step {start_step}, LRs reset to backbone={lr_backbone}, proj={lr_projections}")
        print(f"Resume warmup: {resume_warmup_steps} steps")
        del ckpt

    do_resume_warmup = bool(resume_from) and resume_warmup_steps > 0
    resume_target_lrs = [lr_backbone, lr_projections]

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"Loading dataset: {dataset_name} ({dataset_config})")
    print(f"Batch size: {batch_size},  max_length: {max_length},  grad_accum: {grad_accum}")
    ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=streaming)
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    train_iter = _prefetch_iterator(
        _tokenized_pair_iterator(ds, tokenizer, max_length, batch_size=batch_size),
        maxsize=8,
    )

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
        eval_interval=eval_interval,
        lambada_interval=lambada_interval, num_lambada_pairs=num_lambada_pairs,
        scheduler_type=scheduler_type,
        plateau_signal=plateau_signal,
        plateau_window_evals=plateau_window_evals,
        plateau_threshold_ce_abs=plateau_threshold_ce_abs,
        plateau_cooldown_evals=plateau_cooldown_evals,
        plateau_factor=plateau_factor,
        plateau_window=plateau_window,
        plateau_threshold=plateau_threshold,
        use_loss_balancing=use_loss_balancing,
        balance_signal=balance_signal,
        loss_balance_strength=loss_balance_strength,
        loss_balance_ema_momentum=loss_balance_ema_momentum,
        loss_balance_ema_momentum_eval=loss_balance_ema_momentum_eval,
        loss_balance_warmup_evals=loss_balance_warmup_evals,
        resume_warmup_steps=resume_warmup_steps,
        resume_reset_optimizer=resume_reset_optimizer,
        override_check_interval=override_check_interval,
        checkpoint_last_interval=checkpoint_last_interval,
    )
    config_path = os.path.join(run_dir, "config.json")
    if resume_from and os.path.exists(config_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = os.path.join(run_dir, f"config_resumed_{ts}.json")
    with open(config_path, "w") as f:
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

    def _recompute_balance():
        """Recompute balance_alpha, balance_beta from current EMA values."""
        nonlocal balance_alpha, balance_beta
        if ema_loss_a is None or ema_loss_b is None:
            return
        total = ema_loss_a + ema_loss_b
        w_a = (ema_loss_a / total) ** loss_balance_strength
        w_b = (ema_loss_b / total) ** loss_balance_strength
        norm = w_a + w_b
        balance_alpha = max(loss_balance_clip_min, min(loss_balance_clip_max, w_a / norm * 2.0))
        balance_beta = max(loss_balance_clip_min, min(loss_balance_clip_max, w_b / norm * 2.0))

    def _update_balance_train_loss(la_val: float, lb_val: float, step: int):
        """Per-step EMA update from train loss (legacy signal).

        EMA is always maintained (so turning ``use_loss_balancing`` back on
        via override is smooth). alpha/beta are only recomputed when the
        balancer is actively enabled.
        """
        nonlocal ema_loss_a, ema_loss_b
        if balance_signal != "train_loss":
            return
        mom = loss_balance_ema_momentum
        if ema_loss_a is None:
            ema_loss_a = la_val
            ema_loss_b = lb_val
        else:
            ema_loss_a = mom * ema_loss_a + (1 - mom) * la_val
            ema_loss_b = mom * ema_loss_b + (1 - mom) * lb_val
        if use_loss_balancing and step >= loss_balance_warmup_steps:
            _recompute_balance()

    def _update_balance_eval_ce(ce_a_val: float, ce_b_val: float):
        """Per-eval EMA update from eval CE (new signal, cleaner + lower-noise).

        EMA is always maintained; alpha/beta only change when balancing is enabled.
        """
        nonlocal ema_loss_a, ema_loss_b, balance_eval_count
        if balance_signal != "eval_ce":
            return
        balance_eval_count += 1
        mom = loss_balance_ema_momentum_eval
        if ema_loss_a is None:
            # First sample: initialize EMA but do not apply balance yet —
            # we'd rather wait loss_balance_warmup_evals samples to build a
            # stable estimate before touching alpha/beta.
            ema_loss_a = ce_a_val
            ema_loss_b = ce_b_val
            return
        ema_loss_a = mom * ema_loss_a + (1 - mom) * ce_a_val
        ema_loss_b = mom * ema_loss_b + (1 - mom) * ce_b_val
        if use_loss_balancing and balance_eval_count >= loss_balance_warmup_evals:
            _recompute_balance()

    def _compute_loss(ids_a, ids_b):
        logits_a, logits_b = _forward_split_merge(ids_a, ids_b)
        la = loss_fn(logits_a[:, :-1].contiguous().view(-1, logits_a.size(-1)),
                     ids_a[:, 1:].contiguous().view(-1).to(logits_a.device))
        lb = loss_fn(logits_b[:, :-1].contiguous().view(-1, logits_b.size(-1)),
                     ids_b[:, 1:].contiguous().view(-1).to(logits_b.device))
        loss = 0.5 * (balance_alpha * la + balance_beta * lb)
        return loss, la.item(), lb.item()

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
        ckpt_data = {
            "step": step,
            "model": model.state_dict(),
            "marking": marking.state_dict(),
            "split_proj": split_proj.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler_type": scheduler_type,
            "balance_signal": balance_signal,
            "ema_loss_a": ema_loss_a,
            "ema_loss_b": ema_loss_b,
            "balance_alpha": balance_alpha,
            "balance_beta": balance_beta,
            "balance_eval_count": balance_eval_count,
            "lambada_acc_history": lambada_acc_history,
            "eval_ce_history": eval_ce_history,
            "plateau_cooldown_until": plateau_cooldown_until,
            "plateau_enabled": plateau_enabled,
            "plateau_signal": plateau_signal,
        }
        if scheduler is not None:
            ckpt_data["scheduler"] = scheduler.state_dict()
        torch.save(ckpt_data, path)

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
    if scheduler_type == "plateau_windowed":
        if plateau_signal == "eval_ce":
            _eff_window_steps = plateau_window_evals * eval_interval
            _eff_cooldown_steps = plateau_cooldown_evals * eval_interval
            print(f"Scheduler: {scheduler_type} signal=eval_ce  "
                  f"window={plateau_window_evals} evals ({_eff_window_steps} steps), "
                  f"cooldown={plateau_cooldown_evals} evals ({_eff_cooldown_steps} steps), "
                  f"threshold_abs={plateau_threshold_ce_abs}, factor={plateau_factor}")
        else:
            print(f"Scheduler: {scheduler_type} signal=lambada  "
                  f"window={plateau_window}, factor={plateau_factor}, "
                  f"threshold={plateau_threshold}")
    else:
        print(f"Scheduler: {scheduler_type}")
    if use_loss_balancing:
        if balance_signal == "eval_ce":
            print(f"Loss balancing: eval_ce  strength={loss_balance_strength}, "
                  f"ema_mom={loss_balance_ema_momentum_eval}, "
                  f"warmup_evals={loss_balance_warmup_evals}")
        else:
            print(f"Loss balancing: train_loss  strength={loss_balance_strength}, "
                  f"ema_mom={loss_balance_ema_momentum}")
    else:
        print("Loss balancing: disabled")
    if resume_from:
        print(f"Resume: warmup={resume_warmup_steps} steps, "
              f"reset_optimizer={resume_reset_optimizer}")
    print(f"Batch size: {batch_size},  grad_accum: {grad_accum},  "
          f"effective batch: {batch_size * grad_accum}")
    print(f"Manual override: write JSON to {os.path.join(run_dir, 'override.json')} "
          f"(polled every {override_check_interval} steps)")
    if checkpoint_last_interval > 0:
        print(f"Periodic checkpoint_last.pt every {checkpoint_last_interval} steps.")
    print(f"{'='*90}")

    if start_step == 0:
        ev = _evaluate()
        eval_history.append({"step": 0, **ev})
        _log_eval(0, ev)
        for k, v in ev.items():
            writer.add_scalar(f"eval/{k}", v, 0)

    # ------------------------------------------------------------------
    # Manual-override poller
    # ------------------------------------------------------------------
    override_path = os.path.join(run_dir, "override.json")
    override_applied_log = os.path.join(run_dir, "override_applied.jsonl")

    # Set of keys we know how to apply. Anything else → warning.
    _OVERRIDABLE_FLOAT = {
        "lr_backbone", "lr_projections",
        "balance_alpha", "balance_beta",
        "plateau_factor", "plateau_threshold_ce_abs",
    }
    _OVERRIDABLE_INT = {
        "plateau_window_evals", "plateau_cooldown_evals",
        "plateau_cooldown_until",
        "num_train_steps", "target_last_step",
    }
    _OVERRIDABLE_BOOL = {"plateau_enabled", "use_loss_balancing"}
    _OVERRIDABLE = _OVERRIDABLE_FLOAT | _OVERRIDABLE_INT | _OVERRIDABLE_BOOL

    def _apply_overrides(cur_step: int) -> None:
        """Read override.json, apply in-place, archive it. No side effects
        on any state that isn't explicitly named in the file."""
        nonlocal balance_alpha, balance_beta
        nonlocal plateau_cooldown_until, plateau_enabled
        nonlocal plateau_factor, plateau_threshold_ce_abs
        nonlocal plateau_window_evals, plateau_cooldown_evals
        nonlocal use_loss_balancing, target_last_step
        if not os.path.exists(override_path):
            return
        try:
            with open(override_path) as f:
                ovr = json.load(f)
        except Exception as e:
            print(f"\n!! OVERRIDE unreadable at step {cur_step}: {e!r}, leaving file in place\n")
            return
        if not isinstance(ovr, dict):
            print(f"\n!! OVERRIDE must be a JSON object, got {type(ovr)!r}. Skipping.\n")
            return

        applied: Dict = {}
        unknown = [k for k in ovr.keys() if k not in _OVERRIDABLE]
        for k in unknown:
            print(f"  !! OVERRIDE: unknown key {k!r}, ignored")

        # LR (direct write to optimizer param groups)
        if "lr_backbone" in ovr:
            optimizer.param_groups[0]["lr"] = float(ovr["lr_backbone"])
            applied["lr_backbone"] = float(ovr["lr_backbone"])
        if "lr_projections" in ovr:
            optimizer.param_groups[1]["lr"] = float(ovr["lr_projections"])
            applied["lr_projections"] = float(ovr["lr_projections"])

        # Balance (direct, no EMA touch)
        if "balance_alpha" in ovr:
            balance_alpha = float(ovr["balance_alpha"])
            applied["balance_alpha"] = balance_alpha
        if "balance_beta" in ovr:
            balance_beta = float(ovr["balance_beta"])
            applied["balance_beta"] = balance_beta

        # Plateau behaviour
        if "plateau_enabled" in ovr:
            plateau_enabled = bool(ovr["plateau_enabled"])
            applied["plateau_enabled"] = plateau_enabled
        if "plateau_factor" in ovr:
            plateau_factor = float(ovr["plateau_factor"])
            applied["plateau_factor"] = plateau_factor
        if "plateau_threshold_ce_abs" in ovr:
            plateau_threshold_ce_abs = float(ovr["plateau_threshold_ce_abs"])
            applied["plateau_threshold_ce_abs"] = plateau_threshold_ce_abs
        if "plateau_window_evals" in ovr:
            plateau_window_evals = int(ovr["plateau_window_evals"])
            applied["plateau_window_evals"] = plateau_window_evals
        if "plateau_cooldown_evals" in ovr:
            plateau_cooldown_evals = int(ovr["plateau_cooldown_evals"])
            applied["plateau_cooldown_evals"] = plateau_cooldown_evals
        if "plateau_cooldown_until" in ovr:
            plateau_cooldown_until = int(ovr["plateau_cooldown_until"])
            applied["plateau_cooldown_until"] = plateau_cooldown_until

        # Balancer toggle (EMA keeps updating either way)
        if "use_loss_balancing" in ovr:
            use_loss_balancing = bool(ovr["use_loss_balancing"])
            applied["use_loss_balancing"] = use_loss_balancing

        # Training horizon
        if "target_last_step" in ovr:
            target_last_step = int(ovr["target_last_step"])
            applied["target_last_step"] = target_last_step
        elif "num_train_steps" in ovr:
            target_last_step = start_step + int(ovr["num_train_steps"])
            applied["target_last_step"] = target_last_step
            applied["num_train_steps"] = int(ovr["num_train_steps"])

        ts = datetime.now().isoformat()
        print(f"\n== OVERRIDE applied at step {cur_step} (ts={ts}) ==")
        for k, v in applied.items():
            print(f"   {k}: {v}")
        print("=" * 60 + "\n")
        sys.stdout.flush()

        # Log to TB as a text event (so you can see it in the event timeline)
        writer.add_text("override", json.dumps(applied), cur_step)
        # Append to persistent applied-log
        with open(override_applied_log, "a") as f:
            f.write(json.dumps({"step": cur_step, "ts": ts, "applied": applied}) + "\n")
        # Archive so we don't re-apply
        try:
            os.rename(override_path, override_path + f".applied_{ts.replace(':','-')}")
        except OSError as e:
            print(f"!! couldn't archive override file: {e}, removing instead")
            try:
                os.remove(override_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # SIGINT handler: save checkpoint_last before exit
    # ------------------------------------------------------------------
    import signal as _signal
    interrupt_requested = [False]

    def _sigint_handler(signum, frame):
        if interrupt_requested[0]:
            print("\n!! Second SIGINT received, hard-exiting.\n", flush=True)
            sys.exit(130)
        interrupt_requested[0] = True
        print("\n!! SIGINT received — will save checkpoint_last and exit after this step. "
              "Send SIGINT again to force quit.\n", flush=True)

    try:
        _signal.signal(_signal.SIGINT, _sigint_handler)
    except ValueError:
        pass  # not in main thread, ignore

    model.train()
    marking.train()
    split_proj.train()
    optimizer.zero_grad()
    running_loss = 0.0
    accum_count = 0

    target_last_step = start_step + num_train_steps
    step = start_step
    while step < target_last_step and not interrupt_requested[0]:
        step += 1
        # --- manual override poll (cheap file stat every few steps) ---
        if override_check_interval > 0 and step % override_check_interval == 0:
            _apply_overrides(step)

        # --- get batch ---
        try:
            ids_a, ids_b = next(train_iter)
        except StopIteration:
            ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=streaming)
            ds = ds.shuffle(seed=seed + step, buffer_size=5000)
            train_iter = _prefetch_iterator(
                _tokenized_pair_iterator(ds, tokenizer, max_length, batch_size=batch_size),
                maxsize=8,
            )
            ids_a, ids_b = next(train_iter)

        ids_a = ids_a.to(device)
        ids_b = ids_b.to(device)

        loss, la, lb = _compute_loss(ids_a, ids_b)
        if balance_signal == "train_loss":
            _update_balance_train_loss(la, lb, step)
        (loss / grad_accum).backward()
        accum_count += 1

        if accum_count >= grad_accum:
            # Apply resume warmup BEFORE optimizer.step() for non-cosine schedulers,
            # otherwise the very first post-resume step runs with full target LR.
            if (do_resume_warmup and (step - start_step) <= resume_warmup_steps
                    and scheduler_type != "cosine"):
                warmup_factor = (step - start_step) / max(resume_warmup_steps, 1)
                for i, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = resume_target_lrs[i] * warmup_factor

            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(marking.parameters()) + list(split_proj.parameters()),
                max_grad_norm,
            )
            optimizer.step()
            if scheduler_type == "cosine":
                scheduler.step()
                # Cosine initialises LambdaLR at schedule(0)=0, so warmup can
                # safely be applied *after* optimizer.step() by multiplication.
                if do_resume_warmup and (step - start_step) <= resume_warmup_steps:
                    warmup_factor = (step - start_step) / max(resume_warmup_steps, 1)
                    for i, pg in enumerate(optimizer.param_groups):
                        pg["lr"] *= warmup_factor
            optimizer.zero_grad()
            accum_count = 0

        running_loss += loss.item()

        if step % log_interval == 0:
            avg = running_loss / log_interval
            cur_lr_bb = optimizer.param_groups[0]["lr"]
            cur_lr_proj = optimizer.param_groups[1]["lr"]
            loss_ratio = la / max(lb, 1e-8)
            print(f"[Step {step:>5d}] loss={avg:.4f}  (A={la:.4f} B={lb:.4f} ratio={loss_ratio:.3f})  "
                  f"α={balance_alpha:.3f} β={balance_beta:.3f}  "
                  f"lr_bb={cur_lr_bb:.2e}  lr_proj={cur_lr_proj:.2e}")
            sys.stdout.flush()
            train_losses.append({"step": step, "loss": avg, "loss_a": la, "loss_b": lb})
            writer.add_scalar("train/loss", avg, step)
            writer.add_scalar("train/loss_a", la, step)
            writer.add_scalar("train/loss_b", lb, step)
            writer.add_scalar("train/loss_ratio_a_over_b", loss_ratio, step)
            writer.add_scalar("train/balance_alpha", balance_alpha, step)
            writer.add_scalar("train/balance_beta", balance_beta, step)
            if ema_loss_a is not None:
                writer.add_scalar("train/ema_loss_a", ema_loss_a, step)
                writer.add_scalar("train/ema_loss_b", ema_loss_b, step)
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

            # --- Loss balancing from eval CE ---
            if balance_signal == "eval_ce":
                _update_balance_eval_ce(ev["ce_a"], ev["ce_b"])
                writer.add_scalar("train/balance_alpha", balance_alpha, step)
                writer.add_scalar("train/balance_beta", balance_beta, step)
                if ema_loss_a is not None:
                    writer.add_scalar("train/ema_loss_a", ema_loss_a, step)
                    writer.add_scalar("train/ema_loss_b", ema_loss_b, step)

            # --- Eval-CE-based plateau detector ---
            if scheduler_type == "plateau_windowed" and plateau_signal == "eval_ce":
                # Always keep history up to date so it's available if the user
                # toggles plateau_enabled back on via an override.
                eval_ce_history.append(cur_eval_loss)
                w = plateau_window_evals
                # Only decide if: enabled AND past cooldown AND we have 2 full windows of history
                if (plateau_enabled
                        and step >= plateau_cooldown_until
                        and len(eval_ce_history) >= 2 * w):
                    recent_mean = float(np.mean(eval_ce_history[-w:]))
                    previous_mean = float(np.mean(eval_ce_history[-2*w:-w]))
                    ce_drop = previous_mean - recent_mean   # positive = improved
                    improved = ce_drop > plateau_threshold_ce_abs
                    writer.add_scalar("scheduler/recent_ce_mean", recent_mean, step)
                    writer.add_scalar("scheduler/previous_ce_mean", previous_mean, step)
                    writer.add_scalar("scheduler/ce_drop", ce_drop, step)
                    if not improved:
                        old_lr = optimizer.param_groups[0]["lr"]
                        for pg in optimizer.param_groups:
                            pg["lr"] *= plateau_factor
                        new_lr = optimizer.param_groups[0]["lr"]
                        plateau_cooldown_until = step + plateau_cooldown_evals * eval_interval
                        print(f"  -> Plateau(eval_ce): no improvement "
                              f"(recent={recent_mean:.4f} vs prev={previous_mean:.4f}, "
                              f"drop={ce_drop:+.4f} < {plateau_threshold_ce_abs:.4f}), "
                              f"LR {old_lr:.2e} -> {new_lr:.2e}, "
                              f"cooldown until step {plateau_cooldown_until}")

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
            elif (scheduler_type == "plateau_windowed" and plateau_signal == "lambada"
                    and plateau_enabled):
                lambada_acc_history.append(lambada_res["acc_avg"])
                w = plateau_window
                if len(lambada_acc_history) >= 2 * w:
                    recent_mean = np.mean(lambada_acc_history[-w:])
                    previous_mean = np.mean(lambada_acc_history[-2*w:-w])
                    improved = recent_mean > previous_mean * (1 + plateau_threshold)
                    writer.add_scalar("scheduler/recent_window_mean", recent_mean, step)
                    writer.add_scalar("scheduler/previous_window_mean", previous_mean, step)
                    if not improved:
                        old_lr = optimizer.param_groups[0]["lr"]
                        for pg in optimizer.param_groups:
                            pg["lr"] *= plateau_factor
                        new_lr = optimizer.param_groups[0]["lr"]
                        print(f"  -> Windowed plateau: no improvement "
                              f"(recent={recent_mean:.4f} vs prev={previous_mean:.4f}), "
                              f"LR reduced {old_lr:.2e} -> {new_lr:.2e}")
                        lambada_acc_history.clear()
                    else:
                        print(f"  -> Windowed plateau: improved "
                              f"(recent={recent_mean:.4f} vs prev={previous_mean:.4f})")

        # --- periodic checkpoint_last.pt (so a crash / SIGINT never wipes >N steps) ---
        if checkpoint_last_interval > 0 and step % checkpoint_last_interval == 0:
            tmp_path = os.path.join(run_dir, "checkpoint_last.pt.tmp")
            final_path = os.path.join(run_dir, "checkpoint_last.pt")
            _save_checkpoint(tmp_path, step)
            os.replace(tmp_path, final_path)  # atomic
            print(f"  -> Saved periodic checkpoint_last.pt at step {step}")
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {"config": config, "train_losses": train_losses, "eval_history": eval_history}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Use the actual current step (might be < target_last_step on SIGINT)
    last_step = step
    tmp_path = os.path.join(run_dir, "checkpoint_last.pt.tmp")
    final_path = os.path.join(run_dir, "checkpoint_last.pt")
    _save_checkpoint(tmp_path, last_step)
    os.replace(tmp_path, final_path)

    # Remove best checkpoint if it's the same step as last (no point keeping both)
    best_path = os.path.join(run_dir, "checkpoint_best.pt")
    if best_step == last_step and os.path.exists(best_path):
        os.remove(best_path)
        print(f"Best checkpoint == last checkpoint (step {last_step}), keeping only last.")

    if interrupt_requested[0]:
        print(f"\n!! Exited early at step {last_step} due to SIGINT. "
              f"checkpoint_last.pt saved.\n")

    writer.close()
    print(f"\nDone! Results saved to: {run_dir}")


if __name__ == "__main__":
    fire.Fire(train)
