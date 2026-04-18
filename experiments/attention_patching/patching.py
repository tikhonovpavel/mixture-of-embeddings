"""Donor-attention capture and patched forward pass.

In Transformers v5, every attention module dispatches through
``ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation,
eager_attention_forward)``.  We register our own implementation under a
dedicated key and temporarily flip ``_attn_implementation`` on the model (and
its config). Our implementation:

  * ignores the computed query @ key^T / softmax path
  * takes pre-computed donor post-softmax attention weights indexed by
    ``module.layer_idx`` from a thread-local buffer
  * multiplies those donor weights by the value tensor that was computed
    normally from the current input (i.e. Q, K, V are still computed from the
    live hidden states, but the *weights* come from the donor)

This gives the A2 regime defined in the experiment spec:

    O^{(l,h)}_i = sum_j A^{(l,h)}(X_C)_{ij} * V^{(l,h)}_j(X_A)

For GQA models (Llama/Qwen), the donor weights are already ``[B, H_q, T, T]``
in the shape after ``repeat_kv``, because that is what HF exposes via
``output_attentions=True``.  We therefore apply the same ``repeat_kv`` to V
before the matmul, matching the behaviour of ``eager_attention_forward`` in
``modeling_llama.py``.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from transformers.models.gpt_neox.modeling_gpt_neox import (
    ALL_ATTENTION_FUNCTIONS,
)


# Thread-local storage for per-layer donor attention weights during a patched
# forward pass. Keyed by the integer ``layer_idx`` that every modern HF
# attention module carries.
_donor_state = threading.local()


def _get_donor_weights(layer_idx: int) -> torch.Tensor:
    mapping = getattr(_donor_state, "donor_per_layer", None)
    if mapping is None or layer_idx not in mapping:
        raise RuntimeError(
            f"No donor attention registered for layer_idx={layer_idx}; "
            "patched forward must run inside patched_attention(...) context."
        )
    return mapping[layer_idx]


# --- Key name for our custom attention function in ALL_ATTENTION_FUNCTIONS ---
_IMPL_KEY = "moe_attn_patching_donor"


def _donor_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Custom attention function that substitutes donor post-softmax weights.

    Signature matches ``eager_attention_forward`` across supported
    architectures (GPTNeoX, Llama, Qwen2, ...).  Returns ``(attn_output,
    attn_weights)`` where ``attn_output`` has shape ``[B, T, H_q, head_dim]``
    (needs transpose(1,2) before ``.contiguous()``, matching the HF Llama
    impl) -- more precisely the HF impl returns attn_output already
    transposed, so we mirror that.
    """
    donor = _get_donor_weights(module.layer_idx)  # [B, H_q, T, T]

    # Match dtype/device of query (the forward expects attn_weights in the
    # same dtype for the matmul with value; HF casts softmax to fp32 then
    # back to query.dtype, we just cast donor directly).
    donor = donor.to(device=value.device, dtype=value.dtype)

    # Sanity: head count must match query heads.
    if donor.shape[1] != query.shape[1]:
        raise RuntimeError(
            f"Donor attention has {donor.shape[1]} heads but query has "
            f"{query.shape[1]}. Donor must come from the same model/forward."
        )
    if donor.shape[-1] != key.shape[-2]:
        raise RuntimeError(
            f"Donor attention seq_len {donor.shape[-1]} vs. current key "
            f"seq_len {key.shape[-2]}. They must match exactly."
        )

    # For GQA: replicate value across KV-head groups so that H_kv broadcasts
    # to H_q, matching what eager_attention_forward does with repeat_kv.
    num_key_value_groups = getattr(module, "num_key_value_groups", 1)
    if num_key_value_groups and num_key_value_groups > 1:
        # value: [B, H_kv, T, D] -> [B, H_q, T, D]
        from transformers.models.llama.modeling_llama import repeat_kv
        value_states = repeat_kv(value, num_key_value_groups)
    else:
        value_states = value

    attn_weights = donor  # use donor weights directly (post-softmax, causal)
    attn_output = torch.matmul(attn_weights, value_states)  # [B, H_q, T, D]

    # HF convention: the per-impl function returns attn_output in
    # [B, T, H_q, D] (i.e. transposed) so the outer code can reshape to
    # [B, T, H_q*D]. Verified in modeling_gpt_neox.eager_attention_forward
    # and modeling_llama.eager_attention_forward.
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# Register our function once at import time; harmless if re-registered.
ALL_ATTENTION_FUNCTIONS.register(_IMPL_KEY, _donor_attention_forward)


def _iter_attention_modules(model: nn.Module):
    """Yield every attention submodule that carries a ``layer_idx`` attribute."""
    for module in model.modules():
        if hasattr(module, "layer_idx") and isinstance(module.layer_idx, int):
            # Heuristic: HF attention modules set layer_idx and have a
            # ``config`` attribute pointing at the top-level config.
            if getattr(module, "config", None) is not None:
                yield module


def _set_attn_impl(model: nn.Module, impl: str) -> List[tuple]:
    """Set ``_attn_implementation`` on the top-level config and on every
    attention submodule's config (HF sometimes replicates the attribute).

    Returns a list of ``(cfg_obj, had_attr, prev_value)`` tuples so the
    caller can restore them. HF config objects are not hashable, so a list
    is used instead of a dict.
    """
    touched: List[tuple] = []
    seen_ids = set()

    def _store(cfg_obj):
        if id(cfg_obj) in seen_ids:
            return
        seen_ids.add(id(cfg_obj))
        had_attr = hasattr(cfg_obj, "_attn_implementation")
        prev = getattr(cfg_obj, "_attn_implementation", None)
        touched.append((cfg_obj, had_attr, prev))
        setattr(cfg_obj, "_attn_implementation", impl)

    if hasattr(model, "config"):
        _store(model.config)

    for attn_module in _iter_attention_modules(model):
        if getattr(attn_module, "config", None) is not None:
            _store(attn_module.config)
    return touched


def _restore_attn_impl(touched: List[tuple]) -> None:
    for cfg_obj, had_attr, prev in touched:
        if not had_attr:
            try:
                delattr(cfg_obj, "_attn_implementation")
            except AttributeError:
                pass
        else:
            setattr(cfg_obj, "_attn_implementation", prev)


@contextlib.contextmanager
def patched_attention(
    model: nn.Module,
    donor_per_layer: Dict[int, torch.Tensor],
):
    """Context manager that runs the model with the given donor attention
    weights substituted at every layer.

    ``donor_per_layer`` maps ``layer_idx -> tensor`` of shape
    ``[B, H_q, T, T]`` (post-softmax, causal, already normalized).

    Usage:
        with patched_attention(model, donor):
            out = model(input_ids=..., attention_mask=...)
    """
    prev_mapping = getattr(_donor_state, "donor_per_layer", None)
    _donor_state.donor_per_layer = donor_per_layer
    touched = _set_attn_impl(model, _IMPL_KEY)
    try:
        yield
    finally:
        _restore_attn_impl(touched)
        _donor_state.donor_per_layer = prev_mapping


def capture_donor_attentions(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """Run a normal forward pass with ``output_attentions=True`` and return a
    ``layer_idx -> attention_weights`` mapping.

    The returned tensors are detached and moved to the same device as the
    model (so that the subsequent patched forward can matmul them with
    the value projections on-device without a PCIe round-trip).
    """
    # Force eager attention so that attention weights are actually returned.
    touched = _set_attn_impl(model, "eager")
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        _restore_attn_impl(touched)

    # outputs.attentions is a tuple of length num_layers. We assume attention
    # modules are indexed 0..L-1 by layer_idx (true for every HF decoder-only
    # LM in v5).
    attns = outputs.attentions
    if attns is None:
        raise RuntimeError(
            "Model did not return attentions; ensure eager attention is used "
            "and the architecture supports output_attentions."
        )
    return {i: a.detach() for i, a in enumerate(attns)}


def baseline_forward_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run an unmodified forward pass and return softmax probabilities
    ``[B, T, V]`` in float32 on CPU. Uses eager attention for identical
    numerical behaviour to the patched path (which also uses eager-style
    matmul for attention)."""
    touched = _set_attn_impl(model, "eager")
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        _restore_attn_impl(touched)
    logits = outputs.logits.float()
    return torch.softmax(logits, dim=-1).cpu()


def patched_forward_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    donor_per_layer: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Run a forward on ``input_ids`` with donor attention weights substituted
    at every layer; returns softmax probabilities ``[B, T, V]`` on CPU."""
    with patched_attention(model, donor_per_layer):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    logits = outputs.logits.float()
    return torch.softmax(logits, dim=-1).cpu()


def permute_attentions_causal(
    attn_per_layer: Dict[int, torch.Tensor],
    generator: Optional[torch.Generator] = None,
) -> Dict[int, torch.Tensor]:
    """Return a copy of ``attn_per_layer`` where, for every layer/head/query
    position ``i``, the weights across keys ``j in [0, i]`` are randomly
    permuted among themselves. Keys ``j > i`` remain zero (causal mask is
    preserved), and row-sums stay at 1.

    Input tensor shape per layer: ``[B, H, T, T]`` with
    ``sum_j attn[..., i, j] == 1`` and ``attn[..., i, j] == 0 for j > i``.

    The permutation is independent per ``(layer, batch, head, row)``. Row 0
    has a single valid entry (``j=0``) and is therefore invariant under
    permutation.

    This function is the ``permutation`` control described in the experiment
    spec: it destroys the positional/content structure of attention while
    preserving (a) the multiset of weights in every row, (b) the causal mask
    and (c) row-stochasticity. If the model is equally robust to
    ``permutation`` as to a donor text, the donor result cannot be attributed
    to texts sharing similar attention patterns.
    """
    out: Dict[int, torch.Tensor] = {}
    for layer_idx, attn in attn_per_layer.items():
        # attn: [B, H, T, T] (already on model device)
        B, H, T, _ = attn.shape
        # Build a (T,T) matrix of uniform noise in the causal prefix and
        # -inf elsewhere; argsort along the key axis gives an independent
        # random permutation of the valid keys per row. Broadcast to
        # [B, H, T, T] with independent draws per (B, H, row).
        noise = torch.rand(
            (B, H, T, T),
            device=attn.device,
            dtype=torch.float32,
            generator=generator,
        )
        causal_mask = torch.tril(
            torch.ones(T, T, device=attn.device, dtype=torch.bool)
        )
        noise = noise.masked_fill(~causal_mask, float("inf"))
        # argsort of the masked noise gives, for each row i, a permutation
        # of [0..T-1] whose first (i+1) entries are a random permutation of
        # the causal-valid keys {0..i}, and whose remaining entries index
        # into keys {i+1..T-1}. Gathering ``attn`` along the last axis with
        # this permutation therefore (a) randomly reshuffles the weights on
        # valid keys and (b) pulls zeros (from non-causal keys) into the
        # non-causal positions of the shuffled row. Causal mask and
        # row-sum=1 are both preserved exactly.
        perm = torch.argsort(noise, dim=-1)  # [B, H, T, T] long
        shuffled = torch.gather(attn, dim=-1, index=perm)
        shuffled = shuffled.masked_fill(~causal_mask, 0.0)
        row_sums = shuffled.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        shuffled = shuffled / row_sums
        out[layer_idx] = shuffled.to(dtype=attn.dtype)
        del noise, perm, shuffled, causal_mask, row_sums
    return out


def supports_patching(model: nn.Module) -> bool:
    """Lightweight check: do we see attention modules with integer
    ``layer_idx`` attributes? If not, the patching machinery cannot route
    donor weights correctly."""
    found = 0
    for _ in _iter_attention_modules(model):
        found += 1
        if found > 0:
            return True
    return False
