import contextlib
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


DTypeArg = Literal["auto", "float16", "bfloat16", "float32"]
TruncateStrategy = Literal["min", "a", "b"]


def select_dtype(dtype: DTypeArg) -> torch.dtype:
    if dtype == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def load_model_and_tokenizer(model_name: str, dtype: DTypeArg = "auto", device_map: str = "auto"):

    if "llama" in model_name.lower():
        login(token="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")
    else:
        login(token="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")

    torch_dtype = select_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model


def get_input_embeddings(model: torch.nn.Module) -> torch.nn.Module:
    return model.get_input_embeddings()


@contextlib.contextmanager
def temporarily_scale_rmsnorm_weights(model: torch.nn.Module, scale: float):
    modules = []
    original = []
    for m in model.modules():
        class_name = m.__class__.__name__.lower()
        if "rmsnorm" in class_name and hasattr(m, "weight") and isinstance(m.weight, torch.nn.Parameter):
            modules.append(m)
            original.append(m.weight.data.clone())
    try:
        for m in modules:
            m.weight.data.mul_(scale)
        yield
    finally:
        for m, w in zip(modules, original):
            m.weight.data.copy_(w)


def _get_nested_attr(obj, path: Tuple[str, ...]) -> Optional[torch.nn.Module]:
    current = obj
    for name in path:
        if current is None:
            return None
        current = getattr(current, name, None)
    return current


def _find_pre_norm_module(model: torch.nn.Module) -> torch.nn.Module:
    cached = getattr(model, "_pre_norm_module", None)
    if cached is not None:
        return cached

    candidate_paths = (
        ("model", "norm"),
        ("model", "final_layer_norm"),
        ("model", "final_layernorm"),
        ("model", "final_norm"),
        ("model", "norm_f"),
        ("transformer", "final_layer_norm"),
        ("transformer", "final_layernorm"),
        ("transformer", "norm"),
    )
    for path in candidate_paths:
        module = _get_nested_attr(model, path)
        if isinstance(module, torch.nn.Module):
            setattr(model, "_pre_norm_module", module)
            return module

    found: Optional[torch.nn.Module] = None
    last_norm: Optional[torch.nn.Module] = None
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if ("rmsnorm" in class_name) or ("layernorm" in class_name) or class_name.endswith("norm"):
            last_norm = module
        if "final_layer_norm" in name.lower() or "final_layernorm" in name.lower():
            found = module

    target = found or last_norm
    if target is None:
        raise RuntimeError("Could not locate the final normalization module on the model.")

    setattr(model, "_pre_norm_module", target)
    return target


def _run_with_pre_norm_capture(
    model: torch.nn.Module,
    forward_fn: Callable[[], object],
    apply_rmsnorm_scaling: bool,
) -> Tuple[object, torch.Tensor]:
    norm_module = _find_pre_norm_module(model)
    captured = {}

    def hook(module, inputs):
        if inputs:
            captured["pre_norm"] = inputs[0].detach()

    handle = norm_module.register_forward_pre_hook(hook)
    try:
        if apply_rmsnorm_scaling:
            with temporarily_scale_rmsnorm_weights(model, scale=1):
                outputs = forward_fn()
        else:
            outputs = forward_fn()
    finally:
        handle.remove()

    if "pre_norm" not in captured:
        raise RuntimeError("Failed to capture pre-norm hidden states from the model forward pass.")

    return outputs, captured["pre_norm"]


def _run_with_post_norm_capture(
    model: torch.nn.Module,
    forward_fn: Callable[[], object],
) -> Tuple[object, torch.Tensor]:
    """
    Capture the output of the final normalization module (post-norm hidden states).
    """
    norm_module = _find_pre_norm_module(model)
    captured = {}

    def hook(module, inputs, output):
        if output is not None:
            captured["post_norm"] = output.detach()

    handle = norm_module.register_forward_hook(hook)
    try:
        outputs = forward_fn()
    finally:
        handle.remove()

    if "post_norm" not in captured:
        raise RuntimeError("Failed to capture post-norm hidden states from the model forward pass.")

    return outputs, captured["post_norm"]


@dataclass
class TokenizedPair:
    input_ids_a: torch.Tensor
    input_ids_b: torch.Tensor
    attn_mask_a: torch.Tensor
    attn_mask_b: torch.Tensor
    T: int  # aligned length


def tokenize_and_align_pair(
    tokenizer: AutoTokenizer,
    text_a: str,
    text_b: str,
    truncate: TruncateStrategy = "min",
    max_length: Optional[int] = None,
) -> TokenizedPair:
    enc_a = tokenizer(text_a, return_tensors="pt")
    enc_b = tokenizer(text_b, return_tensors="pt")
    ids_a = enc_a["input_ids"]
    ids_b = enc_b["input_ids"]
    mask_a = enc_a["attention_mask"]
    mask_b = enc_b["attention_mask"]

    len_a = ids_a.shape[1]
    len_b = ids_b.shape[1]

    if truncate == "min":
        T = min(len_a, len_b)
        ids_a = ids_a[:, :T]
        ids_b = ids_b[:, :T]
        mask_a = mask_a[:, :T]
        mask_b = mask_b[:, :T]
    elif truncate == "a":
        T = len_a
        ids_b = ids_b[:, :T]
        mask_b = mask_b[:, :T]
    elif truncate == "b":
        T = len_b
        ids_a = ids_a[:, :T]
        mask_a = mask_a[:, :T]
    else:
        raise ValueError(f"Unknown truncate strategy: {truncate}")

    if max_length is not None:
        T = min(T, max_length)
        ids_a = ids_a[:, :T]
        ids_b = ids_b[:, :T]
        mask_a = mask_a[:, :T]
        mask_b = mask_b[:, :T]

    return TokenizedPair(ids_a, ids_b, mask_a, mask_b, T)


def get_probs_from_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    return_pre_norm: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    with torch.no_grad():
        if return_pre_norm:
            outputs, pre_norm = _run_with_pre_norm_capture(
                model,
                lambda: model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False),
                apply_rmsnorm_scaling=False,
            )
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            pre_norm = None
        logits = outputs.logits  # [B, T, V]
        probs = torch.softmax(logits, dim=-1)
        return probs, pre_norm


def get_probs_from_embeds(
    model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    sqrt2_rmsnorm: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    with torch.no_grad():
        forward = lambda: model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        if sqrt2_rmsnorm:
            outputs, pre_norm = _run_with_pre_norm_capture(
                model,
                forward,
                apply_rmsnorm_scaling=True,
            )
        else:
            outputs, pre_norm = _run_with_pre_norm_capture(
                model,
                forward,
                apply_rmsnorm_scaling=False,
            )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        return probs, pre_norm


def get_post_norm_from_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        _, post_norm = _run_with_post_norm_capture(
            model,
            lambda: model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False),
        )
        return post_norm


def get_post_norm_from_embeds(
    model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        _, post_norm = _run_with_post_norm_capture(
            model,
            lambda: model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False),
        )
        return post_norm


def build_combined_inputs_embeds(
    model: torch.nn.Module,
    input_ids_a: torch.Tensor,
    input_ids_b: torch.Tensor,
) -> torch.Tensor:
    emb = get_input_embeddings(model)
    emb_a = emb(input_ids_a)
    emb_b = emb(input_ids_b)
    return emb_a + emb_b


def to_device_batch(*tensors: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in tensors)


# --- Layerwise residual-stream capture utilities ---

def find_transformer_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Try to locate the list of transformer blocks across common architectures (LLaMA/Gemma/Qwen/GPTNeoX/etc.).
    Returns a list of modules representing the sequential blocks.
    """
    candidate_paths: Tuple[Tuple[str, ...], ...] = (
        ("model", "layers"),                 # LLaMA, Gemma, Qwen
        ("transformer", "layers"),           # Some HF models
        ("transformer", "h"),                # GPT-2 style
        ("gpt_neox", "layers"),              # GPTNeoX/Pythia
        ("model", "decoder", "layers"),     # Some seq2seq decoders
    )

    for path in candidate_paths:
        mod = _get_nested_attr(model, path)
        if isinstance(mod, (list, torch.nn.ModuleList)) and len(mod) > 0:
            return list(mod)

    # Fallback: scan for the first non-empty ModuleList that looks like repeated blocks
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            return list(module)

    raise RuntimeError("Could not locate transformer block list on the model.")


def layer_indices_for_percentages(num_layers: int, percents: List[int]) -> List[int]:
    import math
    indices: List[int] = []
    seen: set = set()
    for p in percents:
        # Map 10..100% to indices 0..num_layers-1
        idx = max(0, min(num_layers - 1, math.ceil(p / 100.0 * num_layers) - 1))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return sorted(indices)


def _run_with_layer_input_capture(
    model: torch.nn.Module,
    forward_fn: Callable[[], object],
    layer_indices: List[int],
) -> Tuple[object, Dict[int, torch.Tensor]]:
    """
    Register forward_pre_hooks on selected transformer blocks to capture their input hidden states
    (residual stream entering the block). Returns model outputs and a dict index->hidden_states.
    """
    blocks = find_transformer_blocks(model)
    captures: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, inputs):
            if inputs:
                captures[layer_idx] = inputs[0].detach()
        return hook

    for idx in layer_indices:
        if 0 <= idx < len(blocks):
            handles.append(blocks[idx].register_forward_pre_hook(make_hook(idx)))

    try:
        outputs = forward_fn()
    finally:
        for h in handles:
            h.remove()

    # Basic validation to ensure we captured something for each requested layer
    for idx in layer_indices:
        if idx not in captures:
            raise RuntimeError(f"Failed to capture input for layer index {idx}.")

    return outputs, captures


def _run_with_layer_input_override(
    model: torch.nn.Module,
    forward_fn: Callable[[], object],
    layer_index: int,
    override_hidden: torch.Tensor,
    capture_pre_norm: bool = True,
) -> Tuple[object, Optional[torch.Tensor]]:
    """
    Register a forward_pre_hook on the specified transformer block to replace its first
    positional argument (hidden states) with `override_hidden`. Optionally capture pre-norm hidden states.
    """
    blocks = find_transformer_blocks(model)
    if not (0 <= layer_index < len(blocks)):
        raise IndexError(f"Layer index {layer_index} out of range (0..{len(blocks)-1}).")

    captured_pre_norm: Dict[str, torch.Tensor] = {}
    pre_norm_handle = None
    if capture_pre_norm:
        norm_module = _find_pre_norm_module(model)

        def pre_norm_hook(module, inputs):
            if inputs:
                captured_pre_norm["pre_norm"] = inputs[0].detach()

        pre_norm_handle = norm_module.register_forward_pre_hook(pre_norm_hook)

    def override_hook(module, inputs):
        if not inputs:
            return inputs
        # Replace the first positional arg (hidden states) and keep the rest unchanged
        new_inputs = (override_hidden,) + tuple(inputs[1:])
        return new_inputs

    layer_handle = blocks[layer_index].register_forward_pre_hook(override_hook)
    try:
        outputs = forward_fn()
    finally:
        layer_handle.remove()
        if pre_norm_handle is not None:
            pre_norm_handle.remove()

    pre_norm = captured_pre_norm.get("pre_norm") if capture_pre_norm else None
    return outputs, pre_norm


def get_post_norm_with_layer_override(
    model: torch.nn.Module,
    base_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_index: int,
    override_hidden: torch.Tensor,
) -> torch.Tensor:
    blocks = find_transformer_blocks(model)
    if not (0 <= layer_index < len(blocks)):
        raise IndexError(f"Layer index {layer_index} out of range (0..{len(blocks)-1}).")

    captured = {}

    def norm_hook(module, inputs, output):
        if output is not None:
            captured["post_norm"] = output.detach()

    norm_module = _find_pre_norm_module(model)
    norm_handle = norm_module.register_forward_hook(norm_hook)

    def override_hook(module, inputs):
        if not inputs:
            return inputs
        return (override_hidden,) + tuple(inputs[1:])

    override_handle = blocks[layer_index].register_forward_pre_hook(override_hook)
    try:
        _ = model(input_ids=base_input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        override_handle.remove()
        norm_handle.remove()

    if "post_norm" not in captured:
        raise RuntimeError("Failed to capture post-norm with layer override.")
    return captured["post_norm"]


def get_logits_with_layer_override(
    model: torch.nn.Module,
    base_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_index: int,
    override_hidden: torch.Tensor,
) -> torch.Tensor:
    blocks = find_transformer_blocks(model)
    if not (0 <= layer_index < len(blocks)):
        raise IndexError(f"Layer index {layer_index} out of range (0..{len(blocks)-1}).")

    def override_hook(module, inputs):
        if not inputs:
            return inputs
        return (override_hidden,) + tuple(inputs[1:])

    handle = blocks[layer_index].register_forward_pre_hook(override_hook)
    try:
        outputs = model(input_ids=base_input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return outputs.logits


def _run_with_layer_override_and_capture_layers(
    model: torch.nn.Module,
    forward_fn: Callable[[], object],
    override_layer_index: int,
    override_hidden: torch.Tensor,
    capture_layer_indices: List[int],
    capture_pre_norm: bool = True,
) -> Tuple[object, Dict[int, torch.Tensor], Optional[torch.Tensor]]:
    """
    Combines an override at one layer with input-captures at a set of layers, plus optional pre-norm capture.
    """
    blocks = find_transformer_blocks(model)
    if not (0 <= override_layer_index < len(blocks)):
        raise IndexError(f"Layer index {override_layer_index} out of range (0..{len(blocks)-1}).")

    captured_pre_norm: Dict[str, torch.Tensor] = {}
    pre_norm_handle = None
    if capture_pre_norm:
        norm_module = _find_pre_norm_module(model)
        def pre_norm_hook(module, inputs):
            if inputs:
                captured_pre_norm["pre_norm"] = inputs[0].detach()
        pre_norm_handle = norm_module.register_forward_pre_hook(pre_norm_hook)

    def override_hook(module, inputs):
        if not inputs:
            return inputs
        return (override_hidden,) + tuple(inputs[1:])

    # Register OVERRIDE first so captures see overridden inputs at the target layer
    override_handle = blocks[override_layer_index].register_forward_pre_hook(override_hook)

    # Then register CAPTURE hooks
    captured_layers: Dict[int, torch.Tensor] = {}
    capture_handles: List[torch.utils.hooks.RemovableHandle] = []
    for idx in capture_layer_indices:
        if 0 <= idx < len(blocks):
            def make_hook(i):
                def hook(module, inputs):
                    if inputs:
                        captured_layers[i] = inputs[0].detach()
                return hook
            capture_handles.append(blocks[idx].register_forward_pre_hook(make_hook(idx)))
    try:
        outputs = forward_fn()
    finally:
        override_handle.remove()
        for h in capture_handles:
            h.remove()
        if pre_norm_handle is not None:
            pre_norm_handle.remove()

    pre_norm = captured_pre_norm.get("pre_norm") if capture_pre_norm else None
    return outputs, captured_layers, pre_norm


def get_probs_from_ids_with_layer_inputs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: List[int],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    with torch.no_grad():
        outputs, layer_inputs = _run_with_layer_input_capture(
            model,
            lambda: model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False),
            layer_indices=layer_indices,
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        return probs, layer_inputs


def get_probs_from_embeds_with_layer_inputs(
    model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: List[int],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    with torch.no_grad():
        outputs, layer_inputs = _run_with_layer_input_capture(
            model,
            lambda: model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False),
            layer_indices=layer_indices,
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        return probs, layer_inputs


def get_probs_with_layer_override(
    model: torch.nn.Module,
    base_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_index: int,
    override_hidden: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Run a forward pass where the input to a specific transformer block is overridden by `override_hidden`.
    Uses input_ids/attention_mask as a base to drive the forward through earlier layers.
    Returns (probs, pre_norm_hidden_states).
    """
    with torch.no_grad():
        outputs, pre_norm = _run_with_layer_input_override(
            model,
            lambda: model(input_ids=base_input_ids, attention_mask=attention_mask, use_cache=False),
            layer_index=layer_index,
            override_hidden=override_hidden,
            capture_pre_norm=True,
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        return probs, pre_norm


def get_probs_with_layer_override_and_layer_inputs(
    model: torch.nn.Module,
    base_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    override_layer_index: int,
    override_hidden: torch.Tensor,
    capture_layer_indices: List[int],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Optional[torch.Tensor]]:
    with torch.no_grad():
        outputs, captured_layers, pre_norm = _run_with_layer_override_and_capture_layers(
            model,
            lambda: model(input_ids=base_input_ids, attention_mask=attention_mask, use_cache=False),
            override_layer_index=override_layer_index,
            override_hidden=override_hidden,
            capture_layer_indices=capture_layer_indices,
            capture_pre_norm=True,
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        return probs, captured_layers, pre_norm


def get_logits_from_embeds(
    model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    sqrt2_rmsnorm: bool = True,
) -> torch.Tensor:
    """Get logits from input embeddings (useful for temperature scaling)."""
    with torch.no_grad():
        forward = lambda: model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        if sqrt2_rmsnorm:
            outputs, _ = _run_with_pre_norm_capture(
                model,
                forward,
                apply_rmsnorm_scaling=True,
            )
        else:
            outputs, _ = _run_with_pre_norm_capture(
                model,
                forward,
                apply_rmsnorm_scaling=False,
            )
        return outputs.logits


def find_all_norm_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Find all RMSNorm and LayerNorm modules in the model.
    Returns list of normalization modules.
    """
    norm_modules = []
    for m in model.modules():
        class_name = m.__class__.__name__.lower()
        # Match RMSNorm, LayerNorm, and similar normalization layers
        if ("rmsnorm" in class_name or "layernorm" in class_name or 
            class_name.endswith("norm")) and len(list(m.children())) == 0:
            # Only include leaf modules (not containers)
            norm_modules.append(m)
    return norm_modules


def capture_norm_layer_input_norms(
    model: torch.nn.Module,
    forward_fn: Callable[[], object],
    norm_modules: List[torch.nn.Module],
) -> Dict[torch.nn.Module, float]:
    """
    Capture mean L2 norms of inputs to each normalization layer during a forward pass.
    
    Args:
        model: The model
        forward_fn: Function that performs forward pass
        norm_modules: List of normalization modules to monitor
        
    Returns:
        Dictionary mapping each norm module to mean L2 norm of its inputs
    """
    captured_norms: Dict[torch.nn.Module, List[float]] = {}
    handles = []
    
    def make_hook(norm_module):
        def hook(module, inputs):
            if inputs and len(inputs) > 0:
                input_tensor = inputs[0] if isinstance(inputs[0], torch.Tensor) else inputs[0][0]
                # Compute L2 norm for each position/token, then average
                # input_tensor shape: [B, T, H] or [B, T]
                if input_tensor.dim() >= 2:
                    # Compute norm across hidden dimension
                    norms = torch.norm(input_tensor, dim=-1)  # [B, T] or [B, T, ...]
                    mean_norm = float(norms.mean().detach().cpu().item())
                    if norm_module not in captured_norms:
                        captured_norms[norm_module] = []
                    captured_norms[norm_module].append(mean_norm)
        return hook
    
    # Register forward_pre_hook to capture inputs before normalization
    for norm_module in norm_modules:
        handle = norm_module.register_forward_pre_hook(make_hook(norm_module))
        handles.append(handle)
    
    try:
        with torch.no_grad():
            forward_fn()
    finally:
        for handle in handles:
            handle.remove()
    
    # Compute mean across all captured values for each module
    mean_norms: Dict[torch.nn.Module, float] = {}
    for norm_module, norms_list in captured_norms.items():
        if norms_list:
            mean_norms[norm_module] = sum(norms_list) / len(norms_list)
        else:
            mean_norms[norm_module] = 1.0  # Default if nothing captured
    
    return mean_norms


def compute_adaptive_norm_scale_factors(
    model: torch.nn.Module,
    input_ids_a: torch.Tensor,
    input_ids_b: torch.Tensor,
    attn_mask_a: torch.Tensor,
    attn_mask_b: torch.Tensor,
    inputs_embeds_mix: Optional[torch.Tensor],
    attn_mask_mix: torch.Tensor,
    norm_modules: List[torch.nn.Module],
    layer_override_index: Optional[int] = None,
    layer_override_hidden: Optional[torch.Tensor] = None,
) -> Dict[torch.nn.Module, float]:
    """
    Compute adaptive scale factors for normalization layers based on activation norm differences.
    
    Args:
        model: The model
        input_ids_a: Input IDs for sentence A
        input_ids_b: Input IDs for sentence B
        attn_mask_a: Attention mask for A
        attn_mask_b: Attention mask for B
        inputs_embeds_mix: Mixed input embeddings (if mixing at input level, None for layer-level)
        attn_mask_mix: Attention mask for mixed pass
        norm_modules: List of normalization modules
        layer_override_index: Layer index to override (for layer-level mixing)
        layer_override_hidden: Hidden state to use at override layer (for layer-level mixing)
        
    Returns:
        Dictionary mapping each norm module to its scale factor (y/x)
    """
    # Capture norms from original passes
    def forward_a():
        return model(input_ids=input_ids_a, attention_mask=attn_mask_a, use_cache=False)
    
    def forward_b():
        return model(input_ids=input_ids_b, attention_mask=attn_mask_b, use_cache=False)
    
    def forward_mix():
        if inputs_embeds_mix is not None:
            # Input-level mixing
            return model(inputs_embeds=inputs_embeds_mix, attention_mask=attn_mask_mix, use_cache=False)
        elif layer_override_index is not None and layer_override_hidden is not None:
            # Layer-level mixing - use layer override
            blocks = find_transformer_blocks(model)
            def override_hook(module, inputs):
                if not inputs:
                    return inputs
                return (layer_override_hidden,) + tuple(inputs[1:])
            handle = blocks[layer_override_index].register_forward_pre_hook(override_hook)
            try:
                outputs = model(input_ids=input_ids_a, attention_mask=attn_mask_mix, use_cache=False)
                return outputs
            finally:
                handle.remove()
        else:
            raise ValueError("Either inputs_embeds_mix or (layer_override_index, layer_override_hidden) must be provided")
    
    norms_a = capture_norm_layer_input_norms(model, forward_a, norm_modules)
    norms_b = capture_norm_layer_input_norms(model, forward_b, norm_modules)
    norms_mix = capture_norm_layer_input_norms(model, forward_mix, norm_modules)
    
    # Compute scale factors: scale_factor = y / x where x = (norm_a + norm_b) / 2
    scale_factors: Dict[torch.nn.Module, float] = {}
    eps = 1e-8
    for norm_module in norm_modules:
        x = (norms_a[norm_module] + norms_b[norm_module]) / 2.0
        y = norms_mix[norm_module]
        scale_factor = y / max(x, eps)  # Avoid division by zero
        scale_factors[norm_module] = float(scale_factor)
    
    return scale_factors


@contextlib.contextmanager
def temporarily_scale_norm_outputs_adaptively(
    norm_modules: List[torch.nn.Module],
    scale_factors: Dict[torch.nn.Module, float],
):
    """
    Context manager that multiplies normalization layer outputs by adaptive scale factors.
    
    Args:
        norm_modules: List of normalization modules
        scale_factors: Dictionary mapping each norm module to its scale factor
    """
    handles = []
    
    def make_hook(norm_module, scale_factor):
        def hook(module, inputs, output):
            return output * scale_factor
        return hook
    
    # Register forward hooks to scale outputs
    for norm_module in norm_modules:
        scale_factor = scale_factors.get(norm_module, 1.0)
        if scale_factor != 1.0:
            handle = norm_module.register_forward_hook(make_hook(norm_module, scale_factor))
            handles.append(handle)
    
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
