import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire
import os

from model import DualStreamLLM


def load_checkpoint(checkpoint_path, model_name, device):
    """Load trained DualStreamLLM from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with same configuration
    model = DualStreamLLM(
        model_name=model_name,
        freeze_backbone=True,
        optimize_memory=True,  # Match training config to avoid lm_head issues
        embedding_init_strategy='copy',
        objective='distillation',
        shared_embeddings=True
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (step {checkpoint.get('step', 'unknown')})")
    return model


@torch.no_grad()
def get_dual_logits(
    dual_model,
    separator_model,
    tokens_a,
    tokens_b,
    tokenizer,
    top_k_mask=30
):
    """
    Get filtered logits for both streams using the separator model.
    
    Args:
        dual_model: Trained DualStreamLLM
        separator_model: Smaller model for creating masks
        tokens_a: Token IDs for stream A [batch, seq_len]
        tokens_b: Token IDs for stream B [batch, seq_len]
        tokenizer: Tokenizer
        top_k_mask: Number of top tokens to keep in mask
        
    Returns:
        filtered_logits_a: Filtered logits for stream A [batch, vocab_size]
        filtered_logits_b: Filtered logits for stream B [batch, vocab_size]
    """
    # Prepare inputs for dual-stream model
    max_len = max(tokens_a.shape[1], tokens_b.shape[1])
    
    # Pad sequences to same length
    input_ids_a = F.pad(tokens_a, (0, max_len - tokens_a.shape[1]), value=tokenizer.pad_token_id)
    input_ids_b = F.pad(tokens_b, (0, max_len - tokens_b.shape[1]), value=tokenizer.pad_token_id)
    
    attention_mask_a = (input_ids_a != tokenizer.pad_token_id).long()
    attention_mask_b = (input_ids_b != tokenizer.pad_token_id).long()
    
    # Forward through dual-stream model to get mixed logits
    mixed_logits = dual_model(
        input_ids_A=input_ids_a,
        attention_mask_A=attention_mask_a,
        input_ids_B=input_ids_b,
        attention_mask_B=attention_mask_b
    )  # Shape: [batch, seq_len, vocab_size]
    
    # Get logits for next token (last position)
    mixed_logits_next = mixed_logits[:, -1, :]  # [batch, vocab_size]
    
    # Get logits from separator model for text A
    sep_outputs_a = separator_model(tokens_a)
    sep_logits_a = sep_outputs_a.logits[:, -1, :]  # [batch, vocab_size]
    
    # Get logits from separator model for text B
    sep_outputs_b = separator_model(tokens_b)
    sep_logits_b = sep_outputs_b.logits[:, -1, :]  # [batch, vocab_size]
    
    # Create masks based on top-K tokens from separator model
    topk_a = torch.topk(sep_logits_a, k=top_k_mask, dim=-1)
    mask_a = torch.zeros_like(mixed_logits_next)
    mask_a.scatter_(1, topk_a.indices, 1.0)
    
    topk_b = torch.topk(sep_logits_b, k=top_k_mask, dim=-1)
    mask_b = torch.zeros_like(mixed_logits_next)
    mask_b.scatter_(1, topk_b.indices, 1.0)
    
    # Apply masks to mixed logits
    filtered_logits_a = mixed_logits_next.clone()
    filtered_logits_a[mask_a == 0] = -float('inf')
    
    filtered_logits_b = mixed_logits_next.clone()
    filtered_logits_b[mask_b == 0] = -float('inf')
    
    return filtered_logits_a, filtered_logits_b


@torch.no_grad()
def get_dual_logits_linear(
    dual_model,
    separator_model,
    tokens_a,
    tokens_b,
    tokenizer,
    alpha=1.0,
    beta=1.0
):
    """
    Get logits for both streams using linear combination.
    
    Formula:
        logits_A = logits_mixed + alpha * logits_small(A) - beta * logits_small(B)
        logits_B = logits_mixed + alpha * logits_small(B) - beta * logits_small(A)
    
    Args:
        dual_model: Trained DualStreamLLM
        separator_model: Smaller model for providing separation signal
        tokens_a: Token IDs for stream A [batch, seq_len]
        tokens_b: Token IDs for stream B [batch, seq_len]
        tokenizer: Tokenizer
        alpha: Weight for own stream logits
        beta: Weight for other stream logits (subtracted)
        
    Returns:
        logits_a: Combined logits for stream A [batch, vocab_size]
        logits_b: Combined logits for stream B [batch, vocab_size]
    """
    # Prepare inputs for dual-stream model
    max_len = max(tokens_a.shape[1], tokens_b.shape[1])
    
    # Pad sequences to same length
    input_ids_a = F.pad(tokens_a, (0, max_len - tokens_a.shape[1]), value=tokenizer.pad_token_id)
    input_ids_b = F.pad(tokens_b, (0, max_len - tokens_b.shape[1]), value=tokenizer.pad_token_id)
    
    attention_mask_a = (input_ids_a != tokenizer.pad_token_id).long()
    attention_mask_b = (input_ids_b != tokenizer.pad_token_id).long()
    
    # Forward through dual-stream model to get mixed logits
    mixed_logits = dual_model(
        input_ids_A=input_ids_a,
        attention_mask_A=attention_mask_a,
        input_ids_B=input_ids_b,
        attention_mask_B=attention_mask_b
    )  # Shape: [batch, seq_len, vocab_size]
    
    # Get logits for next token (last position)
    mixed_logits_next = mixed_logits[:, -1, :]  # [batch, vocab_size]
    
    # Get logits from separator model for text A
    sep_outputs_a = separator_model(tokens_a)
    sep_logits_a = sep_outputs_a.logits[:, -1, :]  # [batch, vocab_size]
    
    # Get logits from separator model for text B
    sep_outputs_b = separator_model(tokens_b)
    sep_logits_b = sep_outputs_b.logits[:, -1, :]  # [batch, vocab_size]
    
    # Apply linear combination
    # logits_A = logits_mixed + alpha * logits_small(A) - beta * logits_small(B)
    logits_a = mixed_logits_next + alpha * sep_logits_a - beta * sep_logits_b
    
    # logits_B = logits_mixed + alpha * logits_small(B) - beta * logits_small(A)
    logits_b = mixed_logits_next + alpha * sep_logits_b - beta * sep_logits_a
    
    return logits_a, logits_b


@torch.no_grad()
def generate_dual(
    dual_model,
    separator_model,
    tokens_a,
    tokens_b,
    tokenizer,
    max_new_tokens=1,
    temperature=0.7,
    top_k_mask=30
):
    """
    Generate multiple tokens for dual streams.
    """
    for _ in range(max_new_tokens):
        filtered_logits_a, filtered_logits_b = get_dual_logits(
            dual_model,
            separator_model,
            tokens_a,
            tokens_b,
            tokenizer,
            top_k_mask=top_k_mask
        )
        
        # Apply temperature and sample
        probs_a = F.softmax(filtered_logits_a / temperature, dim=-1)
        probs_b = F.softmax(filtered_logits_b / temperature, dim=-1)
        
        next_token_a = torch.multinomial(probs_a, num_samples=1)
        next_token_b = torch.multinomial(probs_b, num_samples=1)
        
        tokens_a = torch.cat([tokens_a, next_token_a], dim=1)
        tokens_b = torch.cat([tokens_b, next_token_b], dim=1)
        
    return tokens_a, tokens_b


@torch.no_grad()
def generate_dual_linear(
    dual_model,
    separator_model,
    tokens_a,
    tokens_b,
    tokenizer,
    max_new_tokens=1,
    temperature=0.7,
    alpha=1.0,
    beta=1.0
):
    """
    Generate multiple tokens for dual streams using linear combination.
    """
    for _ in range(max_new_tokens):
        logits_a, logits_b = get_dual_logits_linear(
            dual_model,
            separator_model,
            tokens_a,
            tokens_b,
            tokenizer,
            alpha=alpha,
            beta=beta
        )
        
        # Apply temperature and sample
        probs_a = F.softmax(logits_a / temperature, dim=-1)
        probs_b = F.softmax(logits_b / temperature, dim=-1)
        
        next_token_a = torch.multinomial(probs_a, num_samples=1)
        next_token_b = torch.multinomial(probs_b, num_samples=1)
        
        tokens_a = torch.cat([tokens_a, next_token_a], dim=1)
        tokens_b = torch.cat([tokens_b, next_token_b], dim=1)
        
    return tokens_a, tokens_b


def generate_with_separator(
    checkpoint_path,
    model_name='EleutherAI/pythia-2.8b',
    separator_model_name='EleutherAI/pythia-160m',
    prompt_a="Once upon a time",
    prompt_b="The history of computing",
    max_new_tokens=50,
    temperature=0.7,
    top_k_mask=30,
    device='cuda',
    seed=42
):
    """
    Generate text using dual-stream model with separator model for deconvolution.
    
    Args:
        checkpoint_path: Path to trained DualStreamLLM checkpoint
        model_name: Name of the base model used for training
        separator_model_name: Name of smaller model for creating masks
        prompt_a: Starting prompt for text A
        prompt_b: Starting prompt for text B
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k_mask: Number of top tokens to keep in mask
        device: Device to use
        seed: Random seed
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load trained dual-stream model
    dual_model = load_checkpoint(checkpoint_path, model_name, device)
    
    # Load separator model
    print(f"\nLoading separator model: {separator_model_name}")
    separator_model = AutoModelForCausalLM.from_pretrained(
        separator_model_name,
        dtype=torch.bfloat16,
    ).to(device)
    separator_model.eval()
    print("Separator model loaded successfully")
    
    # Tokenize initial prompts
    tokens_a = tokenizer.encode(prompt_a, return_tensors='pt').to(device)
    tokens_b = tokenizer.encode(prompt_b, return_tensors='pt').to(device)
    
    len_a = tokens_a.shape[1]
    len_b = tokens_b.shape[1]
    
    if len_a != len_b:
        min_len = min(len_a, len_b)
        print(f"\n[INFO] Prompts have different lengths (A: {len_a}, B: {len_b}). Truncating to {min_len} tokens for balance.")
        
        if len_a > min_len:
            truncated_tokens = tokens_a[:, :min_len]
            truncated_text = tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
            print(f"Truncated Prompt A: \"{prompt_a}\" -> \"{truncated_text}\"")
            tokens_a = truncated_tokens
            
        if len_b > min_len:
            truncated_tokens = tokens_b[:, :min_len]
            truncated_text = tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
            print(f"Truncated Prompt B: \"{prompt_b}\" -> \"{truncated_text}\"")
            tokens_b = truncated_tokens

    print(f"\n{'='*80}")
    print(f"GENERATION WITH SEPARATOR MODEL")
    print(f"{'='*80}")
    print(f"Prompt A: {prompt_a}")
    print(f"Prompt B: {prompt_b}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Top-K mask: {top_k_mask}")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        tokens_a, tokens_b = generate_dual(
            dual_model,
            separator_model,
            tokens_a,
            tokens_b,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k_mask=top_k_mask
        )
    
    # Decode results with coloring
    BLUE = "\033[94m"
    RESET = "\033[0m"
    
    def get_colored_text(full_tokens, prompt_len):
        # Decode the full sequence
        full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
        # Decode only the prompt part
        prompt_text = tokenizer.decode(full_tokens[:prompt_len], skip_special_tokens=True)
        
        # Find where prompt ends and generation begins in the full text string
        if full_text.startswith(prompt_text):
            gen_text = full_text[len(prompt_text):]
            return f"{BLUE}{prompt_text}{RESET}{gen_text}", full_text
        else:
            # Fallback for weird edge cases in decoding
            gen_text = tokenizer.decode(full_tokens[prompt_len:], skip_special_tokens=True)
            return f"{BLUE}{prompt_text}{RESET} {gen_text}", full_text

    colored_a, generated_a = get_colored_text(tokens_a[0], len_a if 'min_len' not in locals() else min_len)
    colored_b, generated_b = get_colored_text(tokens_b[0], len_b if 'min_len' not in locals() else min_len)
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"\nText A:\n{colored_a}")
    print(f"\n{'-'*80}")
    print(f"\nText B:\n{colored_b}")
    print(f"\n{'='*80}\n")
    
    return generated_a, generated_b


def generate_with_linear_combination(
    checkpoint_path,
    model_name='EleutherAI/pythia-2.8b',
    separator_model_name='EleutherAI/pythia-160m',
    prompt_a="Once upon a time",
    prompt_b="The history of computing",
    max_new_tokens=50,
    temperature=0.7,
    alpha=1.0,
    beta=1.0,
    device='cuda',
    seed=42
):
    """
    Generate text using dual-stream model with linear combination deconvolution.
    
    Uses formula:
        logits_A = logits_mixed + alpha * logits_small(A) - beta * logits_small(B)
        logits_B = logits_mixed + alpha * logits_small(B) - beta * logits_small(A)
    
    Args:
        checkpoint_path: Path to trained DualStreamLLM checkpoint
        model_name: Name of the base model used for training
        separator_model_name: Name of smaller model for providing separation signal
        prompt_a: Starting prompt for text A
        prompt_b: Starting prompt for text B
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        alpha: Weight for own stream logits
        beta: Weight for other stream logits (subtracted)
        device: Device to use
        seed: Random seed
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load trained dual-stream model
    dual_model = load_checkpoint(checkpoint_path, model_name, device)
    
    # Load separator model
    print(f"\nLoading separator model: {separator_model_name}")
    separator_model = AutoModelForCausalLM.from_pretrained(
        separator_model_name,
        dtype=torch.bfloat16,
    ).to(device)
    separator_model.eval()
    print("Separator model loaded successfully")
    
    # Tokenize initial prompts
    tokens_a = tokenizer.encode(prompt_a, return_tensors='pt').to(device)
    tokens_b = tokenizer.encode(prompt_b, return_tensors='pt').to(device)
    
    len_a = tokens_a.shape[1]
    len_b = tokens_b.shape[1]
    
    if len_a != len_b:
        min_len = min(len_a, len_b)
        print(f"\n[INFO] Prompts have different lengths (A: {len_a}, B: {len_b}). Truncating to {min_len} tokens for balance.")
        
        if len_a > min_len:
            truncated_tokens = tokens_a[:, :min_len]
            truncated_text = tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
            print(f"Truncated Prompt A: \"{prompt_a}\" -> \"{truncated_text}\"")
            tokens_a = truncated_tokens
            
        if len_b > min_len:
            truncated_tokens = tokens_b[:, :min_len]
            truncated_text = tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
            print(f"Truncated Prompt B: \"{prompt_b}\" -> \"{truncated_text}\"")
            tokens_b = truncated_tokens

    print(f"\n{'='*80}")
    print(f"GENERATION WITH LINEAR COMBINATION")
    print(f"{'='*80}")
    print(f"Prompt A: {prompt_a}")
    print(f"Prompt B: {prompt_b}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Alpha: {alpha}")
    print(f"Beta: {beta}")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        tokens_a, tokens_b = generate_dual_linear(
            dual_model,
            separator_model,
            tokens_a,
            tokens_b,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            alpha=alpha,
            beta=beta
        )
    
    # Decode results with coloring
    BLUE = "\033[94m"
    RESET = "\033[0m"
    
    def get_colored_text(full_tokens, prompt_len):
        # Decode the full sequence
        full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
        # Decode only the prompt part
        prompt_text = tokenizer.decode(full_tokens[:prompt_len], skip_special_tokens=True)
        
        # Find where prompt ends and generation begins in the full text string
        if full_text.startswith(prompt_text):
            gen_text = full_text[len(prompt_text):]
            return f"{BLUE}{prompt_text}{RESET}{gen_text}", full_text
        else:
            # Fallback for weird edge cases in decoding
            gen_text = tokenizer.decode(full_tokens[prompt_len:], skip_special_tokens=True)
            return f"{BLUE}{prompt_text}{RESET} {gen_text}", full_text

    colored_a, generated_a = get_colored_text(tokens_a[0], len_a if 'min_len' not in locals() else min_len)
    colored_b, generated_b = get_colored_text(tokens_b[0], len_b if 'min_len' not in locals() else min_len)
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"\nText A:\n{colored_a}")
    print(f"\n{'-'*80}")
    print(f"\nText B:\n{colored_b}")
    print(f"\n{'='*80}\n")
    
    return generated_a, generated_b


if __name__ == '__main__':
    fire.Fire({
        'separator': generate_with_separator,
        'linear': generate_with_linear_combination
    })

