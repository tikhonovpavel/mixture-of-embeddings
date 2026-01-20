import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import fire
import numpy as np
from tqdm import tqdm
import os
import json

from model import DualStreamLLM


def load_checkpoint(checkpoint_path, model_name, device, objective='distillation', shared_embeddings=True):
    """Load trained DualStreamLLM from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Model config: objective={objective}, shared_embeddings={shared_embeddings}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with same configuration
    model = DualStreamLLM(
        model_name=model_name,
        freeze_backbone=True,
        embedding_init_strategy='copy',
        objective=objective,
        shared_embeddings=shared_embeddings
    )
    
    # Load state dict
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as e:
        error_msg = str(e)
        
        # Check if error is due to missing keys (lm_head due to optimize_memory or stream_proj in old runs)
        if 'lm_head' in error_msg or 'stream_proj' in error_msg:
            print(f"Warning: Loading with strict=False due to mismatch: {error_msg.splitlines()[0]}...")
            
            # Handle backward compatibility for old checkpoints missing stream projections
            if 'stream_proj' in error_msg:
                print("Initializing stream projections as Identity for backward compatibility.")
                with torch.no_grad():
                    model.stream_proj_A.weight.copy_(torch.eye(model.stream_proj_A.weight.shape[0]))
                    model.stream_proj_B.weight.copy_(torch.eye(model.stream_proj_B.weight.shape[0]))
            
            # Retry loading with strict=False to ignore missing/extra keys
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            raise e
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (step {checkpoint.get('step', 'unknown')})")
    return model


def compute_token_accuracy(predicted_tokens, target_tokens):
    """Compute exact token match accuracy."""
    min_len = min(len(predicted_tokens), len(target_tokens))
    if min_len == 0:
        return 0.0
    matches = sum(p == t for p, t in zip(predicted_tokens[:min_len], target_tokens[:min_len]))
    return matches / min_len


def generate_and_evaluate_pair(
    dual_model,
    vanilla_model,
    separator_model,
    tokenizer,
    text_a,
    text_b,
    context_len=32,
    eval_len=32,
    top_k_mask=30,
    use_separator_model=True,
    device='cuda'
):
    """
    Generate continuation for a pair of texts and evaluate.
    
    Args:
        dual_model: Trained DualStreamLLM
        separator_model: Smaller model for creating masks (or None)
        tokenizer: Tokenizer
        text_a: Full text A
        text_b: Full text B
        context_len: Number of tokens to use as context
        eval_len: Number of tokens to generate and evaluate
        top_k_mask: Number of top tokens to keep in mask
        use_separator_model: Whether to use separator model
        device: Device
        
    Returns:
        Dictionary with metrics
    """
    # Tokenize full texts
    full_tokens_a = tokenizer.encode(text_a, return_tensors='pt').to(device)
    full_tokens_b = tokenizer.encode(text_b, return_tensors='pt').to(device)
    
    # Split into context and target
    context_tokens_a = full_tokens_a[:, :context_len]
    context_tokens_b = full_tokens_b[:, :context_len]
    
    target_tokens_a = full_tokens_a[:, context_len:context_len+eval_len]
    target_tokens_b = full_tokens_b[:, context_len:context_len+eval_len]
    
    # If target is empty, skip this pair
    if target_tokens_a.shape[1] == 0 or target_tokens_b.shape[1] == 0:
        return None
    
    #
    # Teacher-forcing loop: we feed the ground truth tokens at each step
    #
    
    log_probs_a_hits = []
    log_probs_b_hits = []
    ranks_a_hits = []
    ranks_b_hits = []
    
    vanilla_log_probs_a_hits = []
    vanilla_log_probs_b_hits = []
    vanilla_ranks_a_hits = []
    vanilla_ranks_b_hits = []
    vanilla_correct_predictions_a = 0
    vanilla_correct_predictions_b = 0
    vanilla_misses_a = 0
    vanilla_misses_b = 0
    
    misses_a = 0
    misses_b = 0
    
    correct_predictions_a = 0
    correct_predictions_b = 0
    
    total_steps = 0
    
    with torch.no_grad():
        eval_steps = min(eval_len, target_tokens_a.shape[1], target_tokens_b.shape[1])
        if eval_steps == 0:
            return None
        total_steps = eval_steps
        
        for step in range(total_steps):
            # Context at current step is the ground truth up to this point
            current_context_a = full_tokens_a[:, :context_len + step]
            current_context_b = full_tokens_b[:, :context_len + step]
            
            # Target token for this step
            target_token_a = full_tokens_a[:, context_len + step]
            target_token_b = full_tokens_b[:, context_len + step]
            
            # Prepare inputs (pad to same length)
            max_len = max(current_context_a.shape[1], current_context_b.shape[1])
            
            input_ids_a = F.pad(current_context_a, (0, max_len - current_context_a.shape[1]), value=tokenizer.pad_token_id)
            input_ids_b = F.pad(current_context_b, (0, max_len - current_context_b.shape[1]), value=tokenizer.pad_token_id)
            
            attention_mask_a = (input_ids_a != tokenizer.pad_token_id).long()
            attention_mask_b = (input_ids_b != tokenizer.pad_token_id).long()
            
            # Get mixed logits from dual model
            outputs = dual_model(
                input_ids_A=input_ids_a,
                attention_mask_A=attention_mask_a,
                input_ids_B=input_ids_b,
                attention_mask_B=attention_mask_b
            )
            
            if isinstance(outputs, tuple):
                # Separate objective: returns (logits_A, logits_B)
                logits_a, logits_b = outputs
                
                logits_a_next = logits_a[:, -1, :]
                logits_b_next = logits_b[:, -1, :]
                
                mixed_logits_next = None 
                use_separator_model_current = use_separator_model
            else:
                mixed_logits = outputs
                mixed_logits_next = mixed_logits[:, -1, :]  # [1, vocab_size]
                
                logits_a_next = mixed_logits_next
                logits_b_next = mixed_logits_next
                use_separator_model_current = use_separator_model

            if use_separator_model_current and separator_model is not None:
                # Use separator model to create masks
                sep_outputs_a = separator_model(current_context_a)
                sep_logits_a = sep_outputs_a.logits[:, -1, :]
                
                sep_outputs_b = separator_model(current_context_b)
                sep_logits_b = sep_outputs_b.logits[:, -1, :]
                
                # Create masks
                topk_a = torch.topk(sep_logits_a, k=top_k_mask, dim=-1)
                mask_a = torch.zeros_like(logits_a_next)
                mask_a.scatter_(1, topk_a.indices, 1.0)
                
                topk_b = torch.topk(sep_logits_b, k=top_k_mask, dim=-1)
                mask_b = torch.zeros_like(logits_b_next)
                mask_b.scatter_(1, topk_b.indices, 1.0)
                
                # Apply masks
                filtered_logits_a = logits_a_next.clone()
                filtered_logits_a[mask_a == 0] = -float('inf')
                
                filtered_logits_b = logits_b_next.clone()
                filtered_logits_b[mask_b == 0] = -float('inf')
            else:
                # Without separator model or if mixed_logits_next is not available, use logits directly
                filtered_logits_a = logits_a_next
                filtered_logits_b = logits_b_next
            
            # --- Metrics Calculation ---
            
            # 1. Check for filter misses (if separator is used)
            is_miss_a = use_separator_model_current and (filtered_logits_a[0, target_token_a].item() == -float('inf'))
            is_miss_b = use_separator_model_current and (filtered_logits_b[0, target_token_b].item() == -float('inf'))

            if is_miss_a:
                misses_a += 1
            if is_miss_b:
                misses_b += 1

            # 2. Perplexity (calculated only on hits)
            if not is_miss_a:
                log_probs_dist_a = F.log_softmax(filtered_logits_a, dim=-1)
                log_prob_a = log_probs_dist_a.gather(1, target_token_a.unsqueeze(1)).squeeze().item()
                log_probs_a_hits.append(log_prob_a)
            
            if not is_miss_b:
                log_probs_dist_b = F.log_softmax(filtered_logits_b, dim=-1)
                log_prob_b = log_probs_dist_b.gather(1, target_token_b.unsqueeze(1)).squeeze().item()
                log_probs_b_hits.append(log_prob_b)

            # 3. Accuracy (based on argmax)
            predicted_token_a = torch.argmax(filtered_logits_a, dim=-1)
            predicted_token_b = torch.argmax(filtered_logits_b, dim=-1)
            
            if predicted_token_a.item() == target_token_a.item():
                correct_predictions_a += 1
            if predicted_token_b.item() == target_token_b.item():
                correct_predictions_b += 1

            # 4. Rank of the TRUE token (calculated only on hits)
            # Pre-calculate sorted indices, as they are needed for vanilla rank too
            sorted_indices_a = torch.argsort(filtered_logits_a, dim=-1, descending=True)
            if not is_miss_a:
                rank_a = (sorted_indices_a == target_token_a.unsqueeze(-1)).nonzero(as_tuple=True)[1].item() + 1
                ranks_a_hits.append(rank_a)

            sorted_indices_b = torch.argsort(filtered_logits_b, dim=-1, descending=True)
            if not is_miss_b:
                rank_b = (sorted_indices_b == target_token_b.unsqueeze(-1)).nonzero(as_tuple=True)[1].item() + 1
                ranks_b_hits.append(rank_b)

            if vanilla_model:
                # Get vanilla model's prediction
                vanilla_logits_a = vanilla_model(current_context_a).logits[:, -1, :]
                vanilla_top_token_a = torch.argmax(vanilla_logits_a, dim=-1)
                
                # Check if the vanilla target was missed by the filter
                is_vanilla_miss_a = use_separator_model_current and (filtered_logits_a[0, vanilla_top_token_a].item() == -float('inf'))
                if is_vanilla_miss_a:
                    vanilla_misses_a += 1
                else:
                    # Rank of VANILLA model's top prediction in our distribution (on hits only)
                    vanilla_rank_a = (sorted_indices_a == vanilla_top_token_a.unsqueeze(-1)).nonzero(as_tuple=True)[1].item() + 1
                    vanilla_ranks_a_hits.append(vanilla_rank_a)
                    
                    # Perplexity of our distribution for the VANILLA target token
                    log_probs_dist_a = F.log_softmax(filtered_logits_a, dim=-1)
                    log_prob_a = log_probs_dist_a.gather(1, vanilla_top_token_a.unsqueeze(1)).squeeze().item()
                    vanilla_log_probs_a_hits.append(log_prob_a)

                # Accuracy vs VANILLA target: does our argmax match vanilla's argmax?
                if predicted_token_a.item() == vanilla_top_token_a.item():
                    vanilla_correct_predictions_a += 1

                # --- Stream B ---
                vanilla_logits_b = vanilla_model(current_context_b).logits[:, -1, :]
                vanilla_top_token_b = torch.argmax(vanilla_logits_b, dim=-1)
                
                is_vanilla_miss_b = use_separator_model_current and (filtered_logits_b[0, vanilla_top_token_b].item() == -float('inf'))
                if is_vanilla_miss_b:
                    vanilla_misses_b += 1
                else:
                    # Rank of VANILLA model's top prediction in our distribution (on hits only)
                    vanilla_rank_b = (sorted_indices_b == vanilla_top_token_b.unsqueeze(-1)).nonzero(as_tuple=True)[1].item() + 1
                    vanilla_ranks_b_hits.append(vanilla_rank_b)

                    log_probs_dist_b = F.log_softmax(filtered_logits_b, dim=-1)
                    log_prob_b = log_probs_dist_b.gather(1, vanilla_top_token_b.unsqueeze(1)).squeeze().item()
                    vanilla_log_probs_b_hits.append(log_prob_b)

                if predicted_token_b.item() == vanilla_top_token_b.item():
                    vanilla_correct_predictions_b += 1


    # --- Aggregate results for this pair ---
    
    # Perplexity is inf if all tokens were misses, otherwise calculated on hits
    perplexity_a = np.exp(-np.mean(log_probs_a_hits)) if log_probs_a_hits else float('inf')
    perplexity_b = np.exp(-np.mean(log_probs_b_hits)) if log_probs_b_hits else float('inf')

    accuracy_a = correct_predictions_a / total_steps
    accuracy_b = correct_predictions_b / total_steps
    
    return {
        'perplexity_a': perplexity_a,
        'perplexity_b': perplexity_b,
        'accuracy_a': accuracy_a,
        'accuracy_b': accuracy_b,
        'avg_perplexity': (perplexity_a + perplexity_b) / 2 if perplexity_a != float('inf') and perplexity_b != float('inf') else float('inf'),
        'avg_accuracy': (accuracy_a + accuracy_b) / 2,
        'avg_rank_a': np.mean(ranks_a_hits) if ranks_a_hits else -1,
        'avg_rank_b': np.mean(ranks_b_hits) if ranks_b_hits else -1,
        'avg_rank': np.mean(ranks_a_hits + ranks_b_hits) if (ranks_a_hits or ranks_b_hits) else -1,
        
        'vanilla_accuracy_a': vanilla_correct_predictions_a / total_steps if total_steps > 0 else 0,
        'vanilla_accuracy_b': vanilla_correct_predictions_b / total_steps if total_steps > 0 else 0,
        'avg_vanilla_accuracy': (vanilla_correct_predictions_a + vanilla_correct_predictions_b) / (total_steps * 2) if total_steps > 0 else 0,
        
        'vanilla_perplexity_a': np.exp(-np.mean(vanilla_log_probs_a_hits)) if vanilla_log_probs_a_hits else float('inf'),
        'vanilla_perplexity_b': np.exp(-np.mean(vanilla_log_probs_b_hits)) if vanilla_log_probs_b_hits else float('inf'),
        
        'avg_vanilla_rank_a': np.mean(vanilla_ranks_a_hits) if vanilla_ranks_a_hits else -1,
        'avg_vanilla_rank_b': np.mean(vanilla_ranks_b_hits) if vanilla_ranks_b_hits else -1,
        'avg_vanilla_rank': np.mean(vanilla_ranks_a_hits + vanilla_ranks_b_hits) if (vanilla_ranks_a_hits or vanilla_ranks_b_hits) else -1,
        
        'miss_rate_a': misses_a / total_steps if total_steps > 0 else 0,
        'miss_rate_b': misses_b / total_steps if total_steps > 0 else 0,
        'avg_miss_rate': (misses_a + misses_b) / (total_steps * 2) if total_steps > 0 else 0,
        
        'vanilla_miss_rate_a': vanilla_misses_a / total_steps if total_steps > 0 else 0,
        'vanilla_miss_rate_b': vanilla_misses_b / total_steps if total_steps > 0 else 0,
        'avg_vanilla_miss_rate': (vanilla_misses_a + vanilla_misses_b) / (total_steps * 2) if total_steps > 0 else 0
    }


def evaluate(
    checkpoint_path=None,
    model_name='EleutherAI/pythia-2.8b',
    separator_model_name='EleutherAI/pythia-160m',
    use_separator_model=True,
    dataset_name='HuggingFaceFW/fineweb',
    dataset_config=None,
    num_samples=100,
    context_len=32,
    eval_len=32,
    top_k_mask=30,
    device='cuda',
    seed=42,
    num_train_steps=500_000,
    train_batch_size=8,
    objective='distillation',
    shared_embeddings=True,
    config_path=None,
    cached_dataset_path=None
):
    """
    Evaluate dual-stream generation quality.
    
    Args:
        checkpoint_path: Path to trained DualStreamLLM checkpoint
        model_name: Name of the base model
        separator_model_name: Name of smaller separator model
        use_separator_model: Whether to use separator model (True) or baseline (False)
        dataset_name: Dataset to use for evaluation
        dataset_config: Dataset config
        num_samples: Number of text pairs to evaluate
        context_len: Number of context tokens
        eval_len: Number of tokens to generate
        top_k_mask: Top-K for mask creation
        device: Device to use
        seed: Random seed
        num_train_steps: Number of training steps used during training (to skip train data)
        train_batch_size: Batch size used during training (to skip train data)
        objective: Model objective ('separate', 'distillation', etc.)
        shared_embeddings: Whether embeddings are shared
        config_path: Path to training_config.json to load params automatically
        cached_dataset_path: Path to save/load cached dataset samples
    """
    import json
    if config_path:
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
            objective = config.get('objective', objective)
            shared_embeddings = config.get('shared_embeddings', shared_embeddings)
            model_name = config.get('model_name', model_name)
            # You can override other params here if you want consistency with training config
            
    torch.manual_seed(seed)
    device = torch.device(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dual-stream model
    if checkpoint_path:
        dual_model = load_checkpoint(checkpoint_path, model_name, device, objective, shared_embeddings)
    else:
        print("\nNo checkpoint provided. Loading vanilla pre-trained model into DualStreamLLM structure.")
        dual_model = DualStreamLLM(
            model_name=model_name,
            freeze_backbone=True,
            embedding_init_strategy='copy',
            objective=objective,
            shared_embeddings=shared_embeddings
        ).to(device)
        dual_model.eval()
        print("Vanilla model loaded successfully.")

    
    # Load separator model if needed
    separator_model = None
    if use_separator_model:
        print(f"\nLoading separator model: {separator_model_name}")
        separator_model = AutoModelForCausalLM.from_pretrained(
            separator_model_name,
            dtype=torch.bfloat16,
        ).to(device)
        separator_model.eval()
        print("Separator model loaded successfully")
    else:
        print("\nRunning in BASELINE mode (without separator model)")
        
    # Load full vanilla model for comparison metric
    print(f"\nLoading full vanilla model for comparison: {model_name}")
    vanilla_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).to(device)
    vanilla_model.eval()
    print("Full vanilla model loaded successfully")
    
    # Load dataset logic with caching
    dataset_iter = None
    num_train_samples = num_train_steps * train_batch_size * 2

    if cached_dataset_path and os.path.exists(cached_dataset_path):
        print(f"\nLoading cached dataset from: {cached_dataset_path}")
        with open(cached_dataset_path, 'r') as f:
            cached_data = json.load(f)
        print(f"Loaded {len(cached_data)} samples from cache.")
        dataset_iter = iter(cached_data)
    else:
        print(f"\nLoading dataset: {dataset_name}")
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split='train', streaming=True, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split='train', streaming=True, trust_remote_code=True)
        
        # Use same seed as training to get the same data order
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        
        # Manually skip samples with progress bar
        dataset_iter = iter(dataset)
        print(f"Skipping first {num_train_samples:,} samples (used during training)...")
        for _ in tqdm(range(num_train_samples), desc="Skipping train samples"):
            try:
                next(dataset_iter)
            except StopIteration:
                print("Warning: Dataset exhausted before all training samples could be skipped.")
                break
        print(f"Skipped {num_train_samples:,} samples. Using subsequent data for evaluation.")

        # Save to cache if requested
        if cached_dataset_path:
            print(f"Caching samples to {cached_dataset_path}...")
            # Collect enough samples for evaluation (with some margin)
            # We need num_samples * 2 (for pairs) plus some for potential skips due to length
            samples_to_cache = max(num_samples * 10, 2000)
            cached_data = []
            print(f"Collecting {samples_to_cache} samples for cache...")
            try:
                for _ in tqdm(range(samples_to_cache), desc="Caching"):
                    cached_data.append(next(dataset_iter))
            except StopIteration:
                print("Dataset exhausted during caching.")
            
            with open(cached_dataset_path, 'w') as f:
                json.dump(cached_data, f)
            print(f"Saved {len(cached_data)} samples to cache.")
            
            # Restart iterator from the cached data so we use exactly what's in the file
            dataset_iter = iter(cached_data)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION CONFIGURATION")
    print(f"{'='*80}")
    if checkpoint_path:
        print(f"Mode: Evaluating CHECKPOINT '{checkpoint_path}'")
    else:
        print(f"Mode: Evaluating VANILLA pre-trained model")
    print(f"Using separator model: {use_separator_model}")
    print(f"Base Model: {model_name}")
    print(f"Separator model: {separator_model_name if use_separator_model else 'N/A'}")
    print(f"Dataset: {dataset_name}")
    print(f"Seed: {seed}")
    print(f"Train steps skipped: {num_train_steps:,}")
    print(f"Train batch size: {train_batch_size}")
    print(f"Train samples skipped: {num_train_samples:,}")
    print(f"Number of eval samples: {num_samples}")
    print(f"Context length: {context_len}")
    print(f"Eval length: {eval_len}")
    print(f"Top-K mask: {top_k_mask if use_separator_model else 'N/A'}")
    print(f"{'='*80}\n")
    
    # Evaluate
    results = []
    # The dataset_iter is already advanced past the training samples
    
    with tqdm(total=num_samples, desc="Evaluating") as pbar:
        while len(results) < num_samples:
            try:
                # Get two samples for pair
                sample_a = next(dataset_iter)
                sample_b = next(dataset_iter)
                
                # Extract text
                if 'text' in sample_a:
                    text_a = sample_a['text']
                    text_b = sample_b['text']
                elif 'content' in sample_a:
                    text_a = sample_a['content']
                    text_b = sample_b['content']
                else:
                    continue
                
                # Tokenize and check length
                tokens_a = tokenizer.encode(text_a)
                tokens_b = tokenizer.encode(text_b)
                
                required_len = context_len + eval_len
                if len(tokens_a) < required_len or len(tokens_b) < required_len:
                    continue
                
                # Truncate to required length
                text_a = tokenizer.decode(tokens_a[:required_len])
                text_b = tokenizer.decode(tokens_b[:required_len])
                
                # Evaluate this pair
                metrics = generate_and_evaluate_pair(
                    dual_model=dual_model,
                    vanilla_model=vanilla_model,
                    separator_model=separator_model,
                    tokenizer=tokenizer,
                    text_a=text_a,
                    text_b=text_b,
                    context_len=context_len,
                    eval_len=eval_len,
                    top_k_mask=top_k_mask,
                    use_separator_model=use_separator_model,
                    device=device
                )
                
                if metrics is not None:
                    results.append(metrics)
                    pbar.update(1)
                    
            except StopIteration:
                print("Dataset exhausted before reaching num_samples")
                break
    
    # Compute aggregate statistics
    if len(results) == 0:
        print("No valid results collected!")
        return
    
    avg_metrics = {
        'perplexity_a': np.nanmean([r['perplexity_a'] for r in results if r['perplexity_a'] != float('inf')]),
        'perplexity_b': np.nanmean([r['perplexity_b'] for r in results if r['perplexity_b'] != float('inf')]),
        'avg_perplexity': np.nanmean([r['avg_perplexity'] for r in results if r['avg_perplexity'] != float('inf')]),
        
        'accuracy_a': np.mean([r['accuracy_a'] for r in results]),
        'accuracy_b': np.mean([r['accuracy_b'] for r in results]),
        'avg_accuracy': np.mean([r['avg_accuracy'] for r in results]),
        
        'avg_rank_a': np.mean([r['avg_rank_a'] for r in results if r['avg_rank_a'] != -1]),
        'avg_rank_b': np.mean([r['avg_rank_b'] for r in results if r['avg_rank_b'] != -1]),
        'avg_rank': np.mean([r['avg_rank'] for r in results if r['avg_rank'] != -1]),
        
        'vanilla_accuracy_a': np.mean([r['vanilla_accuracy_a'] for r in results]),
        'vanilla_accuracy_b': np.mean([r['vanilla_accuracy_b'] for r in results]),
        'avg_vanilla_accuracy': np.mean([r['avg_vanilla_accuracy'] for r in results]),
        
        'vanilla_perplexity_a': np.nanmean([r['vanilla_perplexity_a'] for r in results if r['vanilla_perplexity_a'] != float('inf')]),
        'vanilla_perplexity_b': np.nanmean([r['vanilla_perplexity_b'] for r in results if r['vanilla_perplexity_b'] != float('inf')]),
        
        'avg_vanilla_rank_a': np.mean([r['avg_vanilla_rank_a'] for r in results if r['avg_vanilla_rank_a'] != -1]),
        'avg_vanilla_rank_b': np.mean([r['avg_vanilla_rank_b'] for r in results if r['avg_vanilla_rank_b'] != -1]),
        'avg_vanilla_rank': np.mean([r['avg_vanilla_rank'] for r in results if r['avg_vanilla_rank'] != -1]),
        
        'miss_rate_a': np.mean([r['miss_rate_a'] for r in results]),
        'miss_rate_b': np.mean([r['miss_rate_b'] for r in results]),
        'avg_miss_rate': np.mean([r['avg_miss_rate'] for r in results]),
        
        'vanilla_miss_rate_a': np.mean([r['vanilla_miss_rate_a'] for r in results]),
        'vanilla_miss_rate_b': np.mean([r['vanilla_miss_rate_b'] for r in results]),
        'avg_vanilla_miss_rate': np.mean([r['avg_vanilla_miss_rate'] for r in results]),
    }
    
    std_metrics = {
        'perplexity_a': np.nanstd([r['perplexity_a'] for r in results if r['perplexity_a'] != float('inf')]),
        'perplexity_b': np.nanstd([r['perplexity_b'] for r in results if r['perplexity_b'] != float('inf')]),
        'avg_perplexity': np.nanstd([r['avg_perplexity'] for r in results if r['avg_perplexity'] != float('inf')]),
        
        'accuracy_a': np.std([r['accuracy_a'] for r in results]),
        'accuracy_b': np.std([r['accuracy_b'] for r in results]),
        'avg_accuracy': np.std([r['avg_accuracy'] for r in results]),
        
        'avg_rank_a': np.std([r['avg_rank_a'] for r in results if r['avg_rank_a'] != -1]),
        'avg_rank_b': np.std([r['avg_rank_b'] for r in results if r['avg_rank_b'] != -1]),
        'avg_rank': np.std([r['avg_rank'] for r in results if r['avg_rank'] != -1]),
        
        'vanilla_accuracy_a': np.std([r['vanilla_accuracy_a'] for r in results]),
        'vanilla_accuracy_b': np.std([r['vanilla_accuracy_b'] for r in results]),
        'avg_vanilla_accuracy': np.std([r['avg_vanilla_accuracy'] for r in results]),
        
        'vanilla_perplexity_a': np.nanstd([r['vanilla_perplexity_a'] for r in results if r['vanilla_perplexity_a'] != float('inf')]),
        'vanilla_perplexity_b': np.nanstd([r['vanilla_perplexity_b'] for r in results if r['vanilla_perplexity_b'] != float('inf')]),
        
        'avg_vanilla_rank_a': np.std([r['avg_vanilla_rank_a'] for r in results if r['avg_vanilla_rank_a'] != -1]),
        'avg_vanilla_rank_b': np.std([r['avg_vanilla_rank_b'] for r in results if r['avg_vanilla_rank_b'] != -1]),
        'avg_vanilla_rank': np.std([r['avg_vanilla_rank'] for r in results if r['avg_vanilla_rank'] != -1]),
        
        'miss_rate_a': np.std([r['miss_rate_a'] for r in results]),
        'miss_rate_b': np.std([r['miss_rate_b'] for r in results]),
        'avg_miss_rate': np.std([r['avg_miss_rate'] for r in results]),
        
        'vanilla_miss_rate_a': np.std([r['vanilla_miss_rate_a'] for r in results]),
        'vanilla_miss_rate_b': np.std([r['vanilla_miss_rate_b'] for r in results]),
        'avg_vanilla_miss_rate': np.std([r['avg_vanilla_miss_rate'] for r in results]),
    }
    
    median_metrics = {
        'perplexity_a': np.nanmedian([r['perplexity_a'] for r in results if r['perplexity_a'] != float('inf')]),
        'perplexity_b': np.nanmedian([r['perplexity_b'] for r in results if r['perplexity_b'] != float('inf')]),
        'avg_perplexity': np.nanmedian([r['avg_perplexity'] for r in results if r['avg_perplexity'] != float('inf')]),
        
        'accuracy_a': np.median([r['accuracy_a'] for r in results]),
        'accuracy_b': np.median([r['accuracy_b'] for r in results]),
        'avg_accuracy': np.median([r['avg_accuracy'] for r in results]),
        
        'avg_rank_a': np.median([r['avg_rank_a'] for r in results if r['avg_rank_a'] != -1]),
        'avg_rank_b': np.median([r['avg_rank_b'] for r in results if r['avg_rank_b'] != -1]),
        'avg_rank': np.median([r['avg_rank'] for r in results if r['avg_rank'] != -1]),
        
        'vanilla_accuracy_a': np.median([r['vanilla_accuracy_a'] for r in results]),
        'vanilla_accuracy_b': np.median([r['vanilla_accuracy_b'] for r in results]),
        'avg_vanilla_accuracy': np.median([r['avg_vanilla_accuracy'] for r in results]),
        
        'vanilla_perplexity_a': np.nanmedian([r['vanilla_perplexity_a'] for r in results if r['vanilla_perplexity_a'] != float('inf')]),
        'vanilla_perplexity_b': np.nanmedian([r['vanilla_perplexity_b'] for r in results if r['vanilla_perplexity_b'] != float('inf')]),
        
        'avg_vanilla_rank_a': np.median([r['avg_vanilla_rank_a'] for r in results if r['avg_vanilla_rank_a'] != -1]),
        'avg_vanilla_rank_b': np.median([r['avg_vanilla_rank_b'] for r in results if r['avg_vanilla_rank_b'] != -1]),
        'avg_vanilla_rank': np.median([r['avg_vanilla_rank'] for r in results if r['avg_vanilla_rank'] != -1]),
        
        'miss_rate_a': np.median([r['miss_rate_a'] for r in results]),
        'miss_rate_b': np.median([r['miss_rate_b'] for r in results]),
        'avg_miss_rate': np.median([r['avg_miss_rate'] for r in results]),
        
        'vanilla_miss_rate_a': np.median([r['vanilla_miss_rate_a'] for r in results]),
        'vanilla_miss_rate_b': np.median([r['vanilla_miss_rate_b'] for r in results]),
        'avg_vanilla_miss_rate': np.median([r['avg_vanilla_miss_rate'] for r in results]),
    }
    
    # Print results
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Mode: {'WITH separator model' if use_separator_model else 'BASELINE'}")
    print(f"Number of evaluated pairs: {len(results)}")
    print(f"\n{'-'*80}")
    print(f"PERPLEXITY (on non-missed tokens):")
    print(f"  Text A: mean={avg_metrics['perplexity_a']:.2f} ± {std_metrics['perplexity_a']:.2f}, median={median_metrics['perplexity_a']:.2f}")
    print(f"  Text B: mean={avg_metrics['perplexity_b']:.2f} ± {std_metrics['perplexity_b']:.2f}, median={median_metrics['perplexity_b']:.2f}")
    print(f"  Average: mean={avg_metrics['avg_perplexity']:.2f} ± {std_metrics['avg_perplexity']:.2f}, median={median_metrics['avg_perplexity']:.2f}")
    print(f"\n{'-'*80}")
    print(f"FILTER MISS RATE (true token outside top-K mask):")
    print(f"  Text A: mean={avg_metrics['miss_rate_a']*100:.2f}%, median={median_metrics['miss_rate_a']*100:.2f}%")
    print(f"  Text B: mean={avg_metrics['miss_rate_b']*100:.2f}%, median={median_metrics['miss_rate_b']*100:.2f}%")
    print(f"  Average: mean={avg_metrics['avg_miss_rate']*100:.2f}%, median={median_metrics['avg_miss_rate']*100:.2f}%")
    print(f"\n{'-'*80}")
    print(f"TOKEN ACCURACY:")
    print(f"  Text A: mean={avg_metrics['accuracy_a']*100:.2f}% ± {std_metrics['accuracy_a']*100:.2f}%, median={median_metrics['accuracy_a']*100:.2f}%")
    print(f"  Text B: mean={avg_metrics['accuracy_b']*100:.2f}% ± {std_metrics['accuracy_b']*100:.2f}%, median={median_metrics['accuracy_b']*100:.2f}%")
    print(f"  Average: mean={avg_metrics['avg_accuracy']*100:.2f}% ± {std_metrics['avg_accuracy']*100:.2f}%, median={median_metrics['avg_accuracy']*100:.2f}%")
    print(f"\n{'-'*80}")
    print(f"TRUE TOKEN RANK (on non-missed tokens):")
    print(f"  Text A: mean={avg_metrics['avg_rank_a']:.2f} ± {std_metrics['avg_rank_a']:.2f}, median={median_metrics['avg_rank_a']:.2f}")
    print(f"  Text B: mean={avg_metrics['avg_rank_b']:.2f} ± {std_metrics['avg_rank_b']:.2f}, median={median_metrics['avg_rank_b']:.2f}")
    print(f"  Average: mean={avg_metrics['avg_rank']:.2f} ± {std_metrics['avg_rank']:.2f}, median={median_metrics['avg_rank']:.2f}")
    print(f"\n{'-'*80}")
    print(f"--- METRICS AGAINST VANILLA MODEL'S PREDICTION ---")
    print(f"\nVANILLA TARGET MISS RATE (vanilla's top token outside our mask):")
    print(f"  Text A: mean={avg_metrics['vanilla_miss_rate_a']*100:.2f}%, median={median_metrics['vanilla_miss_rate_a']*100:.2f}%")
    print(f"  Text B: mean={avg_metrics['vanilla_miss_rate_b']*100:.2f}%, median={median_metrics['vanilla_miss_rate_b']*100:.2f}%")
    print(f"  Average: mean={avg_metrics['avg_vanilla_miss_rate']*100:.2f}%, median={median_metrics['avg_vanilla_miss_rate']*100:.2f}%")
    print(f"\nVANILLA TARGET ACCURACY (our argmax == vanilla's argmax):")
    print(f"  Text A: mean={avg_metrics['vanilla_accuracy_a']*100:.2f}% ± {std_metrics['vanilla_accuracy_a']*100:.2f}%, median={median_metrics['vanilla_accuracy_a']*100:.2f}%")
    print(f"  Text B: mean={avg_metrics['vanilla_accuracy_b']*100:.2f}% ± {std_metrics['vanilla_accuracy_b']*100:.2f}%, median={median_metrics['vanilla_accuracy_b']*100:.2f}%")
    print(f"  Average: mean={avg_metrics['avg_vanilla_accuracy']*100:.2f}% ± {std_metrics['avg_vanilla_accuracy']*100:.2f}%, median={median_metrics['avg_vanilla_accuracy']*100:.2f}%")
    print(f"\nVANILLA TARGET PERPLEXITY (on non-missed vanilla targets):")
    print(f"  Text A: mean={avg_metrics['vanilla_perplexity_a']:.2f} ± {std_metrics['vanilla_perplexity_a']:.2f}, median={median_metrics['vanilla_perplexity_a']:.2f}")
    print(f"  Text B: mean={avg_metrics['vanilla_perplexity_b']:.2f} ± {std_metrics['vanilla_perplexity_b']:.2f}, median={median_metrics['vanilla_perplexity_b']:.2f}")
    print(f"\nVANILLA MODEL'S TOP TOKEN RANK (on non-missed vanilla targets):")
    print(f"  Text A: mean={avg_metrics['avg_vanilla_rank_a']:.2f} ± {std_metrics['avg_vanilla_rank_a']:.2f}, median={median_metrics['avg_vanilla_rank_a']:.2f}")
    print(f"  Text B: mean={avg_metrics['avg_vanilla_rank_b']:.2f} ± {std_metrics['avg_vanilla_rank_b']:.2f}, median={median_metrics['avg_vanilla_rank_b']:.2f}")
    print(f"  Average: mean={avg_metrics['avg_vanilla_rank']:.2f} ± {std_metrics['avg_vanilla_rank']:.2f}, median={median_metrics['avg_vanilla_rank']:.2f}")
    print(f"{'='*80}\n")
    
    return avg_metrics, std_metrics, median_metrics


def forward_with_embedding_scale(dual_model, input_ids, attention_mask, embedding_scale=1.0):
    """
    Forward pass through dual model. 
    Now that the model has dynamic rescaling and stream tags, we just pass the same input to both streams.
    The embedding_scale parameter is kept for backward compatibility but should usually be 1.0.
    """
    # If scale is 1.0 (default), we use the model's forward pass with same inputs
    # which automatically handles normalization and stream tags correctly.
    if embedding_scale == 1.0:
        outputs = dual_model(
            input_ids_A=input_ids,
            attention_mask_A=attention_mask,
            input_ids_B=input_ids,
            attention_mask_B=attention_mask
        )
        if isinstance(outputs, tuple):
             # For separate objective, return logits from stream A (arbitrary choice for single stream)
             return outputs[0]
        return outputs
    
    # Otherwise, we use the manual logic (only for special testing purposes)
    if dual_model.shared_embeddings:
        embeds = dual_model.emb(input_ids)
    else:
        embeds = dual_model.emb_A(input_ids)
    
    # Add stream tag A (matching how the model was trained)
    embeds = embeds + dual_model.stream_id_A
    
    scaled_embeds = embeds * embedding_scale
    
    transformer_outputs = dual_model.transformer(
        inputs_embeds=scaled_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    
    last_hidden_state = transformer_outputs.last_hidden_state
    
    if dual_model.objective == 'separate':
        logits = dual_model.unemb_A(last_hidden_state)
    else:
        logits = dual_model.unemb(last_hidden_state)
    
    return logits


def evaluate_single_stream_pair(
    dual_model,
    vanilla_model,
    tokenizer,
    text,
    context_len=32,
    eval_len=32,
    embedding_scale=1.0,
    device='cuda'
):
    """
    Evaluate dual model in single-stream mode (same text in both streams).
    Compare with vanilla model predictions.
    
    Args:
        embedding_scale: Scale factor for embeddings. Default is 1.0 as the model
                        now uses Dynamic Rescaling to handle input norms.
    
    Returns:
        Dictionary with metrics
    """
    # Tokenize full text
    full_tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Split into context and target
    context_tokens = full_tokens[:, :context_len]
    target_tokens = full_tokens[:, context_len:context_len+eval_len]
    
    if target_tokens.shape[1] == 0:
        return None
    
    # Metrics accumulators
    dual_log_probs = []
    vanilla_log_probs = []
    dual_correct = 0
    vanilla_correct = 0
    dual_matches_vanilla = 0
    dual_ranks = []
    vanilla_ranks = []
    vanilla_token_rank_in_dual = []  # Rank of vanilla's top-1 in dual's distribution
    
    total_steps = 0
    
    with torch.no_grad():
        eval_steps = min(eval_len, target_tokens.shape[1])
        if eval_steps == 0:
            return None
        total_steps = eval_steps
        
        for step in range(total_steps):
            current_context = full_tokens[:, :context_len + step]
            target_token = full_tokens[:, context_len + step]
            
            # Dual model with embedding scale
            dual_logits = forward_with_embedding_scale(
                dual_model,
                input_ids=current_context,
                attention_mask=torch.ones_like(current_context),
                embedding_scale=embedding_scale
            )
            dual_logits_next = dual_logits[:, -1, :]
            
            # Vanilla model
            vanilla_outputs = vanilla_model(current_context)
            vanilla_logits_next = vanilla_outputs.logits[:, -1, :]
            
            # --- Metrics ---
            
            # Log probs for perplexity
            dual_log_probs_dist = F.log_softmax(dual_logits_next, dim=-1)
            dual_log_prob = dual_log_probs_dist.gather(1, target_token.unsqueeze(1)).squeeze().item()
            dual_log_probs.append(dual_log_prob)
            
            vanilla_log_probs_dist = F.log_softmax(vanilla_logits_next, dim=-1)
            vanilla_log_prob = vanilla_log_probs_dist.gather(1, target_token.unsqueeze(1)).squeeze().item()
            vanilla_log_probs.append(vanilla_log_prob)
            
            # Accuracy (argmax == ground truth)
            dual_pred = torch.argmax(dual_logits_next, dim=-1)
            vanilla_pred = torch.argmax(vanilla_logits_next, dim=-1)
            
            if dual_pred.item() == target_token.item():
                dual_correct += 1
            if vanilla_pred.item() == target_token.item():
                vanilla_correct += 1
            if dual_pred.item() == vanilla_pred.item():
                dual_matches_vanilla += 1
            
            # Rank of true token
            dual_sorted = torch.argsort(dual_logits_next, dim=-1, descending=True)
            dual_rank = (dual_sorted == target_token.unsqueeze(-1)).nonzero(as_tuple=True)[1].item() + 1
            dual_ranks.append(dual_rank)
            
            vanilla_sorted = torch.argsort(vanilla_logits_next, dim=-1, descending=True)
            vanilla_rank = (vanilla_sorted == target_token.unsqueeze(-1)).nonzero(as_tuple=True)[1].item() + 1
            vanilla_ranks.append(vanilla_rank)
            
            # Rank of vanilla's top-1 token in dual's distribution
            vanilla_top1_rank_in_dual = (dual_sorted == vanilla_pred.unsqueeze(-1)).nonzero(as_tuple=True)[1].item() + 1
            vanilla_token_rank_in_dual.append(vanilla_top1_rank_in_dual)
    
    return {
        'dual_perplexity': np.exp(-np.mean(dual_log_probs)),
        'vanilla_perplexity': np.exp(-np.mean(vanilla_log_probs)),
        'dual_accuracy': dual_correct / total_steps,
        'vanilla_accuracy': vanilla_correct / total_steps,
        'dual_matches_vanilla': dual_matches_vanilla / total_steps,
        'dual_avg_rank': np.mean(dual_ranks),
        'vanilla_avg_rank': np.mean(vanilla_ranks),
        'vanilla_top1_rank_in_dual': np.mean(vanilla_token_rank_in_dual),
    }


def evaluate_single_stream(
    checkpoint_path=None,
    model_name='EleutherAI/pythia-2.8b',
    dataset_name='HuggingFaceFW/fineweb',
    dataset_config=None,
    num_samples=100,
    context_len=32,
    eval_len=32,
    embedding_scale=1.0,
    device='cuda',
    seed=42,
    num_train_steps=500_000,
    train_batch_size=8,
    objective='distillation',
    shared_embeddings=True,
    config_path=None,
    cached_dataset_path=None,
    output_json_path=None 
):
    """
    Evaluate dual-stream model in single-stream mode (no mixing).
    This tests if the model has degraded on normal (non-mixed) text.
    
    Args:
        checkpoint_path: Path to trained DualStreamLLM checkpoint
        model_name: Name of the base model
        dataset_name: Dataset to use for evaluation
        dataset_config: Dataset config
        num_samples: Number of texts to evaluate
        context_len: Number of context tokens
        eval_len: Number of tokens to evaluate
        embedding_scale: Scale factor for embeddings. Default is 1.0 as the model
                        now uses Dynamic Rescaling to handle input norms.
        device: Device to use
        seed: Random seed
        num_train_steps: Number of training steps (to skip train data)
        train_batch_size: Batch size used during training
        objective: Model objective
        shared_embeddings: Whether embeddings are shared
        config_path: Path to config file
        cached_dataset_path: Path to save/load cached dataset samples
    """
    import json
    if config_path:
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
            objective = config.get('objective', objective)
            shared_embeddings = config.get('shared_embeddings', shared_embeddings)
            model_name = config.get('model_name', model_name)

    torch.manual_seed(seed)
    device = torch.device(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dual-stream model
    if checkpoint_path:
        dual_model = load_checkpoint(checkpoint_path, model_name, device, objective, shared_embeddings)
    else:
        print("\nNo checkpoint provided. Loading vanilla pre-trained model into DualStreamLLM structure.")
        dual_model = DualStreamLLM(
            model_name=model_name,
            freeze_backbone=True,
            embedding_init_strategy='copy',
            objective=objective,
            shared_embeddings=shared_embeddings
        ).to(device)
        dual_model.eval()
        print("Vanilla model loaded successfully.")
    
    # Load vanilla model for comparison
    print(f"\nLoading vanilla model for comparison: {model_name}")
    vanilla_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).to(device)
    vanilla_model.eval()
    print("Vanilla model loaded successfully")
    
    # Load dataset logic with caching
    dataset_iter = None
    num_train_samples = num_train_steps * train_batch_size * 2

    if cached_dataset_path and os.path.exists(cached_dataset_path):
        print(f"\nLoading cached dataset from: {cached_dataset_path}")
        with open(cached_dataset_path, 'r') as f:
            cached_data = json.load(f)
        print(f"Loaded {len(cached_data)} samples from cache.")
        dataset_iter = iter(cached_data)
    else:
        # Load dataset
        print(f"\nLoading dataset: {dataset_name}")
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split='train', streaming=True, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split='train', streaming=True, trust_remote_code=True)
        
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        
        # Skip training samples
        dataset_iter = iter(dataset)
        print(f"Skipping first {num_train_samples:,} samples (used during training)...")
        for _ in tqdm(range(num_train_samples), desc="Skipping train samples"):
            try:
                next(dataset_iter)
            except StopIteration:
                print("Warning: Dataset exhausted before all training samples could be skipped.")
                break
        
        # Save to cache if requested
        if cached_dataset_path:
            print(f"Caching samples to {cached_dataset_path}...")
            # Collect enough samples for evaluation
            samples_to_cache = max(num_samples * 5, 1000)
            cached_data = []
            print(f"Collecting {samples_to_cache} samples for cache...")
            try:
                for _ in tqdm(range(samples_to_cache), desc="Caching"):
                    cached_data.append(next(dataset_iter))
            except StopIteration:
                print("Dataset exhausted during caching.")
            
            with open(cached_dataset_path, 'w') as f:
                json.dump(cached_data, f)
            print(f"Saved {len(cached_data)} samples to cache.")
            
            dataset_iter = iter(cached_data)
    
    print(f"\n{'='*80}")
    print(f"SINGLE-STREAM EVALUATION (no mixing)")
    print(f"{'='*80}")
    if checkpoint_path:
        print(f"Mode: Evaluating CHECKPOINT '{checkpoint_path}'")
    else:
        print(f"Mode: Evaluating VANILLA pre-trained model")
    print(f"Base Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Seed: {seed}")
    print(f"Number of eval samples: {num_samples}")
    print(f"Context length: {context_len}")
    print(f"Eval length: {eval_len}")
    print(f"Embedding scale: {embedding_scale}")
    print(f"{'='*80}\n")
    
    # Evaluate
    results = []
    
    with tqdm(total=num_samples, desc="Evaluating") as pbar:
        while len(results) < num_samples:
            try:
                sample = next(dataset_iter)
                
                # Extract text
                if 'text' in sample:
                    text = sample['text']
                elif 'content' in sample:
                    text = sample['content']
                else:
                    continue
                
                # Tokenize and check length
                tokens = tokenizer.encode(text)
                required_len = context_len + eval_len
                if len(tokens) < required_len:
                    continue
                
                # Truncate to required length
                text = tokenizer.decode(tokens[:required_len])
                
                # Evaluate
                metrics = evaluate_single_stream_pair(
                    dual_model=dual_model,
                    vanilla_model=vanilla_model,
                    tokenizer=tokenizer,
                    text=text,
                    context_len=context_len,
                    eval_len=eval_len,
                    embedding_scale=embedding_scale,
                    device=device
                )
                
                if metrics is not None:
                    results.append(metrics)
                    pbar.update(1)
                    
            except StopIteration:
                print("Dataset exhausted before reaching num_samples")
                break
    
    if len(results) == 0:
        print("No valid results collected!")
        return
    
    # Aggregate statistics
    avg_metrics = {
        'dual_perplexity': np.mean([r['dual_perplexity'] for r in results]),
        'vanilla_perplexity': np.mean([r['vanilla_perplexity'] for r in results]),
        'dual_accuracy': np.mean([r['dual_accuracy'] for r in results]),
        'vanilla_accuracy': np.mean([r['vanilla_accuracy'] for r in results]),
        'dual_matches_vanilla': np.mean([r['dual_matches_vanilla'] for r in results]),
        'dual_avg_rank': np.mean([r['dual_avg_rank'] for r in results]),
        'vanilla_avg_rank': np.mean([r['vanilla_avg_rank'] for r in results]),
        'vanilla_top1_rank_in_dual': np.mean([r['vanilla_top1_rank_in_dual'] for r in results]),
    }
    
    std_metrics = {
        'dual_perplexity': np.std([r['dual_perplexity'] for r in results]),
        'vanilla_perplexity': np.std([r['vanilla_perplexity'] for r in results]),
        'dual_accuracy': np.std([r['dual_accuracy'] for r in results]),
        'vanilla_accuracy': np.std([r['vanilla_accuracy'] for r in results]),
        'dual_matches_vanilla': np.std([r['dual_matches_vanilla'] for r in results]),
        'dual_avg_rank': np.std([r['dual_avg_rank'] for r in results]),
        'vanilla_avg_rank': np.std([r['vanilla_avg_rank'] for r in results]),
        'vanilla_top1_rank_in_dual': np.std([r['vanilla_top1_rank_in_dual'] for r in results]),
    }
    
    median_metrics = {
        'dual_perplexity': np.median([r['dual_perplexity'] for r in results]),
        'vanilla_perplexity': np.median([r['vanilla_perplexity'] for r in results]),
        'dual_accuracy': np.median([r['dual_accuracy'] for r in results]),
        'vanilla_accuracy': np.median([r['vanilla_accuracy'] for r in results]),
        'dual_matches_vanilla': np.median([r['dual_matches_vanilla'] for r in results]),
        'dual_avg_rank': np.median([r['dual_avg_rank'] for r in results]),
        'vanilla_avg_rank': np.median([r['vanilla_avg_rank'] for r in results]),
        'vanilla_top1_rank_in_dual': np.median([r['vanilla_top1_rank_in_dual'] for r in results]),
    }
    
    # Print results
    print(f"\n{'='*80}")
    print(f"SINGLE-STREAM EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Number of evaluated samples: {len(results)}")
    print(f"\n{'-'*80}")
    print(f"PERPLEXITY (lower is better):")
    print(f"  Dual model:    mean={avg_metrics['dual_perplexity']:.2f} ± {std_metrics['dual_perplexity']:.2f}, median={median_metrics['dual_perplexity']:.2f}")
    print(f"  Vanilla model: mean={avg_metrics['vanilla_perplexity']:.2f} ± {std_metrics['vanilla_perplexity']:.2f}, median={median_metrics['vanilla_perplexity']:.2f}")
    print(f"  Ratio (dual/vanilla): mean={avg_metrics['dual_perplexity']/avg_metrics['vanilla_perplexity']:.3f}, median={median_metrics['dual_perplexity']/median_metrics['vanilla_perplexity']:.3f}")
    print(f"\n{'-'*80}")
    print(f"TOKEN ACCURACY (higher is better):")
    print(f"  Dual model:    mean={avg_metrics['dual_accuracy']*100:.2f}% ± {std_metrics['dual_accuracy']*100:.2f}%, median={median_metrics['dual_accuracy']*100:.2f}%")
    print(f"  Vanilla model: mean={avg_metrics['vanilla_accuracy']*100:.2f}% ± {std_metrics['vanilla_accuracy']*100:.2f}%, median={median_metrics['vanilla_accuracy']*100:.2f}%")
    print(f"\n{'-'*80}")
    print(f"DUAL-VANILLA AGREEMENT (how often argmax matches):")
    print(f"  mean={avg_metrics['dual_matches_vanilla']*100:.2f}% ± {std_metrics['dual_matches_vanilla']*100:.2f}%, median={median_metrics['dual_matches_vanilla']*100:.2f}%")
    print(f"\n{'-'*80}")
    print(f"TRUE TOKEN RANK (lower is better):")
    print(f"  Dual model:    mean={avg_metrics['dual_avg_rank']:.2f} ± {std_metrics['dual_avg_rank']:.2f}, median={median_metrics['dual_avg_rank']:.2f}")
    print(f"  Vanilla model: mean={avg_metrics['vanilla_avg_rank']:.2f} ± {std_metrics['vanilla_avg_rank']:.2f}, median={median_metrics['vanilla_avg_rank']:.2f}")
    print(f"\n{'-'*80}")
    print(f"VANILLA TOP-1 RANK IN DUAL (lower is better):")
    print(f"  mean={avg_metrics['vanilla_top1_rank_in_dual']:.2f} ± {std_metrics['vanilla_top1_rank_in_dual']:.2f}, median={median_metrics['vanilla_top1_rank_in_dual']:.2f}")
    print(f"{'='*80}\n")
    
    if output_json_path:
        results_data = {
            "avg": avg_metrics,
            "std": std_metrics,
            "median": median_metrics,
            "config": {
                "checkpoint": checkpoint_path,
                "model": model_name,
                "dataset": dataset_name,
                "separator": separator_model_name if use_separator_model else None
            }
        }
        with open(output_json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"\nJSON results saved to: {output_json_path}")
    
    return avg_metrics, std_metrics, median_metrics


if __name__ == '__main__':
    fire.Fire({
        'evaluate': evaluate,
        'evaluate_single_stream': evaluate_single_stream
    })

