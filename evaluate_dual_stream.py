import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import fire
import numpy as np
from tqdm import tqdm

from model import DualStreamLLM


def load_checkpoint(checkpoint_path, model_name, device):
    """Load trained DualStreamLLM from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with same configuration
    model = DualStreamLLM(
        model_name=model_name,
        freeze_backbone=True,
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
            mixed_logits = dual_model(
                input_ids_A=input_ids_a,
                attention_mask_A=attention_mask_a,
                input_ids_B=input_ids_b,
                attention_mask_B=attention_mask_b
            )
            
            mixed_logits_next = mixed_logits[:, -1, :]  # [1, vocab_size]
            
            if use_separator_model and separator_model is not None:
                # Use separator model to create masks
                sep_outputs_a = separator_model(current_context_a)
                sep_logits_a = sep_outputs_a.logits[:, -1, :]
                
                sep_outputs_b = separator_model(current_context_b)
                sep_logits_b = sep_outputs_b.logits[:, -1, :]
                
                # Create masks
                topk_a = torch.topk(sep_logits_a, k=top_k_mask, dim=-1)
                mask_a = torch.zeros_like(mixed_logits_next)
                mask_a.scatter_(1, topk_a.indices, 1.0)
                
                topk_b = torch.topk(sep_logits_b, k=top_k_mask, dim=-1)
                mask_b = torch.zeros_like(mixed_logits_next)
                mask_b.scatter_(1, topk_b.indices, 1.0)
                
                # Apply masks
                filtered_logits_a = mixed_logits_next.clone()
                filtered_logits_a[mask_a == 0] = -float('inf')
                
                filtered_logits_b = mixed_logits_next.clone()
                filtered_logits_b[mask_b == 0] = -float('inf')
            else:
                # Without separator model: use mixed logits directly
                filtered_logits_a = mixed_logits_next
                filtered_logits_b = mixed_logits_next
            
            # --- Metrics Calculation ---
            
            # 1. Check for filter misses (if separator is used)
            is_miss_a = use_separator_model and (filtered_logits_a[0, target_token_a].item() == -float('inf'))
            is_miss_b = use_separator_model and (filtered_logits_b[0, target_token_b].item() == -float('inf'))

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
                is_vanilla_miss_a = use_separator_model and (filtered_logits_a[0, vanilla_top_token_a].item() == -float('inf'))
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
                
                is_vanilla_miss_b = use_separator_model and (filtered_logits_b[0, vanilla_top_token_b].item() == -float('inf'))
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
    seed=42
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
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dual-stream model
    if checkpoint_path:
        dual_model = load_checkpoint(checkpoint_path, model_name, device)
    else:
        print("\nNo checkpoint provided. Loading vanilla pre-trained model into DualStreamLLM structure.")
        dual_model = DualStreamLLM(
            model_name=model_name,
            freeze_backbone=True,
            embedding_init_strategy='copy',
            objective='distillation',
            shared_embeddings=True
        ).to(device)
        dual_model.eval()
        print("Vanilla model loaded successfully.")

    
    # Load separator model if needed
    separator_model = None
    if use_separator_model:
        print(f"\nLoading separator model: {separator_model_name}")
        separator_model = AutoModelForCausalLM.from_pretrained(
            separator_model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        separator_model.eval()
        print("Separator model loaded successfully")
    else:
        print("\nRunning in BASELINE mode (without separator model)")
        
    # Load full vanilla model for comparison metric
    print(f"\nLoading full vanilla model for comparison: {model_name}")
    vanilla_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    vanilla_model.eval()
    print("Full vanilla model loaded successfully")
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split='train', streaming=True, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split='train', streaming=True, trust_remote_code=True)
    
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
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
    print(f"Number of samples: {num_samples}")
    print(f"Context length: {context_len}")
    print(f"Eval length: {eval_len}")
    print(f"Top-K mask: {top_k_mask if use_separator_model else 'N/A'}")
    print(f"{'='*80}\n")
    
    # Evaluate
    results = []
    dataset_iter = iter(dataset)
    
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
    
    # Print results
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Mode: {'WITH separator model' if use_separator_model else 'BASELINE'}")
    print(f"Number of evaluated pairs: {len(results)}")
    print(f"\n{'-'*80}")
    print(f"PERPLEXITY (on non-missed tokens):")
    print(f"  Text A: {avg_metrics['perplexity_a']:.2f} ± {std_metrics['perplexity_a']:.2f}")
    print(f"  Text B: {avg_metrics['perplexity_b']:.2f} ± {std_metrics['perplexity_b']:.2f}")
    print(f"  Average: {avg_metrics['avg_perplexity']:.2f} ± {std_metrics['avg_perplexity']:.2f}")
    print(f"\n{'-'*80}")
    print(f"FILTER MISS RATE (true token outside top-K mask):")
    print(f"  Text A: {avg_metrics['miss_rate_a']*100:.2f}%")
    print(f"  Text B: {avg_metrics['miss_rate_b']*100:.2f}%")
    print(f"  Average: {avg_metrics['avg_miss_rate']*100:.2f}%")
    print(f"\n{'-'*80}")
    print(f"TOKEN ACCURACY:")
    print(f"  Text A: {avg_metrics['accuracy_a']*100:.2f}% ± {std_metrics['accuracy_a']*100:.2f}%")
    print(f"  Text B: {avg_metrics['accuracy_b']*100:.2f}% ± {std_metrics['accuracy_b']*100:.2f}%")
    print(f"  Average: {avg_metrics['avg_accuracy']*100:.2f}% ± {std_metrics['avg_accuracy']*100:.2f}%")
    print(f"\n{'-'*80}")
    print(f"TRUE TOKEN RANK (on non-missed tokens):")
    print(f"  Text A: {avg_metrics['avg_rank_a']:.2f} ± {std_metrics['avg_rank_a']:.2f}")
    print(f"  Text B: {avg_metrics['avg_rank_b']:.2f} ± {std_metrics['avg_rank_b']:.2f}")
    print(f"  Average: {avg_metrics['avg_rank']:.2f} ± {std_metrics['avg_rank']:.2f}")
    print(f"\n{'-'*80}")
    print(f"--- METRICS AGAINST VANILLA MODEL'S PREDICTION ---")
    print(f"\nVANILLA TARGET MISS RATE (vanilla's top token outside our mask):")
    print(f"  Text A: {avg_metrics['vanilla_miss_rate_a']*100:.2f}%")
    print(f"  Text B: {avg_metrics['vanilla_miss_rate_b']*100:.2f}%")
    print(f"  Average: {avg_metrics['avg_vanilla_miss_rate']*100:.2f}%")
    print(f"\nVANILLA TARGET ACCURACY (our argmax == vanilla's argmax):")
    print(f"  Text A: {avg_metrics['vanilla_accuracy_a']*100:.2f}% ± {std_metrics['vanilla_accuracy_a']*100:.2f}%")
    print(f"  Text B: {avg_metrics['vanilla_accuracy_b']*100:.2f}% ± {std_metrics['vanilla_accuracy_b']*100:.2f}%")
    print(f"  Average: {avg_metrics['avg_vanilla_accuracy']*100:.2f}% ± {std_metrics['avg_vanilla_accuracy']*100:.2f}%")
    print(f"\nVANILLA TARGET PERPLEXITY (on non-missed vanilla targets):")
    print(f"  Text A: {avg_metrics['vanilla_perplexity_a']:.2f} ± {std_metrics['vanilla_perplexity_a']:.2f}")
    print(f"  Text B: {avg_metrics['vanilla_perplexity_b']:.2f} ± {std_metrics['vanilla_perplexity_b']:.2f}")
    print(f"\nVANILLA MODEL'S TOP TOKEN RANK (on non-missed vanilla targets):")
    print(f"  Text A: {avg_metrics['avg_vanilla_rank_a']:.2f} ± {std_metrics['avg_vanilla_rank_a']:.2f}")
    print(f"  Text B: {avg_metrics['avg_vanilla_rank_b']:.2f} ± {std_metrics['avg_vanilla_rank_b']:.2f}")
    print(f"  Average: {avg_metrics['avg_vanilla_rank']:.2f} ± {std_metrics['avg_vanilla_rank']:.2f}")
    print(f"{'='*80}\n")
    
    return avg_metrics, std_metrics


if __name__ == '__main__':
    fire.Fire(evaluate)

