"""
Evaluation script to measure how well the model can separate mixed embeddings.

For each model (baseline and fine-tuned), we:
1. Take pairs of texts from the dataset
2. Get embeddings for each text
3. Mix them 50:50
4. Pass through the model
5. Measure the rank of the correct next token from text A in the predicted distribution
6. Compute average rank across the dataset

Lower rank = better separation ability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from model import DualStreamLLM
import numpy as np
from tqdm import tqdm
import json
import os
import argparse
from collections import defaultdict


def compute_token_ranks(logits, target_tokens, pad_token_id):
    """
    Compute the rank of target tokens in the predicted distribution.
    
    Args:
        logits: [batch_size, seq_len, vocab_size]
        target_tokens: [batch_size, seq_len] - target token IDs
        pad_token_id: ID of padding token to ignore
        
    Returns:
        ranks: list of ranks for each valid position
        reciprocal_ranks: list of reciprocal ranks (1/rank) for MRR
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Convert logits to probabilities (or just use logits for ranking)
    # We'll use logits directly since rank is all we care about
    
    ranks = []
    reciprocal_ranks = []
    
    for b in range(batch_size):
        for t in range(seq_len):
            target_token = target_tokens[b, t].item()
            
            # Skip padding tokens
            if target_token == pad_token_id or target_token == -100:
                continue
            
            # Get logits for this position
            position_logits = logits[b, t]  # [vocab_size]
            
            # Get score of target token
            target_score = position_logits[target_token].item()
            
            # Count how many tokens have higher score (rank = 1 means highest)
            rank = (position_logits > target_score).sum().item() + 1
            
            ranks.append(rank)
            reciprocal_ranks.append(1.0 / rank)
    
    return ranks, reciprocal_ranks


def evaluate_model(model, tokenizer, dataset, num_samples=1000, max_length=512, device='cuda', batch_size=8):
    """
    Evaluate a model's ability to separate mixed texts.
    
    Returns:
        metrics: dict with mean_rank, median_rank, mrr, top1_acc, top5_acc, top10_acc
    """
    model.eval()
    
    all_ranks = []
    all_reciprocal_ranks = []
    
    # Prepare data iterator
    data_iter = iter(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Evaluating"):
            # Collect batch
            texts_A = []
            texts_B = []
            
            for _ in range(batch_size):
                try:
                    texts_A.append(next(data_iter)['text'])
                    texts_B.append(next(data_iter)['text'])
                except StopIteration:
                    break
            
            if len(texts_A) == 0:
                break
            
            # Tokenize
            encoded_A = tokenizer(
                texts_A,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            encoded_B = tokenizer(
                texts_B,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids_A = encoded_A['input_ids'].to(device)
            attention_mask_A = encoded_A['attention_mask'].to(device)
            input_ids_B = encoded_B['input_ids'].to(device)
            attention_mask_B = encoded_B['attention_mask'].to(device)
            
            # Create target: next token from text A
            # We want to predict input_ids_A[:, 1:] from input_ids_A[:, :-1]
            next_tokens_A = input_ids_A[:, 1:].clone()
            
            # Pass mixed embeddings (50:50) through model
            # For mixup objective, lambda=0.5 means equal mix
            lambdas = torch.full((input_ids_A.size(0),), 0.5, device=device)
            
            # Forward pass
            try:
                logits = model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, lambdas=lambdas)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                continue
            
            # Use logits[:, :-1, :] to align with next_tokens_A
            logits_shifted = logits[:, :-1, :].float()
            
            # Compute ranks
            ranks, reciprocal_ranks = compute_token_ranks(
                logits_shifted, 
                next_tokens_A, 
                tokenizer.pad_token_id
            )
            
            all_ranks.extend(ranks)
            all_reciprocal_ranks.extend(reciprocal_ranks)
            
            # Stop if we've collected enough samples
            if len(all_ranks) >= num_samples * max_length:
                break
    
    # Compute metrics
    all_ranks = np.array(all_ranks)
    all_reciprocal_ranks = np.array(all_reciprocal_ranks)
    
    metrics = {
        'mean_rank': float(np.mean(all_ranks)),
        'median_rank': float(np.median(all_ranks)),
        'std_rank': float(np.std(all_ranks)),
        'mrr': float(np.mean(all_reciprocal_ranks)),  # Mean Reciprocal Rank
        'top1_acc': float(np.mean(all_ranks == 1)),  # Accuracy@1
        'top5_acc': float(np.mean(all_ranks <= 5)),  # Accuracy@5
        'top10_acc': float(np.mean(all_ranks <= 10)),  # Accuracy@10
        'top50_acc': float(np.mean(all_ranks <= 50)),  # Accuracy@50
        'num_samples': len(all_ranks)
    }
    
    return metrics


def load_model_from_checkpoint(checkpoint_path, config, device='cuda'):
    """Load a DualStreamLLM model from checkpoint."""
    
    # Create model
    model = DualStreamLLM(
        model_name=config['model_name'],
        vocab_size=50257,  # Will be overridden by actual vocab size
        hidden_dim=576,     # SmolLM2-135M hidden size
        freeze_backbone=True,  # For evaluation, we don't need gradients
        embedding_init_strategy=config.get('embedding_init_strategy', 'copy'),
        unembedding_init_strategy=config.get('unembedding_init_strategy', 'copy'),
        embedding_init_noise_std=config.get('embedding_init_noise_std', 0.01),
        objective=config['objective'],
        shared_embeddings=config['shared_embeddings']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model


def load_base_model(model_name, objective='mixup', shared_embeddings=True, device='cuda'):
    """Load the base model without fine-tuning."""
    
    model = DualStreamLLM(
        model_name=model_name,
        vocab_size=50257,
        hidden_dim=576,
        freeze_backbone=True,
        embedding_init_strategy='copy',
        unembedding_init_strategy='copy',
        embedding_init_noise_std=0.0,  # No noise for base model
        objective=objective,
        shared_embeddings=shared_embeddings
    )
    
    model = model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate model separation ability')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Base directory containing checkpoint runs')
    parser.add_argument('--runs', type=str, nargs='+', 
                        default=['run_20251028_182407', 'run_20251030_145128', 'run_20251031_181041'],
                        help='List of run directories to evaluate')
    parser.add_argument('--checkpoint_steps', type=int, nargs='+',
                        default=[40000, 80000],
                        help='Checkpoint steps to evaluate (e.g., 40000 80000)')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of text pairs to evaluate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for evaluation')
    parser.add_argument('--output_file', type=str, default='separation_eval_results.json',
                        help='Output file for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for dataset sampling')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    
    # Load tokenizer (same for all models)
    model_name = 'HuggingFaceTB/SmolLM2-135M'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset('roneneldan/TinyStories', split='train', streaming=True, trust_remote_code=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=10000)
    
    results = {}
    
    # Evaluate base model
    print("\n" + "="*80)
    print("Evaluating BASE MODEL (no fine-tuning)")
    print("="*80)
    
    base_model = load_base_model(model_name, objective='mixup', shared_embeddings=True, device=device)
    
    # Reset dataset iterator for each model
    dataset_iter = dataset.shuffle(seed=args.seed, buffer_size=10000)
    
    base_metrics = evaluate_model(
        base_model, 
        tokenizer, 
        dataset_iter, 
        num_samples=args.num_samples,
        max_length=args.max_length,
        device=device,
        batch_size=args.batch_size
    )
    
    results['base_model'] = {
        'config': {
            'model_name': model_name,
            'fine_tuned': False
        },
        'metrics': base_metrics
    }
    
    print(f"\nBase Model Results:")
    print(f"  Mean Rank: {base_metrics['mean_rank']:.2f}")
    print(f"  Median Rank: {base_metrics['median_rank']:.2f}")
    print(f"  MRR: {base_metrics['mrr']:.4f}")
    print(f"  Top-1 Accuracy: {base_metrics['top1_acc']*100:.2f}%")
    print(f"  Top-5 Accuracy: {base_metrics['top5_acc']*100:.2f}%")
    print(f"  Top-10 Accuracy: {base_metrics['top10_acc']*100:.2f}%")
    
    # Clean up
    del base_model
    torch.cuda.empty_cache()
    
    # Evaluate fine-tuned models
    for run_name in args.runs:
        run_dir = os.path.join(args.checkpoint_dir, run_name)
        
        if not os.path.exists(run_dir):
            print(f"\nWarning: Run directory not found: {run_dir}")
            continue
        
        # Load config
        config_path = os.path.join(run_dir, 'training_config.json')
        if not os.path.exists(config_path):
            print(f"\nWarning: Config not found: {config_path}")
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"\n" + "="*80)
        print(f"Evaluating {run_name}")
        print(f"Config: mixup_alpha={config.get('mixup_alpha', 'N/A')}")
        print("="*80)
        
        results[run_name] = {
            'config': config,
            'checkpoints': {}
        }
        
        # Evaluate each checkpoint
        for step in args.checkpoint_steps:
            checkpoint_dir = os.path.join(run_dir, f'checkpoint-{step}')
            checkpoint_path = os.path.join(checkpoint_dir, 'model.pt')
            
            if not os.path.exists(checkpoint_path):
                print(f"\n  Warning: Checkpoint not found: {checkpoint_path}")
                continue
            
            print(f"\n  Checkpoint: step {step}")
            print(f"  Loading model from {checkpoint_path}")
            
            try:
                model = load_model_from_checkpoint(checkpoint_path, config, device=device)
                
                # Reset dataset iterator
                dataset_iter = dataset.shuffle(seed=args.seed, buffer_size=10000)
                
                metrics = evaluate_model(
                    model,
                    tokenizer,
                    dataset_iter,
                    num_samples=args.num_samples,
                    max_length=args.max_length,
                    device=device,
                    batch_size=args.batch_size
                )
                
                results[run_name]['checkpoints'][f'step_{step}'] = metrics
                
                print(f"\n  Results at step {step}:")
                print(f"    Mean Rank: {metrics['mean_rank']:.2f}")
                print(f"    Median Rank: {metrics['median_rank']:.2f}")
                print(f"    MRR: {metrics['mrr']:.4f}")
                print(f"    Top-1 Accuracy: {metrics['top1_acc']*100:.2f}%")
                print(f"    Top-5 Accuracy: {metrics['top5_acc']*100:.2f}%")
                print(f"    Top-10 Accuracy: {metrics['top10_acc']*100:.2f}%")
                
                # Comparison with base model
                rank_improvement = base_metrics['mean_rank'] - metrics['mean_rank']
                mrr_improvement = metrics['mrr'] - base_metrics['mrr']
                
                print(f"\n  Improvement vs Base:")
                print(f"    Mean Rank: {rank_improvement:+.2f} (lower is better)")
                print(f"    MRR: {mrr_improvement:+.4f} (higher is better)")
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error evaluating checkpoint: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results
    print(f"\n" + "="*80)
    print(f"Saving results to {args.output_file}")
    print("="*80)
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nBase Model:")
    print(f"  Mean Rank: {results['base_model']['metrics']['mean_rank']:.2f}")
    print(f"  MRR: {results['base_model']['metrics']['mrr']:.4f}")
    
    for run_name in args.runs:
        if run_name not in results or 'checkpoints' not in results[run_name]:
            continue
        
        mixup_alpha = results[run_name]['config'].get('mixup_alpha', 'N/A')
        print(f"\n{run_name} (mixup_alpha={mixup_alpha}):")
        
        for checkpoint_name, metrics in sorted(results[run_name]['checkpoints'].items()):
            step = checkpoint_name.replace('step_', '')
            mean_rank = metrics['mean_rank']
            mrr = metrics['mrr']
            improvement = results['base_model']['metrics']['mean_rank'] - mean_rank
            
            print(f"  Step {step}: Mean Rank = {mean_rank:.2f} (Δ{improvement:+.2f}), MRR = {mrr:.4f}")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == '__main__':
    main()

