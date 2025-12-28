import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict


def generate_samples(
    model,
    tokenizer,
    prompts: List[Tuple[str, str]],
    device,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    objective: str = 'separate'
) -> List[Dict[str, str]]:
    """Generate text samples from the model.
    
    Args:
        model: The dual-stream model
        tokenizer: Tokenizer for encoding/decoding text
        prompts: List of (prompt_A, prompt_B) tuples
        device: Device to run generation on
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        objective: Training objective ('separate', 'averaged', 'mixup', or 'distillation')
    
    Returns:
        List of dicts with 'prompt_A', 'prompt_B', 'generated_A', 'generated_B'
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt_A, prompt_B in prompts:
            inputs_A = tokenizer(prompt_A, return_tensors='pt', padding=True).to(device)
            inputs_B = tokenizer(prompt_B, return_tensors='pt', padding=True).to(device)
            
            input_ids_A = inputs_A['input_ids']
            attention_mask_A = inputs_A['attention_mask']
            input_ids_B = inputs_B['input_ids']
            attention_mask_B = inputs_B['attention_mask']
            
            if objective == 'separate' or objective == 'separate_distillation':
                # For separate objective, generate from both streams simultaneously
                max_len = max(input_ids_A.size(1), input_ids_B.size(1))
                if input_ids_A.size(1) < max_len:
                    pad_len = max_len - input_ids_A.size(1)
                    input_ids_A = torch.cat([input_ids_A, torch.full((1, pad_len), tokenizer.pad_token_id, device=device)], dim=1)
                    attention_mask_A = torch.cat([attention_mask_A, torch.zeros((1, pad_len), device=device)], dim=1)
                if input_ids_B.size(1) < max_len:
                    pad_len = max_len - input_ids_B.size(1)
                    input_ids_B = torch.cat([input_ids_B, torch.full((1, pad_len), tokenizer.pad_token_id, device=device)], dim=1)
                    attention_mask_B = torch.cat([attention_mask_B, torch.zeros((1, pad_len), device=device)], dim=1)
                
                for _ in range(max_new_tokens):
                    logits_A, logits_B = model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B)
                    
                    logits_A_scaled = logits_A[:, -1, :] / temperature
                    logits_B_scaled = logits_B[:, -1, :] / temperature
                    
                    probs_A = F.softmax(logits_A_scaled, dim=-1)
                    probs_B = F.softmax(logits_B_scaled, dim=-1)
                    
                    next_token_A = torch.multinomial(probs_A, num_samples=1)
                    next_token_B = torch.multinomial(probs_B, num_samples=1)
                    
                    input_ids_A = torch.cat([input_ids_A, next_token_A], dim=1)
                    input_ids_B = torch.cat([input_ids_B, next_token_B], dim=1)
                    attention_mask_A = torch.cat([attention_mask_A, torch.ones((1, 1), device=device)], dim=1)
                    attention_mask_B = torch.cat([attention_mask_B, torch.ones((1, 1), device=device)], dim=1)
                    
                    if next_token_A.item() == tokenizer.eos_token_id and next_token_B.item() == tokenizer.eos_token_id:
                        break
                
                generated_A = tokenizer.decode(input_ids_A[0], skip_special_tokens=True)
                generated_B = tokenizer.decode(input_ids_B[0], skip_special_tokens=True)
            
            elif objective in ['averaged', 'mixup', 'distillation']:
                # For averaged/mixup/distillation objective, generate from each stream independently using pure inputs
                # Stream A: use lambda=1 (pure A)
                for _ in range(max_new_tokens):
                    # For pure A generation, pass same input for both streams (or use lambda=1)
                    lambdas = torch.ones(1, device=device) if objective == 'mixup' else None
                    logits = model(input_ids_A, attention_mask_A, input_ids_A, attention_mask_A, lambdas=lambdas)
                    
                    logits_scaled = logits[:, -1, :] / temperature
                    probs = F.softmax(logits_scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    input_ids_A = torch.cat([input_ids_A, next_token], dim=1)
                    attention_mask_A = torch.cat([attention_mask_A, torch.ones((1, 1), device=device)], dim=1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                # Stream B: use lambda=0 (pure B) 
                for _ in range(max_new_tokens):
                    lambdas = torch.zeros(1, device=device) if objective == 'mixup' else None
                    logits = model(input_ids_B, attention_mask_B, input_ids_B, attention_mask_B, lambdas=lambdas)
                    
                    logits_scaled = logits[:, -1, :] / temperature
                    probs = F.softmax(logits_scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    input_ids_B = torch.cat([input_ids_B, next_token], dim=1)
                    attention_mask_B = torch.cat([attention_mask_B, torch.ones((1, 1), device=device)], dim=1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                generated_A = tokenizer.decode(input_ids_A[0], skip_special_tokens=True)
                generated_B = tokenizer.decode(input_ids_B[0], skip_special_tokens=True)
            
            results.append({
                'prompt_A': prompt_A,
                'prompt_B': prompt_B,
                'generated_A': generated_A,
                'generated_B': generated_B
            })
    
    model.train()
    return results

