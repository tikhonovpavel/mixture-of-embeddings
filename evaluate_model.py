import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

import fire
import json
import os
from tqdm import tqdm
import evaluate as hf_evaluate  # Using Hugging Face's evaluate library

from model import DualStreamLLM


def generate_from_dual_stream_separate(model, tokenizer, prompt_A, prompt_B, device, max_new_tokens=50, temperature=0.7):
    """Generates text from both streams (A and B) based on two different initial prompts."""
    model.eval()

    input_ids_A = tokenizer(prompt_A, return_tensors='pt').input_ids.to(device)
    input_ids_B = tokenizer(prompt_B, return_tensors='pt').input_ids.to(device)
    
    finished_A = False
    finished_B = False
    
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if finished_A and finished_B:
                break
            
            # Pad the shorter sequence to match the longer one for this forward pass
            len_A, len_B = input_ids_A.size(1), input_ids_B.size(1)
            if len_A != len_B:
                max_len = max(len_A, len_B)
                padding_A = torch.full((1, max_len - len_A), pad_token_id, device=device, dtype=torch.long)
                padded_input_A = torch.cat([input_ids_A, padding_A], dim=1)
                
                padding_B = torch.full((1, max_len - len_B), pad_token_id, device=device, dtype=torch.long)
                padded_input_B = torch.cat([input_ids_B, padding_B], dim=1)
                
                attention_mask_A = torch.cat([torch.ones_like(input_ids_A), torch.zeros_like(padding_A)], dim=1)
                attention_mask_B = torch.cat([torch.ones_like(input_ids_B), torch.zeros_like(padding_B)], dim=1)
            else:
                padded_input_A, padded_input_B = input_ids_A, input_ids_B
                attention_mask_A, attention_mask_B = torch.ones_like(input_ids_A), torch.ones_like(input_ids_B)


            logits_A, logits_B = model(padded_input_A, attention_mask_A, padded_input_B, attention_mask_B)

            # --- Process Stream A ---
            if not finished_A:
                # Get logits for the actual last token of sequence A
                logits_A_last = logits_A[:, len_A - 1, :]
                if temperature > 0:
                    probs_A = F.softmax(logits_A_last / temperature, dim=-1)
                    next_token_A = torch.multinomial(probs_A, num_samples=1)
                else:
                    next_token_A = torch.argmax(logits_A_last, dim=-1, keepdim=True)
                
                input_ids_A = torch.cat([input_ids_A, next_token_A], dim=1)

                if next_token_A.item() == eos_token_id:
                    finished_A = True
            
            # --- Process Stream B ---
            if not finished_B:
                # Get logits for the actual last token of sequence B
                logits_B_last = logits_B[:, len_B - 1, :]
                if temperature > 0:
                    probs_B = F.softmax(logits_B_last / temperature, dim=-1)
                    next_token_B = torch.multinomial(probs_B, num_samples=1)
                else:
                    next_token_B = torch.argmax(logits_B_last, dim=-1, keepdim=True)

                input_ids_B = torch.cat([input_ids_B, next_token_B], dim=1)

                if next_token_B.item() == eos_token_id:
                    finished_B = True
            
    return (
        tokenizer.decode(input_ids_A[0], skip_special_tokens=True),
        tokenizer.decode(input_ids_B[0], skip_special_tokens=True)
    )


def generate_from_original(model, tokenizer, prompt, device, max_new_tokens=50, temperature=0.7):
    """Generate text from the original Hugging Face model."""
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_k"] = 50
            generation_kwargs["top_p"] = 0.95
        
        output_ids = model.generate(**inputs, **generation_kwargs)
        
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def calculate_original_model_loss(model, tokenizer, text, device, max_length=512):
    """Calculate the cross-entropy loss of the original model on a given text."""
    model.eval()
    
    inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    labels = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
    return loss.item()


def evaluate(
    model_name: str,
    checkpoint_path: str,
    output_file: str,
    dataset_name: str = 'allenai/c4',
    dataset_config: str = 'en',
    dataset_split: str = 'validation',
    num_samples: int = 100,
    prompt_tokens: int = 20,
    max_gen_tokens: int = 50,
    max_length: int = 512,
    temperature: float = 0.7,
    device: str = 'cuda:0',
):
    """
    Evaluate a trained dual-stream model and compare it with the original backbone model.

    Args:
        model_name (str): The name of the base model (e.g., 'unsloth/Llama-3.2-1B').
        checkpoint_path (str): Path to the .pt checkpoint file for the dual-stream model.
        output_file (str): Path to the output .jsonl file to save results incrementally.
        dataset_name (str): Name of the Hugging Face dataset to use for evaluation.
        dataset_config (str): Configuration for the dataset (e.g., 'en' for C4).
        dataset_split (str): The dataset split to use (e.g., 'validation', 'train').
        num_samples (int): The number of samples from the dataset to evaluate.
        prompt_tokens (int): The number of initial tokens from each sample to use as a prompt.
        max_gen_tokens (int): The maximum number of new tokens to generate.
        max_length (int): Maximum sequence length for calculating the baseline loss.
        temperature (float): Sampling temperature for generation (0 for greedy decoding).
        device (str): The device to run the models on (e.g., 'cuda:0').
    """
    print("Starting evaluation...")
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load original model
    print(f"Loading original model: {model_name}")
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16
    ).to(device)
    original_model.eval()

    # Load dual-stream model
    print(f"Loading dual-stream model from checkpoint: {checkpoint_path}")
    base_model_config = AutoConfig.from_pretrained(model_name)
    vocab_size = base_model_config.vocab_size
    hidden_dim = base_model_config.hidden_size
    
    # We need the full definition of DualStreamLLM to load it
    dual_stream_model = DualStreamLLM(
        model_name=model_name,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        freeze_backbone=True # This doesn't matter for evaluation
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    dual_stream_model.load_state_dict(checkpoint['model_state_dict'])
    dual_stream_model = dual_stream_model.to(device)
    dual_stream_model.eval()
    
    # Load metrics
    print("Loading evaluation metrics: ROUGE, BLEU, BERT-Score")
    rouge = hf_evaluate.load('rouge')
    bleu = hf_evaluate.load('bleu')
    bertscore = hf_evaluate.load('bertscore')

    # Load dataset
    print(f"Loading dataset: {dataset_name} ({dataset_config}), split: {dataset_split}")
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True)
    
    # Ensure output directory exists
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Starting evaluation loop for {num_samples} samples...")
    data_iterator = iter(dataset)
    with open(output_file, 'w') as f:
        for i in tqdm(range(num_samples), total=num_samples, desc="Evaluating samples"):
            try:
                # Fetch two separate samples for A and B
                sample_A = next(data_iterator)
                sample_B = next(data_iterator)
            except StopIteration:
                print("Dataset iterator exhausted.")
                break

            # --- Prepare Prompt and Reference for Stream A ---
            text_A = sample_A['text']
            tokenized_A = tokenizer.encode(text_A, return_tensors='pt')[0]
            if len(tokenized_A) < prompt_tokens + 5: continue
            prompt_A = tokenizer.decode(tokenized_A[:prompt_tokens], skip_special_tokens=True)
            reference_A = tokenizer.decode(tokenized_A[prompt_tokens:], skip_special_tokens=True)

            # --- Prepare Prompt and Reference for Stream B ---
            text_B = sample_B['text']
            tokenized_B = tokenizer.encode(text_B, return_tensors='pt')[0]
            if len(tokenized_B) < prompt_tokens + 5: continue
            prompt_B = tokenizer.decode(tokenized_B[:prompt_tokens], skip_special_tokens=True)
            reference_B = tokenizer.decode(tokenized_B[prompt_tokens:], skip_special_tokens=True)

            # Generate from both models
            generated_dual_A, generated_dual_B = generate_from_dual_stream_separate(dual_stream_model, tokenizer, prompt_A, prompt_B, device, max_gen_tokens, temperature)
            
            # For a fair comparison, generate from the original model on both prompts
            generated_orig_A = generate_from_original(original_model, tokenizer, prompt_A, device, max_gen_tokens, temperature)
            generated_orig_B = generate_from_original(original_model, tokenizer, prompt_B, device, max_gen_tokens, temperature)
            
            # Clean up generations
            if generated_dual_A.startswith(prompt_A): generated_dual_A = generated_dual_A[len(prompt_A):].strip()
            if generated_dual_B.startswith(prompt_B): generated_dual_B = generated_dual_B[len(prompt_B):].strip()
            if generated_orig_A.startswith(prompt_A): generated_orig_A = generated_orig_A[len(prompt_A):].strip()
            if generated_orig_B.startswith(prompt_B): generated_orig_B = generated_orig_B[len(prompt_B):].strip()
                
            # --- Calculate metrics ---
            # Compare A stream against reference A
            rouge_dual_A = rouge.compute(predictions=[generated_dual_A], references=[reference_A])
            bleu_dual_A = bleu.compute(predictions=[generated_dual_A], references=[[reference_A]]) if generated_dual_A else {'bleu': 0.0}
            bertscore_dual_A = bertscore.compute(predictions=[generated_dual_A], references=[reference_A], lang='en', device=device)
            rouge_orig_A = rouge.compute(predictions=[generated_orig_A], references=[reference_A])
            bleu_orig_A = bleu.compute(predictions=[generated_orig_A], references=[[reference_A]]) if generated_orig_A else {'bleu': 0.0}
            bertscore_orig_A = bertscore.compute(predictions=[generated_orig_A], references=[reference_A], lang='en', device=device)

            # Compare B stream against reference B
            rouge_dual_B = rouge.compute(predictions=[generated_dual_B], references=[reference_B])
            bleu_dual_B = bleu.compute(predictions=[generated_dual_B], references=[[reference_B]]) if generated_dual_B else {'bleu': 0.0}
            bertscore_dual_B = bertscore.compute(predictions=[generated_dual_B], references=[reference_B], lang='en', device=device)
            rouge_orig_B = rouge.compute(predictions=[generated_orig_B], references=[reference_B])
            bleu_orig_B = bleu.compute(predictions=[generated_orig_B], references=[[reference_B]]) if generated_orig_B else {'bleu': 0.0}
            bertscore_orig_B = bertscore.compute(predictions=[generated_orig_B], references=[reference_B], lang='en', device=device)

            # Calculate baseline loss for original model on the full text
            loss_A = calculate_original_model_loss(original_model, tokenizer, text_A, device, max_length)
            loss_B = calculate_original_model_loss(original_model, tokenizer, text_B, device, max_length)

            # Prepare results and save incrementally
            result = {
                'sample_id': i,
                'prompt_A': prompt_A,
                'reference_A': reference_A,
                'prompt_B': prompt_B,
                'reference_B': reference_B,
                'generated_dual_stream_A': generated_dual_A,
                'generated_dual_stream_B': generated_dual_B,
                'generated_original_A': generated_orig_A,
                'generated_original_B': generated_orig_B,
                'metrics_dual_stream_A': {
                    'rouge': rouge_dual_A, 'bleu': bleu_dual_A,
                    'bert_score_f1': bertscore_dual_A['f1'][0],
                },
                'metrics_dual_stream_B': {
                    'rouge': rouge_dual_B, 'bleu': bleu_dual_B,
                    'bert_score_f1': bertscore_dual_B['f1'][0],
                },
                'metrics_original_A': {
                    'rouge': rouge_orig_A, 'bleu': bleu_orig_A,
                    'bert_score_f1': bertscore_orig_A['f1'][0],
                },
                'metrics_original_B': {
                     'rouge': rouge_orig_B, 'bleu': bleu_orig_B,
                    'bert_score_f1': bertscore_orig_B['f1'][0],
                },
                'original_model_loss_A': loss_A,
                'original_model_loss_B': loss_B
            }
            
            f.write(json.dumps(result) + '\n')
            f.flush()

    print(f"\nEvaluation finished. Results saved to {output_file}")


if __name__ == '__main__':
    fire.Fire(evaluate)
