# --- START OF FILE run_logit_extraction.py ---

import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

from model_utils import (
    load_model_and_tokenizer,
    tokenize_and_align_pair,
    find_transformer_blocks,
    layer_indices_for_percentages,
    get_logits_from_embeds,
    get_logits_with_layer_override,
    get_probs_from_ids_with_layer_inputs,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract raw logits for Top-K tokens based on single prompts (A and B separately)")
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.2-3B", help="Model identifier")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb", help="Dataset path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--num_pairs", type=int, default=100, help="Number of text pairs to process")
    parser.add_argument("--seq_len", type=int, default=64, help="Fixed sequence length for analysis")
    parser.add_argument("--topk", type=int, default=5, help="Number of tokens to keep per position")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mix-depth-percent", type=int, default=0, help="0 = input embeddings, >0 = layer output")
    parser.add_argument("--mix-mode", type=str, default="sum", choices=["sum", "avg"])
    parser.add_argument("--save_dir", type=str, default="./results_logits")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    return parser.parse_args()

def get_text_pairs(ds, num_pairs, seed, seq_len):
    import random
    random.seed(seed)
    
    candidates = []
    iterator = iter(ds)
    
    print("Extracting candidates...")
    pbar = tqdm(total=num_pairs * 2)
    while len(candidates) < num_pairs * 2:
        try:
            row = next(iterator)
            text = row.get("text", "")
            if not text or len(text) < seq_len * 4: # Heuristic length check
                continue
            candidates.append(text)
            pbar.update(1)
        except StopIteration:
            break
    pbar.close()
            
    random.shuffle(candidates)
    
    pairs = []
    for i in range(0, len(candidates) - 1, 2):
        if len(pairs) >= num_pairs:
            break
        pairs.append((candidates[i], candidates[i+1]))
        
    return pairs

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Loading model: {args.model}")
    tokenizer, model = load_model_and_tokenizer(args.model, dtype=args.dtype)
    
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    pairs = get_text_pairs(ds, args.num_pairs, args.seed, args.seq_len)
    
    if not pairs:
        print("No pairs found.")
        return

    collected_data_A_centric = []
    collected_data_B_centric = []
    device = next(model.parameters()).device
    
    print(f"Processing {len(pairs)} pairs...")
    
    for text_a, text_b in tqdm(pairs):
        pair_data = tokenize_and_align_pair(tokenizer, text_a, text_b, truncate="min", max_length=args.seq_len)
        
        if pair_data.T < args.seq_len:
            continue
            
        input_ids_a = pair_data.input_ids_a[:, :args.seq_len].to(device)
        input_ids_b = pair_data.input_ids_b[:, :args.seq_len].to(device)
        attn_mask = torch.ones((1, args.seq_len), device=device, dtype=torch.long)
        
        with torch.no_grad():
            logits_a = model(input_ids=input_ids_a, attention_mask=attn_mask).logits
            logits_b = model(input_ids=input_ids_b, attention_mask=attn_mask).logits
            
            if args.mix_depth_percent == 0:
                emb = model.get_input_embeddings()
                e_a, e_b = emb(input_ids_a), emb(input_ids_b)
                e_mix = 0.5 * (e_a + e_b) if args.mix_mode == "avg" else (e_a + e_b)
                logits_mix = get_logits_from_embeds(model, e_mix, attn_mask, sqrt2_rmsnorm=False)
            else:
                blocks = find_transformer_blocks(model)
                layer_idx = layer_indices_for_percentages(len(blocks), [args.mix_depth_percent])[0]
                _, inputs_a = get_probs_from_ids_with_layer_inputs(model, input_ids_a, attn_mask, [layer_idx])
                _, inputs_b = get_probs_from_ids_with_layer_inputs(model, input_ids_b, attn_mask, [layer_idx])
                h_a, h_b = inputs_a[layer_idx], inputs_b[layer_idx]
                h_mix = 0.5 * (h_a + h_b) if args.mix_mode == "avg" else (h_a + h_b)
                logits_mix = get_logits_with_layer_override(model, input_ids_a, attn_mask, layer_idx, h_mix)

        L_a, L_b, L_m = logits_a[0], logits_b[0], logits_mix[0]
        P_a, P_b, P_m = F.softmax(L_a, -1), F.softmax(L_b, -1), F.softmax(L_m, -1)
        
        # --- A-centric Extraction ---
        _, indices_a = torch.topk(P_a, k=args.topk, dim=-1)
        
        def g(tensor, idx): return torch.gather(tensor, 1, idx)
            
        la_A, lm_A = g(L_a, indices_a), g(L_m, indices_a)
        pa_A, pm_A = g(P_a, indices_a), g(P_m, indices_a)
        stacked_A = torch.stack([la_A, lm_A, pa_A, pm_A], dim=-1)
        collected_data_A_centric.append(stacked_A.unsqueeze(0).cpu())

        # --- B-centric Extraction ---
        _, indices_b = torch.topk(P_b, k=args.topk, dim=-1)
        lb_B, lm_B = g(L_b, indices_b), g(L_m, indices_b)
        pb_B, pm_B = g(P_b, indices_b), g(P_m, indices_b)
        stacked_B = torch.stack([lb_B, lm_B, pb_B, pm_B], dim=-1)
        collected_data_B_centric.append(stacked_B.unsqueeze(0).cpu())


    if not collected_data_A_centric:
        print("No valid sequences collected.")
        return

    # Create final dictionary of tensors
    final_data = {
        'A_centric': torch.cat(collected_data_A_centric, dim=0),
        'B_centric': torch.cat(collected_data_B_centric, dim=0)
    }
    
    num_collected = final_data['A_centric'].shape[0]
    filename = f"logits_split_topk_N{num_collected}_T{args.seq_len}_K{args.topk}_{args.model.replace('/', '_')}.pt"
    path = os.path.join(args.save_dir, filename)
    
    print(f"Saving dictionary of tensors to {path}...")
    torch.save(final_data, path)
    print("Done.")

if __name__ == "__main__":
    main()