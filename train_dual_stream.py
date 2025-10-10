import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import os
import functools
import fire
from datetime import datetime

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'


class DualStreamLLM(nn.Module):
    def __init__(self, model_name, vocab_size, hidden_dim, freeze_backbone=True):
        super().__init__()
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            use_cache=False
        )
        
        original_embed_weights = self.backbone.get_input_embeddings().weight.data.clone()
        original_lm_head_weights = self.backbone.get_output_embeddings().weight.data.clone()
        
        self.emb_A = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16)
        self.emb_B = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16)
        self.emb_A.weight.data.copy_(original_embed_weights)
        self.emb_B.weight.data.copy_(original_embed_weights)
        
        self.unemb_A = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16)
        self.unemb_B = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16)
        self.unemb_A.weight.data.copy_(original_lm_head_weights)
        self.unemb_B.weight.data.copy_(original_lm_head_weights + torch.randn_like(original_lm_head_weights) * 0.01)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B):
        embeds_A = self.emb_A(input_ids_A)
        embeds_B = self.emb_B(input_ids_B)
        
        fused_embeds = embeds_A + embeds_B
        
        attention_mask = attention_mask_A * attention_mask_B
        
        transformer_outputs = self.backbone.model(
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_hidden_state = transformer_outputs.last_hidden_state
        
        logits_A = self.unemb_A(last_hidden_state)
        logits_B = self.unemb_B(last_hidden_state)
        
        return logits_A, logits_B


class C4DualStreamCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        texts_A = [item['text'] for item in batch[::2]]
        texts_B = [item['text'] for item in batch[1::2]]
        
        min_len = min(len(texts_A), len(texts_B))
        texts_A = texts_A[:min_len]
        texts_B = texts_B[:min_len]
        
        encoded_A = self.tokenizer(
            texts_A,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoded_B = self.tokenizer(
            texts_B,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels_A = encoded_A['input_ids'].clone()
        labels_B = encoded_B['input_ids'].clone()
        labels_A[labels_A == self.tokenizer.pad_token_id] = -100
        labels_B[labels_B == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids_A': encoded_A['input_ids'],
            'attention_mask_A': encoded_A['attention_mask'],
            'labels_A': labels_A,
            'input_ids_B': encoded_B['input_ids'],
            'attention_mask_B': encoded_B['attention_mask'],
            'labels_B': labels_B,
        }


def setup_distributed(device_override=None):
    if device_override is not None:
        return 0, 1, None
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    
    if local_rank is not None:
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def compute_weight_similarity(weight_A, weight_B):
    """Compute cosine similarity between two weight matrices."""
    flat_A = weight_A.view(-1)
    flat_B = weight_B.view(-1)
    similarity = F.cosine_similarity(flat_A.unsqueeze(0), flat_B.unsqueeze(0))
    return similarity.item()


def compute_weight_distance(weight_A, weight_B):
    """Compute normalized Frobenius distance between two weight matrices."""
    diff = weight_A - weight_B
    frobenius_norm = torch.norm(diff, p='fro')
    normalized_distance = frobenius_norm / (weight_A.numel() ** 0.5)
    return normalized_distance.item()


def generate_samples(model, tokenizer, prompts, device, max_new_tokens=50, temperature=0.7):
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
            
            results.append({
                'prompt_A': prompt_A,
                'prompt_B': prompt_B,
                'generated_A': generated_A,
                'generated_B': generated_B
            })
    
    model.train()
    return results


def train(
    model_name='unsloth/Llama-3.2-1B',
    device=None,
    batch_size=2,
    learning_rate=1e-4,
    backbone_lr=1e-5,
    freeze_backbone=True,
    num_train_steps=50_000,
    max_length=2048,
    log_interval=1,
    similarity_log_interval=50,
    eval_interval=500,
    save_interval=5_000,
    max_gen_tokens=50,
    temperature=0.7,
    output_dir='./checkpoints',
    tensorboard_log_dir='./tensorboard_logs'
):
    """Train Dual-Stream LLM.
    
    Args:
        model_name: Base model name
        device: Specific device to use (e.g., cuda:1). If not specified, uses distributed training.
        batch_size: Batch size per GPU
        learning_rate: Learning rate for embeddings and unembeddings
        backbone_lr: Learning rate for backbone (only used if freeze_backbone=False)
        freeze_backbone: Whether to freeze backbone parameters
        num_train_steps: Number of training steps
        max_length: Maximum sequence length
        log_interval: Logging interval
        similarity_log_interval: Weight similarity logging interval
        eval_interval: Evaluation/generation interval
        save_interval: Checkpoint save interval
        max_gen_tokens: Maximum tokens to generate during evaluation
        temperature: Sampling temperature for generation (higher = more random)
        output_dir: Output directory for checkpoints
        tensorboard_log_dir: Tensorboard log directory. Set to empty string to disable tensorboard logging.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create unique subdirectory for this run with timestamp
    if tensorboard_log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
        tensorboard_log_dir = os.path.join(tensorboard_log_dir, run_name)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    rank, world_size, local_rank = setup_distributed(device)
    
    if device is not None:
        device = torch.device(device)
    else:
        device = torch.device(f'cuda:{local_rank}')
    
    writer = None
    if rank == 0:
        print(f"Starting training with world_size={world_size}")
        print(f"Device: {device}")
        print(f"Model: {model_name}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Learning rate (embeddings): {learning_rate}")
        print(f"Learning rate (backbone): {backbone_lr if not freeze_backbone else 'N/A (frozen)'}")
        print(f"Freeze backbone: {freeze_backbone}")
        print(f"Number of training steps: {num_train_steps}")
        
        if tensorboard_log_dir:
            writer = SummaryWriter(log_dir=tensorboard_log_dir)
            print(f"Tensorboard logging enabled: {tensorboard_log_dir}")
            
            writer.add_text('config/model_name', model_name)
            writer.add_scalar('config/batch_size', batch_size)
            writer.add_scalar('config/learning_rate', learning_rate)
            writer.add_scalar('config/backbone_lr', backbone_lr)
            writer.add_scalar('config/freeze_backbone', 1 if freeze_backbone else 0)
            writer.add_scalar('config/num_train_steps', num_train_steps)
            writer.add_scalar('config/max_length', max_length)
            writer.add_scalar('config/world_size', world_size)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    validation_prompts = [
        ("The capital of", "Machine learning is"),
        ("Once upon a", "In the field of"),
        ("The history of", "Modern technology enables")
    ]
    
    dataset = load_dataset('c4', 'en', split='train', streaming=True, trust_remote_code=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10000)
    
    collate_fn = C4DualStreamCollator(tokenizer, max_length=max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size * 2,
        collate_fn=collate_fn,
        num_workers=2,
        prefetch_factor=2
    )
    
    base_model_config = AutoConfig.from_pretrained(model_name)
    vocab_size = base_model_config.vocab_size
    hidden_dim = base_model_config.hidden_size
    
    model = DualStreamLLM(model_name, vocab_size, hidden_dim, freeze_backbone=freeze_backbone)
    
    trainable_params = [
        model.emb_A.parameters(),
        model.emb_B.parameters(),
        model.unemb_A.parameters(),
        model.unemb_B.parameters()
    ]
    embedding_params = [p for group in trainable_params for p in group]
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        embedding_trainable = sum(p.numel() for p in embedding_params)
        
        if not freeze_backbone:
            backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
            total_trainable = embedding_trainable + backbone_trainable
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {total_trainable:,} ({100*total_trainable/total_params:.2f}%)")
            print(f"  - Embeddings/Unembeddings: {embedding_trainable:,} ({100*embedding_trainable/total_params:.2f}%)")
            print(f"  - Backbone: {backbone_trainable:,} ({100*backbone_trainable/total_params:.2f}%)")
        else:
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {embedding_trainable:,} ({100*embedding_trainable/total_params:.2f}%)")
    
    if world_size > 1:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=1e8
        )
        
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            device_id=local_rank,
        )
    else:
        model = model.to(device)
    
    # Create optimizer with parameter groups for different learning rates
    if freeze_backbone:
        optimizer = torch.optim.AdamW(embedding_params, lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW([
            {'params': embedding_params, 'lr': learning_rate},
            {'params': model.backbone.parameters(), 'lr': backbone_lr}
        ])
    
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    data_iter = iter(dataloader)
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("Initial generation samples (step 0 - before training)")
        print('='*80)
        
        samples = generate_samples(model, tokenizer, validation_prompts, device, max_new_tokens=max_gen_tokens, temperature=temperature)
        
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  Prompt A: {sample['prompt_A']}")
            print(f"  Generated A: {sample['generated_A']}")
            print(f"  Prompt B: {sample['prompt_B']}")
            print(f"  Generated B: {sample['generated_B']}")
        
        print('='*80 + '\n')
        
        if writer is not None:
            table_header = "| Sample | Prompt A | Generated A | Prompt B | Generated B |\n"
            table_header += "|--------|----------|-------------|----------|-------------|\n"
            table_rows = ""
            
            for i, sample in enumerate(samples, 1):
                prompt_a = sample['prompt_A'].replace('\n', ' ')[:100]
                gen_a = sample['generated_A'].replace('\n', ' ')[:200]
                prompt_b = sample['prompt_B'].replace('\n', ' ')[:100]
                gen_b = sample['generated_B'].replace('\n', ' ')[:200]
                
                table_rows += f"| {i} | {prompt_a} | {gen_a} | {prompt_b} | {gen_b} |\n"
            
            table_markdown = table_header + table_rows
            writer.add_text(f'generation_samples/step_{0:07d}', table_markdown, 0)
            writer.flush()
    
    for step in range(num_train_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        input_ids_A = batch['input_ids_A'].to(device)
        attention_mask_A = batch['attention_mask_A'].to(device)
        labels_A = batch['labels_A'].to(device)
        input_ids_B = batch['input_ids_B'].to(device)
        attention_mask_B = batch['attention_mask_B'].to(device)
        labels_B = batch['labels_B'].to(device)
        
        optimizer.zero_grad()
        
        logits_A, logits_B = model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B)
        
        loss_A = loss_fn(logits_A.view(-1, logits_A.size(-1)), labels_A.view(-1))
        loss_B = loss_fn(logits_B.view(-1, logits_B.size(-1)), labels_B.view(-1))
        loss = loss_A + loss_B
        
        loss.backward()
        optimizer.step()
        
        if rank == 0 and (step + 1) % log_interval == 0:
            print(f"Step {step+1}/{num_train_steps} | "
                  f"Total Loss: {loss.item():.4f} | "
                  f"Loss A: {loss_A.item():.4f} | "
                  f"Loss B: {loss_B.item():.4f}")
            
            if writer is not None:
                writer.add_scalar('train/loss_total', loss.item(), step + 1)
                writer.add_scalar('train/loss_A', loss_A.item(), step + 1)
                writer.add_scalar('train/loss_B', loss_B.item(), step + 1)
                writer.flush()
        
        if rank == 0 and (step + 1) % similarity_log_interval == 0:
            with torch.no_grad():
                emb_A_weight = model.emb_A.weight if not isinstance(model, FSDP) else model.module.emb_A.weight
                emb_B_weight = model.emb_B.weight if not isinstance(model, FSDP) else model.module.emb_B.weight
                unemb_A_weight = model.unemb_A.weight if not isinstance(model, FSDP) else model.module.unemb_A.weight
                unemb_B_weight = model.unemb_B.weight if not isinstance(model, FSDP) else model.module.unemb_B.weight
                
                emb_similarity = compute_weight_similarity(emb_A_weight, emb_B_weight)
                emb_distance = compute_weight_distance(emb_A_weight, emb_B_weight)
                
                unemb_similarity = compute_weight_similarity(unemb_A_weight, unemb_B_weight)
                unemb_distance = compute_weight_distance(unemb_A_weight, unemb_B_weight)
                
                print(f"  [Similarity] Emb: {emb_similarity:.4f}, Unemb: {unemb_similarity:.4f} | "
                      f"[Distance] Emb: {emb_distance:.4f}, Unemb: {unemb_distance:.4f}")
                
                if writer is not None:
                    writer.add_scalar('similarity/embedding_cosine', emb_similarity, step + 1)
                    writer.add_scalar('similarity/unembedding_cosine', unemb_similarity, step + 1)
                    writer.add_scalar('distance/embedding_frobenius', emb_distance, step + 1)
                    writer.add_scalar('distance/unembedding_frobenius', unemb_distance, step + 1)
                    writer.flush()
        
        if rank == 0 and (step + 1) % eval_interval == 0:
            print(f"\n{'='*80}")
            print(f"Generation samples at step {step+1}")
            print('='*80)
            
            samples = generate_samples(model, tokenizer, validation_prompts, device, max_new_tokens=max_gen_tokens, temperature=temperature)
            
            for i, sample in enumerate(samples, 1):
                print(f"\nSample {i}:")
                print(f"  Prompt A: {sample['prompt_A']}")
                print(f"  Generated A: {sample['generated_A']}")
                print(f"  Prompt B: {sample['prompt_B']}")
                print(f"  Generated B: {sample['generated_B']}")
            
            print('='*80 + '\n')
            
            if writer is not None:
                table_header = "| Sample | Prompt A | Generated A | Prompt B | Generated B |\n"
                table_header += "|--------|----------|-------------|----------|-------------|\n"
                table_rows = ""
                
                for i, sample in enumerate(samples, 1):
                    prompt_a = sample['prompt_A'].replace('\n', ' ')[:100]
                    gen_a = sample['generated_A'].replace('\n', ' ')[:200]
                    prompt_b = sample['prompt_B'].replace('\n', ' ')[:100]
                    gen_b = sample['generated_B'].replace('\n', ' ')[:200]
                    
                    table_rows += f"| {i} | {prompt_a} | {gen_a} | {prompt_b} | {gen_b} |\n"
                
                table_markdown = table_header + table_rows
                writer.add_text(f'generation_samples/step_{step+1:07d}', table_markdown, step + 1)
                writer.flush()
        
        if rank == 0 and (step + 1) % save_interval == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if world_size > 1:
                state_dict = model.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save({
                'step': step + 1,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, 'model.pt'))
            
            print(f"Checkpoint saved at step {step+1}")
    
    if rank == 0:
        print("Training completed!")
        
        if writer is not None:
            writer.close()
            print(f"Tensorboard logs saved to: {tensorboard_log_dir}")
    
    cleanup_distributed()


def main():
    fire.Fire(train)


if __name__ == '__main__':
    main()

