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
import json
import random
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model import DualStreamLLM

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DualStreamCollator:
    def __init__(self, tokenizer, max_length=512, objective='separate', mixup_alpha=1.0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.objective = objective
        self.mixup_alpha = mixup_alpha
        
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
        
        if self.objective == 'separate':
            # Shift labels: predict next token, not current
            labels_A = encoded_A['input_ids'].clone()
            labels_B = encoded_B['input_ids'].clone()
            
            # Shift left: labels[i] = input_ids[i+1]
            labels_A[:, :-1] = encoded_A['input_ids'][:, 1:]
            labels_B[:, :-1] = encoded_B['input_ids'][:, 1:]
            
            # Last position has no next token, so ignore it
            labels_A[:, -1] = -100
            labels_B[:, -1] = -100
            
            # Also ignore padding tokens
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
        elif self.objective == 'averaged':
            # For averaged objective, we need to create target distributions
            # Get next token ids
            next_tokens_A = encoded_A['input_ids'][:, 1:].clone()
            next_tokens_B = encoded_B['input_ids'][:, 1:].clone()
            
            return {
                'input_ids_A': encoded_A['input_ids'],
                'attention_mask_A': encoded_A['attention_mask'],
                'next_tokens_A': next_tokens_A,
                'input_ids_B': encoded_B['input_ids'],
                'attention_mask_B': encoded_B['attention_mask'],
                'next_tokens_B': next_tokens_B,
                'pad_token_id': self.tokenizer.pad_token_id,
            }
        elif self.objective == 'mixup':
            # Sample mixing coefficients from Beta distribution for each sample in batch
            batch_size = encoded_A['input_ids'].size(0)
            # Sample λ ~ Beta(α, α) for each sample
            lambdas = torch.from_numpy(
                np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            ).float()
            
            # Get next token ids for creating soft targets
            next_tokens_A = encoded_A['input_ids'][:, 1:].clone()
            next_tokens_B = encoded_B['input_ids'][:, 1:].clone()
            
            return {
                'input_ids_A': encoded_A['input_ids'],
                'attention_mask_A': encoded_A['attention_mask'],
                'next_tokens_A': next_tokens_A,
                'input_ids_B': encoded_B['input_ids'],
                'attention_mask_B': encoded_B['attention_mask'],
                'next_tokens_B': next_tokens_B,
                'lambdas': lambdas,  # [batch_size]
                'pad_token_id': self.tokenizer.pad_token_id,
            }
        else:
            raise ValueError(f"Unknown objective: {self.objective}")


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


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) 
                      if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    if not checkpoint_dirs:
        return None
    
    # Extract step numbers and find the latest
    steps = []
    for d in checkpoint_dirs:
        try:
            step = int(d.split('-')[1])
            steps.append((step, d))
        except (IndexError, ValueError):
            continue
    
    if not steps:
        return None
    
    latest_step, latest_dir = max(steps, key=lambda x: x[0])
    checkpoint_path = os.path.join(checkpoint_dir, latest_dir, 'model.pt')
    
    if os.path.exists(checkpoint_path):
        return checkpoint_path, latest_step
    
    return None


def copy_tensorboard_events_to_new_run(source_dir, target_writer, max_step, rank=0):
    """Copy TensorBoard events from source to target writer up to max_step."""
    if rank != 0:
        return  # Only main process does this
    
    print(f"\n{'='*80}")
    print(f"Copying TensorBoard events from previous run...")
    print(f"Source: {source_dir}")
    print(f"Max step: {max_step}")
    print('='*80)
    
    try:
        ea = event_accumulator.EventAccumulator(str(source_dir))
        ea.Reload()
        
        # Copy scalars
        scalar_tags = ea.Tags().get('scalars', [])
        print(f"Copying {len(scalar_tags)} scalar tags...")
        
        total_scalar_events = 0
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            for event in events:
                if event.step <= max_step:
                    target_writer.add_scalar(tag, event.value, event.step, walltime=event.wall_time)
                    total_scalar_events += 1
        
        print(f"  → Copied {total_scalar_events} scalar events")
        
        # Copy text/tensor events (including generation samples, script version, etc.)
        tensor_tags = ea.Tags().get('tensors', [])
        if tensor_tags:
            print(f"Copying {len(tensor_tags)} text/tensor tags...")
            total_text_events = 0
            for tag in tensor_tags:
                try:
                    events = ea.Tensors(tag)
                    for event in events:
                        # For text events, copy regardless of step if step <= max_step OR step == 0
                        # (some metadata like script_version is stored at step 0)
                        if event.step <= max_step:
                            try:
                                # Try to decode as text
                                if event.tensor_proto.string_val:
                                    text_value = event.tensor_proto.string_val[0].decode('utf-8')
                                    target_writer.add_text(tag, text_value, event.step, walltime=event.wall_time)
                                    total_text_events += 1
                            except Exception as decode_err:
                                pass  # Skip if can't decode this specific event
                except Exception as tag_err:
                    pass  # Skip if can't load this tag
            
            print(f"  → Copied {total_text_events} text events")
        
        target_writer.flush()
        
        print(f"✅ Successfully copied events up to step {max_step}")
        print('='*80 + '\n')
        
    except Exception as e:
        print(f"Warning: Failed to copy TensorBoard events: {e}")
        print("Continuing without copied logs...")
        print('='*80 + '\n')


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


def load_live_config(config_path):
    """Load live configuration from JSON file."""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load live config: {e}")
    return None


def create_initial_live_config(config_path, learning_rate, backbone_lr, temperature, 
                                log_interval, similarity_log_interval, eval_interval,
                                loss_balance_strength, loss_balance_ema_momentum, 
                                loss_balance_clip_min, loss_balance_clip_max):
    """Create initial live configuration file."""
    config = {
        "learning_rate": learning_rate,
        "backbone_lr": backbone_lr,
        "temperature": temperature,
        "log_interval": log_interval,
        "similarity_log_interval": similarity_log_interval,
        "eval_interval": eval_interval,
        "loss_balance_strength": loss_balance_strength,
        "loss_balance_ema_momentum": loss_balance_ema_momentum,
        "loss_balance_clip_min": loss_balance_clip_min,
        "loss_balance_clip_max": loss_balance_clip_max
    }
    with open(config_path, 'w') as f:
        json.dump(config, indent=2, fp=f)
    return config


def apply_config_changes(current_config, new_config, optimizer, embedding_params, 
                         backbone_params, freeze_backbone):
    """Apply configuration changes and return dict of what changed."""
    changes = {}
    
    # Check and apply learning rate changes
    if "learning_rate" in new_config and new_config["learning_rate"] != current_config["learning_rate"]:
        old_lr = current_config["learning_rate"]
        new_lr = new_config["learning_rate"]
        # Update optimizer learning rate for embedding parameters
        optimizer.param_groups[0]['lr'] = new_lr
        current_config["learning_rate"] = new_lr
        changes["learning_rate"] = (old_lr, new_lr)
    
    # Check and apply backbone learning rate changes (if not frozen)
    if not freeze_backbone and "backbone_lr" in new_config and new_config["backbone_lr"] != current_config["backbone_lr"]:
        old_lr = current_config["backbone_lr"]
        new_lr = new_config["backbone_lr"]
        # Update optimizer learning rate for backbone parameters (second param group)
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = new_lr
        current_config["backbone_lr"] = new_lr
        changes["backbone_lr"] = (old_lr, new_lr)
    
    # Check and apply temperature changes
    if "temperature" in new_config and new_config["temperature"] != current_config["temperature"]:
        old_temp = current_config["temperature"]
        new_temp = new_config["temperature"]
        current_config["temperature"] = new_temp
        changes["temperature"] = (old_temp, new_temp)
    
    # Check and apply loss balance parameter changes
    for param_name in ["loss_balance_strength", "loss_balance_ema_momentum", 
                       "loss_balance_clip_min", "loss_balance_clip_max"]:
        if param_name in new_config and new_config[param_name] != current_config[param_name]:
            old_val = current_config[param_name]
            new_val = new_config[param_name]
            current_config[param_name] = new_val
            changes[param_name] = (old_val, new_val)
    
    # Check and apply interval changes
    for interval_name in ["log_interval", "similarity_log_interval", "eval_interval"]:
        if interval_name in new_config and new_config[interval_name] != current_config[interval_name]:
            old_val = current_config[interval_name]
            new_val = new_config[interval_name]
            current_config[interval_name] = new_val
            changes[interval_name] = (old_val, new_val)
    
    return changes


def generate_samples(model, tokenizer, prompts, device, max_new_tokens=50, temperature=0.7, objective='separate'):
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
            
            if objective == 'separate':
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
            
            elif objective in ['averaged', 'mixup']:
                # For averaged/mixup objective, generate from each stream independently using pure inputs
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


def train(
    model_name='HuggingFaceTB/SmolLM2-135M',
    dataset_name='roneneldan/TinyStories',
    dataset_config=None,
    dataset_split='train',
    device=None,
    batch_size=32,
    learning_rate=1e-3,
    backbone_lr=1e-4,
    freeze_backbone=False,
    num_train_steps=200_000,
    max_length=4096,
    log_interval=1,
    similarity_log_interval=50,
    eval_interval=500,
    save_interval=20_000,
    max_gen_tokens=50,
    temperature=0.7,
    seed=42,
    output_dir='./checkpoints',
    tensorboard_log_dir='./tensorboard_logs',
    # Training objective
    objective='separate',  # 'separate', 'averaged', or 'mixup'
    use_loss_balancing=True,
    shared_embeddings=False,  # Use single shared embedding matrix instead of separate emb_A and emb_B
    # Embedding initialization
    embedding_init_strategy='noisy_copy', # 'random', 'copy', or 'noisy_copy'
    unembedding_init_strategy=None, # 'random', 'copy', or 'noisy_copy'. If None, uses same as embedding_init_strategy
    embedding_init_noise_std=0.01,
    # Mixup parameters
    mixup_alpha=0.4,  # Alpha parameter for Beta distribution in mixup: λ ~ Beta(α, α)
    # Loss balancing parameters
    loss_balance_strength=5.0,
    loss_balance_ema_momentum=0.99,
    loss_balance_warmup_steps=100,
    loss_balance_clip_min=0.1,
    loss_balance_clip_max=10.0,
    # LR Scheduler parameters
    lr_scheduler_patience=10,
    lr_scheduler_factor=0.5,
    # Resume parameters
    resume_from_checkpoint=None,
    continue_run=None,
    gradient_accumulation_steps=1
):
    # Set seed for reproducibility
    set_seed(seed)
    
    # Initialize variables for resume functionality
    resume_step = 0
    checkpoint_data = None
    
    # Handle continue_run and resume logic
    if continue_run:
        # When continuing a run, create a NEW run directory and copy logs
        if not os.path.exists(continue_run):
            raise ValueError(f"Continue run directory not found: {continue_run}")
        
        old_run_name = os.path.basename(continue_run)
        old_checkpoint_dir = continue_run
        
        # Construct old tensorboard directory path
        # tensorboard_log_dir is the BASE directory (e.g., './tensorboard_logs')
        old_tensorboard_dir = None
        if tensorboard_log_dir:
            old_tensorboard_dir = os.path.join(tensorboard_log_dir, old_run_name)
        
        # Load existing live config
        old_live_config_path = os.path.join(old_checkpoint_dir, 'live_config.json')
        existing_live_config = load_live_config(old_live_config_path)
        
        # Handle resume from checkpoint
        if resume_from_checkpoint:
            if resume_from_checkpoint == 'auto':
                checkpoint_result = find_latest_checkpoint(old_checkpoint_dir)
                if checkpoint_result:
                    checkpoint_path, resume_step = checkpoint_result
                    print(f"Found latest checkpoint: {checkpoint_path} at step {resume_step}")
                else:
                    raise ValueError(f"No checkpoints found in {old_checkpoint_dir}")
            else:
                checkpoint_path = resume_from_checkpoint
                if not os.path.exists(checkpoint_path):
                    raise ValueError(f"Checkpoint not found: {checkpoint_path}")
                # Extract step from checkpoint path or filename
                try:
                    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                    resume_step = checkpoint_data.get('step', 0)
                except Exception as e:
                    raise ValueError(f"Failed to load checkpoint {checkpoint_path}: {e}")
            
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            resume_step = checkpoint_data.get('step', 0)
            print(f"Resuming from step {resume_step}")
        
        # Create NEW run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{old_run_name}_resumed_{timestamp}"
        
        output_dir = os.path.join(os.path.dirname(old_checkpoint_dir), run_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created new run directory: {output_dir}")
        
        # Create NEW tensorboard directory and copy events
        if tensorboard_log_dir and old_tensorboard_dir and os.path.exists(old_tensorboard_dir):
            # Create new tensorboard directory in the BASE tensorboard_log_dir
            tensorboard_log_dir = os.path.join(tensorboard_log_dir, run_name)
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            
            # Note: We'll create writer later and copy events into it
            # to ensure everything goes into one event file
        elif tensorboard_log_dir:
            tensorboard_log_dir = os.path.join(tensorboard_log_dir, run_name)
            os.makedirs(tensorboard_log_dir, exist_ok=True)
        
        # Copy live config to new run
        live_config_path = os.path.join(output_dir, 'live_config.json')
        
        # Update parameters from existing live config if available
        if existing_live_config:
            learning_rate = existing_live_config.get('learning_rate', learning_rate)
            backbone_lr = existing_live_config.get('backbone_lr', backbone_lr)
            temperature = existing_live_config.get('temperature', temperature)
            log_interval = existing_live_config.get('log_interval', log_interval)
            similarity_log_interval = existing_live_config.get('similarity_log_interval', similarity_log_interval)
            eval_interval = existing_live_config.get('eval_interval', eval_interval)
            loss_balance_strength = existing_live_config.get('loss_balance_strength', loss_balance_strength)
            loss_balance_ema_momentum = existing_live_config.get('loss_balance_ema_momentum', loss_balance_ema_momentum)
            loss_balance_clip_min = existing_live_config.get('loss_balance_clip_min', loss_balance_clip_min)
            loss_balance_clip_max = existing_live_config.get('loss_balance_clip_max', loss_balance_clip_max)
            
    else:
        # Create new run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
        
        # Create run-specific checkpoint directory
        output_dir = os.path.join(output_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create run-specific tensorboard directory
        if tensorboard_log_dir:
            tensorboard_log_dir = os.path.join(tensorboard_log_dir, run_name)
            os.makedirs(tensorboard_log_dir, exist_ok=True)
        
        # Create initial live config
        live_config_path = os.path.join(output_dir, 'live_config.json')
    
    # Ensure live config exists with all parameters
    live_config = {
        "learning_rate": learning_rate,
        "backbone_lr": backbone_lr,
        "temperature": temperature,
        "log_interval": log_interval,
        "similarity_log_interval": similarity_log_interval,
        "eval_interval": eval_interval,
        "loss_balance_strength": loss_balance_strength,
        "loss_balance_ema_momentum": loss_balance_ema_momentum,
        "loss_balance_clip_min": loss_balance_clip_min,
        "loss_balance_clip_max": loss_balance_clip_max
    }
    
    create_initial_live_config(live_config_path, learning_rate, backbone_lr, temperature,
                               log_interval, similarity_log_interval, eval_interval,
                               loss_balance_strength, loss_balance_ema_momentum, 
                               loss_balance_clip_min, loss_balance_clip_max)
    live_config_check_interval = 10  # Check for config changes every N steps
    
    rank, world_size, local_rank = setup_distributed(device)
    
    if device is not None:
        device = torch.device(device)
    else:
        device = torch.device(f'cuda:{local_rank}')
    
    # Enforce shared embeddings for mixup objective
    if objective == 'mixup':
        shared_embeddings = True
        if rank == 0:
            print(f"Note: Mixup objective requires shared embeddings. Setting shared_embeddings=True")
    
    # Save full training configuration to JSON
    if rank == 0:
        config_json = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'dataset_config': dataset_config,
            'dataset_split': dataset_split,
            'device': str(device) if device else 'distributed',
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'backbone_lr': backbone_lr,
            'freeze_backbone': freeze_backbone,
            'num_train_steps': num_train_steps,
            'max_length': max_length,
            'log_interval': log_interval,
            'similarity_log_interval': similarity_log_interval,
            'eval_interval': eval_interval,
            'save_interval': save_interval,
            'max_gen_tokens': max_gen_tokens,
            'temperature': temperature,
            'seed': seed,
            'objective': objective,
            'use_loss_balancing': use_loss_balancing,
            'shared_embeddings': shared_embeddings,
            'embedding_init_strategy': embedding_init_strategy,
            'unembedding_init_strategy': unembedding_init_strategy,
            'embedding_init_noise_std': embedding_init_noise_std,
            'mixup_alpha': mixup_alpha,
            'loss_balance_strength': loss_balance_strength,
            'loss_balance_ema_momentum': loss_balance_ema_momentum,
            'loss_balance_warmup_steps': loss_balance_warmup_steps,
            'loss_balance_clip_min': loss_balance_clip_min,
            'loss_balance_clip_max': loss_balance_clip_max,
            'lr_scheduler_patience': lr_scheduler_patience,
            'lr_scheduler_factor': lr_scheduler_factor,
            'gradient_accumulation_steps': gradient_accumulation_steps
        }
        
        config_json_path = os.path.join(output_dir, 'training_config.json')
        with open(config_json_path, 'w') as f:
            json.dump(config_json, f, indent=2)
        print(f"Training configuration saved to: {config_json_path}")
    
    writer = None
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"DISTRIBUTED TRAINING CONFIGURATION")
        print(f"{'='*80}")
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Device: {device}")
        print(f"\n{'='*80}")
        print(f"MODEL AND DATA CONFIGURATION")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}" + (f" ({dataset_config})" if dataset_config else ""))
        print(f"Objective: {objective}")
        print(f"Use loss balancing: {use_loss_balancing}")
        print(f"Shared embeddings: {shared_embeddings}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Effective global batch size: {batch_size * world_size * gradient_accumulation_steps}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"\n{'='*80}")
        print(f"TRAINING PARAMETERS")
        print(f"{'='*80}")
        print(f"Learning rate (embeddings): {learning_rate}")
        print(f"Learning rate (backbone): {backbone_lr if not freeze_backbone else 'N/A (frozen)'}")
        print(f"Freeze backbone: {freeze_backbone}")
        print(f"Number of training steps: {num_train_steps}")
        print(f"Base seed: {seed}")
        print(f"Dataset seeds: rank_0={seed}, rank_1={seed+1}, ..." if world_size > 1 else f"Dataset seed: {seed}")
        print(f"Live config path: {live_config_path}")
        print(f"{'='*80}\n")
        
        if tensorboard_log_dir:
            # Create writer
            writer = SummaryWriter(log_dir=tensorboard_log_dir)
            if resume_step > 0:
                print(f"Tensorboard logging enabled: {tensorboard_log_dir} (resuming from step {resume_step})")
            else:
                print(f"Tensorboard logging enabled: {tensorboard_log_dir}")
            
            # Copy old TensorBoard events into the same writer (if resuming from continue_run)
            if continue_run and old_tensorboard_dir and os.path.exists(old_tensorboard_dir) and resume_step > 0:
                copy_tensorboard_events_to_new_run(old_tensorboard_dir, writer, resume_step, rank=0)
            
            # Save script version to tensorboard
            script_path = __file__
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_content = f.read()
                writer.add_text('script_version/train_dual_stream.py', f"```python\n{script_content}\n```")
                print(f"Script version saved to tensorboard")
            except Exception as e:
                print(f"Warning: Failed to save script version: {e}")
            
            # Save full training config as text
            config_json_text = json.dumps(config_json, indent=2)
            writer.add_text('config/training_config_json', f"```json\n{config_json_text}\n```")
            
            writer.add_text('config/model_name', model_name)
            writer.add_text('config/dataset_name', dataset_name + (f" ({dataset_config})" if dataset_config else ""))
            writer.add_text('config/objective', objective)
            writer.add_scalar('config/use_loss_balancing', 1 if use_loss_balancing else 0)
            writer.add_scalar('config/shared_embeddings', 1 if shared_embeddings else 0)
            writer.add_scalar('config/batch_size', batch_size)
            writer.add_scalar('config/learning_rate_initial', learning_rate)
            writer.add_scalar('config/backbone_lr_initial', backbone_lr)
            writer.add_scalar('config/freeze_backbone', 1 if freeze_backbone else 0)
            writer.add_scalar('config/num_train_steps', num_train_steps)
            writer.add_scalar('config/max_length', max_length)
            writer.add_scalar('config/seed', seed)
            writer.add_scalar('config/world_size', world_size)
            writer.add_scalar('config/mixup_alpha', mixup_alpha)
            writer.add_scalar('config/loss_balance_strength', loss_balance_strength)
            writer.add_scalar('config/loss_balance_ema_momentum', loss_balance_ema_momentum)
            writer.add_scalar('config/loss_balance_warmup_steps', loss_balance_warmup_steps)
            writer.add_scalar('config/loss_balance_clip_min', loss_balance_clip_min)
            writer.add_scalar('config/loss_balance_clip_max', loss_balance_clip_max)
            writer.add_scalar('config/lr_scheduler_patience', lr_scheduler_patience)
            writer.add_scalar('config/lr_scheduler_factor', lr_scheduler_factor)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    validation_prompts = [
        ("The capital of", "Machine learning is"),
        ("Once upon a", "In the field of"),
        ("The history of", "Modern technology enables")
    ]
    
    # Load dataset with optional config
    if dataset_config is not None:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split, streaming=True, trust_remote_code=True)
    
    # Use different seed for each rank to ensure different data
    dataset_seed = seed + rank
    dataset = dataset.shuffle(seed=dataset_seed, buffer_size=10000)
    
    if rank == 0:
        print(f"Dataset shuffle seed for rank {rank}: {dataset_seed}")
    
    collate_fn = DualStreamCollator(tokenizer, max_length=max_length, objective=objective, mixup_alpha=mixup_alpha)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size * 2,
        collate_fn=collate_fn
    )
    
    base_model_config = AutoConfig.from_pretrained(model_name)
    vocab_size = base_model_config.vocab_size
    hidden_dim = base_model_config.hidden_size
    
    model = DualStreamLLM(
        model_name=model_name,
        freeze_backbone=freeze_backbone,
        embedding_init_strategy=embedding_init_strategy,
        unembedding_init_strategy=unembedding_init_strategy,
        embedding_init_noise_std=embedding_init_noise_std,
        objective=objective,
        shared_embeddings=shared_embeddings
    )
    
    if objective == 'separate':
        if shared_embeddings:
            trainable_params = [
                model.emb.parameters(),
                model.unemb_A.parameters(),
                model.unemb_B.parameters()
            ]
        else:
            trainable_params = [
                model.emb_A.parameters(),
                model.emb_B.parameters(),
                model.unemb_A.parameters(),
                model.unemb_B.parameters()
            ]
    else:  # averaged or mixup
        if shared_embeddings:
            trainable_params = [
                model.emb.parameters(),
                model.unemb.parameters()
            ]
        else:
            trainable_params = [
                model.emb_A.parameters(),
                model.emb_B.parameters(),
                model.unemb.parameters()
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
    
    # LR Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Restore model and optimizer states from checkpoint if resuming
    if checkpoint_data:
        print("Loading model state dict...")
        model.load_state_dict(checkpoint_data['model_state_dict'])
        print("Loading optimizer state dict...")  
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            print("Loading scheduler state dict...")
        print("Model, optimizer, and scheduler states restored successfully")
    
    # Initialize loss balancing variables
    ema_loss_A = None
    ema_loss_B = None
    alpha = 1.0
    beta = 1.0
    
    # Restore loss balancing variables from checkpoint if resuming
    if checkpoint_data:
        ema_loss_A = checkpoint_data.get('ema_loss_A', ema_loss_A)
        ema_loss_B = checkpoint_data.get('ema_loss_B', ema_loss_B)
        alpha = checkpoint_data.get('alpha', alpha)
        beta = checkpoint_data.get('beta', beta)
        def _fmt_val(x):
            try:
                return f"{float(x):.4f}"
            except (TypeError, ValueError):
                return "None"
        print(
            "Restored loss balancing state: "
            f"ema_A={_fmt_val(ema_loss_A)}, "
            f"ema_B={_fmt_val(ema_loss_B)}, "
            f"alpha={_fmt_val(alpha)}, "
            f"beta={_fmt_val(beta)}"
        )
    
    if rank == 0 and objective == 'separate' and use_loss_balancing:
        print(f"\nLoss balancing configuration:")
        print(f"  Strength: {loss_balance_strength}")
        print(f"  EMA momentum: {loss_balance_ema_momentum}")
        print(f"  Warmup steps: {loss_balance_warmup_steps}")
        print(f"  Clip range: [{loss_balance_clip_min}, {loss_balance_clip_max}]")
    
    model.train()
    data_iter = iter(dataloader)

    # Exact resume: fast-forward the per-rank dataloader by resume_step batches
    if resume_step > 0:
        try:
            print(f"[Rank {rank}] Fast-forwarding dataloader by {resume_step} batches to resume position...")
            if rank == 0:
                ffbar = tqdm(range(resume_step), desc=f"Fast-forward dataloader (rank {rank})")
            else:
                ffbar = range(resume_step)
            for _ in ffbar:
                try:
                    _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    _ = next(data_iter)
            print(f"[Rank {rank}] Fast-forward complete. Resuming training from step {resume_step}.")
        except Exception as _e:
            print(f"[Rank {rank}] Warning: failed to fast-forward dataloader ({_e}).")
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("Initial generation samples (step 0 - before training)")
        print('='*80)
        
        samples = generate_samples(model, tokenizer, validation_prompts, device, max_new_tokens=max_gen_tokens, temperature=live_config['temperature'], objective=objective)
        
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
    
    if rank == 0 and writer is not None:
        # Log initial hyperparameters
        writer.add_scalar('hyperparams/learning_rate', live_config['learning_rate'], 0)
        writer.add_scalar('hyperparams/backbone_lr', live_config['backbone_lr'], 0)
        writer.add_scalar('hyperparams/temperature', live_config['temperature'], 0)
        writer.add_scalar('hyperparams/loss_balance_strength', live_config['loss_balance_strength'], 0)
        writer.add_scalar('hyperparams/loss_balance_ema_momentum', live_config['loss_balance_ema_momentum'], 0)
        writer.add_scalar('hyperparams/loss_balance_clip_min', live_config['loss_balance_clip_min'], 0)
        writer.add_scalar('hyperparams/loss_balance_clip_max', live_config['loss_balance_clip_max'], 0)
        
        # Log initial loss balancing coefficients (only for separate objective)
        if objective == 'separate' and use_loss_balancing:
            writer.add_scalar('loss_balance/alpha', alpha, 0)
            writer.add_scalar('loss_balance/beta', beta, 0)
            if ema_loss_A is not None and ema_loss_B is not None:
                writer.add_scalar('loss_balance/losses_ratio', ema_loss_A / ema_loss_B if ema_loss_B > 0 else 0, 0)
        
        writer.flush()
    
    total_loss_acc_log = 0.0
    loss_A_acc_log = 0.0
    loss_B_acc_log = 0.0
    
    total_loss_acc_scheduler = 0.0
    scheduler_steps_acc = 0
    
    num_optimizer_steps = num_train_steps // gradient_accumulation_steps
    
    # Initialize tqdm progress bar
    pbar = None
    if rank == 0:
        pbar = tqdm(range(resume_step, num_train_steps), 
                    initial=resume_step, 
                    total=num_train_steps, 
                    desc="Training Steps")
    else:
        pbar = range(resume_step, num_train_steps)

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        input_ids_A = batch['input_ids_A'].to(device)
        attention_mask_A = batch['attention_mask_A'].to(device)
        input_ids_B = batch['input_ids_B'].to(device)
        attention_mask_B = batch['attention_mask_B'].to(device)
        
        
        if objective == 'separate':
            labels_A = batch['labels_A'].to(device)
            labels_B = batch['labels_B'].to(device)
            
            logits_A, logits_B = model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B)
            
            loss_A = loss_fn(logits_A.view(-1, logits_A.size(-1)), labels_A.view(-1))
            loss_B = loss_fn(logits_B.view(-1, logits_B.size(-1)), labels_B.view(-1))
            
            # Update EMA for loss balancing (use live config values)
            if use_loss_balancing:
                current_ema_momentum = live_config.get('loss_balance_ema_momentum', loss_balance_ema_momentum)
                if ema_loss_A is None:
                    ema_loss_A = loss_A.item()
                    ema_loss_B = loss_B.item()
                else:
                    ema_loss_A = current_ema_momentum * ema_loss_A + (1 - current_ema_momentum) * loss_A.item()
                    ema_loss_B = current_ema_momentum * ema_loss_B + (1 - current_ema_momentum) * loss_B.item()
                
                # Compute loss balancing coefficients after warmup (use live config values)
                if step >= loss_balance_warmup_steps:
                    current_strength = live_config.get('loss_balance_strength', loss_balance_strength)
                    current_clip_min = live_config.get('loss_balance_clip_min', loss_balance_clip_min)
                    current_clip_max = live_config.get('loss_balance_clip_max', loss_balance_clip_max)
                    
                    # Compute weights based on EMA losses
                    total_loss = ema_loss_A + ema_loss_B
                    weight_A = (ema_loss_A / total_loss) ** current_strength
                    weight_B = (ema_loss_B / total_loss) ** current_strength
                    
                    # Normalize so alpha + beta = 2
                    norm = weight_A + weight_B
                    alpha = weight_A / norm * 2.0
                    beta = weight_B / norm * 2.0
                    
                    # Apply clipping
                    alpha = torch.clamp(torch.tensor(alpha), current_clip_min, current_clip_max).item()
                    beta = torch.clamp(torch.tensor(beta), current_clip_min, current_clip_max).item()
                else:
                    alpha = 1.0
                    beta = 1.0
            else:
                alpha = 1.0
                beta = 1.0
            
            loss = alpha * loss_A + beta * loss_B

        elif objective == 'averaged':
            next_tokens_A = batch['next_tokens_A'].to(device)
            next_tokens_B = batch['next_tokens_B'].to(device)
            pad_token_id = batch['pad_token_id']
            
            logits = model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B)
            
            # Use logits[:, :-1, :] to match with next_tokens
            # Cast to float32 for numerically stable log_softmax on bf16 models
            logits_shifted = logits[:, :-1, :].float()
            
            # Create mask for valid (non-padding) positions
            valid_mask = (input_ids_A[:, :-1] != pad_token_id) & (input_ids_B[:, :-1] != pad_token_id)
            valid_mask = valid_mask & (next_tokens_A != pad_token_id) & (next_tokens_B != pad_token_id)
            
            # Compute log probabilities
            log_probs = F.log_softmax(logits_shifted, dim=-1)
            
            # Goal: compute cross-entropy H(Q, P), where P is the model's prediction
            # and Q is the target distribution 0.5 * e_A + 0.5 * e_B.
            # This loss is equivalent to: L = - (0.5 * log(P_A) + 0.5 * log(P_B)).
            # This can be interpreted as the average of two standard cross-entropy losses.

            # Gather log probs for both target tokens
            log_prob_A = torch.gather(log_probs, dim=-1, index=next_tokens_A.unsqueeze(-1)).squeeze(-1)
            log_prob_B = torch.gather(log_probs, dim=-1, index=next_tokens_B.unsqueeze(-1)).squeeze(-1)
            
            # The cross-entropy loss if the target were only token A
            loss_ce_A = -log_prob_A
            # The cross-entropy loss if the target were only token B
            loss_ce_B = -log_prob_B
            
            # Averaging these two losses is mathematically equivalent to the cross-entropy
            # with the soft target distribution [..., 0.5, ..., 0.5, ...].
            # This formula correctly handles the case where t_A == t_B,
            # because in that case loss_ce_A == loss_ce_B, and their average is just themselves.
            loss = 0.5 * (loss_ce_A + loss_ce_B)
            
            # Apply the mask to only compute loss for valid positions
            loss = loss * valid_mask.float()
            
            # Compute the average loss over all valid tokens in the batch
            # Add a small epsilon to avoid division by zero if the batch is empty
            loss = loss.sum() / (valid_mask.float().sum() + 1e-8)
            
            # For logging purposes
            loss_A = loss
            loss_B = loss
            alpha = 1.0
            beta = 1.0

        elif objective == 'mixup':
            next_tokens_A = batch['next_tokens_A'].to(device)
            next_tokens_B = batch['next_tokens_B'].to(device)
            lambdas = batch['lambdas'].to(device)  # [batch_size]
            pad_token_id = batch['pad_token_id']
            
            logits = model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, lambdas=lambdas)
            
            # Use logits[:, :-1, :] to match with next_tokens
            # Cast to float32 for numerically stable log_softmax on bf16 models
            logits_shifted = logits[:, :-1, :].float()
            
            # Create mask for valid (non-padding) positions
            valid_mask = (input_ids_A[:, :-1] != pad_token_id) & (input_ids_B[:, :-1] != pad_token_id)
            valid_mask = valid_mask & (next_tokens_A != pad_token_id) & (next_tokens_B != pad_token_id)
            
            # Compute log probabilities
            log_probs = F.log_softmax(logits_shifted, dim=-1)
            
            # Mixup loss: weighted cross-entropy with soft targets
            # Target distribution: λ * one_hot(token_A) + (1-λ) * one_hot(token_B)
            # Loss: -[λ * log(P[token_A]) + (1-λ) * log(P[token_B])]
            
            # Gather log probs for both target tokens
            log_prob_A = torch.gather(log_probs, dim=-1, index=next_tokens_A.unsqueeze(-1)).squeeze(-1)
            log_prob_B = torch.gather(log_probs, dim=-1, index=next_tokens_B.unsqueeze(-1)).squeeze(-1)
            
            # Expand lambdas to [batch_size, seq_len]
            lambdas_expanded = lambdas.view(-1, 1).to(log_prob_A.dtype)
            
            # Mixup loss: -[λ * log(P_A) + (1-λ) * log(P_B)]
            loss = -(lambdas_expanded * log_prob_A + (1 - lambdas_expanded) * log_prob_B)
            
            # Apply the mask to only compute loss for valid positions
            loss = loss * valid_mask.float()
            
            # Compute the average loss over all valid tokens in the batch
            loss = loss.sum() / (valid_mask.float().sum() + 1e-8)
            
            # For logging purposes
            loss_A = loss
            loss_B = loss
            alpha = 1.0
            beta = 1.0

        # Non-finite loss guard
        if not torch.isfinite(loss):
            if rank == 0:
                msg = f"Non-finite loss detected at step {step+1}: loss={loss.item()}"
                print(f"\n{'='*80}\n{msg}\n{'='*80}")
                if 'input_ids_A' in batch and 'input_ids_B' in batch:
                    try:
                        print("Sample A:", tokenizer.decode(batch['input_ids_A'][0], skip_special_tokens=False))
                        print("Sample B:", tokenizer.decode(batch['input_ids_B'][0], skip_special_tokens=False))
                    except Exception:
                        pass
                problem_batch_path = os.path.join(output_dir, f"nonfinite_loss_batch_step_{step+1}.pt")
                try:
                    torch.save(batch, problem_batch_path)
                    print(f"Saved batch to {problem_batch_path}")
                except Exception:
                    pass
                if writer is not None:
                    writer.add_text('events/nonfinite_loss', msg, step+1)
                    writer.flush()
            raise RuntimeError(f"Non-finite loss at step {step+1}")

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
            
        loss.backward()

        # Accumulate losses for logging
        total_loss_acc_log += loss.item()
        if objective == 'separate':
            loss_A_acc_log += loss_A.item()
            loss_B_acc_log += loss_B.item()
        else:
            loss_A_acc_log += loss.item() # For averaged, log the same loss
            loss_B_acc_log += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer_step = (step + 1) // gradient_accumulation_steps

            # Gradient clipping and grad-norm logging
            grad_norm_value = None
            try:
                if isinstance(model, FSDP):
                    grad_norm_value = FSDP.clip_grad_norm_(model.parameters(), max_norm=1.0)
                else:
                    grad_norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            except Exception:
                # Fallback: ignore clipping errors
                pass

            if grad_norm_value is not None:
                try:
                    grad_norm_scalar = float(grad_norm_value)
                except Exception:
                    grad_norm_scalar = grad_norm_value.item() if hasattr(grad_norm_value, 'item') else None
                if grad_norm_scalar is not None:
                    if writer is not None:
                        writer.add_scalar('train/grad_norm', grad_norm_scalar, optimizer_step)
                    if not np.isfinite(grad_norm_scalar):
                        if rank == 0:
                            msg = f"Non-finite grad norm at optimizer step {optimizer_step}: grad_norm={grad_norm_scalar}"
                            print(f"\n{'='*80}\n{msg}\n{'='*80}")
                            problem_batch_path = os.path.join(output_dir, f"nonfinite_grad_batch_step_{step+1}.pt")
                            try:
                                torch.save(batch, problem_batch_path)
                                print(f"Saved batch to {problem_batch_path}")
                            except Exception:
                                pass
                            if writer is not None:
                                writer.add_text('events/nonfinite_grad_norm', msg, optimizer_step)
                                writer.flush()
                        raise RuntimeError(f"Non-finite grad norm at step {step+1}")

            optimizer.step()
            optimizer.zero_grad()
            
            # Accumulate loss for scheduler
            avg_loss_for_step = total_loss_acc_log if gradient_accumulation_steps == 1 else total_loss_acc_log * gradient_accumulation_steps / gradient_accumulation_steps
            total_loss_acc_scheduler += avg_loss_for_step
            scheduler_steps_acc += 1

            # Check for live config changes
            if rank == 0 and (step + 1) % live_config_check_interval == 0:
                new_config = load_live_config(live_config_path)
                if new_config is not None:
                    changes = apply_config_changes(
                        live_config, new_config, optimizer, embedding_params, 
                        model.backbone.parameters() if not isinstance(model, FSDP) else model.module.backbone.parameters(),
                        freeze_backbone
                    )
                    
                    if changes:
                        change_description = ", ".join([f"{k}: {v[0]} -> {v[1]}" for k, v in changes.items()])
                        print(f"\n{'='*80}")
                        print(f"[Step {step+1}] Configuration changed: {change_description}")
                        print(f"{'='*80}\n")
                        
                        if writer is not None:
                            # Log the change event as text
                            event_text = f"Step {step+1}: {change_description}"
                            writer.add_text('events/config_changes', event_text, optimizer_step)
                            
                            # Log changed hyperparameters as scalars for graph visualization
                            if 'learning_rate' in changes:
                                writer.add_scalar('hyperparams/learning_rate', changes['learning_rate'][1], optimizer_step)
                            if 'backbone_lr' in changes:
                                writer.add_scalar('hyperparams/backbone_lr', changes['backbone_lr'][1], optimizer_step)
                            if 'temperature' in changes:
                                writer.add_scalar('hyperparams/temperature', changes['temperature'][1], optimizer_step)
                            if 'loss_balance_strength' in changes:
                                writer.add_scalar('hyperparams/loss_balance_strength', changes['loss_balance_strength'][1], optimizer_step)
                            if 'loss_balance_ema_momentum' in changes:
                                writer.add_scalar('hyperparams/loss_balance_ema_momentum', changes['loss_balance_ema_momentum'][1], optimizer_step)
                            if 'loss_balance_clip_min' in changes:
                                writer.add_scalar('hyperparams/loss_balance_clip_min', changes['loss_balance_clip_min'][1], optimizer_step)
                            if 'loss_balance_clip_max' in changes:
                                writer.add_scalar('hyperparams/loss_balance_clip_max', changes['loss_balance_clip_max'][1], optimizer_step)
                            
                            writer.flush()
                    
                    # Update intervals from live config
                    log_interval = live_config.get("log_interval", log_interval)
                    similarity_log_interval = live_config.get("similarity_log_interval", similarity_log_interval)
                    eval_interval = live_config.get("eval_interval", eval_interval)
            
            # Logging logic - now tied to optimizer steps
            if rank == 0 and optimizer_step % log_interval == 0:
                avg_total_loss = avg_loss_for_step
                avg_loss_A = loss_A_acc_log / gradient_accumulation_steps
                avg_loss_B = loss_B_acc_log / gradient_accumulation_steps
                current_lr = optimizer.param_groups[0]['lr']
                current_backbone_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None

                if objective == 'separate':
                    postfix_metrics = {
                        'Opt Step': f"{optimizer_step}/{num_optimizer_steps}",
                        'Loss': f"{avg_total_loss:.4f}",
                        'Loss A': f"{avg_loss_A:.4f}",
                        'Loss B': f"{avg_loss_B:.4f}",
                        'LR': f"{current_lr:.2e}",
                        'Backbone LR': f"{current_backbone_lr:.2e}" if current_backbone_lr is not None else 'N/A',
                        'Alpha': f"{alpha:.4f}",
                        'Beta': f"{beta:.4f}",
                    }
                    if pbar: pbar.set_postfix(postfix_metrics)
                else:
                    postfix_metrics = {
                        'Opt Step': f"{optimizer_step}/{num_optimizer_steps}",
                        'Loss': f"{avg_total_loss:.4f}",
                        'LR': f"{current_lr:.2e}",
                        'Backbone LR': f"{current_backbone_lr:.2e}" if current_backbone_lr is not None else 'N/A',
                    }
                    if pbar: pbar.set_postfix(postfix_metrics)
                
                if writer is not None:
                    writer.add_scalar('train/loss_total', avg_total_loss, optimizer_step)
                    writer.add_scalar('hyperparams/learning_rate', current_lr, optimizer_step)
                    if current_backbone_lr is not None:
                        writer.add_scalar('hyperparams/backbone_current_lr', current_backbone_lr, optimizer_step)
                    
                    if objective == 'separate':
                        writer.add_scalar('train/loss_A', avg_loss_A, optimizer_step)
                        writer.add_scalar('train/loss_B', avg_loss_B, optimizer_step)
                        
                        if ema_loss_A is not None and ema_loss_B is not None:
                            writer.add_scalar('train/ema_loss_A', ema_loss_A, optimizer_step)
                            writer.add_scalar('train/ema_loss_B', ema_loss_B, optimizer_step)
                        
                        if use_loss_balancing:
                            writer.add_scalar('loss_balance/alpha', alpha, optimizer_step)
                            writer.add_scalar('loss_balance/beta', beta, optimizer_step)
                            writer.add_scalar('loss_balance/alpha_beta_ratio', alpha / beta if beta > 0 else 0, optimizer_step)
                            
                            if ema_loss_A is not None and ema_loss_B is not None:
                                writer.add_scalar('loss_balance/losses_ratio', ema_loss_A / ema_loss_B if ema_loss_B > 0 else 0, optimizer_step)
                    
                    writer.flush()

                # Reset logging accumulators
                total_loss_acc_log = 0.0
                loss_A_acc_log = 0.0
                loss_B_acc_log = 0.0
            
            # Similarity logging
            if rank == 0 and optimizer_step > 0 and optimizer_step % similarity_log_interval == 0 and objective == 'separate' and not shared_embeddings:
                # Only log similarity when we have separate embedding matrices
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
                        writer.add_scalar('similarity/embedding_cosine', emb_similarity, optimizer_step)
                        writer.add_scalar('similarity/unembedding_cosine', unemb_similarity, optimizer_step)
                        writer.add_scalar('distance/embedding_frobenius', emb_distance, optimizer_step)
                        writer.add_scalar('distance/unembedding_frobenius', unemb_distance, optimizer_step)
                        writer.flush()
            
            # Evaluation samples and scheduler step
            if rank == 0 and optimizer_step > 0 and optimizer_step % eval_interval == 0:
                # Step the scheduler based on the average loss over the eval period
                if scheduler_steps_acc > 0:
                    avg_loss_for_scheduler = total_loss_acc_scheduler / scheduler_steps_acc
                    scheduler.step(avg_loss_for_scheduler)
                    if writer is not None:
                        writer.add_scalar('train/loss_for_scheduler', avg_loss_for_scheduler, optimizer_step)
                    # Reset scheduler accumulators
                    total_loss_acc_scheduler = 0.0
                    scheduler_steps_acc = 0

                print(f"\n{'='*80}")
                print(f"Generation samples at step {optimizer_step}")
                print('='*80)
                
                samples = generate_samples(model, tokenizer, validation_prompts, device, max_new_tokens=max_gen_tokens, temperature=live_config['temperature'], objective=objective)
                
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
                    writer.add_text(f'generation_samples/step_{optimizer_step:07d}', table_markdown, optimizer_step)
                    writer.flush()
            
            # Checkpointing
            if rank == 0 and optimizer_step > 0 and optimizer_step % save_interval == 0:
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
                    'scheduler_state_dict': scheduler.state_dict(),
                    'alpha': alpha,
                    'beta': beta,
                    'ema_loss_A': ema_loss_A,
                    'ema_loss_B': ema_loss_B,
                }, os.path.join(checkpoint_dir, 'model.pt'))
                
                print(f"Checkpoint saved at step {step+1} (optimizer step {optimizer_step})")
    
    if rank == 0:
        if pbar: pbar.close()
        print("Training completed!")
        
        if writer is not None:
            writer.close()
            print(f"Tensorboard logs saved to: {tensorboard_log_dir}")
    
    cleanup_distributed()

if __name__ == '__main__':
    fire.Fire(train)
