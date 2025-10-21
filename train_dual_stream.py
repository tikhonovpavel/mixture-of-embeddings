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


class DualStreamLLM(nn.Module):
    def __init__(self, model_name, vocab_size, hidden_dim, freeze_backbone=True):
        super().__init__()
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            use_cache=False
        )
        
        # Random initialization for embeddings and unembeddings
        self.emb_A = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16)
        self.emb_B = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16)
        
        self.unemb_A = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16)
        self.unemb_B = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16)
        
        # Initialize with normal distribution (std=0.02 is common for embeddings)
        nn.init.normal_(self.emb_A.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.emb_B.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.unemb_A.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.unemb_B.weight, mean=0.0, std=0.02)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B):
        embeds_A = self.emb_A(input_ids_A)
        embeds_B = self.emb_B(input_ids_B)
        
        fused_embeds = (embeds_A + embeds_B) * 0.5
        
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
    batch_size=4,
    learning_rate=1e-3,
    backbone_lr=2e-5,
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
    # Loss balancing parameters
    loss_balance_strength=2.0,
    loss_balance_ema_momentum=0.99,
    loss_balance_warmup_steps=100,
    loss_balance_clip_min=0.1,
    loss_balance_clip_max=10.0,
    # Resume parameters
    resume_from_checkpoint=None,
    continue_run=None
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
        seed: Random seed for reproducibility
        output_dir: Output directory for checkpoints
        tensorboard_log_dir: Tensorboard log directory. Set to empty string to disable tensorboard logging.
        loss_balance_strength: Strength of loss balancing (higher = more aggressive)
        loss_balance_ema_momentum: EMA momentum for loss tracking (0.99 = smooth, 0.9 = responsive)
        loss_balance_warmup_steps: Steps before enabling loss balancing
        loss_balance_clip_min: Minimum value for alpha/beta coefficients
        loss_balance_clip_max: Maximum value for alpha/beta coefficients
        resume_from_checkpoint: Path to checkpoint to resume from, or 'auto' to find latest
        continue_run: Path to existing run directory to continue (preserves run name and logs)
    
    Live Configuration:
        A live_config.json file is created in the run directory. You can edit this file during training
        to adjust hyperparameters in real-time. Changes are checked every 10 steps. Supported parameters:
        - learning_rate: Learning rate for embeddings/unembeddings
        - backbone_lr: Learning rate for backbone (if not frozen)
        - temperature: Sampling temperature for generation
        - log_interval: How often to log metrics
        - similarity_log_interval: How often to log weight similarities
        - eval_interval: How often to run generation samples
        - loss_balance_strength: Strength of loss balancing
        - loss_balance_ema_momentum: EMA momentum for loss tracking
        - loss_balance_clip_min: Minimum value for alpha/beta coefficients
        - loss_balance_clip_max: Maximum value for alpha/beta coefficients
    """
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
        print(f"Seed: {seed}")
        print(f"Live config path: {live_config_path}")
        
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
            
            writer.add_text('config/model_name', model_name)
            writer.add_scalar('config/batch_size', batch_size)
            writer.add_scalar('config/learning_rate_initial', learning_rate)
            writer.add_scalar('config/backbone_lr_initial', backbone_lr)
            writer.add_scalar('config/freeze_backbone', 1 if freeze_backbone else 0)
            writer.add_scalar('config/num_train_steps', num_train_steps)
            writer.add_scalar('config/max_length', max_length)
            writer.add_scalar('config/seed', seed)
            writer.add_scalar('config/world_size', world_size)
            writer.add_scalar('config/loss_balance_strength', loss_balance_strength)
            writer.add_scalar('config/loss_balance_ema_momentum', loss_balance_ema_momentum)
            writer.add_scalar('config/loss_balance_warmup_steps', loss_balance_warmup_steps)
            writer.add_scalar('config/loss_balance_clip_min', loss_balance_clip_min)
            writer.add_scalar('config/loss_balance_clip_max', loss_balance_clip_max)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    validation_prompts = [
        ("The capital of", "Machine learning is"),
        ("Once upon a", "In the field of"),
        ("The history of", "Modern technology enables")
    ]
    
    dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True, trust_remote_code=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
    collate_fn = C4DualStreamCollator(tokenizer, max_length=max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size * 2,
        collate_fn=collate_fn
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
    
    # Restore model and optimizer states from checkpoint if resuming
    if checkpoint_data:
        print("Loading model state dict...")
        model.load_state_dict(checkpoint_data['model_state_dict'])
        print("Loading optimizer state dict...")  
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        print("Model and optimizer states restored successfully")
    
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
        print(f"Restored loss balancing state: ema_A={ema_loss_A:.4f}, ema_B={ema_loss_B:.4f}, alpha={alpha:.4f}, beta={beta:.4f}")
    
    if rank == 0:
        print(f"\nLoss balancing configuration:")
        print(f"  Strength: {loss_balance_strength}")
        print(f"  EMA momentum: {loss_balance_ema_momentum}")
        print(f"  Warmup steps: {loss_balance_warmup_steps}")
        print(f"  Clip range: [{loss_balance_clip_min}, {loss_balance_clip_max}]")
    
    model.train()
    data_iter = iter(dataloader)
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("Initial generation samples (step 0 - before training)")
        print('='*80)
        
        samples = generate_samples(model, tokenizer, validation_prompts, device, max_new_tokens=max_gen_tokens, temperature=live_config['temperature'])
        
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
            
            # Log initial hyperparameters
            writer.add_scalar('hyperparams/learning_rate', live_config['learning_rate'], 0)
            writer.add_scalar('hyperparams/backbone_lr', live_config['backbone_lr'], 0)
            writer.add_scalar('hyperparams/temperature', live_config['temperature'], 0)
            writer.add_scalar('hyperparams/loss_balance_strength', live_config['loss_balance_strength'], 0)
            writer.add_scalar('hyperparams/loss_balance_ema_momentum', live_config['loss_balance_ema_momentum'], 0)
            writer.add_scalar('hyperparams/loss_balance_clip_min', live_config['loss_balance_clip_min'], 0)
            writer.add_scalar('hyperparams/loss_balance_clip_max', live_config['loss_balance_clip_max'], 0)
            
            # Log initial loss balancing coefficients
            writer.add_scalar('loss_balance/alpha', alpha, 0)
            writer.add_scalar('loss_balance/beta', beta, 0)
            writer.add_scalar('loss_balance/alpha_beta_ratio', alpha / beta if beta > 0 else 0, 0)
            if ema_loss_A is not None and ema_loss_B is not None:
                writer.add_scalar('loss_balance/losses_ratio', ema_loss_A / ema_loss_B if ema_loss_B > 0 else 0, 0)
            
            writer.flush()
    
    for step in range(resume_step, num_train_steps):
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
        
        # Update EMA for loss balancing (use live config values)
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
        
        loss = alpha * loss_A + beta * loss_B
        
        loss.backward()
        optimizer.step()
        
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
                        writer.add_text('events/config_changes', event_text, step + 1)
                        
                        # Log changed hyperparameters as scalars for graph visualization
                        if 'learning_rate' in changes:
                            writer.add_scalar('hyperparams/learning_rate', changes['learning_rate'][1], step + 1)
                        if 'backbone_lr' in changes:
                            writer.add_scalar('hyperparams/backbone_lr', changes['backbone_lr'][1], step + 1)
                        if 'temperature' in changes:
                            writer.add_scalar('hyperparams/temperature', changes['temperature'][1], step + 1)
                        if 'loss_balance_strength' in changes:
                            writer.add_scalar('hyperparams/loss_balance_strength', changes['loss_balance_strength'][1], step + 1)
                        if 'loss_balance_ema_momentum' in changes:
                            writer.add_scalar('hyperparams/loss_balance_ema_momentum', changes['loss_balance_ema_momentum'][1], step + 1)
                        if 'loss_balance_clip_min' in changes:
                            writer.add_scalar('hyperparams/loss_balance_clip_min', changes['loss_balance_clip_min'][1], step + 1)
                        if 'loss_balance_clip_max' in changes:
                            writer.add_scalar('hyperparams/loss_balance_clip_max', changes['loss_balance_clip_max'][1], step + 1)
                        
                        writer.flush()
                
                # Update intervals from live config
                log_interval = live_config.get("log_interval", log_interval)
                similarity_log_interval = live_config.get("similarity_log_interval", similarity_log_interval)
                eval_interval = live_config.get("eval_interval", eval_interval)
        
        if rank == 0 and (step + 1) % log_interval == 0:
            print(f"Step {step+1}/{num_train_steps} | "
                  f"Total Loss: {loss.item():.4f} | "
                  f"Loss A: {loss_A.item():.4f} | "
                  f"Loss B: {loss_B.item():.4f} | "
                  f"Alpha: {alpha:.4f} | "
                  f"Beta: {beta:.4f}")
            
            if writer is not None:
                writer.add_scalar('train/loss_total', loss.item(), step + 1)
                writer.add_scalar('train/loss_A', loss_A.item(), step + 1)
                writer.add_scalar('train/loss_B', loss_B.item(), step + 1)
                writer.add_scalar('train/ema_loss_A', ema_loss_A, step + 1)
                writer.add_scalar('train/ema_loss_B', ema_loss_B, step + 1)
                writer.add_scalar('loss_balance/alpha', alpha, step + 1)
                writer.add_scalar('loss_balance/beta', beta, step + 1)
                writer.add_scalar('loss_balance/alpha_beta_ratio', alpha / beta if beta > 0 else 0, step + 1)
                writer.add_scalar('loss_balance/losses_ratio', ema_loss_A / ema_loss_B if ema_loss_B > 0 else 0, step + 1)
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
            
            samples = generate_samples(model, tokenizer, validation_prompts, device, max_new_tokens=max_gen_tokens, temperature=live_config['temperature'])
            
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
                'alpha': alpha,
                'beta': beta,
                'ema_loss_A': ema_loss_A,
                'ema_loss_B': ema_loss_B,
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

