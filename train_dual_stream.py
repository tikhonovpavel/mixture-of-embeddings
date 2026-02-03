import sys
import types

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from datasets import load_dataset
import os
import functools
import fire
from datetime import datetime
import json

from model import DualStreamLLM
from collator import DualStreamCollator
from trainer import Trainer
from checkpoint_manager import CheckpointManager
from config_manager import ConfigManager
from logger import TrainingLogger
from utils import set_seed, setup_distributed, cleanup_distributed

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

def train(
    model_name='HuggingFaceTB/SmolLM2-135M',
    dataset_name='HuggingFaceFW/fineweb',
    dataset_config=None,
    dataset_split='train',
    device=None,
    batch_size=16,
    learning_rate=1e-3,
    backbone_lr=1e-4,
    freeze_backbone=False,
    freeze_embeddings=False,
    optimize_memory=False,
    num_train_steps=1_000_000,
    max_length=128,
    log_interval=1,
    similarity_log_interval=50,
    eval_interval=500,
    save_interval=20_000,
    max_gen_tokens=50,
    temperature=0.95,
    seed=42,
    output_dir='./checkpoints',
    tensorboard_log_dir='./tensorboard_logs',
    # Training objective
    objective='separate',  # 'separate', 'averaged', 'mixup', or 'distillation'
    use_loss_balancing=True,
    shared_embeddings=False,  # Use single shared embedding matrix instead of separate emb_A and emb_B
    # Embedding initialization
    embedding_init_strategy='noisy_copy', # 'random', 'copy', or 'noisy_copy'
    unembedding_init_strategy=None, # 'random', 'copy', or 'noisy_copy'. If None, uses same as embedding_init_strategy
    embedding_init_noise_std=0.01,
    # Mixup parameters
    mixup_alpha=0.4,  # Alpha parameter for Beta distribution in mixup: λ ~ Beta(α, α)
    # Distillation parameters
    teacher_model_name=None,  # Teacher model name. If None, uses same as model_name
    teacher_load_in_8bit=True,
    distillation_temperature=2.0,  # Temperature for softening teacher distributions
    distillation_type='kl_geodesic',  # 'naive_average' (average probabilities) or 'kl_geodesic' (average logits)
    # Same text probability
    same_text_prob=0.2,  # Probability of using same text for both streams
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
    gradient_accumulation_steps=2,
    grad_clip=1.0,
    # Resume warmup (helps optimizer adapt when changing hyperparameters)
    resume_warmup_steps=0,  # Number of steps to warmup after resume (0 = disabled)
    resume_warmup_lr_factor=0.1,  # LR multiplier at start of warmup (linearly increases to 1.0)
):
    """Main training function."""
    # Set seed for reproducibility
    set_seed(seed)
    
    # Initialize variables for resume functionality
    resume_step = 0
    checkpoint_data = None
    old_tensorboard_dir = None
    
    # Handle continue_run and resume logic
    resume_num_samples = 0
    if continue_run:
        output_dir, tensorboard_log_dir, resume_step, resume_num_samples, checkpoint_data, old_tensorboard_dir = \
            _handle_continue_run(
                continue_run, resume_from_checkpoint, tensorboard_log_dir,
                batch_size, gradient_accumulation_steps
            )
    else:
        # Create new run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
        
        # Create run-specific directories
        output_dir = os.path.join(output_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)
        
        if tensorboard_log_dir:
            tensorboard_log_dir = os.path.join(tensorboard_log_dir, run_name)
            os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed(device)
    
    if device is not None:
        device = torch.device(device)
    else:
        device = torch.device(f'cuda:{local_rank}')
        
    # Set specific devices for student and teacher if not distributed and GPUs are available
    teacher_device = device
    if world_size == 1 and torch.cuda.device_count() >= 2:
        device = torch.device('cuda:0')
        teacher_device = torch.device('cuda:1')
        print(f"Using separate devices: Student on {device}, Teacher on {teacher_device}")
    else:
        print(f"Using device: {device}")
    
    # Enforce shared embeddings for mixup objective
    if objective == 'mixup':
        shared_embeddings = True
        if rank == 0:
            print(f"Note: Mixup objective requires shared embeddings. Setting shared_embeddings=True")
    
    # Enforce shared embeddings for distillation objective
    if objective == 'distillation':
        shared_embeddings = True
        if rank == 0:
            print(f"Note: Distillation objective requires shared embeddings. Setting shared_embeddings=True")
            
    # Suggest shared embeddings for separate_distillation objective
    if objective == 'separate_distillation' and not shared_embeddings:
        if rank == 0:
            print(f"Note: Separate distillation objective usually benefits from shared embeddings. Consider setting shared_embeddings=True")
    
    # Save full training configuration to JSON
    if rank == 0:
        config_json = _create_config_dict(
            model_name, dataset_name, dataset_config, dataset_split, device,
            batch_size, learning_rate, backbone_lr, freeze_backbone, freeze_embeddings, optimize_memory,
            num_train_steps, max_length, log_interval, similarity_log_interval, eval_interval, save_interval,
            max_gen_tokens, temperature, seed, objective, use_loss_balancing, shared_embeddings,
            embedding_init_strategy, unembedding_init_strategy, embedding_init_noise_std,
            mixup_alpha, same_text_prob, loss_balance_strength, loss_balance_ema_momentum,
            loss_balance_warmup_steps, loss_balance_clip_min, loss_balance_clip_max,
            lr_scheduler_patience, lr_scheduler_factor, gradient_accumulation_steps,
            teacher_model_name, teacher_load_in_8bit, distillation_temperature, distillation_type, grad_clip
        )
        
        config_json_path = os.path.join(output_dir, 'training_config.json')
        with open(config_json_path, 'w') as f:
            json.dump(config_json, f, indent=2)
        print(f"Training configuration saved to: {config_json_path}")
        
        _print_training_info(rank, world_size, device, model_name, dataset_name, dataset_config,
                             objective, use_loss_balancing, shared_embeddings, batch_size,
                             gradient_accumulation_steps, learning_rate, backbone_lr, freeze_backbone,
                             freeze_embeddings, optimize_memory, num_train_steps, seed, output_dir, grad_clip)
    
    # Setup logger and log configuration
    logger = TrainingLogger(tensorboard_log_dir, rank)
    if rank == 0:
        logger.log_training_config(config_json, project_root=os.path.dirname(__file__))
        
        # Copy old TensorBoard events if resuming
        if continue_run and old_tensorboard_dir and os.path.exists(old_tensorboard_dir) and resume_step > 0:
            logger.copy_tensorboard_events(old_tensorboard_dir, resume_step)
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Validation prompts
    validation_prompts = [
        ("The capital of", "Machine learning is"),
        ("Once upon a", "In the field of"),
        ("The history of", "Modern technology enables")
    ]
    
    # Load dataset
    if dataset_config is not None:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split, streaming=True, trust_remote_code=True)
    
    # Use different seed for each rank to ensure different data
    dataset_seed = seed + rank
    dataset = dataset.shuffle(seed=dataset_seed, buffer_size=10000)
    
    if rank == 0:
        print(f"Dataset shuffle seed for rank {rank}: {dataset_seed}")
    
    # Setup collator and dataloader
    collate_fn = DualStreamCollator(tokenizer, max_length=max_length, objective=objective, 
                                    mixup_alpha=mixup_alpha, same_text_prob=same_text_prob)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    model = DualStreamLLM(
        model_name=model_name,
        freeze_backbone=freeze_backbone,
        freeze_embeddings=freeze_embeddings,
        optimize_memory=optimize_memory,
        embedding_init_strategy=embedding_init_strategy,
        unembedding_init_strategy=unembedding_init_strategy,
        embedding_init_noise_std=embedding_init_noise_std,
        objective=objective,
        shared_embeddings=shared_embeddings
    )
    
    # Print parameter counts
    if rank == 0:
        _print_parameter_counts(model, freeze_backbone, objective, shared_embeddings)
    
    # Load teacher model for distillation
    teacher_model = None
    if objective == 'distillation' or objective == 'separate_distillation':
        teacher_model = _load_teacher_model(
            teacher_model_name if teacher_model_name else model_name,
            teacher_load_in_8bit,
            teacher_device,
            rank
        )
    
    # Wrap model with FSDP if distributed
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
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=device,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        output_dir=output_dir,
        tensorboard_log_dir=tensorboard_log_dir,
        learning_rate=learning_rate,
        backbone_lr=backbone_lr,
        freeze_backbone=freeze_backbone,
        num_train_steps=num_train_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        grad_clip=grad_clip,
        objective=objective,
        use_loss_balancing=use_loss_balancing,
        shared_embeddings=shared_embeddings,
        loss_balance_strength=loss_balance_strength,
        loss_balance_ema_momentum=loss_balance_ema_momentum,
        loss_balance_warmup_steps=loss_balance_warmup_steps,
        loss_balance_clip_min=loss_balance_clip_min,
        loss_balance_clip_max=loss_balance_clip_max,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor,
        log_interval=log_interval,
        similarity_log_interval=similarity_log_interval,
        eval_interval=eval_interval,
        save_interval=save_interval,
        max_gen_tokens=max_gen_tokens,
        temperature=temperature,
        validation_prompts=validation_prompts,
        resume_step=resume_step,
        resume_num_samples=resume_num_samples,
        checkpoint_data=checkpoint_data,
        teacher_model=teacher_model,
        distillation_temperature=distillation_temperature,
        distillation_type=distillation_type,
        resume_warmup_steps=resume_warmup_steps,
        resume_warmup_lr_factor=resume_warmup_lr_factor,
    )
    
    # Train
    trainer.train()
    
    # Cleanup
    if rank == 0:
        logger.close()
        if tensorboard_log_dir:
            print(f"Tensorboard logs saved to: {tensorboard_log_dir}")
    
    cleanup_distributed()

def _handle_continue_run(
    continue_run, resume_from_checkpoint, tensorboard_log_dir,
    current_batch_size, current_grad_accum
):
    """Handle continue_run logic."""
    if not os.path.exists(continue_run):
        raise ValueError(f"Continue run directory not found: {continue_run}")
    
    old_run_name = os.path.basename(continue_run)
    old_checkpoint_dir = continue_run
    
    # Construct old tensorboard directory path
    old_tensorboard_dir = None
    if tensorboard_log_dir:
        old_tensorboard_dir = os.path.join(tensorboard_log_dir, old_run_name)
    
    # Handle resume from checkpoint
    checkpoint_data = None
    resume_step = 0
    resume_num_samples = 0
    if resume_from_checkpoint:
        checkpoint_manager = CheckpointManager(old_checkpoint_dir)
        if resume_from_checkpoint == 'auto':
            checkpoint_result = checkpoint_manager.find_latest_checkpoint()
            if checkpoint_result:
                checkpoint_path, resume_step = checkpoint_result
                print(f"Found latest checkpoint: {checkpoint_path} at step {resume_step}")
            else:
                raise ValueError(f"No checkpoints found in {old_checkpoint_dir}")
        else:
            checkpoint_path = resume_from_checkpoint
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Checkpoint not found: {checkpoint_path}")
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                resume_step = checkpoint_data.get('step', 0)
            except Exception as e:
                raise ValueError(f"Failed to load checkpoint {checkpoint_path}: {e}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        resume_step = checkpoint_data.get('step', 0)
        
        # Get num_samples_seen from checkpoint, or calculate from old step if not available
        resume_num_samples = checkpoint_data.get('num_samples_seen', None)
        if resume_num_samples is None:
            # Backward compatibility: calculate from old training config
            old_config_path = os.path.join(old_checkpoint_dir, 'training_config.json')
            if os.path.exists(old_config_path):
                with open(old_config_path, 'r') as f:
                    old_config = json.load(f)
                old_batch_size = old_config.get('batch_size', current_batch_size)
                old_grad_accum = old_config.get('gradient_accumulation_steps', current_grad_accum)
                old_effective_batch = old_batch_size * old_grad_accum
                resume_num_samples = resume_step * old_effective_batch
                print(f"[Backward compatibility] Calculated num_samples_seen from old config: {resume_num_samples}")
            else:
                # Fallback: use current config
                current_effective_batch = current_batch_size * current_grad_accum
                resume_num_samples = resume_step * current_effective_batch
                print(f"[Warning] No training_config.json found, using current batch settings to estimate samples")
        
        print(f"Resuming from optimizer step {resume_step} ({resume_num_samples} samples seen)")
    
    # Create NEW run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{old_run_name}_resumed_{timestamp}"
    
    output_dir = os.path.join(os.path.dirname(old_checkpoint_dir), run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created new run directory: {output_dir}")
    
    # Create NEW tensorboard directory
    if tensorboard_log_dir:
        tensorboard_log_dir = os.path.join(tensorboard_log_dir, run_name)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    return output_dir, tensorboard_log_dir, resume_step, resume_num_samples, checkpoint_data, old_tensorboard_dir


def _load_teacher_model(model_name: str, load_in_8bit: bool, device, rank: int):
    """Load teacher model for distillation."""
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Loading teacher model: {model_name}")
        print(f"8-bit quantization: {load_in_8bit}")
        print(f"{'='*80}\n")
    
    if load_in_8bit:
        # Use BitsAndBytesConfig for 8-bit quantization
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            quantization_config=quantization_config,
            dtype=torch.float16,
        )
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
        )
        teacher = teacher.to(device)
    
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    if rank == 0:
        print(f"Teacher model loaded successfully\n")
    
    return teacher

def _create_config_dict(
    model_name, dataset_name, dataset_config, dataset_split, device,
    batch_size, learning_rate, backbone_lr, freeze_backbone, freeze_embeddings, optimize_memory,
    num_train_steps, max_length, log_interval, similarity_log_interval, eval_interval, save_interval,
    max_gen_tokens, temperature, seed, objective, use_loss_balancing, shared_embeddings,
    embedding_init_strategy, unembedding_init_strategy, embedding_init_noise_std,
    mixup_alpha, same_text_prob, loss_balance_strength, loss_balance_ema_momentum,
    loss_balance_warmup_steps, loss_balance_clip_min, loss_balance_clip_max,
    lr_scheduler_patience, lr_scheduler_factor, gradient_accumulation_steps,
    teacher_model_name, teacher_load_in_8bit, distillation_temperature, distillation_type, grad_clip
):
    """Create configuration dictionary."""
    return {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'dataset_config': dataset_config,
        'dataset_split': dataset_split,
        'device': str(device) if device else 'distributed',
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'backbone_lr': backbone_lr,
        'freeze_backbone': freeze_backbone,
        'freeze_embeddings': freeze_embeddings,
        'optimize_memory': optimize_memory,
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
        'same_text_prob': same_text_prob,
        'teacher_model_name': teacher_model_name,
        'teacher_load_in_8bit': teacher_load_in_8bit,
        'distillation_temperature': distillation_temperature,
        'distillation_type': distillation_type,
        'loss_balance_strength': loss_balance_strength,
        'loss_balance_ema_momentum': loss_balance_ema_momentum,
        'loss_balance_warmup_steps': loss_balance_warmup_steps,
        'loss_balance_clip_min': loss_balance_clip_min,
        'loss_balance_clip_max': loss_balance_clip_max,
        'lr_scheduler_patience': lr_scheduler_patience,
        'lr_scheduler_factor': lr_scheduler_factor,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'grad_clip': grad_clip,
        'world_size': 1  # Will be updated by trainer
    }

def _print_training_info(
    rank, world_size, device, model_name, dataset_name, dataset_config,
    objective, use_loss_balancing, shared_embeddings, batch_size,
    gradient_accumulation_steps, learning_rate, backbone_lr, freeze_backbone,
    freeze_embeddings, optimize_memory,
    num_train_steps, seed, output_dir, grad_clip
):
    """Print training information."""
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
    print(f"Optimize memory: {optimize_memory}")
    print(f"Batch size per GPU: {batch_size}")
    print(f"Effective global batch size: {batch_size * world_size * gradient_accumulation_steps}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Gradient clipping norm: {grad_clip}")
    print(f"\n{'='*80}")
    print(f"TRAINING PARAMETERS")
    print(f"{'='*80}")
    print(f"Learning rate (embeddings): {learning_rate}")
    print(f"Learning rate (backbone): {backbone_lr if not freeze_backbone else 'N/A (frozen)'}")
    print(f"Freeze backbone: {freeze_backbone}")
    print(f"Freeze embeddings: {freeze_embeddings}")
    print(f"Number of training steps: {num_train_steps}")
    print(f"Base seed: {seed}")
    print(f"Dataset seeds: rank_0={seed}, rank_1={seed+1}, ..." if world_size > 1 else f"Dataset seed: {seed}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


def _print_parameter_counts(model, freeze_backbone, objective, shared_embeddings):
    """Print parameter counts."""
    # Use a dict with param IDs to handle shared parameters (deduplication)
    # This ensures we don't double-count memory when optimize_memory=True
    unique_params = {id(p): p for p in model.parameters()}
    total_params = sum(p.numel() for p in unique_params.values())
    
    # Calculate total trainable params
    unique_trainable = {id(p): p for p in unique_params.values() if p.requires_grad}
    total_trainable = sum(p.numel() for p in unique_trainable.values())
    
    # Categorize trainable parameters by checking their module names
    backbone_trainable = 0
    embedding_related_trainable = 0
    
    # Track processed IDs to avoid double counting during categorization
    processed_ids = set()
    
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in processed_ids:
            processed_ids.add(id(p))
            if name.startswith('backbone'):
                backbone_trainable += p.numel()
            else:
                # Includes emb (if separate), unemb, stream_proj
                embedding_related_trainable += p.numel()

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {total_trainable:,} ({100*total_trainable/total_params:.2f}%)")
    print(f"  - Embeddings/Heads/Projections: {embedding_related_trainable:,} ({100*embedding_related_trainable/total_params:.2f}%)")
    
    if not freeze_backbone:
        print(f"  - Backbone: {backbone_trainable:,} ({100*backbone_trainable/total_params:.2f}%)")
    else:
        print(f"  - Backbone: 0 (0.00%) [Frozen]")


if __name__ == '__main__':
    fire.Fire(train)

