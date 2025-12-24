import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import functools
import numpy as np
from datetime import datetime

from checkpoint_manager import CheckpointManager
from config_manager import ConfigManager
from logger import TrainingLogger
from generation import generate_samples
from utils import compute_weight_similarity, compute_weight_distance


class Trainer:
    """Main trainer class for dual-stream LLM training."""
    
    def __init__(
        self,
        model,
        tokenizer,
        dataloader: DataLoader,
        device,
        rank: int,
        world_size: int,
        local_rank: int,
        output_dir: str,
        tensorboard_log_dir: str,
        # Training parameters
        learning_rate: float,
        backbone_lr: float,
        freeze_backbone: bool,
        num_train_steps: int,
        gradient_accumulation_steps: int,
        # Objectives and loss
        objective: str,
        use_loss_balancing: bool,
        shared_embeddings: bool,
        loss_balance_strength: float,
        loss_balance_ema_momentum: float,
        loss_balance_warmup_steps: int,
        loss_balance_clip_min: float,
        loss_balance_clip_max: float,
        # LR Scheduler
        lr_scheduler_patience: int,
        lr_scheduler_factor: float,
        # Logging intervals
        log_interval: int,
        similarity_log_interval: int,
        eval_interval: int,
        save_interval: int,
        # Generation
        max_gen_tokens: int,
        temperature: float,
        validation_prompts: list,
        # Resume
        resume_step: int = 0,
        checkpoint_data: dict = None,
        # Distillation
        teacher_model = None,
        distillation_temperature: float = 2.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.output_dir = output_dir
        
        self.learning_rate = learning_rate
        self.backbone_lr = backbone_lr
        self.freeze_backbone = freeze_backbone
        self.num_train_steps = num_train_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.objective = objective
        self.use_loss_balancing = use_loss_balancing
        self.shared_embeddings = shared_embeddings
        self.loss_balance_strength = loss_balance_strength
        self.loss_balance_ema_momentum = loss_balance_ema_momentum
        self.loss_balance_warmup_steps = loss_balance_warmup_steps
        self.loss_balance_clip_min = loss_balance_clip_min
        self.loss_balance_clip_max = loss_balance_clip_max
        
        self.log_interval = log_interval
        self.similarity_log_interval = similarity_log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.validation_prompts = validation_prompts
        
        self.resume_step = resume_step
        
        self.teacher_model = teacher_model
        self.distillation_temperature = distillation_temperature
        
        # Setup managers
        self.checkpoint_manager = CheckpointManager(output_dir)
        self.config_manager = ConfigManager(os.path.join(output_dir, 'live_config.json'))
        self.logger = TrainingLogger(tensorboard_log_dir, rank)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience
        )
        
        # Setup loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Initialize loss balancing variables
        self.ema_loss_A = None
        self.ema_loss_B = None
        self.alpha = 1.0
        self.beta = 1.0
        
        # Load checkpoint if provided
        if checkpoint_data:
            self._load_checkpoint(checkpoint_data)
        
        # Setup live config
        self.live_config = self._setup_live_config()
        self.live_config_check_interval = 10
    
    def _setup_optimizer(self):
        """Setup optimizer with parameter groups."""
        if self.objective == 'separate':
            if self.shared_embeddings:
                trainable_params = [
                    self.model.emb.parameters(),
                    self.model.unemb_A.parameters(),
                    self.model.unemb_B.parameters()
                ]
            else:
                trainable_params = [
                    self.model.emb_A.parameters(),
                    self.model.emb_B.parameters(),
                    self.model.unemb_A.parameters(),
                    self.model.unemb_B.parameters()
                ]
        else:  # averaged, mixup, or distillation
            if self.shared_embeddings:
                trainable_params = [
                    self.model.emb.parameters(),
                    self.model.unemb.parameters()
                ]
            else:
                trainable_params = [
                    self.model.emb_A.parameters(),
                    self.model.emb_B.parameters(),
                    self.model.unemb.parameters()
                ]
        
        # Add stream tags to trainable parameters
        stream_params = list(self.model.stream_proj_A.parameters()) + \
                        list(self.model.stream_proj_B.parameters())
        self.embedding_params = [p for group in trainable_params for p in group] + stream_params
        
        if self.freeze_backbone:
            self.optimizer = torch.optim.AdamW(self.embedding_params, lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW([
                {'params': self.embedding_params, 'lr': self.learning_rate},
                {'params': self.model.backbone.parameters(), 'lr': self.backbone_lr}
            ])
    
    def _setup_live_config(self):
        """Setup or load live configuration."""
        config = {
            "learning_rate": self.learning_rate,
            "backbone_lr": self.backbone_lr,
            "temperature": self.temperature,
            "log_interval": self.log_interval,
            "similarity_log_interval": self.similarity_log_interval,
            "eval_interval": self.eval_interval,
            "loss_balance_strength": self.loss_balance_strength,
            "loss_balance_ema_momentum": self.loss_balance_ema_momentum,
            "loss_balance_clip_min": self.loss_balance_clip_min,
            "loss_balance_clip_max": self.loss_balance_clip_max
        }
        self.config_manager.save_config(config)
        return config
    
    def _load_checkpoint(self, checkpoint_data: dict):
        """Load model, optimizer, scheduler, and loss balancing state from checkpoint."""
        print("Loading model state dict...")
        state_dict = checkpoint_data['model_state_dict']
        # Filter out duplicate transformer keys if they exist (backward compatibility)
        keys_to_remove = [k for k in state_dict.keys() if k.startswith('transformer.')]
        if keys_to_remove:
            print(f"Removing {len(keys_to_remove)} duplicate 'transformer' keys from checkpoint for backward compatibility.")
            for k in keys_to_remove:
                del state_dict[k]
        self.model.load_state_dict(state_dict)
        print("Loading optimizer state dict...")  
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint_data:
            self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            print("Loading scheduler state dict...")
        
        # Restore loss balancing variables
        self.ema_loss_A = checkpoint_data.get('ema_loss_A', None)
        self.ema_loss_B = checkpoint_data.get('ema_loss_B', None)
        self.alpha = checkpoint_data.get('alpha', 1.0)
        self.beta = checkpoint_data.get('beta', 1.0)
        
        def _fmt_val(x):
            try:
                return f"{float(x):.4f}"
            except (TypeError, ValueError):
                return "None"
        
        print(
            "Restored loss balancing state: "
            f"ema_A={_fmt_val(self.ema_loss_A)}, "
            f"ema_B={_fmt_val(self.ema_loss_B)}, "
            f"alpha={_fmt_val(self.alpha)}, "
            f"beta={_fmt_val(self.beta)}"
        )
        print("Model, optimizer, and scheduler states restored successfully")
    
    def _compute_loss_separate(self, batch):
        """Compute loss for 'separate' objective."""
        input_ids_A = batch['input_ids_A'].to(self.device)
        attention_mask_A = batch['attention_mask_A'].to(self.device)
        input_ids_B = batch['input_ids_B'].to(self.device)
        attention_mask_B = batch['attention_mask_B'].to(self.device)
        labels_A = batch['labels_A'].to(self.device)
        labels_B = batch['labels_B'].to(self.device)
        is_same_text = batch.get('is_same_text', None)
        
        logits_A, logits_B = self.model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B)
        
        # Compute per-sample losses for separate logging
        batch_size = input_ids_A.size(0)
        seq_len = input_ids_A.size(1)
        vocab_size = logits_A.size(-1)
        
        # Reshape for per-sample loss computation
        loss_A_per_sample = F.cross_entropy(
            logits_A.view(batch_size * seq_len, vocab_size),
            labels_A.view(batch_size * seq_len),
            reduction='none'
        ).view(batch_size, seq_len)
        loss_B_per_sample = F.cross_entropy(
            logits_B.view(batch_size * seq_len, vocab_size),
            labels_B.view(batch_size * seq_len),
            reduction='none'
        ).view(batch_size, seq_len)
        
        # Mask out padding (-100)
        valid_mask_A = (labels_A != -100).float()
        valid_mask_B = (labels_B != -100).float()
        
        # Per-sample mean loss
        loss_A_sample = (loss_A_per_sample * valid_mask_A).sum(dim=1) / (valid_mask_A.sum(dim=1) + 1e-8)
        loss_B_sample = (loss_B_per_sample * valid_mask_B).sum(dim=1) / (valid_mask_B.sum(dim=1) + 1e-8)
        
        # Total loss (mean over batch)
        loss_A = loss_A_sample.mean()
        loss_B = loss_B_sample.mean()
        
        # Compute separate losses for same_text and different_text cases
        loss_same = None
        loss_diff = None
        if is_same_text is not None:
            is_same_text = is_same_text.to(self.device)
            same_mask = is_same_text.float()
            diff_mask = (~is_same_text).float()
            
            num_same = same_mask.sum()
            num_diff = diff_mask.sum()
            
            total_loss_sample = loss_A_sample + loss_B_sample
            
            if num_same > 0:
                loss_same = (total_loss_sample * same_mask).sum() / num_same
            if num_diff > 0:
                loss_diff = (total_loss_sample * diff_mask).sum() / num_diff
        
        # Update EMA for loss balancing
        if self.use_loss_balancing:
            current_ema_momentum = self.live_config.get('loss_balance_ema_momentum', self.loss_balance_ema_momentum)
            if self.ema_loss_A is None:
                self.ema_loss_A = loss_A.item()
                self.ema_loss_B = loss_B.item()
            else:
                self.ema_loss_A = current_ema_momentum * self.ema_loss_A + (1 - current_ema_momentum) * loss_A.item()
                self.ema_loss_B = current_ema_momentum * self.ema_loss_B + (1 - current_ema_momentum) * loss_B.item()
        
        return loss_A, loss_B, loss_same, loss_diff
    
    def _compute_loss_averaged(self, batch):
        """Compute loss for 'averaged' objective."""
        input_ids_A = batch['input_ids_A'].to(self.device)
        attention_mask_A = batch['attention_mask_A'].to(self.device)
        input_ids_B = batch['input_ids_B'].to(self.device)
        attention_mask_B = batch['attention_mask_B'].to(self.device)
        next_tokens_A = batch['next_tokens_A'].to(self.device)
        next_tokens_B = batch['next_tokens_B'].to(self.device)
        pad_token_id = batch['pad_token_id']
        is_same_text = batch.get('is_same_text', None)
        
        logits = self.model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B)
        
        # Use logits[:, :-1, :] to match with next_tokens
        logits_shifted = logits[:, :-1, :].float()
        
        # Create mask for valid (non-padding) positions
        valid_mask = (input_ids_A[:, :-1] != pad_token_id) & (input_ids_B[:, :-1] != pad_token_id)
        valid_mask = valid_mask & (next_tokens_A != pad_token_id) & (next_tokens_B != pad_token_id)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits_shifted, dim=-1)
        
        # Gather log probs for both target tokens
        log_prob_A = torch.gather(log_probs, dim=-1, index=next_tokens_A.unsqueeze(-1)).squeeze(-1)
        log_prob_B = torch.gather(log_probs, dim=-1, index=next_tokens_B.unsqueeze(-1)).squeeze(-1)
        
        # Per-position loss
        loss_per_pos = 0.5 * (-log_prob_A + -log_prob_B)
        
        # Per-sample loss
        valid_mask_float = valid_mask.float()
        loss_per_sample = (loss_per_pos * valid_mask_float).sum(dim=1) / (valid_mask_float.sum(dim=1) + 1e-8)
        
        # Total loss
        loss = loss_per_sample.mean()
        
        # Compute separate losses for same_text and different_text cases
        loss_same = None
        loss_diff = None
        if is_same_text is not None:
            is_same_text = is_same_text.to(self.device)
            same_mask = is_same_text.float()
            diff_mask = (~is_same_text).float()
            
            num_same = same_mask.sum()
            num_diff = diff_mask.sum()
            
            if num_same > 0:
                loss_same = (loss_per_sample * same_mask).sum() / num_same
            if num_diff > 0:
                loss_diff = (loss_per_sample * diff_mask).sum() / num_diff
        
        return loss, loss, loss_same, loss_diff
    
    def _compute_loss_mixup(self, batch):
        """Compute loss for 'mixup' objective."""
        input_ids_A = batch['input_ids_A'].to(self.device)
        attention_mask_A = batch['attention_mask_A'].to(self.device)
        input_ids_B = batch['input_ids_B'].to(self.device)
        attention_mask_B = batch['attention_mask_B'].to(self.device)
        next_tokens_A = batch['next_tokens_A'].to(self.device)
        next_tokens_B = batch['next_tokens_B'].to(self.device)
        lambdas = batch['lambdas'].to(self.device)
        pad_token_id = batch['pad_token_id']
        is_same_text = batch.get('is_same_text', None)
        
        logits = self.model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, lambdas=lambdas)
        
        # Use logits[:, :-1, :] to match with next_tokens
        logits_shifted = logits[:, :-1, :].float()
        
        # Create mask for valid (non-padding) positions
        valid_mask = (input_ids_A[:, :-1] != pad_token_id) & (input_ids_B[:, :-1] != pad_token_id)
        valid_mask = valid_mask & (next_tokens_A != pad_token_id) & (next_tokens_B != pad_token_id)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits_shifted, dim=-1)
        
        # Gather log probs for both target tokens
        log_prob_A = torch.gather(log_probs, dim=-1, index=next_tokens_A.unsqueeze(-1)).squeeze(-1)
        log_prob_B = torch.gather(log_probs, dim=-1, index=next_tokens_B.unsqueeze(-1)).squeeze(-1)
        
        # Expand lambdas to [batch_size, seq_len]
        lambdas_expanded = lambdas.view(-1, 1).to(log_prob_A.dtype)
        
        # Mixup loss: -[λ * log(P_A) + (1-λ) * log(P_B)]
        loss_per_pos = -(lambdas_expanded * log_prob_A + (1 - lambdas_expanded) * log_prob_B)
        
        # Per-sample loss
        valid_mask_float = valid_mask.float()
        loss_per_sample = (loss_per_pos * valid_mask_float).sum(dim=1) / (valid_mask_float.sum(dim=1) + 1e-8)
        
        # Total loss
        loss = loss_per_sample.mean()
        
        # Compute separate losses for same_text and different_text cases
        loss_same = None
        loss_diff = None
        if is_same_text is not None:
            is_same_text = is_same_text.to(self.device)
            same_mask = is_same_text.float()
            diff_mask = (~is_same_text).float()
            
            num_same = same_mask.sum()
            num_diff = diff_mask.sum()
            
            if num_same > 0:
                loss_same = (loss_per_sample * same_mask).sum() / num_same
            if num_diff > 0:
                loss_diff = (loss_per_sample * diff_mask).sum() / num_diff
        
        return loss, loss, loss_same, loss_diff
    
    def _compute_loss_distillation(self, batch):
        """Compute loss for 'distillation' objective using KL divergence."""
        input_ids_A = batch['input_ids_A'].to(self.device)
        attention_mask_A = batch['attention_mask_A'].to(self.device)
        input_ids_B = batch['input_ids_B'].to(self.device)
        attention_mask_B = batch['attention_mask_B'].to(self.device)
        pad_token_id = batch['pad_token_id']
        is_same_text = batch.get('is_same_text', None)
        
        # Get teacher predictions for both streams separately
        with torch.no_grad():
            # Teacher predictions for stream A
            teacher_outputs_A = self.teacher_model(
                input_ids=input_ids_A,
                attention_mask=attention_mask_A,
            )
            teacher_logits_A = teacher_outputs_A.logits[:, :-1, :].float()
            
            # Teacher predictions for stream B
            teacher_outputs_B = self.teacher_model(
                input_ids=input_ids_B,
                attention_mask=attention_mask_B,
            )
            teacher_logits_B = teacher_outputs_B.logits[:, :-1, :].float()
            
            # Apply temperature and compute soft targets
            teacher_probs_A = F.softmax(teacher_logits_A / self.distillation_temperature, dim=-1)
            teacher_probs_B = F.softmax(teacher_logits_B / self.distillation_temperature, dim=-1)
            
            # Average the teacher distributions: P_target = 0.5 * P_A + 0.5 * P_B
            target_probs = 0.5 * teacher_probs_A + 0.5 * teacher_probs_B
        
        # Get student predictions on mixed embeddings
        student_logits = self.model(input_ids_A, attention_mask_A, input_ids_B, attention_mask_B)
        student_logits = student_logits[:, :-1, :].float()
        
        # Apply temperature to student logits
        student_log_probs = F.log_softmax(student_logits / self.distillation_temperature, dim=-1)
        
        # Create mask for valid (non-padding) positions
        valid_mask = (input_ids_A[:, :-1] != pad_token_id) & (input_ids_B[:, :-1] != pad_token_id)
        
        # Compute KL divergence: KL(target || student) = sum(target * log(target/student))
        # = sum(target * (log(target) - log(student)))
        kl_div_per_pos = (target_probs * (torch.log(target_probs + 1e-10) - student_log_probs)).sum(dim=-1)
        
        # Per-sample loss
        valid_mask_float = valid_mask.float()
        loss_per_sample = (kl_div_per_pos * valid_mask_float).sum(dim=1) / (valid_mask_float.sum(dim=1) + 1e-8)
        
        # Total loss
        loss = loss_per_sample.mean()
        
        # Scale loss by temperature^2 (standard practice in distillation)
        loss = loss * (self.distillation_temperature ** 2)
        loss_per_sample = loss_per_sample * (self.distillation_temperature ** 2)
        
        # Compute separate losses for same_text and different_text cases
        loss_same = None
        loss_diff = None
        if is_same_text is not None:
            is_same_text = is_same_text.to(self.device)
            same_mask = is_same_text.float()
            diff_mask = (~is_same_text).float()
            
            num_same = same_mask.sum()
            num_diff = diff_mask.sum()
            
            if num_same > 0:
                loss_same = (loss_per_sample * same_mask).sum() / num_same
            if num_diff > 0:
                loss_diff = (loss_per_sample * diff_mask).sum() / num_diff
        
        return loss, loss, loss_same, loss_diff
    
    def _update_loss_balancing_coefficients(self, step: int):
        """Update alpha and beta coefficients for loss balancing."""
        if not self.use_loss_balancing or self.objective != 'separate':
            return 1.0, 1.0
        
        if step >= self.loss_balance_warmup_steps:
            current_strength = self.live_config.get('loss_balance_strength', self.loss_balance_strength)
            current_clip_min = self.live_config.get('loss_balance_clip_min', self.loss_balance_clip_min)
            current_clip_max = self.live_config.get('loss_balance_clip_max', self.loss_balance_clip_max)
            
            # Compute weights based on EMA losses
            total_loss = self.ema_loss_A + self.ema_loss_B
            weight_A = (self.ema_loss_A / total_loss) ** current_strength
            weight_B = (self.ema_loss_B / total_loss) ** current_strength
            
            # Normalize so alpha + beta = 2
            norm = weight_A + weight_B
            alpha = weight_A / norm * 2.0
            beta = weight_B / norm * 2.0
            
            # Apply clipping
            alpha = torch.clamp(torch.tensor(alpha), current_clip_min, current_clip_max).item()
            beta = torch.clamp(torch.tensor(beta), current_clip_min, current_clip_max).item()
            
            return alpha, beta
        else:
            return 1.0, 1.0
    
    def _check_and_apply_config_changes(self, step: int):
        """Check for live config changes and apply them."""
        if self.rank != 0 or (step + 1) % self.live_config_check_interval != 0:
            return
        
        new_config = self.config_manager.load_config()
        if new_config is None:
            return
        
        changes = self.config_manager.apply_config_changes(
            self.live_config, new_config, self.optimizer, self.freeze_backbone
        )
        
        if changes:
            change_description = ", ".join([f"{k}: {v[0]} -> {v[1]}" for k, v in changes.items()])
            print(f"\n{'='*80}")
            print(f"[Step {step+1}] Configuration changed: {change_description}")
            print(f"{'='*80}\n")
            
            # Log changes
            optimizer_step = (step + 1) // self.gradient_accumulation_steps
            self.logger.log_config_change(optimizer_step, changes)
        
        # Update intervals from live config
        self.log_interval = self.live_config.get("log_interval", self.log_interval)
        self.similarity_log_interval = self.live_config.get("similarity_log_interval", self.similarity_log_interval)
        self.eval_interval = self.live_config.get("eval_interval", self.eval_interval)
    
    def _log_similarity_metrics(self, optimizer_step: int):
        """Log embedding similarity metrics."""
        if self.rank != 0 or self.objective != 'separate' or self.shared_embeddings:
            return
        
        with torch.no_grad():
            emb_A_weight = self.model.emb_A.weight if not isinstance(self.model, FSDP) else self.model.module.emb_A.weight
            emb_B_weight = self.model.emb_B.weight if not isinstance(self.model, FSDP) else self.model.module.emb_B.weight
            unemb_A_weight = self.model.unemb_A.weight if not isinstance(self.model, FSDP) else self.model.module.unemb_A.weight
            unemb_B_weight = self.model.unemb_B.weight if not isinstance(self.model, FSDP) else self.model.module.unemb_B.weight
            
            emb_similarity = compute_weight_similarity(emb_A_weight, emb_B_weight)
            emb_distance = compute_weight_distance(emb_A_weight, emb_B_weight)
            unemb_similarity = compute_weight_similarity(unemb_A_weight, unemb_B_weight)
            unemb_distance = compute_weight_distance(unemb_A_weight, unemb_B_weight)
            
            print(f"  [Similarity] Emb: {emb_similarity:.4f}, Unemb: {unemb_similarity:.4f} | "
                  f"[Distance] Emb: {emb_distance:.4f}, Unemb: {unemb_distance:.4f}")
            
            self.logger.log_similarity_metrics(optimizer_step, emb_similarity, emb_distance, 
                                               unemb_similarity, unemb_distance)
    
    def _generate_and_log_samples(self, optimizer_step: int):
        """Generate and log text samples."""
        if self.rank != 0:
            return
        
        print(f"\n{'='*80}")
        print(f"Generation samples at step {optimizer_step}")
        print('='*80)
        
        samples = generate_samples(
            self.model, self.tokenizer, self.validation_prompts, 
            self.device, max_new_tokens=self.max_gen_tokens, 
            temperature=self.live_config['temperature'], 
            objective=self.objective
        )
        
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  Prompt A: {sample['prompt_A']}")
            print(f"  Generated A: {sample['generated_A']}")
            print(f"  Prompt B: {sample['prompt_B']}")
            print(f"  Generated B: {sample['generated_B']}")
        
        print('='*80 + '\n')
        
        self.logger.log_generation_samples(samples, optimizer_step)
    
    def train(self):
        """Main training loop."""
        self.model.train()
        data_iter = iter(self.dataloader)
        
        # Fast-forward dataloader if resuming
        if self.resume_step > 0:
            self._fast_forward_dataloader(data_iter)
        
        # Initial generation samples
        if self.rank == 0:
            print(f"\n{'='*80}")
            print("Initial generation samples (step 0 - before training)")
            print('='*80)
            self._generate_and_log_samples(0)
        
        # Log initial hyperparameters
        if self.rank == 0:
            self.logger.log_initial_hyperparams(
                self.live_config, self.alpha, self.beta, 
                self.objective, self.use_loss_balancing,
                self.ema_loss_A, self.ema_loss_B
            )
        
        # Accumulator variables
        total_loss_acc_log = 0.0
        loss_A_acc_log = 0.0
        loss_B_acc_log = 0.0
        total_loss_acc_scheduler = 0.0
        scheduler_steps_acc = 0
        
        # Same/different text loss accumulators
        loss_same_acc_log = 0.0
        loss_diff_acc_log = 0.0
        num_same_batches = 0
        num_diff_batches = 0
        
        num_optimizer_steps = self.num_train_steps // self.gradient_accumulation_steps
        
        # Progress bar
        pbar = None
        if self.rank == 0:
            pbar = tqdm(range(self.resume_step, self.num_train_steps), 
                        initial=self.resume_step, 
                        total=self.num_train_steps, 
                        desc="Training Steps")
        else:
            pbar = range(self.resume_step, self.num_train_steps)
        
        for step in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
            
            # Compute loss based on objective
            if self.objective == 'separate':
                loss_A, loss_B, loss_same, loss_diff = self._compute_loss_separate(batch)
                self.alpha, self.beta = self._update_loss_balancing_coefficients(step)
                loss = self.alpha * loss_A + self.beta * loss_B
            elif self.objective == 'averaged':
                loss, loss_A, loss_same, loss_diff = self._compute_loss_averaged(batch)
                loss_B = loss_A
                self.alpha, self.beta = 1.0, 1.0
            elif self.objective == 'mixup':
                loss, loss_A, loss_same, loss_diff = self._compute_loss_mixup(batch)
                loss_B = loss_A
                self.alpha, self.beta = 1.0, 1.0
            elif self.objective == 'distillation':
                loss, loss_A, loss_same, loss_diff = self._compute_loss_distillation(batch)
                loss_B = loss_A
                self.alpha, self.beta = 1.0, 1.0
            
            # Non-finite loss guard
            if not torch.isfinite(loss):
                self._handle_nonfinite_loss(step, batch, loss)
            
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            # Accumulate losses for logging
            total_loss_acc_log += loss.item()
            loss_A_acc_log += loss_A.item()
            loss_B_acc_log += loss_B.item()
            
            # Accumulate same/different text losses
            if loss_same is not None:
                loss_same_acc_log += loss_same.item()
                num_same_batches += 1
            if loss_diff is not None:
                loss_diff_acc_log += loss_diff.item()
                num_diff_batches += 1
            
            # Optimizer step
            if (step + 1) % self.gradient_accumulation_steps == 0:
                optimizer_step = (step + 1) // self.gradient_accumulation_steps
                
                # Gradient clipping
                grad_norm_value = self._clip_gradients()
                if grad_norm_value is not None and self.rank == 0:
                    self._check_gradient_norm(optimizer_step, grad_norm_value, step, batch)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Accumulate loss for scheduler
                avg_loss_for_step = total_loss_acc_log if self.gradient_accumulation_steps == 1 else total_loss_acc_log * self.gradient_accumulation_steps / self.gradient_accumulation_steps
                total_loss_acc_scheduler += avg_loss_for_step
                scheduler_steps_acc += 1
                
                # Check for live config changes
                self._check_and_apply_config_changes(step)
                
                # Logging
                if self.rank == 0 and optimizer_step % self.log_interval == 0:
                    # Compute average same/diff losses
                    avg_loss_same = loss_same_acc_log / num_same_batches if num_same_batches > 0 else None
                    avg_loss_diff = loss_diff_acc_log / num_diff_batches if num_diff_batches > 0 else None
                    
                    self._log_training_step(optimizer_step, num_optimizer_steps, pbar, 
                                            avg_loss_for_step, loss_A_acc_log, loss_B_acc_log, 
                                            grad_norm_value, avg_loss_same, avg_loss_diff)
                    total_loss_acc_log = 0.0
                    loss_A_acc_log = 0.0
                    loss_B_acc_log = 0.0
                    loss_same_acc_log = 0.0
                    loss_diff_acc_log = 0.0
                    num_same_batches = 0
                    num_diff_batches = 0
                
                # Similarity logging
                if self.rank == 0 and optimizer_step > 0 and optimizer_step % self.similarity_log_interval == 0:
                    self._log_similarity_metrics(optimizer_step)
                
                # Evaluation and scheduler step
                if self.rank == 0 and optimizer_step > 0 and optimizer_step % self.eval_interval == 0:
                    # Step scheduler
                    if scheduler_steps_acc > 0:
                        avg_loss_for_scheduler = total_loss_acc_scheduler / scheduler_steps_acc
                        self.scheduler.step(avg_loss_for_scheduler)
                        self.logger.log_scheduler_loss(optimizer_step, avg_loss_for_scheduler)
                        total_loss_acc_scheduler = 0.0
                        scheduler_steps_acc = 0
                    
                    # Generate samples
                    self._generate_and_log_samples(optimizer_step)
                
                # Checkpointing
                if self.rank == 0 and optimizer_step > 0 and optimizer_step % self.save_interval == 0:
                    self._save_checkpoint(step + 1)
        
        # Training complete
        if self.rank == 0:
            if pbar and hasattr(pbar, 'close'):
                pbar.close()
            print("Training completed!")
    
    def _fast_forward_dataloader(self, data_iter):
        """Fast-forward dataloader to resume position."""
        try:
            print(f"[Rank {self.rank}] Fast-forwarding dataloader by {self.resume_step} batches to resume position...")
            if self.rank == 0:
                ffbar = tqdm(range(self.resume_step), desc=f"Fast-forward dataloader (rank {self.rank})")
            else:
                ffbar = range(self.resume_step)
            for _ in ffbar:
                try:
                    _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    _ = next(data_iter)
            print(f"[Rank {self.rank}] Fast-forward complete. Resuming training from step {self.resume_step}.")
        except Exception as e:
            print(f"[Rank {self.rank}] Warning: failed to fast-forward dataloader ({e}).")
    
    def _clip_gradients(self):
        """Clip gradients and return gradient norm."""
        try:
            if isinstance(self.model, FSDP):
                return FSDP.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            else:
                return torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        except Exception:
            return None
    
    def _check_gradient_norm(self, optimizer_step: int, grad_norm_value, step: int, batch):
        """Check gradient norm and handle non-finite values."""
        try:
            grad_norm_scalar = float(grad_norm_value)
        except Exception:
            grad_norm_scalar = grad_norm_value.item() if hasattr(grad_norm_value, 'item') else None
        
        if grad_norm_scalar is not None:
            if not np.isfinite(grad_norm_scalar):
                msg = f"Non-finite grad norm at optimizer step {optimizer_step}: grad_norm={grad_norm_scalar}"
                print(f"\n{'='*80}\n{msg}\n{'='*80}")
                problem_batch_path = os.path.join(self.output_dir, f"nonfinite_grad_batch_step_{step+1}.pt")
                try:
                    torch.save(batch, problem_batch_path)
                    print(f"Saved batch to {problem_batch_path}")
                except Exception:
                    pass
                raise RuntimeError(f"Non-finite grad norm at step {step+1}")
    
    def _handle_nonfinite_loss(self, step: int, batch, loss):
        """Handle non-finite loss."""
        if self.rank == 0:
            msg = f"Non-finite loss detected at step {step+1}: loss={loss.item()}"
            print(f"\n{'='*80}\n{msg}\n{'='*80}")
            if 'input_ids_A' in batch and 'input_ids_B' in batch:
                try:
                    print("Sample A:", self.tokenizer.decode(batch['input_ids_A'][0], skip_special_tokens=False))
                    print("Sample B:", self.tokenizer.decode(batch['input_ids_B'][0], skip_special_tokens=False))
                except Exception:
                    pass
            problem_batch_path = os.path.join(self.output_dir, f"nonfinite_loss_batch_step_{step+1}.pt")
            try:
                torch.save(batch, problem_batch_path)
                print(f"Saved batch to {problem_batch_path}")
            except Exception:
                pass
        raise RuntimeError(f"Non-finite loss at step {step+1}")
    
    def _log_training_step(self, optimizer_step: int, num_optimizer_steps: int, pbar, 
                           avg_total_loss: float, loss_A_acc: float, loss_B_acc: float,
                           grad_norm_value, avg_loss_same: float = None, avg_loss_diff: float = None):
        """Log training metrics for current step."""
        avg_loss_A = loss_A_acc / self.gradient_accumulation_steps
        avg_loss_B = loss_B_acc / self.gradient_accumulation_steps
        current_lr = self.optimizer.param_groups[0]['lr']
        current_backbone_lr = self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else None
        
        # Update progress bar
        if self.objective == 'separate':
            postfix_metrics = {
                'Opt Step': f"{optimizer_step}/{num_optimizer_steps}",
                'Loss': f"{avg_total_loss:.4f}",
                'Loss A': f"{avg_loss_A:.4f}",
                'Loss B': f"{avg_loss_B:.4f}",
                'LR': f"{current_lr:.2e}",
                'Backbone LR': f"{current_backbone_lr:.2e}" if current_backbone_lr is not None else 'N/A',
                'Alpha': f"{self.alpha:.4f}",
                'Beta': f"{self.beta:.4f}",
            }
            if avg_loss_same is not None:
                postfix_metrics['Same'] = f"{avg_loss_same:.4f}"
            if avg_loss_diff is not None:
                postfix_metrics['Diff'] = f"{avg_loss_diff:.4f}"
            if pbar and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(postfix_metrics)
        else:
            postfix_metrics = {
                'Opt Step': f"{optimizer_step}/{num_optimizer_steps}",
                'Loss': f"{avg_total_loss:.4f}",
                'LR': f"{current_lr:.2e}",
                'Backbone LR': f"{current_backbone_lr:.2e}" if current_backbone_lr is not None else 'N/A',
            }
            if avg_loss_same is not None:
                postfix_metrics['Same'] = f"{avg_loss_same:.4f}"
            if avg_loss_diff is not None:
                postfix_metrics['Diff'] = f"{avg_loss_diff:.4f}"
            if pbar and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(postfix_metrics)
        
        # Log to TensorBoard
        grad_norm_scalar = None
        if grad_norm_value is not None:
            try:
                grad_norm_scalar = float(grad_norm_value)
            except Exception:
                grad_norm_scalar = grad_norm_value.item() if hasattr(grad_norm_value, 'item') else None
        
        self.logger.log_training_metrics(
            optimizer_step, avg_total_loss, avg_loss_A, avg_loss_B,
            self.ema_loss_A, self.ema_loss_B, self.alpha, self.beta,
            current_lr, current_backbone_lr, grad_norm_scalar,
            self.objective, self.use_loss_balancing,
            avg_loss_same, avg_loss_diff
        )
    
    def _save_checkpoint(self, step: int):
        """Save a checkpoint."""
        if self.world_size > 1:
            state_dict = self.model.state_dict()
        else:
            state_dict = self.model.state_dict()
        
        self.checkpoint_manager.save_checkpoint(
            step, state_dict, self.optimizer.state_dict(),
            self.scheduler.state_dict(), self.alpha, self.beta,
            self.ema_loss_A, self.ema_loss_B
        )

