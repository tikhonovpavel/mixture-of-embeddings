import os
import json
from typing import Optional, Dict, Any, List
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator


class TrainingLogger:
    """Manages TensorBoard logging and event copying."""
    
    def __init__(self, log_dir: Optional[str] = None, rank: int = 0):
        self.log_dir = log_dir
        self.rank = rank
        self.writer = None
        
        if log_dir and rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir)
    
    def copy_tensorboard_events(self, source_dir: str, max_step: int):
        """Copy TensorBoard events from source to target writer up to max_step."""
        if self.rank != 0 or self.writer is None:
            return
        
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
                        self.writer.add_scalar(tag, event.value, event.step, walltime=event.wall_time)
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
                            if event.step <= max_step:
                                try:
                                    # Try to decode as text
                                    if event.tensor_proto.string_val:
                                        text_value = event.tensor_proto.string_val[0].decode('utf-8')
                                        self.writer.add_text(tag, text_value, event.step, walltime=event.wall_time)
                                        total_text_events += 1
                                except Exception:
                                    pass  # Skip if can't decode this specific event
                    except Exception:
                        pass  # Skip if can't load this tag
                
                print(f"  → Copied {total_text_events} text events")
            
            self.writer.flush()
            
            print(f"✅ Successfully copied events up to step {max_step}")
            print('='*80 + '\n')
            
        except Exception as e:
            print(f"Warning: Failed to copy TensorBoard events: {e}")
            print("Continuing without copied logs...")
            print('='*80 + '\n')
    
    def log_training_config(self, config: Dict[str, Any], script_path: Optional[str] = None):
        """Log training configuration to TensorBoard."""
        if self.writer is None:
            return
        
        # Save full training config as text
        config_json_text = json.dumps(config, indent=2)
        self.writer.add_text('config/training_config_json', f"```json\n{config_json_text}\n```")
        
        # Log individual config values
        self.writer.add_text('config/model_name', config['model_name'])
        dataset_text = config['dataset_name']
        if config.get('dataset_config'):
            dataset_text += f" ({config['dataset_config']})"
        self.writer.add_text('config/dataset_name', dataset_text)
        self.writer.add_text('config/objective', config['objective'])
        
        # Log scalar configs
        self.writer.add_scalar('config/use_loss_balancing', 1 if config['use_loss_balancing'] else 0)
        self.writer.add_scalar('config/shared_embeddings', 1 if config['shared_embeddings'] else 0)
        self.writer.add_scalar('config/batch_size', config['batch_size'])
        self.writer.add_scalar('config/learning_rate_initial', config['learning_rate'])
        self.writer.add_scalar('config/backbone_lr_initial', config['backbone_lr'])
        self.writer.add_scalar('config/freeze_backbone', 1 if config['freeze_backbone'] else 0)
        self.writer.add_scalar('config/num_train_steps', config['num_train_steps'])
        self.writer.add_scalar('config/max_length', config['max_length'])
        self.writer.add_scalar('config/seed', config['seed'])
        self.writer.add_scalar('config/world_size', config.get('world_size', 1))
        self.writer.add_scalar('config/mixup_alpha', config['mixup_alpha'])
        self.writer.add_scalar('config/loss_balance_strength', config['loss_balance_strength'])
        self.writer.add_scalar('config/loss_balance_ema_momentum', config['loss_balance_ema_momentum'])
        self.writer.add_scalar('config/loss_balance_warmup_steps', config['loss_balance_warmup_steps'])
        self.writer.add_scalar('config/loss_balance_clip_min', config['loss_balance_clip_min'])
        self.writer.add_scalar('config/loss_balance_clip_max', config['loss_balance_clip_max'])
        self.writer.add_scalar('config/lr_scheduler_patience', config['lr_scheduler_patience'])
        self.writer.add_scalar('config/lr_scheduler_factor', config['lr_scheduler_factor'])
        
        # Save script version if provided
        if script_path and os.path.exists(script_path):
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_content = f.read()
                self.writer.add_text('script_version/train_dual_stream.py', f"```python\n{script_content}\n```")
                print(f"Script version saved to tensorboard")
            except Exception as e:
                print(f"Warning: Failed to save script version: {e}")
        
        self.writer.flush()
    
    def log_generation_samples(self, samples: List[Dict[str, str]], step: int):
        """Log generation samples to TensorBoard."""
        if self.writer is None:
            return
        
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
        self.writer.add_text(f'generation_samples/step_{step:07d}', table_markdown, step)
        self.writer.flush()
    
    def log_training_metrics(
        self,
        step: int,
        total_loss: float,
        loss_A: Optional[float] = None,
        loss_B: Optional[float] = None,
        ema_loss_A: Optional[float] = None,
        ema_loss_B: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        learning_rate: Optional[float] = None,
        backbone_lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        objective: str = 'separate',
        use_loss_balancing: bool = False
    ):
        """Log training metrics to TensorBoard."""
        if self.writer is None:
            return
        
        self.writer.add_scalar('train/loss_total', total_loss, step)
        
        if learning_rate is not None:
            self.writer.add_scalar('hyperparams/learning_rate', learning_rate, step)
        if backbone_lr is not None:
            self.writer.add_scalar('hyperparams/backbone_current_lr', backbone_lr, step)
        if grad_norm is not None:
            self.writer.add_scalar('train/grad_norm', grad_norm, step)
        
        if objective == 'separate':
            if loss_A is not None:
                self.writer.add_scalar('train/loss_A', loss_A, step)
            if loss_B is not None:
                self.writer.add_scalar('train/loss_B', loss_B, step)
            
            if ema_loss_A is not None and ema_loss_B is not None:
                self.writer.add_scalar('train/ema_loss_A', ema_loss_A, step)
                self.writer.add_scalar('train/ema_loss_B', ema_loss_B, step)
            
            if use_loss_balancing and alpha is not None and beta is not None:
                self.writer.add_scalar('loss_balance/alpha', alpha, step)
                self.writer.add_scalar('loss_balance/beta', beta, step)
                self.writer.add_scalar('loss_balance/alpha_beta_ratio', alpha / beta if beta > 0 else 0, step)
                
                if ema_loss_A is not None and ema_loss_B is not None:
                    self.writer.add_scalar('loss_balance/losses_ratio', ema_loss_A / ema_loss_B if ema_loss_B > 0 else 0, step)
        
        self.writer.flush()
    
    def log_similarity_metrics(
        self,
        step: int,
        emb_similarity: float,
        emb_distance: float,
        unemb_similarity: float,
        unemb_distance: float
    ):
        """Log embedding similarity metrics."""
        if self.writer is None:
            return
        
        self.writer.add_scalar('similarity/embedding_cosine', emb_similarity, step)
        self.writer.add_scalar('similarity/unembedding_cosine', unemb_similarity, step)
        self.writer.add_scalar('distance/embedding_frobenius', emb_distance, step)
        self.writer.add_scalar('distance/unembedding_frobenius', unemb_distance, step)
        self.writer.flush()
    
    def log_config_change(self, step: int, changes: Dict[str, tuple]):
        """Log configuration changes."""
        if self.writer is None or not changes:
            return
        
        change_description = ", ".join([f"{k}: {v[0]} -> {v[1]}" for k, v in changes.items()])
        event_text = f"Step {step}: {change_description}"
        self.writer.add_text('events/config_changes', event_text, step)
        
        # Log changed hyperparameters as scalars
        if 'learning_rate' in changes:
            self.writer.add_scalar('hyperparams/learning_rate', changes['learning_rate'][1], step)
        if 'backbone_lr' in changes:
            self.writer.add_scalar('hyperparams/backbone_lr', changes['backbone_lr'][1], step)
        if 'temperature' in changes:
            self.writer.add_scalar('hyperparams/temperature', changes['temperature'][1], step)
        if 'loss_balance_strength' in changes:
            self.writer.add_scalar('hyperparams/loss_balance_strength', changes['loss_balance_strength'][1], step)
        if 'loss_balance_ema_momentum' in changes:
            self.writer.add_scalar('hyperparams/loss_balance_ema_momentum', changes['loss_balance_ema_momentum'][1], step)
        if 'loss_balance_clip_min' in changes:
            self.writer.add_scalar('hyperparams/loss_balance_clip_min', changes['loss_balance_clip_min'][1], step)
        if 'loss_balance_clip_max' in changes:
            self.writer.add_scalar('hyperparams/loss_balance_clip_max', changes['loss_balance_clip_max'][1], step)
        
        self.writer.flush()
    
    def log_initial_hyperparams(self, config: Dict[str, Any], alpha: float, beta: float, 
                                 objective: str, use_loss_balancing: bool,
                                 ema_loss_A: Optional[float] = None, ema_loss_B: Optional[float] = None):
        """Log initial hyperparameters at step 0."""
        if self.writer is None:
            return
        
        self.writer.add_scalar('hyperparams/learning_rate', config['learning_rate'], 0)
        self.writer.add_scalar('hyperparams/backbone_lr', config['backbone_lr'], 0)
        self.writer.add_scalar('hyperparams/temperature', config['temperature'], 0)
        self.writer.add_scalar('hyperparams/loss_balance_strength', config['loss_balance_strength'], 0)
        self.writer.add_scalar('hyperparams/loss_balance_ema_momentum', config['loss_balance_ema_momentum'], 0)
        self.writer.add_scalar('hyperparams/loss_balance_clip_min', config['loss_balance_clip_min'], 0)
        self.writer.add_scalar('hyperparams/loss_balance_clip_max', config['loss_balance_clip_max'], 0)
        
        # Log initial loss balancing coefficients (only for separate objective)
        if objective == 'separate' and use_loss_balancing:
            self.writer.add_scalar('loss_balance/alpha', alpha, 0)
            self.writer.add_scalar('loss_balance/beta', beta, 0)
            if ema_loss_A is not None and ema_loss_B is not None:
                self.writer.add_scalar('loss_balance/losses_ratio', ema_loss_A / ema_loss_B if ema_loss_B > 0 else 0, 0)
        
        self.writer.flush()
    
    def log_scheduler_loss(self, step: int, loss: float):
        """Log loss for scheduler."""
        if self.writer is None:
            return
        self.writer.add_scalar('train/loss_for_scheduler', loss, step)
        self.writer.flush()
    
    def close(self):
        """Close the writer."""
        if self.writer is not None:
            self.writer.close()

