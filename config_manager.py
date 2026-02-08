import json
import os
from typing import Optional, Dict, Any


class ConfigManager:
    """Manages training configuration including live config updates."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
    
    def load_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load config: {e}")
        return None
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to JSON file."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def create_initial_config(
        self,
        learning_rate: float,
        backbone_lr: float,
        temperature: float,
        log_interval: int,
        similarity_log_interval: int,
        eval_interval: int,
        grad_clip: float,
        loss_balance_strength: float,
        loss_balance_ema_momentum: float,
        loss_balance_clip_min: float,
        loss_balance_clip_max: float
    ) -> Dict[str, Any]:
        """Create and save initial configuration."""
        config = {
            "learning_rate": learning_rate,
            "backbone_lr": backbone_lr,
            "temperature": temperature,
            "log_interval": log_interval,
            "similarity_log_interval": similarity_log_interval,
            "eval_interval": eval_interval,
            "grad_clip": grad_clip,
            "loss_balance_strength": loss_balance_strength,
            "loss_balance_ema_momentum": loss_balance_ema_momentum,
            "loss_balance_clip_min": loss_balance_clip_min,
            "loss_balance_clip_max": loss_balance_clip_max
        }
        self.save_config(config)
        return config
    
    def apply_config_changes(
        self,
        current_config: Dict[str, Any],
        new_config: Dict[str, Any],
        optimizer,
        freeze_backbone: bool
    ) -> Dict[str, tuple]:
        """Apply configuration changes and return dict of what changed.
        
        Returns:
            Dict mapping parameter names to (old_value, new_value) tuples.
        """
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

        # Check and apply grad_clip changes
        if "grad_clip" in new_config and new_config["grad_clip"] != current_config.get("grad_clip", 1.0):
            old_val = current_config.get("grad_clip", 1.0)
            new_val = new_config["grad_clip"]
            current_config["grad_clip"] = new_val
            changes["grad_clip"] = (old_val, new_val)
        
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

