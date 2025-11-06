import os
import torch
from typing import Optional, Tuple


class CheckpointManager:
    """Manages checkpoint saving and loading."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def find_latest_checkpoint(self) -> Optional[Tuple[str, int]]:
        """Find the latest checkpoint in the output directory.
        
        Returns:
            Tuple of (checkpoint_path, step) if found, None otherwise.
        """
        if not os.path.exists(self.output_dir):
            return None
        
        checkpoint_dirs = [d for d in os.listdir(self.output_dir) 
                          if d.startswith('checkpoint-') and 
                          os.path.isdir(os.path.join(self.output_dir, d))]
        
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
        checkpoint_path = os.path.join(self.output_dir, latest_dir, 'model.pt')
        
        if os.path.exists(checkpoint_path):
            return checkpoint_path, latest_step
        
        return None
    
    def save_checkpoint(
        self,
        step: int,
        model_state_dict: dict,
        optimizer_state_dict: dict,
        scheduler_state_dict: dict,
        alpha: float,
        beta: float,
        ema_loss_A: Optional[float],
        ema_loss_B: Optional[float]
    ):
        """Save a checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save({
            'step': step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'scheduler_state_dict': scheduler_state_dict,
            'alpha': alpha,
            'beta': beta,
            'ema_loss_A': ema_loss_A,
            'ema_loss_B': ema_loss_B,
        }, os.path.join(checkpoint_dir, 'model.pt'))
        
        print(f"Checkpoint saved at step {step}")
    
    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """Load a checkpoint from path."""
        return torch.load(checkpoint_path, map_location='cpu')

