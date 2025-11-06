import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import os


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed(device_override=None):
    """Setup distributed training environment.
    
    Returns:
        Tuple of (rank, world_size, local_rank)
    """
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
    """Cleanup distributed training environment."""
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

