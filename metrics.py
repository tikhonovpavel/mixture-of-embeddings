import torch
from typing import Dict


def _ensure_probs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = torch.clamp(x, min=eps)
    x = x / x.sum(dim=-1, keepdim=True)
    return x


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = _ensure_probs(p, eps)
    q = _ensure_probs(q, eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = _ensure_probs(p, eps)
    q = _ensure_probs(q, eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def total_variation_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = _ensure_probs(p, eps)
    q = _ensure_probs(q, eps)
    return 0.5 * torch.sum(torch.abs(p - q), dim=-1)


def l2_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = _ensure_probs(p, eps)
    q = _ensure_probs(q, eps)
    return torch.sqrt(torch.sum((p - q) ** 2, dim=-1))


def cosine_similarity(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = _ensure_probs(p, eps)
    q = _ensure_probs(q, eps)
    p_norm = torch.linalg.norm(p, dim=-1)
    q_norm = torch.linalg.norm(q, dim=-1)
    denom = torch.clamp(p_norm * q_norm, min=eps)
    return torch.sum(p * q, dim=-1) / denom


def hellinger_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = _ensure_probs(p, eps)
    q = _ensure_probs(q, eps)
    return torch.linalg.norm(torch.sqrt(p) - torch.sqrt(q), dim=-1) / torch.sqrt(torch.tensor(2.0, device=p.device, dtype=p.dtype))


def compute_all_metrics(p: torch.Tensor, q: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "kl": kl_divergence(p, q),
        "js": js_divergence(p, q),
        "tvd": total_variation_distance(p, q),
        "l2": l2_distance(p, q),
        "cosine": cosine_similarity(p, q),
        "hellinger": hellinger_distance(p, q),
    }
