"""Extra distributional metrics used by the attention-patching experiment.

In particular, a top-256 Wasserstein distance with cosine distance between
token embeddings as the ground metric, matching the definition in
``icml_submission.tex`` (§6.2, tab:dist_metrics).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    import ot  # POT: Python Optimal Transport
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The Wasserstein metric requires POT. Install with: pip install pot"
    ) from e


def _top_k_union(
    p: torch.Tensor,
    q: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Return the union of top-k token indices from both distributions.

    Using the union (rather than, say, top-k of p only) ensures that both
    distributions have most of their mass covered by the restricted support
    regardless of which one you treat as the 'target'. Any token outside
    the union has at most ``min(p.sort()[-k-1], q.sort()[-k-1])`` probability
    in both distributions -- small for k>=256 and typical vocab sizes.
    """
    topp = torch.topk(p, k=k, dim=-1).indices  # [k]
    topq = torch.topk(q, k=k, dim=-1).indices
    union = torch.unique(torch.cat([topp, topq]))
    return union


def wasserstein_topk_cosine(
    p: torch.Tensor,
    q: torch.Tensor,
    token_embeddings: torch.Tensor,
    k: int = 256,
    normalize: bool = True,
) -> float:
    """Wasserstein-1 distance between ``p`` and ``q`` (each shape ``[V]``),
    restricted to the top-k support, with cosine distance between token
    embeddings as the ground metric.

    Args:
        p, q:
            1-D probability vectors over the vocabulary (same length).
        token_embeddings:
            ``[V, d]`` tensor of token embeddings (e.g. the input embedding
            matrix). Used to form the ground metric on the restricted support.
        k:
            Top-k mass to keep per distribution. Union of top-k yields the
            support of size ``|support| <= 2k`` in the worst case.
        normalize:
            If True, renormalize ``p`` and ``q`` on the restricted support so
            the transport problem is balanced. Captures the total mass lost
            to the tail only indirectly (via renormalization).

    Returns:
        Scalar Wasserstein distance (in cosine-distance units, i.e. in
        ``[0, 2]``).
    """
    assert p.dim() == 1 and q.dim() == 1, "p, q must be 1-D prob vectors"
    assert p.shape == q.shape, "p, q must have identical vocab size"
    V = p.shape[0]
    assert token_embeddings.shape[0] == V, (
        f"token_embeddings has {token_embeddings.shape[0]} rows but vocab "
        f"size is {V}"
    )

    support = _top_k_union(p, q, k)
    p_sub = p[support].to(torch.float64)
    q_sub = q[support].to(torch.float64)

    if normalize:
        p_sub = p_sub / torch.clamp(p_sub.sum(), min=1e-12)
        q_sub = q_sub / torch.clamp(q_sub.sum(), min=1e-12)
    else:
        # POT requires identical total mass; if we don't renormalize we need
        # to equalize by adding a dummy sink. Keep it simple -- always
        # renormalize when using this function and expose this as a flag.
        raise NotImplementedError("normalize=False not supported yet")

    emb = token_embeddings[support].to(torch.float64)
    emb_norm = emb / torch.clamp(
        torch.linalg.norm(emb, dim=-1, keepdim=True), min=1e-12
    )
    # Cosine distance: 1 - cos_sim, clamped to [0, 2] for numerical safety.
    cos_sim = emb_norm @ emb_norm.T  # [|S|, |S|]
    M = torch.clamp(1.0 - cos_sim, min=0.0, max=2.0).cpu().numpy()

    a = p_sub.cpu().numpy()
    b = q_sub.cpu().numpy()

    # ``ot.emd2`` returns the Wasserstein-1 cost (scalar).
    cost = float(ot.emd2(a, b, M, numItermax=200_000))
    return cost


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy in nats. Accepts any shape; reduces over last dim."""
    p = torch.clamp(p, min=eps)
    return -(p * torch.log(p)).sum(dim=-1)
