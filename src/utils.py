# utils.py
"""
Utility functions for CLIP-from-scratch:
- vector normalization
- contrastive loss
- input normalization
- retrieval evaluation metrics
"""

import torch
import torch.nn as nn

def cosine_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize vectors along the last dimension to unit L2 norm.

    Args:
        x (torch.Tensor): Input tensor of shape (..., dim).

    Returns:
        torch.Tensor: Normalized tensor with same shape, where each vector
        has unit norm.
    """
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def clip_contrastive_loss(logits_per_image: torch.Tensor,
                          logits_per_text: torch.Tensor) -> torch.Tensor:
    """
    Compute the symmetric CLIP contrastive loss.

    For a batch of N (image, text) pairs:
    - logits_per_image: similarity matrix (N x N), image->text
    - logits_per_text: similarity matrix (N x N), text->image

    The correct pair is always on the diagonal.

    Args:
        logits_per_image (torch.Tensor): [N, N] image-to-text logits.
        logits_per_text (torch.Tensor): [N, N] text-to-image logits.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    n = logits_per_image.size(0)
    labels = torch.arange(n, device=logits_per_image.device)
    loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)


def clip_normalize_inplace(images: torch.Tensor,
                           mean=(0.48145466, 0.4578275, 0.40821073),
                           std=(0.26862954, 0.26130258, 0.27577711)) -> None:
    """
    In-place normalize images using CLIP's mean and std.

    Args:
        images (torch.Tensor): Input of shape (B, 3, H, W) in [0,1].
        mean (tuple): Channel mean values.
        std (tuple): Channel std values.

    Returns:
        None (operation is performed in-place).
    """
    device = images.device
    mean = torch.tensor(mean, device=device).view(1,3,1,1)
    std  = torch.tensor(std, device=device).view(1,3,1,1)
    images.sub_(mean).div_(std)


@torch.no_grad()
def recall_at_k(sim_matrix: torch.Tensor, k: int) -> float:
    """
    Compute Recall@K for retrieval.

    Args:
        sim_matrix (torch.Tensor): Similarity matrix [N, N], rows = queries,
                                   cols = candidates. Assumes ground truth is
                                   along the diagonal.
        k (int): Top-k cutoff.

    Returns:
        float: Recall@k score between 0 and 1.
    """
    ranks = sim_matrix.argsort(dim=1, descending=True)
    correct = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    return (ranks[:, :k] == correct.view(-1,1)).any(dim=1).float().mean().item()


@torch.no_grad()
def eval_retrieval(sim_matrix: torch.Tensor) -> dict:
    """
    Evaluate retrieval performance with Recall@1,5,10.

    Args:
        sim_matrix (torch.Tensor): Similarity matrix [N, N].

    Returns:
        dict: Mapping {"R@1 (I→T)": ..., "R@5 (I→T)": ..., "R@1 (T→I)": ...}
    """
    ks = [1,5,10]
    ks = [k for k in ks if k <= sim_matrix.size(0)]
    out = {}
    for k in ks:
        out[f"R@{k} (I→T)"] = recall_at_k(sim_matrix, k)
        out[f"R@{k} (T→I)"] = recall_at_k(sim_matrix.t(), k)
    return out
