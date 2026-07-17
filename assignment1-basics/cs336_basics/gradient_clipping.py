import torch

from collections.abc import Iterable
def gradient_clipping(parameters: Iterable[torch.Tensor], max_l2_norm: float) -> None:
    """
    Clips the gradients of the given parameters to have a maximum L2 norm.

    Args:
        parameters: An iterable of tensors whose gradients will be clipped.
        max_l2_norm: The maximum L2 norm for the gradients.

    Returns:
        None
    """
    eps = 1e-6  # Small value to prevent division by zero
    total_norm = sum([p.grad.data.detach().norm(2) ** 2 for p in parameters if p.grad is not None])
    total_norm = total_norm ** 0.5
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)