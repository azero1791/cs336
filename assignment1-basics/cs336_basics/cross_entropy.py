from jaxtyping import Float, Int
import torch
def cross_entropy(inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]):
    """
    Computes the cross-entropy loss between the predicted inputs and the true targets.

    Args:
        inputs (torch.Tensor): The predicted (logits) from the model.
        targets (torch.Tensor): The true labels (one-hot encoded or class indices).

    Returns:
        torch.Tensor: The computed cross-entropy loss.
    """
    logits = inputs - inputs.max(dim=-1, keepdim=True).values

    log_denominator = logits.exp().sum(dim=-1).log()

    target_logits = logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    return (log_denominator - target_logits).mean()