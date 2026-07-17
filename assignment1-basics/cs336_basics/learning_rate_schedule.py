import math
def learning_rate_schedule(current_step: int, max_lr: float, min_lr: float, warmup_steps: int, cos_steps: int) -> float:
    """
    Computes the learning rate based on the current step using a warmup and cosine decay schedule.

    Args:
        current_step (int): The current training step.
        max_lr (float): The maximum learning rate.
        min_lr (float): The minimum learning rate.
        warmup_steps (int): The number of steps for the warmup phase.
        cos_steps (int): The finial steps for the cosine decay phase.

    Returns:
        float: The computed learning rate for the current step.
    """
    if current_step < warmup_steps:
        # Linear warmup
        lr = max_lr * (current_step / warmup_steps)
    elif current_step < cos_steps:
        # Cosine decay
        progress = (current_step - warmup_steps) / (cos_steps - warmup_steps)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    else:
        # After cosine decay, keep the learning rate at min_lr
        lr = min_lr

    return lr
