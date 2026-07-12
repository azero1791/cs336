import math

import torch
import torch.nn as nn
from collections.abc import Callable
from jaxtyping import List, Optional

class SGD(torch.optim.Optimizer):
    def __init__(self, params: List[nn.Parameter], lr=1e-3) -> None:
        assert lr > 0.0, "Learning rate must be positive."
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None) -> None:
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        # Here we don't use closure, just for standard API
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                # Update the parameters using SGD update rule
                state = self.state[param]
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = param.grad.data # Get the gradient of loss with respect to p.
                param.data -= group['lr'] / math.sqrt(1 + t) * grad
                state["t"] = t + 1 # Increment iteration number.

        return None
    
