import torch
import math
from collections.abc import Iterable, Callable
from typing import Optional


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3, eps: float = 1e-8, betas: tuple[float, float] = (0.9, 0.999), weight_decay: float=0.01) -> None:
        assert lr > 0.0, "Learning rate must be positive."
        
        assert betas[0] > 0 and betas[0] < 1, "beta1 should between 0 and 1."

        assert betas[1] > 0 and betas[1] < 1, "beta2 should between 0 and 1."
        
        defaults = {
            "lr": lr,
            "eps": eps,
            "betas": betas,
            "weight_decay": weight_decay
        }

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None) -> None:
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """


        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                
                # Update the parameters using AdamW update rule 
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                # update step 
                state["step"] += 1

                # get current gradient of loss 
                grad = param.grad.data

                # compute learn_rate at step t
                beta_1 = group["betas"][0]
                beta_2 = group["betas"][1]
                lr_t = group["lr"] * (math.sqrt(1-beta_2 ** state["step"])) / (1 - beta_1 ** state["step"])

                # apply weight decay
                param.data -= group["lr"] * group["weight_decay"] * param.data

                # update m of param
                state["m"] = beta_1 * state["m"] + (1 - beta_1) * grad
                
                # update v of param
                state["v"] = beta_2 * state["v"] + (1 - beta_2) * (grad ** 2)  

                # update param
                param.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + group["eps"])

        return None
