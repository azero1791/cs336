import torch
from jaxtyping import Float

def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    x_max = x.max(dim=dim, keepdim=True).values

    x_stable = x - x_max

    x_exp = torch.exp(x_stable)
    sum_x_exp = x_exp.sum(dim=dim, keepdim=True)

    return x_exp / sum_x_exp
