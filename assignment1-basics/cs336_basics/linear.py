import torch
import torch.nn as nn
import math
from jaxtyping import Float
from einops import einsum 

class Linear(nn.Module):
    def __init__(self, d_in : int, d_out : int, device : torch.device | None=None, dtype : torch.dtype | None=None) -> None:
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(torch.empty(d_out, d_in))

        w_std = math.sqrt(2.0/(d_in + d_out))
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=w_std, a= (-3) * w_std, b= 3 * w_std)

    def forward(self, in_features : Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out"]:
        output = einsum(in_features, self.weights, "... d_in, d_out d_in -> ... d_out")

        return output

    