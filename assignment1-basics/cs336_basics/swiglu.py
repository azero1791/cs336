import torch
import torch.nn as nn
from torch import sigmoid
from cs336_basics.linear import Linear
from jaxtyping import Float
from einops import einsum

class Swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # linearNumber corresponed to weightNumber
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.linear3 = Linear(d_model, d_ff)

    def forward(self, in_features : Float[torch.Tensor, "... d_in"], load_weight=False) -> Float[torch.Tensor, "... d_out"]:
        o1 = self.linear1(in_features) * sigmoid(self.linear1(in_features))
        o3 = self.linear3(in_features)
        o2 = self.linear2(o1 + o3)

        return o2
