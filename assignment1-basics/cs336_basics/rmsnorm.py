import torch.nn as nn
import torch
from jaxtyping import Float

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None=None, dtype: torch.dtype | None=None) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:

        assert x.shape[-1] == self.d_model, f"shape of x {x.shape[0]} should be equal to {self.d_model}"
        # save original precision of x type
        ori_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        nor_x = x / rms * self.weights

        nor_x = nor_x.to(ori_dtype)
        return nor_x