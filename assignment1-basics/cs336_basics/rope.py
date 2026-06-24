import torch.nn as nn
import torch 
from jaxtyping import Float, Int

class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None=None) -> None:
        super().__init__()

        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # precompute cos and sin 
        positions = torch.arange(max_seq_len)

        # calculate 2k -2
        d_indices = torch.arange(0, d_k, 2)

        inv_freq = 1.0 /(torch.pow(theta, (d_indices / d_k)))

        # freqs[i, k] = positions[i] * inv_freq[k]
        freqs = torch.outer(positions, inv_freq)

        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)


    def forward(self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions: Int[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len d_k"]:

        assert self.d_k == x.shape[-1], "Dimension of query or key is not equal to rope's"

        assert len(token_positions) <= self.max_seq_len, f"Expected max_seq_len: {self.max_seq_len}, but get actual max_seq_len: {len(token_positions)}"

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos

        x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)
        x_rotated = x_rotated.flatten(-2)

        return x_rotated