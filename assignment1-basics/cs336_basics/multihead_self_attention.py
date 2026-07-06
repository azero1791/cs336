from cs336_basics.rope import Rope
import torch
import torch.nn as nn

from jaxtyping import Float
from cs336_basics.linear import Linear, Int
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from einops import rearrange

class Multihead_self_attention(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # q: [... d_k], k: [... d_k], v: [... d_v]\
        # *_embedding: all heads concatenated, so [... num_heads * d_k] and [... num_heads * d_v]
        self.q_embedding = Linear(d_model, num_heads * self.d_k)
        self.k_embedding = Linear(d_model, num_heads * self.d_k)
        self.v_embedding = Linear(d_model, num_heads * self.d_v)

        # rearrange to separate heads: [... num_heads * d_k] -> [... num_heads d_k]
        self.q_embedding = rearrange(self.q_embedding, "... (num_heads d_k) -> ... num_heads d_k", num_heads=num_heads, d_k=self.d_k)
        self.k_embedding = rearrange(self.k_embedding, "... (num_heads d_k) -> ... num_heads d_k", num_heads=num_heads, d_k=self.d_k)
        self.v_embedding = rearrange(self.v_embedding, "... (num_heads d_v) -> ... num_heads d_v", num_heads=num_heads, d_v=self.d_v)

        # rope
        self.rope = Rope(theta=10000, d_k=self.d_k, max_seq_len=max_seq_len)

    def forward(self, in_features: Float[torch.Tensor, "... d_model"], use_rope: bool = False, token_positions: Int[torch.Tensor, "... sequence_length"] | None = None) -> Float[torch.Tensor, "... d_model"]:
        q = self.q_embedding(in_features)
        k = self.k_embedding(in_features)
        v = self.v_embedding(in_features)

        if use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.ones(q.shape[:-1], dtype=torch.bool, device=q.device)  # no masking for now
        # compute attention scores
        scores = scaled_dot_product_attention(q, k, v)       
     