from cs336_basics.rope import Rope
import torch
import torch.nn as nn

from jaxtyping import Float, Int
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from einops import rearrange

class Multihead_self_attention(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None=None, theta: int | None=None) -> None:
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

        # output projection: [... num_heads d_v] -> [... num_heads * d_v] -> [... d_model]
        self.output_projection = Linear(num_heads * self.d_v, d_model)

        # rope
        if max_seq_len is not None and theta is not None:
            self.rope = Rope(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        

    def forward(self, in_features: Float[torch.Tensor, "... d_model"], use_rope: bool = False, token_positions: Int[torch.Tensor, "... seq_len"] | None = None) -> Float[torch.Tensor, "... d_model"]:
        q = self.q_embedding(in_features)
        k = self.k_embedding(in_features)
        v = self.v_embedding(in_features)

                # rearrange to separate heads: [... num_heads * d_k] -> [... num_heads d_k]
        q = rearrange(q, "... queries (num_heads d_k) -> ... num_heads queries d_k", num_heads=self.num_heads, d_k=self.d_k)
        k = rearrange(k, "... keys (num_heads d_k) -> ... num_heads keys d_k", num_heads=self.num_heads, d_k=self.d_k)
        v = rearrange(v, "... keys (num_heads d_v) -> ... num_heads keys d_v", num_heads=self.num_heads, d_v=self.d_v)
        
        if use_rope:
            # suppose k and q have the same token positions, so we can use the same token_positions for both
            assert token_positions is not None, "token_positions must be provided when use_rope is True"
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones(
            q.shape[:-1] + (k.shape[-2],), dtype=torch.bool, device=q.device
        ))
        
        # compute attention scores : [... num_heads queries d_v]
        scores = scaled_dot_product_attention(q, k, v, mask)    
        scores = rearrange(scores, "... num_heads queries d_v -> ... queries (num_heads d_v)", num_heads=self.num_heads, d_v=self.d_v)
        scores = self.output_projection(scores)
        return scores  
     