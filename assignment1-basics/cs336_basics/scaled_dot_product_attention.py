import torch
import math
from jaxtyping import Float, Bool
from cs336_basics.softmax import softmax
from einops import einsum

def scaled_dot_product_attention(queries: Float[torch.Tensor, "... queries d_k"], keys: Float[torch.Tensor, "... keys d_k"], values: Float[torch.Tensor, "... keys d_v"], mask: Bool[torch.Tensor, "... queries keys"]) -> Float[torch.Tensor, "... seq_len d_v"]:
    d_k = queries.shape[-1]
    scores = einsum(queries, keys, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    scores_masked = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores_masked, dim=-1)

    output = einsum(weights, values, "... queries keys, ... keys d_v -> ... queries d_v")

    return output