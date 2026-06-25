import torch.nn as nn

class Multihead_self_attention(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention
    """
    def __init__(self, d_model: int, num_heads: int) -> None:
