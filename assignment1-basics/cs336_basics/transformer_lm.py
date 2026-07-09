import torch.nn as nn
import torch

from cs336_basics.transformer import Transformer
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax

class Transformer_LM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float) -> None:
        """
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        """
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_block_lst = nn.ModuleList([Transformer(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)])
        self.rmsnorm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)
        

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        output_embedding = self.embedding(in_features)
        input_transformer = output_embedding
        for transformer_block in self.transformer_block_lst:
            output_transformer = transformer_block(input_transformer)
            input_transformer = output_transformer
        output_rmsnorm = self.rmsnorm(output_transformer)
        output = self.linear(output_rmsnorm)

        return output
    