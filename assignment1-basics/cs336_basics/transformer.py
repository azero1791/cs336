import torch.nn as nn
import torch
from jaxtyping import Float
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import Multihead_self_attention
from cs336_basics.swiglu import Swiglu

class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float) -> None:
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # define rmsnorm
        self.rmsnorm_multihead = RMSNorm(d_model)
        self.rmsnorm_feedforward = RMSNorm(d_model)
        
        # define multihead self attention
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.multihead_self_attention = Multihead_self_attention(d_model, num_heads, max_seq_len, theta)

        # define feed forward network
        self.feedforward_network = Swiglu(d_model, d_ff)


    def forward(self, in_features: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        # layer normalization
        in_features_rmsnorm = self.rmsnorm_multihead(in_features)

        # multihead self attention
        attention_output = self.multihead_self_attention(in_features_rmsnorm, use_rope=True, token_positions=torch.arange(in_features.shape[-2], device=in_features.device))

        # attention + residual connection
        output1 = in_features + attention_output

        # rmsnorm after attention
        output1_rmsnorm = self.rmsnorm_feedforward(output1)

        # feed forward network
        output2 = self.feedforward_network(output1_rmsnorm)

        # feed forward + residual connection
        output = output1 + output2

        return output
        
        

