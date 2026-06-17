import torch.nn as nn
import torch
import math

from jaxtyping import Int, Float

class Embedding(nn.Module):
    def __init__(self, num_embeddings : int, embedding_dim : int, device : torch.device | None=None, dtype : torch.dtype | None=None) -> None:

        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype 

        self.lookup = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        lookup_std = math.sqrt(2.0/(num_embeddings + embedding_dim))
        torch.nn.init.trunc_normal_(self.lookup, mean=0, std= lookup_std, a= (-3) * lookup_std, b= 3 * lookup_std)
 
    def forward(self, token_ids : Int[torch.Tensor, "batchs sequence_length"]) -> Float[torch.Tensor, "batchs sequence_length embedding_dim"]:

        embeddings = []
        for batch in token_ids:
            batch_embed = []
            for token in batch:
                batch_embed.append(self.lookup[token])
            batch_embed = torch.stack(batch_embed, dim=0)
            embeddings.append(batch_embed)
        
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings
