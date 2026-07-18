import numpy as np
import torch
from typing import Tuple
from jaxtyping import Int
from einops import rearrange

def data_loading(x: Int[np.ndarray, "token_ids"], batch_size: int, context_length: int, device : torch.device | None=None) ->Tuple[
    Int[torch.Tensor, "batch_size context_length"],
    Int[torch.Tensor, "batch_size context_length"]
]:
    """
    Loads the data into batches of specified size and context length,
    and return paired next token IDs for each sequence in the batch. The data is loaded onto the specified device.

    Args:
        x: A 1D numpy array of token IDs.
        batch_size: The number of sequences in each batch.
        context_length: The length of each sequence.
        device: The device to load the data onto (e.g., 'cpu' or 'cuda').
    """
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)
    x_batch = np.stack([x[start:start + context_length] for start in starts])
    x_next_tokens = np.stack([x[start + 1:start + context_length + 1] for start in starts])
    x_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
    x_next_tokens = torch.tensor(x_next_tokens, dtype=torch.long, device=device)
    return (x_batch, x_next_tokens)
