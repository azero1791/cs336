import torch
import os
import typing
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    """
    save states of learnable parameters of model and optimizer into out path
    """
    model_learnable_params = model.state_dict()
    optim_states = optimizer.state_dict()

    checkpoint_dict = {}

    checkpoint_dict["model"] = model_learnable_params
    checkpoint_dict["optimizer"] = optim_states
    checkpoint_dict["iteration"] = iteration

    torch.save(checkpoint_dict, out)

    return None

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:

    checkpoint_dict = torch.load(src)
    model.load_state_dict(checkpoint_dict["model"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])

    return checkpoint_dict["iteration"]
