
from pathlib import Path
from cs336_basics.training_config import TrainingConfig
from cs336_basics.data_loading import data_loading

import numpy as np
from torch import device

def training_together(training_config: TrainingConfig) -> None:
    """
    This function is used to train the model together with the training configuration.
    """

    data_config = training_config.data
    model_config = training_config.model
    optimizer_config = training_config.optimizer
    scheduler_config = training_config.scheduler
    runtime_config = training_config.runtime
    output_config = training_config.output
    checkpoint_config = training_config.checkpoint

    # TODO: implement the training loop here, using the provided configurations.

    # load data using lazy copy of memmap or np.load
    # train data pretokenized is a sequence of token ids
    train_data_path = data_config.train_data_path
    dtype = data_config.dtype
    if Path(train_data_path).suffix == ".npy":
        train_data = np.load(train_data_path, mmap_mode="r")
    elif Path(train_data_path).suffix == "bin":
        train_data = np.memmap(train_data_path, dtype=dtype, mode="r")

    batch_size = runtime_config.batch_size
    context_length = model_config.context_length
    device = runtime_config.device

    # construct batched data
    batched_data =  data_loading(train_data, batch_size, context_length, device)

    checkpoint_src = checkpoint_config.src
    if checkpoint_src is None:
        model = Transformer_LM()
