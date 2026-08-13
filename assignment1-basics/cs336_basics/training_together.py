
from pathlib import Path
from cs336_basics.training_config import TrainingConfig
from cs336_basics.transformer_lm import Transformer_LM
from cs336_basics.adamw import AdamW
from cs336_basics.ops_checkpoint import load_checkpoint
from cs336_basics.data_loading import data_loading
from cs336_basics.cross_entropy import cross_entropy

import numpy as np

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

    # extract model configurations
    vocab_size = model_config.vocab_size
    d_model = model_config.d_model
    num_layers = model_config.num_layers
    num_heads = model_config.num_heads
    d_ff = model_config.d_ff
    rope_theta = model_config.rope_theta

    # extract optimizer configurations
    lr = optimizer_config.lr
    eps = optimizer_config.eps
    betas = optimizer_config.betas
    weight_decay = optimizer_config.weight_decay

    # config the model from scratch
    model = Transformer_LM(vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)

    # config the optimizer
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps, betas=betas, weight_decay=weight_decay)

    if checkpoint_src is not None:
        # load the model from checkpoint
        iteration = load_checkpoint(checkpoint_src, model, optimizer)

    training_epoches = runtime_config.epoches

    for epoch in range(training_epoches):

        batched_x, batched_labels = batched_data.unbind(dim=-1)
        # forward pass
        predictions = model(batched_x)
        loss = cross_entropy(predictions, batched_labels)

        # eliminate gradient from previous batch
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()




