#######
# Config classes for training
#######

from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float

@dataclass 
class OptimizerConfig:
    lr: float
    eps: float
    betas: tuple[float, float]
    weight_decay: float

@dataclass
class LrSchedulerConfig:
    current_step: int
    max_lr: float
    min_lr: float
    warmup_steps: int
    cos_steps: int

@dataclass
class DataConfig:
    train_data_path: str
    val_data_path: str
    dtype: str

@dataclass
class RuntimeConfig:
    device: str
    max_steps: int
    batch_size: int
    max_l2_norm: float

@dataclass
class OutputConfig:
    log_interval: int
    validation_inerval: int

@dataclass
class CheckpointConfig:
    src: str
    out: str
    checkpoint_interval: int

@dataclass
class TrainingConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: LrSchedulerConfig
    data: DataConfig
    runtime: RuntimeConfig
    output: OutputConfig
    checkpoint: CheckpointConfig