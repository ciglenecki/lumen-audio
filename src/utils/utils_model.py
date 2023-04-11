from typing import Callable

import pytorch_lightning as pl
import pytorch_lightning.callbacks
import torch.nn as nn


def filter_modules(module: nn.Module, module_type=nn.Linear):
    output = []
    for submodule in pytorch_lightning.callbacks.BaseFinetuning.flatten_modules(module):
        if isinstance(submodule, module_type):
            output.append(submodule)
    return output


def initialize_weights(module: nn.Module, initialization: Callable, kwargs: dict):
    for submodule in module:
        initialization(submodule.weight, **kwargs)


def get_linear_init(activation: nn.Module) -> tuple[Callable, dict]:
    """Gets the appropriate initialization for a nn.Linear for a given activation, along side some
    important keyword arguments fo the initialization function."""
    if isinstance(activation, nn.ReLU):
        return nn.init.kaiming_normal, {"nonlinearity": "relu"}
    elif isinstance(activation, nn.LeakyReLU):
        return nn.init.kaiming_normal, {"nonlinearity": "leaky_relu"}
    elif isinstance(activation, nn.Sigmoid) or isinstance(activation, nn.Tanh):
        return nn.init.xavier_normal, {}
    else:
        return (nn.init.kaiming_uniform,)


def count_module_params(module: nn.Module):
    total_params = float(sum(p.numel() for p in module.parameters()))
    trainable_params = float(
        sum(p.numel() for p in module.parameters() if p.requires_grad)
    )
    non_trainable_params = float(
        sum(p.numel() for p in module.parameters() if not p.requires_grad)
    )
    return dict(
        total_params=total_params,
        trainable_params=trainable_params,
        non_trainable_params=non_trainable_params,
    )
