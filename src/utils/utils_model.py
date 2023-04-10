from typing import Callable

import torch.nn as nn


def filter_modules(module: nn.Module, module_type=nn.Linear):
    return [submodule for submodule in module if isinstance(submodule, module_type)]


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
