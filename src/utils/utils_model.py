from typing import Callable, Union

import pytorch_lightning.callbacks
import torch.nn as nn
from torchmetrics.metric import Metric

from src.utils.utils_exceptions import InvalidModuleStr


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


def find_model_parameter(module: Union[nn.ModuleList, nn.Module], module_str: str):
    """Returns all consequive submodules after and including the submodule `module_str`"""

    for sub_module_name, sub_module in module.named_modules():
        if type(sub_module) is Metric:
            continue
        if module_str in sub_module_name:
            return sub_module_name, sub_module
    raise ValueError(f"module_str '{module_str}' not found. should be (e.g. layer3.2)")


def get_all_modules_after(
    module: Union[nn.ModuleList, nn.Module], module_str: str
) -> nn.ModuleDict:
    """Returns all consequive submodules after and including the submodule `module_str`"""
    found_layer = False
    modules = nn.ModuleDict()

    for sub_module_name, sub_module in module.named_modules():
        if type(sub_module) is Metric:
            continue
        if module_str in sub_module_name:
            found_layer = True
        if found_layer:
            modules.add_module(sub_module_name.replace(".", "_"), sub_module)

    if not found_layer:
        raise ValueError(
            f"module_str '{module_str}' not found. should be (e.g. layer3.2)"
        )

    if len(modules) == 0:
        print_modules(module)
        raise InvalidModuleStr(
            f"get_all_modules_after return no elements because of invalid module '{module_str}', use one of the above."
        )

    return modules


def print_params(module: Union[nn.ModuleList, nn.Module]):
    """Print params."""
    for sub_module_name, sub_module in module.named_modules():
        for param_name, param in sub_module.named_parameters():
            print("requires_grad:", param.requires_grad, param.numel(), param_name)


def print_modules(module: Union[nn.ModuleList, nn.Module]):
    """Print model's paramteres."""
    print()
    for sub_module_name, sub_module in module.named_modules():
        sub_module_req_grad = any(
            [x[1].requires_grad for x in sub_module.named_parameters()]
        )
        print(sub_module_name, "requires_grad:", sub_module_req_grad)
    print()
