import argparse
import torch
from src.config.config_defaults import get_default_config
from src.enums.enums import SupportedModels
import simple_parsing
from src.config.config_defaults import ConfigDefault
import argparse

def parse():
    destination_str = "user_args"
    parser = simple_parsing.ArgumentParser(
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH
    )
    parser.add_argument(
        "--path_to_model", type=str, required=True, 
        help="Path to a trained model."
    )
    parser.add_argument(
        "--target_layer", type=str, required=True, 
        help="Full name of the layer used for tracking gradients (if not sure, use the final feature map/embedding)"
    )
    parser.add_argument(
        "--device", 
        type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", 
        help="The device to be used eg. cuda:0."
    )
    parser.add_argument(
        "--label", type=str, default=None, help="The label to track the gradients for. If None, tracked for all"
    )
    parser.add_arguments(ConfigDefault, dest=destination_str)
    args = parser.parse_args()

    # Razdvoji config argumente i preostale/dodatne argumente
    args_dict = vars(args)
    config: ConfigDefault = args_dict.pop(destination_str)
    args = argparse.Namespace(**args_dict)

    return args, config