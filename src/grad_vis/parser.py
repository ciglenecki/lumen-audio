import argparse

import simple_parsing

from src.config.config_defaults import ConfigDefault


def parse():
    destination_str = "user_args"
    parser = simple_parsing.ArgumentParser(
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH
    )
    parser.add_arguments(ConfigDefault, dest=destination_str)
    args = parser.parse_args()

    # Razdvoji config argumente i preostale/dodatne argumente
    args_dict = vars(args)
    config: ConfigDefault = args_dict.pop(destination_str)
    args = argparse.Namespace(**args_dict)

    return args, config
