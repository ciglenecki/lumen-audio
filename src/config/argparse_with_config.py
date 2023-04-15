import argparse

import pytorch_lightning as pl
import simple_parsing
import torch
from simple_parsing import DashVariant

from src.config.config_defaults import ConfigDefault


class ArgParseWithConfig(simple_parsing.ArgumentParser):
    config_dest_str = "config_args"
    args_dest_group = "args_additional"
    pl_group_name = "pl.Trainer"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            add_option_string_dash_variants=DashVariant.DASH, *args, **kwargs
        )
        self.add_arguments(ConfigDefault, dest=ArgParseWithConfig.config_dest_str)
        self.additional_args_group = self.add_argument_group(
            ArgParseWithConfig.args_dest_group
        )

        lightning_parser = pl.Trainer.add_argparse_args(self)
        lightning_parser.set_defaults(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=-1,
        )

    def add_argument(self, *name_or_flags: str, **kwargs):
        return self.additional_args_group.add_argument(*name_or_flags, **kwargs)

    def parse_args(
        self,
    ) -> tuple[argparse.Namespace, ConfigDefault, argparse.Namespace]:
        args = super().parse_args()

        config = getattr(args, ArgParseWithConfig.config_dest_str)
        delattr(args, ArgParseWithConfig.config_dest_str)

        args_dict: dict[str, argparse.Namespace] = {}
        for group in self._action_groups:
            group_dict = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
            if group.title:
                args_dict[group.title] = argparse.Namespace(**group_dict)

        args, pl_args = (
            args_dict[ArgParseWithConfig.args_dest_group],
            args_dict[ArgParseWithConfig.pl_group_name],
        )
        return args, config, pl_args


def test_args_parse_with_config():
    parser = ArgParseWithConfig()
    parser.add_argument("--my-cool-arg", type=str, default=3)
    parser.add_argument("--hello-there", type=str, default=4)
    args, config, pl_args = parser.parse_args()

    assert args.my_cool_arg == 3
    assert args.hello_there == 4
    assert config == ConfigDefault()
