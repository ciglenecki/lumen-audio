import argparse
from operator import attrgetter

import pytorch_lightning as pl
import simple_parsing
import torch
from simple_parsing import DashVariant

from src.config.config_defaults import ConfigDefault


class ArgParseWithConfig(simple_parsing.ArgumentParser):
    """Class which connects the ConfigDefault class and argparse.

    Every field in ConfigDefault will become exposed via the argparse.
    """

    config_dest_str = "Config arguments"
    args_dest_group = "Script arguments"
    pl_group_title = "pl.Trainer"
    config_group_title = f"ConfigDefault ['{config_dest_str}']"

    def __init__(
        self,
        config_pl_args: None | list[str] = None,
        add_lightning_args=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            add_option_string_dash_variants=DashVariant.DASH, *args, **kwargs
        )

        self.add_lightning_args = add_lightning_args

        self.config_pl_args = (
            set(config_pl_args) if config_pl_args is not None else None
        )

        self.add_arguments(
            ConfigDefault,
            dest=ArgParseWithConfig.config_dest_str,
        )
        self.additional_args_group = self.add_argument_group(
            ArgParseWithConfig.args_dest_group
        )

        # Add PyTorch Lightning arguments
        if add_lightning_args:
            lightning_parser = pl.Trainer.add_argparse_args(self)
            lightning_parser.set_defaults(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=-1 if torch.cuda.is_available() else 1,
            )

    def add_argument(self, *name_or_flags: str, **kwargs):
        return self.additional_args_group.add_argument(*name_or_flags, **kwargs)

    def parse_args(
        self,
        *n_args,
        **kwargs,
    ) -> tuple[argparse.Namespace, ConfigDefault, argparse.Namespace]:
        args = super().parse_args(*n_args, **kwargs)
        # Extract Config from args
        config: ConfigDefault = getattr(args, ArgParseWithConfig.config_dest_str)

        # Remove Config from args
        delattr(args, ArgParseWithConfig.config_dest_str)

        # Apply after_init
        config.after_init()

        args_dict: dict[str, argparse.Namespace] = {}

        for idx, group in enumerate(self._action_groups):
            # Get group dict as dictionary
            group_dict = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
            print(group.title, group_dict)
            if group.title:
                args_dict[group.title] = argparse.Namespace(**group_dict)

        args = args_dict[ArgParseWithConfig.args_dest_group]
        pl_args = (
            args_dict[ArgParseWithConfig.pl_group_title]
            if self.add_lightning_args
            else None
        )
        return args, config, pl_args

    def sort_action_groups(self):
        user_idx = None
        for idx, action_group in enumerate(self._action_groups):
            if action_group.title == ArgParseWithConfig.args_dest_group:
                user_idx = idx
                break
        assert user_idx is not None
        action_group = self._action_groups.pop(user_idx)
        self._action_groups.append(action_group)

    def format_help(self):
        self.sort_action_groups()
        formatter = self._get_formatter()
        # Filter actions based on self.config_pl_args
        filtered_actions = set()

        if self.config_pl_args is not None:
            for action_group in self._action_groups:
                # Add all options which are not from Config or PyTorch Lightning
                if action_group.title not in [
                    ArgParseWithConfig.config_group_title,
                    ArgParseWithConfig.pl_group_title,
                ]:
                    filtered_actions.update(action_group._group_actions)
                    continue

                # Add the argument only if it's in config_pl_args
                for action in action_group._group_actions:
                    action_option_strings = set(action.option_strings)
                    if len(action_option_strings.intersection(self.config_pl_args)) > 0:
                        filtered_actions.add(action)
        else:
            filtered_actions.update(self._actions)

        # Add usage
        formatter.add_usage(
            self.usage, filtered_actions, self._mutually_exclusive_groups
        )

        # Add description
        formatter.add_text(self.description)

        # positionals, optionals and user-defined groups
        for action_group in self._action_groups:
            # Filter actions from the current group by using the existing set
            if self.config_pl_args is not None:
                group_actions = set(action_group._group_actions)
                filtered_group_actions = list(
                    group_actions.intersection(filtered_actions)
                )
            else:
                filtered_group_actions = action_group._group_actions

            formatter.start_section(action_group.title)
            # formatter.add_text(action_group.description)
            formatter.add_arguments(filtered_group_actions)
            formatter.end_section()

        # epilog
        formatter.add_text(self.epilog)

        # determine help from format above
        return formatter.format_help()


class SortingHelpFormatter(
    simple_parsing.SimpleHelpFormatter, argparse.RawTextHelpFormatter
):
    """Alphabetically sort -h."""

    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter("option_strings"))
        super().add_arguments(actions)


def test_args_parse_with_config():
    # This is a test, dont use this function!
    fake_cli_args = ["--my-cool-arg", "3", "--hello-there", "4"]
    parser = ArgParseWithConfig()
    parser.add_argument("--my-cool-arg", type=int, default=3)
    parser.add_argument("--hello-there", type=int, default=4)
    args, config, pl_args = parser.parse_args(fake_cli_args)
    assert args.my_cool_arg == 3
    assert args.hello_there == 4


if __name__ == "__main__":
    parser = ArgParseWithConfig(add_lightning_args=True, config_pl_args=["--val-paths"])
    parser.add_argument("--my-cool-arg", type=str, default=3)
    parser.add_argument("--hello-there", type=str, default=4)
    args, config, pl_args = parser.parse_args()
    print(args, config, pl_args)
