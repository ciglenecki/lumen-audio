import torch.nn as nn


class FCBlock(nn.Sequential):
    def __init__(
        self,
        input_size,
        output_size,
        activation,
        normalization=nn.LayerNorm,
        dropout=0.0,
    ):
        super().__init__()
        if dropout != 0.0:
            self.add_module("dropout", nn.Dropout(p=dropout))

        self.add_module("norm", normalization(input_size))
        self.add_module("activation", activation)
        self.add_module(
            "linear", nn.Linear(in_features=input_size, out_features=output_size)
        )


class BnActConv2d(nn.Sequential):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        activation=nn.ReLU(),
        bias=False,
        padding=2,
        stride=1,
        switch=False,
    ):
        super().__init__()
        if switch:
            self.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding=padding,
                    stride=stride,
                ),
            )
            self.add_module("batchNorm", nn.BatchNorm2d(out_features))
            self.add_module("activation", activation)
        else:
            self.add_module("batchNorm", nn.BatchNorm2d(in_features))
            self.add_module("activation", activation)
            self.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding=padding,
                    stride=stride,
                ),
            )


class BnActConv1d(nn.Sequential):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        activation=nn.ReLU(),
        bias=False,
        padding=2,
        stride=1,
        switch=False,
    ):
        super().__init__()
        if switch:
            self.add_module(
                "conv",
                nn.Conv1d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding=padding,
                    stride=stride,
                ),
            )
            self.add_module("batchNorm", nn.BatchNorm1d(out_features))
            self.add_module("activation", activation)
        else:
            self.add_module("batchNorm", nn.BatchNorm1d(in_features))
            self.add_module("activation", activation)
            self.add_module(
                "conv",
                nn.Conv1d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding=padding,
                    stride=stride,
                ),
            )
