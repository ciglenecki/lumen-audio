import torch.nn as nn


class DeepHead(nn.Module):
    def __init__(self, dimensions: list[int], dropout_p=0.3) -> None:
        """List of input and output features which will create N - 1 fully connected layers.

        Args:
            dimensions: list[int]
        """
        assert len(dimensions) >= 2, "Dimensions have to contain at least two ints"

        super().__init__()
        self.dimensions = dimensions
        modules = nn.Sequential()
        if len(self.dimensions) == 2:
            in_features, out_features = self.dimensions[0], self.dimensions[1]
            layer_norm = nn.LayerNorm(in_features)
            relu = nn.ReLU()
            linear = nn.Linear(in_features, out_features)
            modules.add_module("layernorm", layer_norm)
            modules.add_module("relu", relu)
            modules.add_module("linear", linear)
        else:
            for i in range(len(self.dimensions) - 1):

                layer = nn.Sequential()

                in_features, out_features = self.dimensions[i], self.dimensions[i + 1]
                layer_norm = nn.LayerNorm(in_features)
                relu = nn.ReLU()
                linear = nn.Linear(in_features, out_features)

                layer.add_module("layernorm", layer_norm)
                layer.add_module("relu", relu)
                layer.add_module("linear", linear)

                if i != len(self.dimensions) - 1:
                    # don't add dropout to the classifier itself!
                    dropout = nn.Dropout(p=dropout_p)
                    layer.add_module("dropout", dropout)

                modules.add_module(str(i), layer)

        self.deep_head = modules

    def forward(self, x):
        return self.deep_head(x)
