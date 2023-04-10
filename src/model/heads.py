import torch.nn as nn

from src.model.attention import AttentionLayer
from src.utils.utils_model import filter_modules, get_linear_init, initialize_weights


class DeepHead(nn.Module):
    def __init__(
        self, dimensions: list[int], dropout_p=0.3, activation=nn.ReLU()
    ) -> None:
        """List of input and output features which will create N - 1 fully connected layers.

        Args:
            dimensions: list[int], e.g [128, 64]
        """
        assert len(dimensions) >= 2, "Dimensions have to contain at least two ints"

        super().__init__()
        self.dimensions = dimensions
        modules = nn.Sequential()
        if len(self.dimensions) == 2:
            in_features, out_features = self.dimensions[0], self.dimensions[1]
            layer_norm = nn.LayerNorm(in_features)
            linear = nn.Linear(in_features, out_features)
            modules.add_module("layernorm", layer_norm)
            modules.add_module("activation", activation)
            modules.add_module("linear", linear)
        else:
            for i in range(len(self.dimensions) - 1):
                layer = nn.Sequential()

                in_features, out_features = self.dimensions[i], self.dimensions[i + 1]
                layer_norm = nn.LayerNorm(in_features)
                linear = nn.Linear(in_features, out_features)

                layer.add_module("layernorm", layer_norm)
                layer.add_module("activation", activation)
                layer.add_module("linear", linear)

                if i != len(self.dimensions) - 1:
                    # don't add dropout to the classifier itself!
                    dropout = nn.Dropout(p=dropout_p)
                    layer.add_module("dropout", dropout)

                modules.add_module(str(i), layer)

        initialization, initialization_kwargs = get_linear_init(activation)
        linears = filter_modules(modules, nn.Linear)
        assert len(linears) != 0, "Why is this empty? Y u runnin?"
        initialize_weights(linears, initialization, initialization_kwargs)

        self.deep_head = modules

    def forward(self, x):
        return self.deep_head(x)


class AttentionHead(nn.Module):
    """Attention heads perform attetional pooling on feature vectors.

    This head returns a single scalar value used for binary classification. (use this for single
    instrument classification only!)
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_size: int,
        activation=nn.ReLU(),
        dropout_p=0.3,
    ) -> None:
        super().__init__()
        self.attention_layer = AttentionLayer(input_size=input_size)

        dimensions = [input_size]
        dimensions.extend([hidden_size] * num_layers)
        dimensions.append(1)

        self.classifer = DeepHead(
            dimensions=dimensions,
            dropout_p=dropout_p,
            activation=activation,
        )

    def forward(self, features):
        """
        Args:
            features: torch.Tensor (N,T,D): N is batch size, T is the temporal length,
            D is the feature vector dimension
        """
        attention = self.attention_layer(features)
        return self.classifer(attention)
