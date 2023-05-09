from typing import Any

import torch
import torch.nn as nn
import torchaudio
from pytorch_lightning.loggers import TensorBoardLogger

import src.config.config_defaults as config_defaults
from src.model.attention import AttentionLayer
from src.model.model_base import ModelBase
from src.model.spec_conv import SpecConv


class ConvLSTM(ModelBase):
    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        convolution=None,
        hidden_dim=64,
        num_layers=2,
        num_classes=config_defaults.DEFAULT_NUM_LABELS,
        bidirectional=False,
        dropout_p=0.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if convolution is None:
            self.convolution = SpecConv(
                dimensions=[1, 32, 64, 128], dropout_p=dropout_p
            )
            embedding_dim = 128
        else:
            self.convolution = convolution
            embedding_dim = self.convolution.embedding_dim

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_p,
        )
        self.attn = AttentionLayer(input_size=hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.save_hyperparameters()

    def load_convolution_from_pretrained(self, conv_state_dict):
        self.convolution.load_state_dict(conv_state_dict)

    def forward(self, x):
        features = self.convolution(x)
        # N C, T, F  -> N, T, C, F
        features = features.permute(0, 2, 1, 3)  # change from NCHW to NHWC for LSTM
        # batch_size, seq_len, num_channels, height = features.shape
        features = torch.mean(features, dim=-1, keepdim=False)
        # features = features.reshape(batch_size, seq_len, -1)
        output, _ = self.lstm(features)
        output, att_weights = self.attn(output)
        output = self.linear(output)
        return output, att_weights
