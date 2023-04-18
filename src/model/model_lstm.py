import math
from typing import Any

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MultilabelF1Score

import src.config.config_defaults as config_defaults
from src.model.fluffy import Fluffy

from src.model.attention import AttentionLayer
from src.model.model_base import ModelBase
from src.model.optimizers import our_configure_optimizers



class LSTMWrapper(ModelBase):

    def __init__(self,
                 activation = nn.ReLU(),
                 **lstm_kwargs
                 ) -> None:
        super().__init_()
        lstm_kwargs.update({"batch_first":True})
        self.activation = activation
        self.classifier = nn.LSTM(
            **lstm_kwargs
        )
        self.attention = AttentionLayer(
            input_size = lstm_kwargs["hidden_size"]
        )
        self.batch_norm = nn.BatchNorm1d(lstm_kwargs["input_size"])
        # 2 times the hidden size because of the concatenation
        # with the attention layer output
        self.layer_norm = nn.LayerNorm(2*lstm_kwargs["hidden_size"])

        self.linear = nn.Linear(2*lstm_kwargs["hidden_size"],self.num_labels)

    def forward(self,x):
        x = self.batch_norm(x.permute(0,2,1)).permute(0,2,1)
        outputs, (h,c) = self.lstm(x)
        attn = self.attention(outputs)
        h = torch.cat([h[-1],attn],dim = 1)
        out = self.layer_norm(h)
        out = self.activation(out)
        return self.linear(out).view(-1)



