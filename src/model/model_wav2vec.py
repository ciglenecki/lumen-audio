import torch
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import Wav2Vec2Config, Wav2Vec2Model

import src.config.config_defaults as config_defaults
from src.model.model_base import ModelBase


class Wav2VecWrapper(ModelBase):
    """Original Wav2Vec2 model."""

    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        time_dim_pooling_mode="mean",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.time_dim_pooling_mode = time_dim_pooling_mode

        config_wav2vec = Wav2Vec2Config(
            pretrained_model_name_or_path=self.pretrained_tag,
            id2label=config_defaults.IDX_TO_INSTRUMENT,
            label2id=config_defaults.INSTRUMENT_TO_IDX,
            num_labels=self.num_labels,
        )
        self.backbone: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(
            self.pretrained_tag,
            config=config_wav2vec,
            ignore_mismatched_sizes=True,
        )

        self.classifier = self.create_head(config_wav2vec.hidden_size)

        self.save_hyperparameters()

    def time_dim_pooling(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
            )

        return outputs

    def forward(self, audio: torch.Tensor):
        hidden_states = self.backbone.forward(
            input_values=audio,
            output_attentions=False,
            return_dict=True,
        ).last_hidden_state
        hidden_states = self.time_dim_pooling(
            hidden_states, mode=self.time_dim_pooling_mode
        )

        logits_pred = self.classifier(hidden_states)
        return logits_pred
