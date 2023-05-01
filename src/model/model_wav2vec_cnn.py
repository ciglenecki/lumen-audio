import torch
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import Wav2Vec2Config, Wav2Vec2Model

import src.config.config_defaults as config_defaults
from src.model.heads import AttentionHead
from src.model.model_base import ForwardInput, ForwardOut, ModelBase


class Wav2VecCnnWrapper(ModelBase):
    """Wav2Vec2's convolution (no transformer!) model."""

    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        time_dim_pooling_mode="mean",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.time_dim_pooling_mode = (
            "attention"
            if self.head_constructor == AttentionHead
            else time_dim_pooling_mode
        )

        self.wav2vec_config = Wav2Vec2Config(
            pretrained_model_name_or_path=self.pretrained_tag,
            id2label=config_defaults.IDX_TO_INSTRUMENT,
            label2id=config_defaults.INSTRUMENT_TO_IDX,
            num_labels=self.num_labels,
            finetuning_task="audio-classification",
            problem_type="multi_label_classification",
        )

        self.backbone = Wav2Vec2Model.from_pretrained(
            self.pretrained_tag,
            config=self.wav2vec_config,
            ignore_mismatched_sizes=True,
        ).feature_extractor

        output_size = self.wav2vec_config.conv_dim[-1]
        self.classifier = self.create_head(head_input_size=output_size)
        self.save_hyperparameters()

    def time_dim_pooling(self, hidden_states):
        """Reduce the temporal features of the backbone last hidden state."""

        # Original shape: [Batch size, Feature Dimension, Time]
        # Result shape: [Batch size, Time, Feature Dimension]
        hidden_states = torch.permute(hidden_states, [0, 2, 1])
        if self.time_dim_pooling_mode == "attention":
            return hidden_states
        elif self.time_dim_pooling_mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif self.time_dim_pooling_mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif self.time_dim_pooling_mode == "max":
            outputs, _ = torch.max(hidden_states, dim=1)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max', 'attention']"
            )

        return outputs

    def forward(self, audio: torch.Tensor):
        cnn_features = self.backbone.forward(audio)
        hidden_states = self.time_dim_pooling(cnn_features)
        logits_pred = self.classifier(hidden_states)
        return logits_pred

    def forward_wrapper(self, forward_input: ForwardInput) -> ForwardOut:
        audio, y_true = forward_input.feature, forward_input.y_true
        logits_pred = self.forward(audio)
        if y_true is not None:
            loss = self.loss_function(logits_pred, y_true)
        else:
            loss = None
        return ForwardOut(logits=logits_pred, loss=loss)
