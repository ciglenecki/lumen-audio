import math
from typing import Any

import librosa
import numpy as np
import torch
import torchaudio
import torchmetrics
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MultilabelF1Score
from transformers import (
    ASTConfig,
    ASTFeatureExtractor,
    ASTForAudioClassification,
    Wav2Vec2FeatureExtractor,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import src.config.config_defaults as config_defaults
from src.model.deep_head import DeepHead
from src.model.model_base import ModelBase
from src.train.metrics import get_metrics
from src.utils.utils_audio import (
    example_audio_mel_audio,
    load_audio_from_file,
    play_audio,
    plot_spec_general,
    plot_spec_general_no_scale,
)
from src.utils.utils_functions import add_prefix_to_keys, print_tensor


class ASTModelWrapper(ModelBase):
    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        model_name: str = config_defaults.DEFAULT_AST_PRETRAINED_TAG,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_name = model_name

        config = ASTConfig.from_pretrained(
            pretrained_model_name_or_path=model_name,
            id2label=config_defaults.IDX_TO_INSTRUMENT,
            label2id=config_defaults.IDX_TO_INSTRUMENT,
            num_labels=self.num_labels,
            finetuning_task="audio-classification",
            problem_type="multi_label_classification",
        )

        self.backbone: ASTForAudioClassification = (
            ASTForAudioClassification.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
        )

        middle_size = int(
            math.sqrt(config.hidden_size * self.num_labels) + self.num_labels
        )

        self.backbone.classifier = DeepHead(
            [config.hidden_size, middle_size, self.num_labels]
        )

        self.save_hyperparameters()

    def forward(self, audio: torch.Tensor, labels: torch.Tensor):
        out: SequenceClassifierOutput = self.backbone.forward(
            audio,
            output_attentions=True,
            return_dict=True,
            labels=labels,
        )
        return out.loss, out.logits

    def _step(self, batch, batch_idx, type: str):
        audio, y = batch

        loss, logits_pred = self.forward(audio, labels=y)
        y_pred_prob = torch.sigmoid(logits_pred)
        y_pred = y_pred_prob >= 0.5
        return self.log_and_return_loss_step(
            loss=loss, y_pred=y_pred, y_true=y, type=type
        )

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="test")

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        # TODO:
        pass


def invert_ast_feature(
    spectrogram: torch.Tensor,
    n_fft=400,
    hop=160,
    target_sr=config_defaults.DEFAULT_SAMPLING_RATE,
):
    """play_audio(invert_ast_feature(features_torch, n_fft, hop), sr=target_sr, max_seconds=3)"""
    spectrogram = spectrogram.T
    spectrogram.requires_grad = True
    inv = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, sample_rate=target_sr
    )
    grif = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop, n_iter=54)
    # torchaudio.functional.inverse_spectrogram()
    tmp = inv(spectrogram)
    audio = grif(tmp)
    return audio


if __name__ == "__main__":
    config = ASTConfig.from_pretrained(
        pretrained_model_name_or_path=config_defaults.DEFAULT_AST_PRETRAINED_TAG,
        id2label=config_defaults.IDX_TO_INSTRUMENT,
        label2id=config_defaults.IDX_TO_INSTRUMENT,
        num_labels=11,
        finetuning_task="audio-classification",
        problem_type="multi_label_classification",
    )

    backbone: ASTForAudioClassification = ASTForAudioClassification.from_pretrained(
        config_defaults.DEFAULT_AST_PRETRAINED_TAG,
        config=config,
        ignore_mismatched_sizes=True,
    )
    target_sr = 16_000
    n_fft = 400
    hop = 160
    mel_bins = 128
    fname = "data/irmas_sample/1 - Hank's Other Bag-1.wav"

    feature_extractor = ASTFeatureExtractor.from_pretrained(
        config_defaults.DEFAULT_AST_PRETRAINED_TAG,
        do_normalize=False,
    )

    audio_torch, org_sample_rate = torchaudio.load(fname)
    resampler = torchaudio.transforms.Resample(
        org_sample_rate, target_sr, dtype=audio_torch.dtype
    )
    audio_torch = resampler(audio_torch)
    audio_torch = audio_torch.mean(dim=0, keepdim=False)

    audio_lib, org_sample_rate = load_audio_from_file(fname, method="librosa")
    audio_lib = librosa.resample(
        y=audio_lib, orig_sr=org_sample_rate, target_sr=target_sr
    )

    features_torch = feature_extractor(
        audio_torch,
        sampling_rate=target_sr,
        return_tensors="np",
    )["input_values"]

    features_librosa = feature_extractor(
        audio_lib,
        sampling_rate=target_sr,
        return_tensors="np",
    )["input_values"]

    torch_spec = features_torch
    lib_spec = features_librosa

    torch_spec = torch_spec.squeeze(0).T
    lib_spec = lib_spec.squeeze(0).T
    inv = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, sample_rate=target_sr
    )
    grif = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop, n_iter=54)
    inv_torch = grif(inv(torch.tensor(torch_spec)))
    inv_lib = grif(inv(torch.tensor(lib_spec)))

    torch_reconstructed = librosa.feature.inverse.mel_to_audio(
        torch_spec, sr=target_sr, n_fft=n_fft, hop_length=hop
    )
    lib_reconstruct = librosa.feature.inverse.mel_to_audio(
        lib_spec, sr=target_sr, n_fft=n_fft, hop_length=hop
    )
    play_audio(torch_reconstructed, sr=target_sr, max_seconds=3)
    play_audio(lib_reconstruct, sr=target_sr, max_seconds=3)
    play_audio(inv_torch.squeeze(0).numpy(), sr=target_sr, max_seconds=3)
    play_audio(inv_lib.squeeze(0).numpy(), sr=target_sr, max_seconds=3)
