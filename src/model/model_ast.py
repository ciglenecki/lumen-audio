from typing import Any

import librosa
import torch
import torchaudio
from pytorch_lightning.loggers import TensorBoardLogger
from torch_scatter import scatter_max
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import src.config.config_defaults as config_defaults
from src.config.config_train import config
from src.model.heads import DeepHead
from src.model.model_base import ModelBase
from src.utils.utils_audio import (
    ast_spec_to_audio,
    load_audio_from_file,
    play_audio,
    plot_spectrograms,
)


class ASTModelWrapper(ModelBase):
    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        ast_config = ASTConfig.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_tag,
            id2label=config_defaults.IDX_TO_INSTRUMENT,
            label2id=config_defaults.INSTRUMENT_TO_IDX,
            num_labels=self.num_labels,
            finetuning_task="audio-classification",
            problem_type="multi_label_classification",
        )

        self.backbone: ASTForAudioClassification = (
            ASTForAudioClassification.from_pretrained(
                self.pretrained_tag,
                config=ast_config,
                ignore_mismatched_sizes=True,
            )
        )

        # middle_size = int(
        #     math.sqrt(config.hidden_size * self.num_labels) + self.num_labels
        # )
        # self.backbone.classifier = DeepHead([ast_config.hidden_size, self.num_labels])
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
        spectrogram, y, file_indices = batch
        # plot_spectrograms(spectrogram, y_axis=None)
        # play_audio(
        #     ast_spec_to_audio(spectrogram[0].unsqueeze(0)), sr=config.sampling_rate
        # )
        loss, logits_pred = self.forward(spectrogram, labels=y)
        y_pred_prob = torch.sigmoid(logits_pred)
        y_pred = (y_pred_prob >= 0.5).float()

        if type != "train":
            """
            >>> a
            tensor([[0.6744, 0.7307, 0.6614],
                    [0.1346, 0.0142, 0.5730],
                    [0.3153, 0.0235, 0.7663],
                    [0.4487, 0.9715, 0.9067],
                    [0.3930, 0.9055, 0.6433]])
            >>> ids = torch.tensor([0,0,0,1,2])
            >>> ids
            tensor([0, 0, 0, 1, 2])
            >>> scatter_max(a,ids,dim=0)
            (tensor([[0.6744, 0.7307, 0.7663],
                    [0.4487, 0.9715, 0.9067],
                    [0.3930, 0.9055, 0.6433]]), tensor([[0, 0, 2],
                    [3, 3, 3],
                    [4, 4, 4]]))
            """

            y_final_out, _ = scatter_max(y_pred, file_indices, dim=0)

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


if __name__ == "__main__":
    # example_audio_mel_audio()
    config_ = ASTConfig.from_pretrained(
        pretrained_model_name_or_path=config_defaults.DEFAULT_AST_PRETRAINED_TAG,
        id2label=config_defaults.IDX_TO_INSTRUMENT,
        label2id=config_defaults.INSTRUMENT_TO_IDX,
        num_labels=config_defaults.DEFAULT_NUM_LABELS,
        finetuning_task="audio-classification",
        problem_type="multi_label_classification",
    )

    backbone: ASTForAudioClassification = ASTForAudioClassification.from_pretrained(
        config_defaults.DEFAULT_AST_PRETRAINED_TAG,
        config=config_,
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
    grif = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop, n_iter=54, power=2
    )
    inv_torch = grif(inv(torch.tensor(torch_spec)))
    inv_lib = grif(inv(torch.tensor(lib_spec)))

    torch_reconstructed = librosa.feature.inverse.mel_to_audio(
        torch_spec, sr=target_sr, n_fft=n_fft, hop_length=hop
    )
    lib_reconstruct = librosa.feature.inverse.mel_to_audio(
        lib_spec, sr=target_sr, n_fft=n_fft, hop_length=hop
    )

    print(librosa.get_duration(y=lib_reconstruct, sr=16_000))
    print(len(lib_reconstruct) / 16_000)

    # play_audio(torch_reconstructed, sr=target_sr, max_seconds=3)
    # play_audio(lib_reconstruct, sr=target_sr, max_seconds=3)
    play_audio(inv_torch.squeeze(0).numpy(), sr=target_sr, max_seconds=3)
    play_audio(inv_lib.squeeze(0).numpy(), sr=target_sr, max_seconds=3)
