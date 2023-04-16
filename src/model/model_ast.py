from typing import Any

import librosa
import torch
import torchaudio
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import src.config.config_defaults as config_defaults
from src.config.argparse_with_config import ArgParseWithConfig
from src.model.model_base import ModelBase
from src.utils.utils_audio import load_audio_from_file, play_audio
from src.utils.utils_dataset import get_example_val_sample


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
        """
        - batch_size: 4.
        - `batch` size can actually be bigger (10) because of chunking.
        - Split the `batch` with batch_size
        - sub_batches = [[4, height, width], [4, height, width], [2, height, width]]
        """

        images, y, file_indices, item_indices = batch

        # plot_spectrograms(
        #     images,
        #     sampling_rate=self.config.sampling_rate,
        #     n_fft=self.config.n_fft,
        #     n_mels=self.config.n_mels,
        #     hop_length=self.config.hop_length,
        #     y_axis=None,
        # )

        sub_batches = torch.split(images, self.batch_size, dim=0)
        loss = 0
        y_pred_prob = torch.zeros((len(images), self.num_labels), device=self.device)
        y_pred = torch.zeros((len(images), self.num_labels), device=self.device)

        passed_images = 0
        for sub_batch_image in sub_batches:
            b_size = len(sub_batch_image)
            start = passed_images
            end = passed_images + b_size

            b_y = y[start:end]
            b_loss, b_logits_pred = self.forward(sub_batch_image, labels=b_y)
            b_y_pred_prob = torch.sigmoid(b_logits_pred)
            b_y_pred = (b_y_pred_prob >= 0.5).float()

            loss += b_loss * b_size
            y_pred_prob[start:end] = b_y_pred_prob
            y_pred[start:end] = b_y_pred

            passed_images += b_size
        if type != "train":
            pass
            # y_final_out, _ = scatter_max(y_pred, file_indices, dim=0)

        loss = loss / len(images)
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
    parser = ArgParseWithConfig()
    args, config, pl_args = parser.parse_args()
    audio = get_example_val_sample(config.sampling_rate)

    # example_audio_mel_audio()
    config_ = ASTConfig.from_pretrained(
        pretrained_model_name_or_path=config_defaults.TAG_AST_AUDIOSET,
        id2label=config_defaults.IDX_TO_INSTRUMENT,
        label2id=config_defaults.INSTRUMENT_TO_IDX,
        num_labels=config_defaults.DEFAULT_NUM_LABELS,
        finetuning_task="audio-classification",
        problem_type="multi_label_classification",
    )

    backbone: ASTForAudioClassification = ASTForAudioClassification.from_pretrained(
        config_defaults.TAG_AST_AUDIOSET,
        config=config_,
        ignore_mismatched_sizes=True,
    )
    target_sr = 16_000
    n_fft = 400
    hop = 160
    mel_bins = 128
    fname = "data/irmas_sample/1 - Hank's Other Bag-1.wav"

    feature_extractor = ASTFeatureExtractor.from_pretrained(
        config_defaults.TAG_AST_AUDIOSET,
        normalize=False,
    )

    audio_torch, org_sample_rate = torchaudio.load(fname)
    resampler = torchaudio.transforms.Resample(
        org_sample_rate, target_sr, dtype=audio_torch.dtype
    )
    audio_torch = resampler(audio_torch)
    audio_torch = audio_torch.mean(dim=0, keepdim=False)

    audio_lib, org_sample_rate = load_audio_from_file(
        fname, target_sr=target_sr, method="librosa"
    )
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
    play_audio(inv_torch.squeeze(0).numpy(), sampling_rate=target_sr, max_seconds=3)
    play_audio(inv_lib.squeeze(0).numpy(), sampling_rate=target_sr, max_seconds=3)
