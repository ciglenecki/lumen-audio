import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torch_scatter import scatter_max
from torchmetrics.classification import MultilabelF1Score
from torchvision.models import (
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
    resnext50_32x4d,
    resnext101_32x8d,
    resnext101_64x4d,
)

from src.model.heads import DeepHead
from src.model.model import SupportedModels
from src.model.model_base import ModelBase
from src.utils.utils_audio import plot_spectrograms
from src.utils.utils_dataset import decode_instruments
from src.utils.utils_exceptions import UnsupportedModel

TORCHVISION_CONSTRUCTOR_DICT = {
    SupportedModels.EFFICIENT_NET_V2_S: efficientnet_v2_s,
    SupportedModels.EFFICIENT_NET_V2_M: efficientnet_v2_m,
    SupportedModels.EFFICIENT_NET_V2_L: efficientnet_v2_l,
    SupportedModels.RESNEXT50_32X4D: resnext50_32x4d,
    SupportedModels.RESNEXT101_32X8D: resnext101_32x8d,
    SupportedModels.RESNEXT101_64X4D: resnext101_64x4d,
}
import time

import src.config.config_defaults as config_defaults
from src.utils.utils_functions import timeit


class TorchvisionModel(ModelBase):
    """Implementation of a torchvision model accessed using a string."""

    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        torch.set_float32_matmul_precision("medium")

        self.hamming_distance = torchmetrics.HammingDistance(
            task="multilabel", num_labels=self.num_labels
        )

        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)

        if self.model_enum not in TORCHVISION_CONSTRUCTOR_DICT:
            raise UnsupportedModel(
                f"If you want to use {self.model_enum} in TorchvisionModel you need to add the enum to TORCHVISION_CONSTRUCTOR_DICT map."
            )

        backbone_constructor = TORCHVISION_CONSTRUCTOR_DICT[self.model_enum]
        backbone_kwargs = {}

        if backbone_constructor in {
            resnext50_32x4d,
            resnext101_32x8d,
            resnext101_64x4d,
        }:
            backbone_kwargs.update({"zero_init_residual": True})

        self.backbone = backbone_constructor(
            weights=self.pretrained_tag, progress=True, **backbone_kwargs
        )

        print("------------------------------------------")
        print("\n")
        print("Backbone before changing the classifier:")
        print(list(self.backbone.children())[-1])
        print("\n")
        print("------------------------------------------")

        last_module_name = [
            i[0]
            for i in self.backbone.named_modules()
            if "." not in i[0] and i[0] != ""
        ][-1]
        last_module = getattr(self.backbone, last_module_name)
        last_dim = (
            last_module[-1].in_features
            if isinstance(last_module, nn.Sequential)
            else last_module.in_features
        )

        setattr(self.backbone, last_module_name, DeepHead([last_dim, self.num_labels]))

        print("\n")
        print("Backbone after changing the classifier:")
        print(list(self.backbone.children())[-1])
        print("\n")
        print("------------------------------------------")

        self.save_hyperparameters()

    def forward(self, audio: torch.Tensor):
        out = self.backbone.forward(audio)
        return out

    def _step(self, batch, batch_idx, type: str):
        """
        - batch_size: 4.
        - `batch` size can actually be bigger (10) because of chunking.
        - Split the `batch` with batch_size
        - sub_batches = [[4, height, width], [4, height, width], [2, height, width]]
        """
        # print("START", "_step")
        # ts = time.time()

        images, y, _, item_index = batch

        # Uncomment if y ou want to plot and play audio
        # if type == "train":
        #     irmas_dataset = self.trainer.train_dataloader.dataset.datasets.datasets[0]
        # else:
        #     irmas_dataset = self.trainer.val_dataloaders[0].dataset.datasets[0]
        # transform = irmas_dataset.audio_transform
        # audio_label_path = [irmas_dataset.load_sample(i) for i in item_index]
        # instrument_name = [
        #     config_defaults.INSTRUMENT_TO_FULLNAME[decode_instruments(x[1])[0]]
        #     for x in audio_label_path
        # ]
        # paths = [x[2] for x in audio_label_path]
        # titles = [
        #     f"{i} {n} {p}" for i, (p, n) in enumerate(zip(paths, instrument_name))
        # ]
        # plot_spectrograms(
        #     transform.undo(images),
        #     sampling_rate=self.config.sampling_rate,
        #     n_fft=self.config.n_fft,
        #     n_mels=self.config.n_mels,
        #     hop_length=self.config.hop_length,
        #     y_axis="mel",
        #     titles=titles,
        #     # block_plot=False,
        # )
        # item_index = list(set(item_index))
        # orig_audios = [irmas_dataset.load_sample(i)[0] for i in item_index]
        # titles = list(set(titles))

        # while True:
        #     for audio, title in zip(orig_audios, titles):
        #         print(title)
        #         play_audio(audio, transform.sampling_rate)
        #     pass

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
            b_logits_pred = self.forward(sub_batch_image)
            b_loss = self.loss_function(b_logits_pred, b_y)

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

        # te = time.time()
        # print("END", "_step", "time:", round((te - ts) * 1000, 1), "ms")

        return self.log_and_return_loss_step(
            loss=loss, y_pred=y_pred, y_true=y, type=type
        )

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="test")

    # def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
    #     images, y, file_indices, item_index = batch

    #     sub_batches = torch.split(images, self.batch_size, dim=0)
    #     loss = 0
    #     y_pred_prob = torch.zeros((len(images), self.num_labels), device=self.device)
    #     y_pred = torch.zeros((len(images), self.num_labels), device=self.device)

    #     passed_images = 0
    #     for sub_batch_image in sub_batches:
    #         b_size = len(sub_batch_image)
    #         start = passed_images
    #         end = passed_images + b_size

    #         b_y = y[start:end]
    #         b_logits_pred = self.forward(sub_batch_image)
    #         b_loss = self.loss_function(b_logits_pred, b_y)

    #         b_y_pred_prob = torch.sigmoid(b_logits_pred)
    #         b_y_pred = (b_y_pred_prob >= 0.5).float()

    #         loss += b_loss * b_size
    #         y_pred_prob[start:end] = b_y_pred_prob
    #         y_pred[start:end] = b_y_pred

    #         passed_images += b_size

    #     y_final_out, _ = scatter_max(y_pred, file_indices, dim=0)

    #     loss = loss / len(images)
    #     return self.log_and_return_loss_step(
    #         loss=loss, y_pred=y_pred, y_true=y, type=type
    #     )
