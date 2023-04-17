import lovely_tensors as lt
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
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
from src.utils.utils_exceptions import UnsupportedModel

TORCHVISION_CONSTRUCTOR_DICT = {
    SupportedModels.EFFICIENT_NET_V2_S: efficientnet_v2_s,
    SupportedModels.EFFICIENT_NET_V2_M: efficientnet_v2_m,
    SupportedModels.EFFICIENT_NET_V2_L: efficientnet_v2_l,
    SupportedModels.RESNEXT50_32X4D: resnext50_32x4d,
    SupportedModels.RESNEXT101_32X8D: resnext101_32x8d,
    SupportedModels.RESNEXT101_64X4D: resnext101_64x4d,
}


class TorchvisionModel(ModelBase):
    """Implementation of a torchvision model accessed using a string."""

    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.hamming_distance = torchmetrics.HammingDistance(
            task="multilabel", num_labels=self.num_labels
        )

        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)

        if self.model_enum not in TORCHVISION_CONSTRUCTOR_DICT:
            raise UnsupportedModel(
                f"If you want to use {self.model_enum} in TorchvisionModel you need to add the enum to TORCHVISION_CONSTRUCTOR_DICT map."
            )
        self.backbone = TORCHVISION_CONSTRUCTOR_DICT[self.model_enum](
            weights=self.pretrained_tag, progress=True
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
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, audio: torch.Tensor):
        out = self.backbone.forward(audio)
        return out

    def _eval_step(self, batch, batch_idx, type: str):
        images, y, _, _ = batch

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

            y_pred_prob[start:end] = b_y_pred_prob
            y_pred[start:end] = b_y_pred

            passed_images += b_size
            loss += float(b_loss) * b_size

        if type != "train":
            pass
            # y_final_out, _ = scatter_max(y_pred, file_indices, dim=0)

        loss = loss / len(images)
        return self.log_and_return_loss_step(
            loss=loss, y_pred=y_pred, y_true=y, type=type
        )

    def _step(self, batch, batch_idx, type: str):
        """
        - batch_size: 4.
        - `batch` size can actually be bigger (10) because of chunking.
        - Split the `batch` with batch_size
        - sub_batches = [[4, height, width], [4, height, width], [2, height, width]]
        """

        images, y, _, _ = batch

        # Uncomment to plot
        # irmas_dataset = self.trainer.train_dataloader.dataset.datasets.datasets[0]
        # transform = irmas_dataset.audio_transform
        # plot_spectrograms(
        #     transform.undo(images),
        #     sampling_rate=self.config.sampling_rate,
        #     n_fft=self.config.n_fft,
        #     n_mels=self.config.n_mels,
        #     hop_length=self.config.hop_length,
        #     y_axis="mel",
        # )

        # Accumulation variables
        sub_batches = torch.split(images, self.batch_size, dim=0)
        loss = 0
        y_pred_prob = torch.zeros((len(images), self.num_labels), device=self.device)
        y_pred = torch.zeros((len(images), self.num_labels), device=self.device)
        passed_images = 0

        for sub_batch_image in sub_batches:
            for o in self.optimizers_list:
                o.zero_grad()

            b_size = len(sub_batch_image)
            start = passed_images
            end = passed_images + b_size

            b_y = y[start:end]
            b_logits_pred = self.forward(sub_batch_image)
            b_loss = self.loss_function(b_logits_pred, b_y)

            b_y_pred_prob = torch.sigmoid(b_logits_pred)
            b_y_pred = (b_y_pred_prob >= 0.5).float()

            y_pred_prob[start:end] = b_y_pred_prob
            y_pred[start:end] = b_y_pred

            passed_images += b_size
            self.manual_backward(b_loss)

            for optimizer in self.optimizers_list:
                optimizer.step()

            loss += float(b_loss) * b_size

        # Perform scheduler step only when all subbatches are over
        for optim_idx, scheduler in enumerate(self.schedulers_list):
            self.lr_scheduler_step(scheduler, optim_idx, metric=None)

        if type != "train":
            pass
            # y_final_out, _ = scatter_max(y_pred, file_indices, dim=0)

        loss = loss / len(images)
        self.log_and_return_loss_step(loss=loss, y_pred=y_pred, y_true=y, type=type)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="train")

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, type="val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, type="test")
