import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_metric_learning import losses


class ArcFaceModel(pl.LightningModule):
    def __init__(
        self,
        embdedding_model: nn.Module | pl.LightningModule,
        num_classes: int,
        embedding_space_size: int,
    ) -> None:
        super().__init__()
        self.embedding_model = embdedding_model
        self.arcface = losses.ArcFaceLoss(
            embedding_size=embedding_space_size,
            num_classes=num_classes,
            scale=10,
            margin=32,
        )

    def forward(self, x, labels):
        labels = torch.nonzero(labels)[:, 1]
        embeddings = self.embedding_model(x)
        logits = self.arcface.get_logits(embeddings)
        logits = torch.mm(embeddings, self.loss_function.W)
        return logits
