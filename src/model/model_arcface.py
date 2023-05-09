import pytorch_lightning as pl
from pytorch_metric_learning import losses


class ArcFaceModel(pl.LightningModule):
    def __init__(
        self,
        embedding_space_size: int,
        num_classes: int = 11,
    ) -> None:
        super().__init__()
        self.arcface = losses.ArcFaceLoss(
            embedding_size=embedding_space_size,
            num_classes=num_classes,
            scale=10,
            margin=32,
        )

    def forward(self, x):
        logits = self.arcface.get_logits(x)
        return logits
