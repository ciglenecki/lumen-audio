from typing import Any
import torch
import torch.nn as nn
import pytorch_lightning as pl

from model.attention import AttentionLayer
from model.heads import AttentionHead
from src.model.model_base import ModelBase
from src.train.metrics import get_metrics


import torchopenl3

    
class OpenL3Full(ModelBase):

                 
    def __init__(self,
                 sampling_rate,
                 embedding_size = 512,
                 head_dimensions = [128],
                 *args,
                 **kwargs
                 ) -> None:
        super(*args,**kwargs).__init__()
        self.embedding_size = embedding_size
        self.backbone = torchopenl3.models.load_audio_embedding_model(
        content_type = "music",embedding_size = 512,input_repr="mel128"
                            )
        self.sampling_rate = sampling_rate
                 
        self.head = AttentionHead(
                [self.backbone.embedding_size]+head_dimensions+[self.num_labels]
            )
    

    def forward(self,audio):
        embeddings,timesamps = torchopenl3.get_audio_embedding(
            audio=audio,
            model=self.model,
            center=False,
            sr = self.sampling_rate
        )
  
        out, attention_weights = self.head(embeddings)
        return out
    




    def log_and_return_loss_step(self, loss, y_pred, y_true, type):
        """Has to return dictionary with 'loss'.

        Lightning uses loss variable to perform backwards

        metric_dict = {
            "train/loss": ...,
            "train/f1": ...,
            "val/loss"...,
        }
        """

        metric_dict = get_metrics(
            y_pred=y_pred,
            y_true=y_true,
            num_labels=11,
            return_per_instrument=True,
        )

        # add "loss" metric which will be converted to "train/loss", "val/loss"...
        metric_dict.update({"loss": loss})

        # add prefix "trian" or "test" but skip adding "train" / "test" to per instrument metrics to avoid clutter in tensorboard
        metric_dict = {
            f"{k[:11]}/{type}{k[11:]}"
            if k.startswith("instruments")
            else f"{type}/{k}": v
            for k, v in metric_dict.items()
        }

        self.log_dict(
            metric_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        # add "loss" metric which is required for gradient caculation
        metric_dict.update({"loss": loss})
        return metric_dict

    

    def _step(self, batch, batch_idx, type: str):
        audio, y = batch

        logits_pred = self.forward(audio)
        y_pred_prob = torch.sigmoid(logits_pred)
        y_pred = y_pred_prob >= 0.5
        loss = self.loss_function(logits_pred, y)

        return self.log_and_return_loss_step(
            loss=loss, y_pred=y_pred, y_true=y, type=type
        )
    
    
    def training_step(self,batch, batch_idx):
        return self._step(batch, batch_idx, type="train")
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="test")
