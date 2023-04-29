import torch
import torch.nn as nn
from pytorch_metric_learning import losses,miners
import pytorch_lightning as pl


from  src.model.blocks import BnActConv2d
from src.train.metrics import get_metrics

class ArcFaceModel(pl.LightningModule):
    def __init__(self,
                 embdedding_model:nn.Module|pl.LightningModule,
                 num_classes:int,
                 embedding_space_size:int
                 ) -> None:
        super().__init__()
        self.embedding_model = embdedding_model
        self.loss_function = losses.ArcFaceLoss(
            embedding_size=embedding_space_size,num_classes=num_classes,
            scale=10,margin=32
        )

    def forward(self,x,labels):
        embeddings = self.embedding_model(x)
        loss = self.loss_function(embeddings,labels)
        #Ova operacija se izvodi dva put ne znam kak to zaobic
        #Jeeeebiga . . .
        logits = torch.mm(embeddings,self.loss_function.W)
        return logits,loss
    

    def _step(self, batch, batch_idx, type: str):
        audio, y = batch
        y = torch.nonzero(y)[:,1]
        loss = self.forward(audio,y)
        return self.log_and_return_loss_step(
            loss=loss, y_pred=y_pred, y_true=y, type=type
        )

    
    def training_step(self,batch, batch_idx):
        return self._step(batch, batch_idx, type="train")
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="test")

    def predict_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="predict")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr = 1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,gamma = 1-1e-2)
        return [optimizer],[scheduler]




