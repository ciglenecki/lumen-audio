import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.model.model_spectral_conv import SpecConv
from src.model.attention import AttentionLayer
from src.train.metrics import get_metrics
from src.model.model_base import ModelBase


class CRNN_Attention(ModelBase):
    def __init__(self,
                convolution_base = None,
                lstm_input_dim = 128,
                lstm_hidden_dim = 64,
                lstm_num_layers = 2,
                lstm_bidirectional=False,
                *args,
                **kwargs,
                 ):
        super(*args,**kwargs).__init__()
        
        # Model:
        if convolution_base is None:
            self.convolution = SpecConv(dimensions=[1,32,64,128],dropout_p=0.2)
        else:
            self.convoluton_base = convolution_base
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size = lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional = lstm_bidirectional,
            batch_first = True,
        )
        self.attn = AttentionLayer(input_size=lstm_hidden_dim)
        self.linear = nn.Linear(lstm_hidden_dim,self.num_labels)


    def load_convolution_from_pretrained(self,conv_state_dict):
        self.convolution.load_state_dict(conv_state_dict)


    def forward(self, x):
        features = self.convolution(x)
        # N C, T, F  -> N, T, C, F
        features = features.permute(0, 2, 1, 3)  # change from NCHW to NHWC for LSTM
        #batch_size, seq_len, num_channels, height = features.shape
        features = torch.mean(features,dim = -1,keepdim=False)
        #features = features.reshape(batch_size, seq_len, -1) 
        output, _ = self.lstm(features)
        output, att_weights = self.attn(output)
        output = self.linear(output)
        return output,att_weights
    

    def _step(self, batch, batch_idx, type: str):
        audio, y = batch

        logits_pred,attention_weights = self.forward(audio)
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

    def predict_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="predict")
