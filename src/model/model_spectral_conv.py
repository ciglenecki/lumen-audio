import torch
import torch.nn as nn
from src.model.blocks import BnActConv2d


class SpecConv(nn.Module):
    def __init__(self,
                 dimensions = [1,32,64,128],
                activation=nn.LeakyReLU(),
                kernel_size = 5,
                num_pools = 3,
                dropout_p = 0.0
            
                 ):
        super().__init__()

        self.output_size = dimensions[-1]
        self.activation = activation
        layers = self.get_layers(
            dimensions=dimensions,
            kernel_size=kernel_size,
            activation = self.activation,
            num_pools=num_pools,
            dropout_p=dropout_p
            )
        self.convolutions = nn.Sequential(*layers)
        
    def forward(self,melspec):
        #print(melspec.shape)
        features=torch.unsqueeze(melspec,1)
        #print(features.shape)
        features = self.convolutions(features)
        return features



    def get_layers(self,dimensions,kernel_size,activation,dropout_p=0.0,num_pools=3):
        layers = []
        for i in range(len(dimensions)-1):
            layers.append(
                BnActConv2d(
                in_features=dimensions[i],
                out_features=dimensions[i+1],
                kernel_size=kernel_size,
                stride = 1,
                activation=activation,
                padding=2,
                switch=True if i==0 else False
                )
            )
            if i<num_pools:
                layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,4)))
            if dropout_p != 0.0:
                layers.append(nn.Dropout(p=dropout_p))
        return layers


