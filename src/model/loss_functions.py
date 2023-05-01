import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses, miners
import src.config.config_defaults as config_defaults

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=self.pos_weight
            )

    def forward(self, inputs, targets):
        """
        Args:
            inputs: tensor of shape (N, C) representing the predicted logits
            targets: tensor of shape (N, C) representing the ground truth labels
        Returns:
            focal_loss: scalar tensor representing the computed focal loss
        """
        probs = torch.sigmoid(inputs)
        focal_weight = (1 - probs) ** self.gamma
        cross_entropy = self.bce(inputs, targets)
        focal_loss = focal_weight * cross_entropy
        return focal_loss.mean()
    

class InstrumentFamilyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss=nn.BCELoss()
        self.label_transform_matrix = get_label_transform_matrix()
        
    def forward(self,predictions,targets):
        prediction_family = self.instrument_to_family(predictions)
        target_family = self.instrument_to_family(targets)
        return self.loss(prediction_family,target_family)

    def instrument_to_family(self,instrument_labels):
        family_labels = torch.mm(self.label_transform_matrix,instrument_labels).astype(bool).astype(int)
        return family_labels



def get_label_transform_matrix():
    label_transform_matrix = torch.zeros(
        [len(config_defaults.InstrumentFamily),len(config_defaults.InstrumentEnums)]
        )
    for inst_idx,instrument in config_defaults.IDX_TO_INSTRUMENT.items():
        for family_idx,family in config_defaults.IDX_TO_FAMILY.items():
            if config_defaults.INSTRUMENT_TO_FAMILY[instrument] == family:
                label_transform_matrix[family_idx,inst_idx] = 1
            else:
                label_transform_matrix[family_idx,inst_idx] = 0
    return label_transform_matrix

