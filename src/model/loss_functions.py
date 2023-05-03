import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses, miners
import src.config.config_defaults as config_defaults

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='none', pos_weight=None):
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
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "none":
            return focal_loss
    
class InstrumentFamilyLoss(nn.Module):
    def __init__(self,add_instrument_loss) -> None:
        super().__init__()
        self.family_loss = nn.BCEWithLogitsLoss()
        self.family_indices = nn.Parameter(get_family_indices(),requires_grad=False)
        self.num_families = len(config_defaults.InstrumentFamily)

    def forward(self,prediction_logits,targets):
        prediction_family = self.instrument_to_family(prediction_logits)
        target_family = self.instrument_to_family(targets)
        fam_loss = self.family_loss(prediction_family,target_family)
        return fam_loss
    
    def instrument_to_family(self,instrument_labels):
        fam_labels = []
        for i in range(self.num_families):
            fam_labels.append(instrument_labels[:,self.family_indices[i]].max(dim=1).values.unsqueeze(1))
        fam_labels=torch.cat(fam_labels,dim=1)
        return fam_labels

def get_family_indices():
    family_indices = [torch.zeros(len(config_defaults.InstrumentEnums)) for _ in config_defaults.InstrumentFamily]
    for inst in config_defaults.InstrumentEnums:
        inst_idx = config_defaults.INSTRUMENT_TO_IDX[inst.value]
        fam_idx = config_defaults.FAMILIY_TO_IDX[
        config_defaults.INSTRUMENT_TO_FAMILY[inst.value]
        ]
        family_indices[fam_idx][inst_idx] = 1

    family_indices=torch.stack(family_indices).bool()
    return torch.tensor(family_indices)


