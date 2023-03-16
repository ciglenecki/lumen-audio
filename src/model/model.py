from pytorch_lightning.callbacks import ModelSummary
from torchsummary import summary

import src.config.config_defaults as config_defaults
from src.model.model_ast import ASTModelWrapper
from src.utils.utils_exceptions import UnsupportedModel
from src.utils.utils_functions import EnumStr


class SupportedModels(EnumStr):
    AST = "ast"
    EFFICIENT_NET_V2_S = "efficient_net_v2_s"
    EFFICIENT_NET_V2_M = "efficient_net_v2_m"
    EFFICIENT_NET_V2_L = "efficient_net_v2_l"
    RESNEXT50_32X4D = "resnext50_32x4d"
    RESNEXT101_32X8D = "resnext101_32x8d"
    RESNEXT101_64X4D = "resnext101_64x4d"


def get_model(args, pl_args):
    from src.model.model_torch import TORCHVISION_CONSTRUCTOR_DICT, TorchvisionModel

    model_enum = args.model
    if model_enum == SupportedModels.AST:
        model = ASTModelWrapper(
            pretrained=args.pretrained,
            lr=args.lr,
            batch_size=args.batch_size,
            scheduler_type=args.scheduler,
            epochs=args.epochs,
            warmup_lr=args.warmup_lr,
            optimizer_type=args.optimizer,
            model_name=config_defaults.DEFAULT_AST_PRETRAINED_TAG,
            num_labels=args.num_labels,
            optimization_metric=args.metric,
            weight_decay=config_defaults.DEFAULT_WEIGHT_DECAY,
            metric_mode=args.metric_mode,
            epoch_patience=args.patience,
            unfreeze_at_epoch=args.unfreeze_at_epoch,
            backbone_after=args.backbone_after,
            head_after=args.head_after,
        )
        return model
    elif model_enum in TORCHVISION_CONSTRUCTOR_DICT:
        model = TorchvisionModel(
            model_enum=model_enum,
            pretrained=args.pretrained,
            lr=args.lr,
            batch_size=args.batch_size,
            scheduler_type=args.scheduler,
            epochs=args.epochs,
            warmup_lr=args.warmup_lr,
            optimizer_type=args.optimizer,
            num_labels=args.num_labels,
            optimization_metric=args.metric,
            weight_decay=config_defaults.DEFAULT_WEIGHT_DECAY,
            metric_mode=args.metric_mode,
            early_stopping_epoch=args.patience,
            unfreeze_at_epoch=args.unfreeze_at_epoch,
            fc=args.fc,
            pretrained_weights=args.pretrained_weights,
            epoch_patience=args.patience,
            backbone_after=args.backbone_after,
            head_after=args.head_after,
        )
        return model
    raise UnsupportedModel(f"Model {model_enum} is not supported")


if __name__ == "__main__":
    pass
