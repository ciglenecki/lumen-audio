from pytorch_lightning.callbacks import ModelSummary
from torchsummary import summary

import src.config.config_defaults as config_defaults
from src.model.model_ast import ASTModelWrapper
from src.model.model_effcientnetv2 import EfficientNetV2SmallModel
from src.utils.utils_exceptions import UnsupportedModel
from src.utils.utils_train import SupportedModels


def get_model(args, pl_args):
    model_enum = args.model
    if model_enum == SupportedModels.AST and args.pretrained:
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
    elif model_enum == SupportedModels.EFFICIENT_NET_V2_S and args.pretrained:
        model = EfficientNetV2SmallModel(
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
            epoch_patience=args.patience,
            unfreeze_at_epoch=args.unfreeze_at_epoch,
            backbone_after=args.backbone_after,
            head_after=args.head_after,
        )
        return model
    raise UnsupportedModel(f"Model {model_enum.value} is not supported")


if __name__ == "__main__":
    # python3 -m src.train --accelerator gpu --devices -1 --dataset-dir data/raw/train --audio-transform mel_spectrogram --model efficient_net_v2_s
    model = ASTModelWrapper()
    summary(
        model,
    )
    ModelSummary(model, max_depth=-1)
