import src.config.config_defaults as config_defaults
from src.model.model_ast import ASTModelWrapper
from src.model.model_wav2vec import Wav2VecWrapper
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
    WAV2VEC = "wav2vec"


def get_model(args, pl_args):
    from src.model.model_torch import TORCHVISION_CONSTRUCTOR_DICT, TorchvisionModel

    model_enum = args.model
    model_base_kwargs = dict(
        pretrained=args.pretrained,
        lr=args.lr,
        batch_size=args.batch_size,
        scheduler_type=args.scheduler,
        epochs=args.epochs,
        lr_warmup=args.lr_warmup,
        optimizer_type=args.optimizer,
        num_labels=args.num_labels,
        optimization_metric=args.metric,
        weight_decay=config_defaults.DEFAULT_WEIGHT_DECAY,
        metric_mode=args.metric_mode,
        plateau_epoch_patience=args.patience,
        unfreeze_at_epoch=args.unfreeze_at_epoch,
        backbone_after=args.backbone_after,
        head_after=args.head_after,
        lr_onecycle_max=args.lr_onecycle_max,
    )
    if model_enum == SupportedModels.AST:
        model = ASTModelWrapper(
            model_name=config_defaults.DEFAULT_AST_PRETRAINED_TAG,
            **model_base_kwargs,
        )
        return model
    elif model_enum in TORCHVISION_CONSTRUCTOR_DICT:
        model = TorchvisionModel(
            model_enum=model_enum,
            fc=args.fc,
            pretrained_weights=args.pretrained_weights,
            **model_base_kwargs,
        )
        return model
    elif model_enum == SupportedModels.WAV2VEC:
        model = Wav2VecWrapper(
            model_name=config_defaults.DEFAULT_WAV2VEC_PRETRAINED_TAG,
            **model_base_kwargs,
        )
        return model
    raise UnsupportedModel(f"Model {model_enum} is not supported")


if __name__ == "__main__":
    pass
