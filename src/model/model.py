import pytorch_lightning as pl
import torch

import src.config.config_defaults as config_defaults
from src.model.model_ast import ASTModelWrapper
from src.model.model_base import ModelInputDataType, SupportedModels
from src.model.model_wav2vec import Wav2VecWrapper
from src.model.model_wav2vec_cnn_only import Wav2VecCNNWrapper
from src.utils.utils_exceptions import UnsupportedModel


def get_data_input_type(model_enum: SupportedModels) -> ModelInputDataType:
    from src.model.model_torch import TORCHVISION_CONSTRUCTOR_DICT

    if model_enum == SupportedModels.AST:
        return ModelInputDataType.IMAGE
    elif model_enum in TORCHVISION_CONSTRUCTOR_DICT:
        return ModelInputDataType.IMAGE
    elif model_enum == SupportedModels.WAV2VEC:
        return ModelInputDataType.IMAGE
    elif model_enum == SupportedModels.WAV2VECCNN:
        return ModelInputDataType.WAVEFORM
    raise UnsupportedModel(
        f"Each model should have it's own ModelInputDataType. Create a new `elif` for the model {model_enum}."
    )


def get_model(
    args, pl_args, loss_function=torch.nn.modules.loss
) -> tuple[pl.LightningModule, ModelInputDataType]:
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
        log_per_instrument_metrics=args.log_per_instrument_metrics,
        freeze_train_bn=args.freeze_train_bn,
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
    elif model_enum == SupportedModels.WAV2VECCNN:
        model = Wav2VecCNNWrapper(
            model_name=config_defaults.DEFAULT_WAV2VEC_PRETRAINED_TAG,
            loss_function=loss_function,
            **model_base_kwargs,
        )
        return model
    raise UnsupportedModel(f"Model {model_enum} is not supported")


if __name__ == "__main__":
    pass
