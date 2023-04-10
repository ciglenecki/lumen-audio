import pytorch_lightning as pl
import torch

from src.enums.enums import ModelInputDataType, SupportedModels
from src.model.fluffy import FluffyConfig
from src.model.heads import get_head_constructor
from src.model.model_ast import ASTModelWrapper
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
    config, pl_args, loss_function=torch.nn.modules.loss
) -> tuple[pl.LightningModule, ModelInputDataType]:
    from src.model.model_torch import TORCHVISION_CONSTRUCTOR_DICT, TorchvisionModel

    fluffy_config = FluffyConfig(
        use_multiple_optimizers=config.use_multiple_optimizers,
        classifer_constructor=get_head_constructor(head_enum=config.head),
    )

    model_enum = config.model
    model_base_kwargs = dict(
        pretrained=config.pretrained,
        pretrained_tag=config.pretrained_tag,
        lr=config.lr,
        batch_size=config.batch_size,
        scheduler_type=config.scheduler,
        epochs=config.epochs,
        lr_warmup=config.lr_warmup,
        optimizer_type=config.optimizer,
        num_labels=config.num_labels,
        optimization_metric=config.metric,
        weight_decay=config.weight_decay,
        metric_mode=config.metric_mode,
        plateau_epoch_patience=config.plateau_epoch_patience,
        finetune_head_epochs=config.finetune_head_epochs,
        finetune_head=config.finetune_head,
        backbone_after=config.backbone_after,
        head_after=config.head_after,
        lr_onecycle_max=config.lr_onecycle_max,
        log_per_instrument_metrics=config.log_per_instrument_metrics,
        freeze_train_bn=config.freeze_train_bn,
        model_enum=model_enum,
        loss_function=loss_function,
        fluffy_config=fluffy_config,
        use_fluffy=config.use_fluffy,
    )

    if model_enum == SupportedModels.AST:
        model = ASTModelWrapper(
            **model_base_kwargs,
        )
        return model
    elif model_enum in TORCHVISION_CONSTRUCTOR_DICT:
        model = TorchvisionModel(
            **model_base_kwargs,
        )
        return model
    elif model_enum == SupportedModels.WAV2VEC:
        model = Wav2VecWrapper(
            **model_base_kwargs,
        )
        return model
    elif model_enum == SupportedModels.WAV2VECCNN:
        model = Wav2VecCNNWrapper(
            **model_base_kwargs,
        )
        return model
    raise UnsupportedModel(f"Model {model_enum} is not supported")


if __name__ == "__main__":
    pass
