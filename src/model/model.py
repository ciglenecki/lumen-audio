import pytorch_lightning as pl
import torch

from src.config.config_defaults import ConfigDefault
from src.enums.enums import ModelInputDataType, SupportedModels
from src.model.fluffy import FluffyConfig
from src.model.heads import get_head_constructor
from src.model.model_ast import ASTModelWrapper
from src.model.model_torch import TorchvisionModel
from src.model.model_wav2vec import Wav2VecWrapper
from src.model.model_wav2vec_cnn import Wav2VecCnnWrapper
from src.utils.utils_exceptions import UnsupportedModel


def get_data_input_type(model_enum: SupportedModels) -> ModelInputDataType:
    model_data_input_type = {
        SupportedModels.EFFICIENT_NET_V2_S: ModelInputDataType.IMAGE,
        SupportedModels.EFFICIENT_NET_V2_M: ModelInputDataType.IMAGE,
        SupportedModels.EFFICIENT_NET_V2_L: ModelInputDataType.IMAGE,
        SupportedModels.RESNEXT50_32X4D: ModelInputDataType.IMAGE,
        SupportedModels.RESNEXT101_32X8D: ModelInputDataType.IMAGE,
        SupportedModels.RESNEXT101_64X4D: ModelInputDataType.IMAGE,
        SupportedModels.AST: ModelInputDataType.IMAGE,
        SupportedModels.WAV2VEC: ModelInputDataType.WAVEFORM,
        SupportedModels.WAV2VEC_CNN: ModelInputDataType.WAVEFORM,
        SupportedModels.CONVNEXT_TINY: ModelInputDataType.IMAGE,
        SupportedModels.CONVNEXT_SMALL: ModelInputDataType.IMAGE,
        SupportedModels.CONVNEXT_LARGE: ModelInputDataType.IMAGE,
        SupportedModels.CONVNEXT_BASE: ModelInputDataType.IMAGE,
        SupportedModels.MOBILENET_V3_LARGE: ModelInputDataType.IMAGE,
    }

    if model_enum not in model_data_input_type:
        raise UnsupportedModel(
            f"Model {model_enum} doesn't exist in model_data_input_type. Please add the enum to the map."
        )
    return model_data_input_type[model_enum]


model_constructor_map = {
    SupportedModels.AST: ASTModelWrapper,
    SupportedModels.WAV2VEC: Wav2VecWrapper,
    SupportedModels.WAV2VEC_CNN: Wav2VecCnnWrapper,
    SupportedModels.EFFICIENT_NET_V2_S: TorchvisionModel,
    SupportedModels.EFFICIENT_NET_V2_M: TorchvisionModel,
    SupportedModels.EFFICIENT_NET_V2_L: TorchvisionModel,
    SupportedModels.RESNEXT50_32X4D: TorchvisionModel,
    SupportedModels.RESNEXT101_32X8D: TorchvisionModel,
    SupportedModels.RESNEXT101_64X4D: TorchvisionModel,
    SupportedModels.CONVNEXT_TINY: ModelInputDataType.IMAGE,
    SupportedModels.CONVNEXT_SMALL: ModelInputDataType.IMAGE,
    SupportedModels.CONVNEXT_LARGE: ModelInputDataType.IMAGE,
    SupportedModels.CONVNEXT_BASE: ModelInputDataType.IMAGE,
    SupportedModels.MOBILENET_V3_LARGE: ModelInputDataType.IMAGE,
}


def get_model(
    config: ConfigDefault,
    loss_function: torch.nn.modules.loss,
) -> tuple[pl.LightningModule, ModelInputDataType]:
    fluffy_config = FluffyConfig(
        use_multiple_optimizers=config.use_multiple_optimizers,
        classifer_constructor=get_head_constructor(head_enum=config.head),
    )

    model_enum = config.model
    model_kwargs = dict(
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
        early_stopping_metric_patience=config.early_stopping_metric_patience,
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
        config=config,
    )

    if model_enum not in model_constructor_map:
        raise UnsupportedModel(
            f"Model {model_enum} is not in the model_constructor_map. Add the model enum to the model_constructor_map."
        )

    model_constructor = model_constructor_map[model_enum]

    # Want to change/add new kwargs for your specific model?
    # if model_enum == SupportedModels.EFFICIENT_NET_V2_L:
    #     model_kwargs.update({"arg_i_want_to_change": "some new value"})

    model = model_constructor(**model_kwargs)
    return model


if __name__ == "__main__":
    pass
