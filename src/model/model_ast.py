import librosa
import torch
import torchaudio
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

import src.config.config_defaults as config_defaults
from src.config.argparse_with_config import ArgParseWithConfig
from src.enums.enums import SupportedHeads, SupportedModels
from src.model.heads import get_head_constructor
from src.model.model_base import ModelBase
from src.utils.utils_audio import load_audio_from_file, play_audio
from src.utils.utils_dataset import get_example_val_sample


class ASTModelWrapper(ModelBase):
    """Audio Spectrogram Transformer model."""

    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # ast_config = ASTConfig.from_pretrained(
        #     pretrained_model_name_or_path=self.pretrained_tag,
        #     id2label=config_defaults.IDX_TO_INSTRUMENT,
        #     label2id=config_defaults.INSTRUMENT_TO_IDX,
        #     num_labels=self.num_labels,
        #     finetuning_task="audio-classification",
        #     problem_type="multi_label_classification",
        # )

        # self.backbone: ASTForAudioClassification = (
        #     ASTForAudioClassification.from_pretrained(
        #         self.pretrained_tag,
        #         config=ast_config,
        #         ignore_mismatched_sizes=True,
        #     )
        # )

        # ast_config = ASTConfig.from_pretrained(
        #     pretrained_model_name_or_path=self.pretrained_tag,
        #     finetuning_task="audio-classification",
        #     problem_type="multi_label_classification",
        # )

        # self.backbone: ASTForAudioClassification = (
        #     ASTForAudioClassification.from_pretrained(
        #         self.pretrained_tag,
        #         config=ast_config,
        #     )
        # )
        # self.subclassifier = torch.nn.Linear(
        #     ast_config.num_labels,
        #     config_defaults.DEFAULT_NUM_LABELS,
        # )
        # new_weights = torch.zeros(self.subclassifier.weight.shape)
        # new_bias = torch.zeros(self.subclassifier.bias.shape)

        # for irmas_idx, ast_idx in enumerate(
        #     config_defaults.AST_INSTRUMENTS_IRMAS.keys()
        # ):
        #     new_weights[irmas_idx, ast_idx] = 1.0
        # with torch.no_grad():
        #     self.subclassifier.weight.copy_(new_weights)
        #     self.subclassifier.bias.copy_(new_bias)

        ast_config: ASTConfig = ASTConfig.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_tag,
            finetuning_task="audio-classification",
            problem_type="multi_label_classification",
        )

        self.backbone: ASTForAudioClassification = (
            ASTForAudioClassification.from_pretrained(
                self.pretrained_tag,
                config=ast_config,
            )
        )
        new_classifier = torch.nn.Linear(
            ast_config.hidden_size,
            config_defaults.DEFAULT_NUM_LABELS,
        )
        ast_indices = list(config_defaults.AST_INSTRUMENTS_IRMAS.keys())
        with torch.no_grad():
            new_classifier.weight.copy_(
                self.backbone.classifier.dense.weight[ast_indices, :]
            )
            new_classifier.bias.copy_(self.backbone.classifier.dense.bias[ast_indices])
        # with torch.no_grad():
        #     self.classifier.dense.weight.copy_(new_weights)
        #     self.classifier.bias.copy_(new_bias)

        self.backbone.classifier = new_classifier
        self.save_hyperparameters()

    def forward(self, audio: torch.Tensor):
        out = self.backbone.forward(
            audio,
            output_attentions=True,
            return_dict=True,
        )
        return out.logits

    # def forward(self, image: torch.Tensor):
    #     out: BaseModelOutputWithPooling = self.backbone.audio_spectrogram_transformer(
    #         image,
    #         output_attentions=True,
    #         return_dict=True,
    #     )
    #     out = self.backbone.classifier(out.pooler_output)
    #     logits = self.subclassifier(out)
    #     return logits


if __name__ == "__main__":
    parser = ArgParseWithConfig()
    args, config, pl_args = parser.parse_args()
    audio = get_example_val_sample(config.sampling_rate)
    head_constructor = get_head_constructor(SupportedHeads.DEEP_HEAD)
    ast_our = ASTModelWrapper(
        pretrained=config.pretrained,
        pretrained_tag=config_defaults.TAG_AST_AUDIOSET,
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
        finetune_train_bn=config.finetune_train_bn,
        model_enum=SupportedModels.AST,
        loss_function=torch.nn.BCEWithLogitsLoss(reduction="none"),
        head_constructor=head_constructor,
        use_fluffy=config.use_fluffy,
        config=config,
        head_hidden_dim=config.head_hidden_dim,
        add_instrument_loss=config.add_instrument_loss,
    )
    # trainer = Trainer()
    # trainer.strategy.connect(ast_our)
    # trainer.save_checkpoint("models/ast_our.pt")

    # example_audio_mel_audio()
    config_ = ASTConfig.from_pretrained(
        pretrained_model_name_or_path=config_defaults.TAG_AST_AUDIOSET,
        id2label=config_defaults.IDX_TO_INSTRUMENT,
        label2id=config_defaults.INSTRUMENT_TO_IDX,
        num_labels=config_defaults.DEFAULT_NUM_LABELS,
        finetuning_task="audio-classification",
        problem_type="multi_label_classification",
    )

    backbone: ASTForAudioClassification = ASTForAudioClassification.from_pretrained(
        config_defaults.TAG_AST_AUDIOSET,
        config=config_,
        ignore_mismatched_sizes=True,
    )
    target_sr = 16_000
    n_fft = 400
    hop = 160
    mel_bins = 128
    fname = "data/irmas_sample/1 - Hank's Other Bag-1.wav"

    feature_extractor = ASTFeatureExtractor.from_pretrained(
        config_defaults.TAG_AST_AUDIOSET,
        normalize=False,
    )

    audio_torch, org_sample_rate = torchaudio.load(fname)
    resampler = torchaudio.transforms.Resample(
        org_sample_rate, target_sr, dtype=audio_torch.dtype
    )
    audio_torch = resampler(audio_torch)
    audio_torch = audio_torch.mean(dim=0, keepdim=False)

    audio_lib, org_sample_rate = load_audio_from_file(
        fname, target_sr=target_sr, method="librosa"
    )
    audio_lib = librosa.resample(
        y=audio_lib, orig_sr=org_sample_rate, target_sr=target_sr
    )

    features_torch = feature_extractor(
        audio_torch,
        sampling_rate=target_sr,
        return_tensors="np",
    )["input_values"]

    features_librosa = feature_extractor(
        audio_lib,
        sampling_rate=target_sr,
        return_tensors="np",
    )["input_values"]

    torch_spec = features_torch
    lib_spec = features_librosa

    torch_spec = torch_spec.squeeze(0).T
    lib_spec = lib_spec.squeeze(0).T
    inv = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, sample_rate=target_sr
    )
    grif = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop, n_iter=54, power=2
    )
    inv_torch = grif(inv(torch.tensor(torch_spec)))
    inv_lib = grif(inv(torch.tensor(lib_spec)))

    torch_reconstructed = librosa.feature.inverse.mel_to_audio(
        torch_spec, sr=target_sr, n_fft=n_fft, hop_length=hop
    )
    lib_reconstruct = librosa.feature.inverse.mel_to_audio(
        lib_spec, sr=target_sr, n_fft=n_fft, hop_length=hop
    )

    print(librosa.get_duration(y=lib_reconstruct, sr=16_000))
    print(len(lib_reconstruct) / 16_000)

    # play_audio(torch_reconstructed, sr=target_sr, max_seconds=3)
    # play_audio(lib_reconstruct, sr=target_sr, max_seconds=3)
    play_audio(inv_torch.squeeze(0).numpy(), sampling_rate=target_sr, max_seconds=3)
    play_audio(inv_lib.squeeze(0).numpy(), sampling_rate=target_sr, max_seconds=3)
