import operator

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import config_defaults
from src.data.datamodule import IRMASDataModule
from src.enums.enums import SupportedAugmentations
from src.features.audio_transform import get_audio_transform
from src.features.chunking import get_collate_fn
from src.grad_vis.parser import parse
from src.model.model import get_model


class MultiLabelBinaryClassifierOutputTarget:
    def __init__(self, output_index):
        self.output_index = output_index

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            model_output = model_output.unsqueeze(0)
        return -model_output[:, self.output_index]


if __name__ == "__main__":
    args, config = parse()
    model_type = get_model(config, torch.nn.BCEWithLogitsLoss())
    model = model_type.load_from_checkpoint(args.path_to_model)
    model.eval()

    target_module = [operator.attrgetter(args.target_layer)(model)]
    cam = GradCAM(model=model, target_layers=target_module, use_cuda=args.device)

    datamodule = IRMASDataModule(
        train_dirs=config.train_dirs,
        val_dirs=config.val_dirs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=config.drop_last,
        train_audio_transform=get_audio_transform(config, config.audio_transform),
        val_audio_transform=get_audio_transform(config, config.audio_transform),
        collate_fn=get_collate_fn(config),
        normalize_audio=config.normalize_audio,
        train_only_dataset=config.train_only_dataset,
        concat_n_samples=(
            config.aug_kwargs["concat_n_samples"]
            if SupportedAugmentations.CONCAT_N_SAMPLES in config.augmentations
            else None
        ),
        sum_two_samples=SupportedAugmentations.SUM_TWO_SAMPLES in config.augmentations,
        use_weighted_train_sampler=config.use_weighted_train_sampler,
    )

    test_dataloader = datamodule.test_dataloader()
    for inputs, labels, ids, _ in test_dataloader:
        instrument_idx = config_defaults.INSTRUMENT_TO_IDX[args.label]
        targets = [MultiLabelBinaryClassifierOutputTarget(instrument_idx)]
        res = cam(input_tensor=inputs, targets=targets)

        unique_ids = torch.unique(ids)
        grouped_tensors = []
        for id in unique_ids:
            indices = torch.where(ids == id)

            grouped_example = torch.cat([torch.tensor(i) for i in res[indices]], dim=-1)

            grouped_example_rgb = torch.cat(
                [torch.tensor(i) for i in inputs[indices]], dim=-1
            )

            grouped_example = (grouped_example - grouped_example.min()) / (
                grouped_example.max() - grouped_example.min()
            )
            grouped_example_rgb = (grouped_example_rgb - grouped_example_rgb.min()) / (
                grouped_example_rgb.max() - grouped_example_rgb.min()
            )

            final_img = visualization = show_cam_on_image(
                grouped_example_rgb.detach().cpu().permute(1, 2, 0).numpy(),
                grouped_example.detach().cpu().numpy(),
            )

            cv2.imshow("img", final_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
