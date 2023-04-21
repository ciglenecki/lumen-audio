"""python3 src/grad_vis/visualize_grads.py --input-dirs irmastrain:data/irmas/train --path_to_model
models_quick/04-14-15-25-32_CalmAlan_resnext50_32x4d/checkpoints/04-14-15-25-
32_CalmAlan_resnext50_32x4d_val_acc_0.0000_val_loss_1.1923.ckpt --target_layer backbone.avgpool.

--model RESNEXT50_32X4D --audio-transform MEL_SPECTROGRAM --label vio --batch-size 1 --image-size
256 256.
"""
import operator

import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import config_defaults
from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import parse_dataset_paths
from src.data.datamodule import IRMASDataModule
from src.features.audio_transform import get_audio_transform
from src.features.chunking import get_collate_fn
from src.model.model import get_model


def parse_args():
    parser = ArgParseWithConfig()
    parser.add_argument(
        "--input-dirs",
        type=parse_dataset_paths,
        required=True,
        help="Directories which will be used to render visualize_grads.",
    )
    parser.add_argument(
        "--path_to_model",
        type=str,
        required=True,
        help="Path to a trained model.",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        required=True,
        help="Full name of the layer used for tracking gradients (if not sure, use the final feature map/embedding)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="The device to be used eg. cuda:0.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="The label to track the gradients for. If None, tracked for all",
    )

    args, config, pl_args = parser.parse_args()
    config.test_paths = args.input_dirs
    config.required_model()
    config.required_audio_transform()
    config.required_test_paths()
    return args, config, pl_args


class MultiLabelBinaryClassifierOutputTarget:
    def __init__(self, output_index):
        self.output_index = output_index

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            model_output = model_output.unsqueeze(0)
        return -model_output[:, self.output_index]


if __name__ == "__main__":
    args, config, pl_args = parse_args()
    model_type = get_model(config, torch.nn.BCEWithLogitsLoss())
    model = model_type.load_from_checkpoint(args.path_to_model)
    model.eval()

    target_module = [operator.attrgetter(args.target_layer)(model)]
    cam = GradCAM(model=model, target_layers=target_module, use_cuda=args.device)

    audio_transform = get_audio_transform(
        config, spectrogram_augmentation=None, waveform_augmentation=None
    )
    datamodule = IRMASDataModule(
        train_paths=config.train_paths,
        val_paths=config.val_paths,
        test_paths=config.test_paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=config.drop_last,
        train_audio_transform=audio_transform,
        val_audio_transform=audio_transform,
        collate_fn=get_collate_fn(config),
        normalize_audio=config.normalize_audio,
        normalize_image=config.normalize_image,
        train_only_dataset=False,
        concat_n_samples=None,
        sum_two_samples=False,
        use_weighted_train_sampler=config.use_weighted_train_sampler,
    )
    datamodule.setup("test")

    test_dataloader = datamodule.test_dataloader()
    for inputs, labels, ids, _ in test_dataloader:
        instrument_idx = config_defaults.INSTRUMENT_TO_IDX[args.label]
        targets = [MultiLabelBinaryClassifierOutputTarget(instrument_idx)]
        res = cam(input_tensor=inputs, targets=targets)

        unique_ids = torch.unique(ids)
        grouped_tensors = []
        for idx in unique_ids:
            indices = torch.where(ids == idx)

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
