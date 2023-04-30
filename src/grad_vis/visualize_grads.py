"""python3 src/grad_vis/visualize_grads.py --input-dirs irmastrain:data/irmas/train --path_to_model
models_quick/04-14-15-25-32_CalmAlan_resnext50_32x4d/checkpoints/04-14-15-25-
32_CalmAlan_resnext50_32x4d_val_acc_0.0000_val_loss_1.1923.ckpt --target_layer backbone.avgpool.

--model RESNEXT50_32X4D --audio-transform MEL_SPECTROGRAM --label vio --batch-size 1 --image-size
256 256.
"""
import operator

import cv2
import librosa
import librosa.display
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import config_defaults
from src.config.argparse_with_config import ArgParseWithConfig
from src.data.datamodule import OurDataModule
from src.features.audio_to_spectrogram import MelSpectrogram
from src.features.audio_transform import get_audio_transform
from src.features.chunking import collate_fn_feature
from src.model.model import get_model
from src.utils.utils_audio import plot_spectrograms
from src.utils.utils_dataset import add_rgb_channel, concat_images
from utils.utils_functions import min_max_scale


def parse_args():
    parser = ArgParseWithConfig()

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
    config.required_model()
    config.required_audio_transform()
    config.required_dataset_paths()

    return args, config, pl_args


def img_to_rgb():
    # Get current figure
    fig = plt.gcf()
    fig.axes[0].get_xaxis().set_visible(False)
    fig.axes[0].get_yaxis().set_visible(False)
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    cb = plt.colorbar()  # remove colorbar
    cb.remove()
    plt.axis("off")  # remove axis
    plt.savefig("wtf.png")
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(h, w, 3)
    return img_array


class MultiLabelBinaryClassifierOutputTarget:
    def __init__(self, output_index):
        self.output_index = output_index

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            model_output = model_output.unsqueeze(0)
        return (-1) * model_output[:, self.output_index]


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
    datamodule = OurDataModule(
        train_paths=None,
        val_paths=None,
        test_paths=config.dataset_paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=config.drop_last,
        train_audio_transform=None,
        val_audio_transform=audio_transform,
        collate_fn=collate_fn_feature,
        normalize_audio=config.normalize_audio,
        normalize_image=config.normalize_image,
        train_only_dataset=False,
        concat_n_samples=None,
        sum_two_samples=False,
        use_weighted_train_sampler=config.use_weighted_train_sampler,
        sampling_rate=config.sampling_rate,
    )
    datamodule.setup_for_inference()

    test_dataloader = datamodule.test_dataloader()
    for images, labels, file_indices, _ in test_dataloader:
        # images = torchvision.transforms.Resize(size=(384, 384))(
        #     torchvision.transforms.ToTensor()(Image.open("cat.jpg")).unsqueeze_(0)
        # )
        instrument_idx = config_defaults.INSTRUMENT_TO_IDX[args.label]
        targets = [MultiLabelBinaryClassifierOutputTarget(instrument_idx)]
        grads_mask = cam(input_tensor=images, targets=targets)

        if len(grads_mask.shape) == 2:
            regrads_masks = grads_mask[np.newaxis, ...]

        unique_indices = torch.unique(file_indices)
        grouped_tensors = []

        for file_idx in unique_indices:
            batch_file_indices = torch.where(file_indices == file_idx)
            image_grouped = images[batch_file_indices]
            grads_mask_grouped = grads_mask[batch_file_indices]

            grads_mask_catted = concat_images(grads_mask_grouped)
            image_catted = concat_images(image_grouped)

            image_catted = audio_transform.undo(image_catted.unsqueeze(dim=0))[0]

            my_dpi = 150
            width = 1200
            height = 600

            plt.figure(figsize=(width / my_dpi, height / my_dpi), dpi=my_dpi)
            S_db = librosa.power_to_db(image_catted, ref=np.max)
            librosa.display.specshow(
                S_db,
                y_axis="mel",
                x_axis=None,
                sr=config.sampling_rate,
                hop_length=config.hop_length,
                n_fft=config.n_fft,
            )
            image_catted = img_to_rgb() / 255

            grads_width = grads_mask_catted.shape[-1]
            grads_mask_catted = (
                torch.nn.functional.interpolate(
                    grads_mask_catted.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

            grads_mask_catted = min_max_scale(grads_mask_catted)
            image_catted = cv2.cvtColor(
                image_catted.astype("float32"), cv2.COLOR_BGR2RGB
            )
            final_img = show_cam_on_image(
                image_catted, grads_mask_catted, use_rgb=True, image_weight=0.5
            )
            # plt.imshow(final_img)
            cv2.imshow("img", final_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
