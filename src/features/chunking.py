from functools import partial
from typing import Callable

import torch
import torch.nn.functional
import torchvision.transforms.functional

from src.config.config_defaults import NUM_RGB_CHANNELS, ConfigDefault
from src.enums.enums import ModelInputDataType
from src.model.model import get_data_input_type


def create_and_repeat_channel(images: torch.Tensor, num_repeat: int):
    # Create new dimension then repeat along it.
    return images.unsqueeze(dim=1).repeat(1, num_repeat, 1, 1)


def add_rgb_channel(images: torch.Tensor):
    return create_and_repeat_channel(images, NUM_RGB_CHANNELS)


def remove_rgb_channel(images: torch.Tensor):
    # Pick only one channel out of 3..
    return images[:, 0, :, :]


def chunk_image_by_width(target_image_size: tuple[int, int], image: torch.Tensor):
    """Target image size: (384, 384)

    Image:
    (128 x 1024)
    |========================================|
    |                                        |
    |========================================|

    Step 1: scale height only
    (384 x 1024)
    |========================================|
    |                                        |
    |                                        |
    |========================================|

    Step 2: split image
    |==============|==============|==========|
    |  (384, 384)  |  (384, 384)  |(384, 256)|
    |              |              |          |
    |==============|==============|==========|

    Step 3: pad with zeros on the right
    |==============|==============|==============|
    |  (384, 384)  |  (384, 384)  |  (384, 384)0 |
    |              |              |            0 |
    |==============|==============|==============|

    Returns 3x (384, 384)
    """
    image_height, image_width = target_image_size
    # Simulate that this is a batch of images
    # [Batch, height, width] = [1, 128, 1024]
    image = image.unsqueeze(0)

    # Scale only the height (freqs) and don't touch the width (time) because the `time` will get chunked.
    full_width = image.shape[-1]
    pre_resize_height = image_height  # change the height!
    pre_resize_width = full_width  # don't change the width!

    # [1, 384, 2048]
    # [Batch, height, width]
    image = torchvision.transforms.functional.resize(
        image, size=(pre_resize_height, pre_resize_width), antialias=False
    )

    # Chunk by last dimension (width)
    # list([1, 384, 384]), length = 5
    # list([batch, height, width]), length = full_width // image_width
    chunks = list(torch.tensor(image).split(image_width, dim=-1))

    # Last chunk might be cut off which means the time dimension (image width) will be smaller
    # Add zero padding if last chunk's width is shorter than image width
    last_chunk = chunks[-1]  # [Batch]
    last_chunk_width = last_chunk.shape[-1]
    diff = image_width - last_chunk_width  # e.g. 384 - 200 = 184
    if diff > 0:
        # we add 0 pads on the left, and diff on the right side, pad=(padding_left,padding_right)
        chunks[-1] = torch.nn.functional.pad(
            input=chunks[-1], pad=(0, diff), mode="constant", value=0
        )

    if len(chunks) == 1:
        return chunks[0].float()

    # list([1, 384, 384], [1, 384, 384], ...) -> [5, 384, 384]
    # list([Batch, height, width]) -> ([Batch + chunks, height, width])
    chunks = torch.stack(chunks, dim=1).squeeze(0).float()
    chunks = chunks.float()
    return chunks


def undo_image_chunking(spectrogram: torch.Tensor, n_mel_bins: int) -> torch.Tensor:
    spectrogram = torch.cat(tuple(spectrogram), dim=-1).unsqueeze(0)
    spectrogram = torchvision.transforms.functional.resize(
        spectrogram, size=(n_mel_bins, spectrogram.shape[-1]), antialias=False
    )
    return spectrogram


def collate_fn_spectrogram(
    examples: list[tuple[torch.Tensor], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Example:
    Image shape: [Chunks, Channel, Height, Width]
    Label shape: [Label]
    Dataset index: [1]

    Batch is tuple (image, label, dataset index)
    batch = [
        [[3,4], [5,6]], [1,0,0], 851
        [[2,3], [1,7]], [0,0,1], 532
        [[9,3], [2,1]], [0,1,0], 91
    ]
    """

    # Count total number of images (sum of all chunks)
    num_images = 0
    for e in examples:
        image_chunks = e[0]
        num_images += image_chunks.shape[0]

    example_item = examples[0]
    example_image = example_item[0]
    example_label = example_item[1]
    image_shape = tuple(example_image.shape[1:])

    # Create empty matrices
    images = torch.empty((num_images, *image_shape))
    labels = torch.empty((num_images, example_label.shape[-1]))
    file_indices = torch.empty(num_images, dtype=torch.int64)
    item_indices = torch.empty(num_images, dtype=torch.int64)

    images_passed = 0
    for unique_file_idx, item in enumerate(examples):
        image_chunks, label, dataset_index = item
        num_chunks = image_chunks.shape[0]

        start = images_passed
        end = images_passed + num_chunks

        images[start:end] = image_chunks
        labels[start:end] = label
        file_indices[start:end] = torch.full((num_chunks,), unique_file_idx)
        item_indices[start:end] = torch.full((num_chunks,), int(dataset_index))

        images_passed += num_chunks

    return images, labels, file_indices, item_indices


def collate_fn_audio(
    batch: list[tuple[torch.Tensor, torch.Tensor]], max_num_width_samples
):
    """Batch is tuple (waveform, label)

    Example:

        batch size = 3
        max_size = 2

        batch = [
            [5, 6, 1, 2, 3], [1,0,0]
            [1, 5, 3, 8, 9], [0,0,1]
            [2, 5, 3, 1, 7], [0,1,0]
        ]

        output_audio_batch = [
            [5, 6],
            [1, 2],
            [3, 0],
              ....,
            [3, 1],
            [7, 0],
        ]

        output_lables_batch = [
            [1,0,0],
            [1,0,0],
            [1,0,0],
            .......,
            [0,1,0],
            [0,1,0]
        ]

        output_labels = [
            0,
            0,
            0,
            ...,
            2,
            2
        ]
        output = output_image_batch, output_lables_batch, output_labels
    """
    audio_chunks, labels, file_indices = [], [], []
    for file_idx, (audio, label) in enumerate(batch):
        chunks = list(torch.split(audio, max_num_width_samples))
        audio_chunks.extend(chunks)  # [[5,6], [1,2], [3]]
        repeat_times = len(chunks)

        repeated_labels = label.repeat([repeat_times, 1])
        labels.extend(repeated_labels)  # [[1,0,0], [1,0,0], [1,0,0]]

        file_id_repeated = torch.full([repeat_times], file_idx)
        file_indices.extend(file_id_repeated)  # [0, 0, 0]

    audio_chunks = torch.nn.utils.rnn.pad_sequence(
        audio_chunks,
        batch_first=True,
        padding_value=0.0,
    )  # Shape [chunks, width, height]

    labels = torch.vstack(labels)  # Shape [audio_chunks, labels]
    file_indices = torch.vstack(file_indices)  # Shape [audio_chunks, 1]

    return audio_chunks, labels, file_indices


def get_collate_fn(config: ConfigDefault) -> Callable:
    data_input_type = get_data_input_type(model_enum=config.model)
    if data_input_type == ModelInputDataType.IMAGE:
        return collate_fn_spectrogram
    elif data_input_type == ModelInputDataType.WAVEFORM:
        return partial(
            collate_fn_audio,
            max_num_width_samples=config.max_num_width_samples * config.sampling_rate,
        )
    else:
        raise Exception(f"Unsupported data input type {data_input_type}")


####################################################################################
####################################################################################
# TEST
####################################################################################
####################################################################################


def test_collate_fn_spectrogram():
    num_chunks = 3
    image_width = 384
    image_height = 384

    spectrogram_1 = torch.rand(num_chunks, image_width, image_height)
    spectrogram_2 = torch.rand(num_chunks, image_width, image_height)
    spectrogram_3 = torch.rand(num_chunks, image_width, image_height)
    label_1 = torch.zeros(11)
    label_1[1] = 1
    label_2 = torch.zeros(11)
    label_2[5] = 1
    label_3 = torch.zeros(11)
    label_2[5:6] = 1

    dataset_id_1 = 532
    dataset_id_2 = 591
    dataset_id_3 = 0

    examples = [
        (spectrogram_1, label_1, dataset_id_1),
        (spectrogram_2, label_2, dataset_id_2),
        (spectrogram_3, label_3, dataset_id_3),
    ]

    batch = collate_fn_spectrogram(examples)
    images, labels, file_indices, item_indices = batch
    assert torch.all(file_indices[0:3] == 0)
    assert torch.all(file_indices[3:6] == 1)
    assert torch.all(file_indices[6:9] == 2)

    id_1 = file_indices == 0
    id_2 = file_indices == 1
    id_3 = file_indices == 2

    assert torch.all(images[id_1] == spectrogram_1)
    assert torch.all(images[id_2] == spectrogram_2)
    assert torch.all(images[id_3] == spectrogram_3)

    assert torch.all(labels[id_1] == label_1)
    assert torch.all(labels[id_2] == label_2)
    assert torch.all(labels[id_3] == label_3)

    assert torch.all(item_indices[id_1] == dataset_id_1)
    assert torch.all(item_indices[id_2] == dataset_id_2)
    assert torch.all(item_indices[id_3] == dataset_id_3)


def test_collate_fn_spectrogram_rgb():
    num_chunks = 3
    image_width = 384
    image_height = 384

    spectrogram_1 = torch.rand(num_chunks, NUM_RGB_CHANNELS, image_width, image_height)
    spectrogram_2 = torch.rand(num_chunks, NUM_RGB_CHANNELS, image_width, image_height)
    spectrogram_3 = torch.rand(num_chunks, NUM_RGB_CHANNELS, image_width, image_height)
    label_1 = torch.zeros(11)
    label_1[1] = 1
    label_2 = torch.zeros(11)
    label_2[5] = 1
    label_3 = torch.zeros(11)
    label_2[5:6] = 1

    dataset_id_1 = 532
    dataset_id_2 = 591
    dataset_id_3 = 0

    examples = [
        (spectrogram_1, label_1, dataset_id_1),
        (spectrogram_2, label_2, dataset_id_2),
        (spectrogram_3, label_3, dataset_id_3),
    ]

    batch = collate_fn_spectrogram(examples)
    images, labels, file_indices, item_indices = batch
    assert torch.all(file_indices[0:3] == 0)
    assert torch.all(file_indices[3:6] == 1)
    assert torch.all(file_indices[6:9] == 2)

    id_1 = file_indices == 0
    id_2 = file_indices == 1
    id_3 = file_indices == 2

    assert torch.all(images[id_1] == spectrogram_1)
    assert torch.all(images[id_2] == spectrogram_2)
    assert torch.all(images[id_3] == spectrogram_3)

    assert torch.all(labels[id_1] == label_1)
    assert torch.all(labels[id_2] == label_2)
    assert torch.all(labels[id_3] == label_3)

    assert torch.all(item_indices[id_1] == dataset_id_1)
    assert torch.all(item_indices[id_2] == dataset_id_2)
    assert torch.all(item_indices[id_3] == dataset_id_3)


def test_collate_fn_spectrogram_greyscale():
    num_chunks = 3
    image_width = 384
    image_height = 384

    spectrogram_1 = torch.rand(num_chunks, 1, image_width, image_height)
    spectrogram_2 = torch.rand(num_chunks, 1, image_width, image_height)
    spectrogram_3 = torch.rand(num_chunks, 1, image_width, image_height)
    label_1 = torch.zeros(11)
    label_1[1] = 1
    label_2 = torch.zeros(11)
    label_2[5] = 1
    label_3 = torch.zeros(11)
    label_2[5:6] = 1

    dataset_id_1 = 532
    dataset_id_2 = 591
    dataset_id_3 = 0

    examples = [
        (spectrogram_1, label_1, dataset_id_1),
        (spectrogram_2, label_2, dataset_id_2),
        (spectrogram_3, label_3, dataset_id_3),
    ]

    batch = collate_fn_spectrogram(examples)
    images, labels, file_indices, item_indices = batch
    assert torch.all(file_indices[0:3] == 0)
    assert torch.all(file_indices[3:6] == 1)
    assert torch.all(file_indices[6:9] == 2)

    id_1 = file_indices == 0
    id_2 = file_indices == 1
    id_3 = file_indices == 2

    assert torch.all(images[id_1] == spectrogram_1)
    assert torch.all(images[id_2] == spectrogram_2)
    assert torch.all(images[id_3] == spectrogram_3)

    assert torch.all(labels[id_1] == label_1)
    assert torch.all(labels[id_2] == label_2)
    assert torch.all(labels[id_3] == label_3)

    assert torch.all(item_indices[id_1] == dataset_id_1)
    assert torch.all(item_indices[id_2] == dataset_id_2)
    assert torch.all(item_indices[id_3] == dataset_id_3)


def test_collate_fn_spectrogram_single():
    num_chunks = 1
    image_width = 1
    image_height = 1

    spectrogram_1 = torch.rand(num_chunks, image_width, image_height)
    label_1 = torch.zeros(11)
    label_1[1] = 1

    dataset_id_1 = 532

    examples = [
        (spectrogram_1, label_1, dataset_id_1),
    ]

    batch = collate_fn_spectrogram(examples)
    images, labels, file_indices, item_indices = batch
    assert torch.all(file_indices[0:1] == 0)

    id_1 = file_indices == 0

    assert torch.all(images[id_1] == spectrogram_1)
    assert torch.all(labels[id_1] == label_1)
    assert torch.all(item_indices[id_1] == dataset_id_1)


def test_collate_fn_spectrogram_diff_sizes():
    image_width = 384
    image_height = 384

    spectrogram_1 = torch.rand(4, image_width, image_height)
    spectrogram_2 = torch.rand(5, image_width, image_height)
    spectrogram_3 = torch.rand(6, image_width, image_height)
    label_1 = torch.zeros(11)
    label_1[1] = 1
    label_2 = torch.zeros(11)
    label_2[5] = 1
    label_3 = torch.zeros(11)
    label_2[5:6] = 1

    dataset_id_1 = 532
    dataset_id_2 = 591
    dataset_id_3 = 0

    examples = [
        (spectrogram_1, label_1, dataset_id_1),
        (spectrogram_2, label_2, dataset_id_2),
        (spectrogram_3, label_3, dataset_id_3),
    ]

    batch = collate_fn_spectrogram(examples)
    images, labels, file_indices, item_indices = batch
    assert torch.all(file_indices[0:4] == 0)
    assert torch.all(file_indices[4 : 4 + 5] == 1)
    assert torch.all(file_indices[4 + 5 : 4 + 5 + 6] == 2)

    id_1 = file_indices == 0
    id_2 = file_indices == 1
    id_3 = file_indices == 2

    assert torch.all(images[id_1] == spectrogram_1)
    assert torch.all(images[id_2] == spectrogram_2)
    assert torch.all(images[id_3] == spectrogram_3)

    assert torch.all(labels[id_1] == label_1)
    assert torch.all(labels[id_2] == label_2)
    assert torch.all(labels[id_3] == label_3)

    assert torch.all(item_indices[id_1] == dataset_id_1)
    assert torch.all(item_indices[id_2] == dataset_id_2)
    assert torch.all(item_indices[id_3] == dataset_id_3)


test_collate_fn_spectrogram_diff_sizes()
