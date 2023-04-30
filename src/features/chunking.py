import torch
import torch.nn.functional
import torchvision.transforms.functional

from src.config.config_defaults import NUM_RGB_CHANNELS


def chunk_image_by_width(
    target_image_size: tuple[int, int],
    image: torch.Tensor,
    pad_value: float | str = "repeat",
):
    """EXAMPLE 1 Target image size: (384, 384)

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

    Step 3a): pad with zeros on the right
    |==============|==============|==============|
    |  (384, 384)  |  (384, 384)  |  (384, 384)0 |
    |              |              |            0 |
    |==============|==============|==============|

    Step 3b): repeat with first chunk
    |==============|==============|==============|
    |1 (384, 384)  |  (384, 384)  |  (384, 384)1 |
    |1             |              |            1 |
    |==============|==============|==============|

    Returns 3x (384, 384)


    EXAMPLE 2
    Target image size: (384, 384)

    Image:
    (128 x 100)
    |====|
    |    |
    |====|

    Step 1: scale height only
    (384 x 100)
    |====|
    |    |
    |    |
    |====|

    Step 2: repeat images
    |====================|
    | 100  100  100  100 |
    |                    |
    |====================|

    Step 3: cut excess
    |=================|
    |100  100  100  84|
    |                 |
    |=================|

    Returns 1x (384, 384)
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
    interpolation = torchvision.transforms.functional.InterpolationMode.NEAREST_EXACT
    image = torchvision.transforms.functional.resize(
        image,
        size=(pre_resize_height, pre_resize_width),
        interpolation=interpolation,
        antialias=False,
    )

    # Chunk by last dimension (width)
    # list([1, 384, 384]), length = 5
    # list([batch, height, width]), length = full_width // image_width
    chunks = list(image.split(image_width, dim=-1))
    # Last chunk might be cut off which means the time dimension (image width) will be smaller
    # Add zero padding if last chunk's width is shorter than image width
    last_chunk = chunks[-1]  # [Batch]
    last_chunk_width = last_chunk.shape[-1]
    diff = image_width - last_chunk_width  # e.g. 384 - 200 = 184
    if diff > 0 and type(pad_value) is int or type(pad_value) is float:
        # we add 0 pads on the left, and diff on the right side, pad=(padding_left,padding_right)
        chunks[-1] = torch.nn.functional.pad(
            input=last_chunk, pad=(0, diff), mode="constant", value=pad_value
        )

    elif diff > 0 and type(pad_value) is str:
        """Take the first chunk, glue it to the last (which is shorter).

        If first chunk is last chunk then repeat it until the size is large enough.
        """
        # diff = 384 - 50 = 334
        first_chunk: torch.Tensor = chunks[0]  # [384, 50] if first chunk == first chunk
        first_chunk_width = first_chunk.shape[-1]  # 50
        num_first_chunk_repeats = max(1, int(diff / first_chunk_width))  # 8
        repeated_first_chunk = torch.cat(
            [first_chunk] * num_first_chunk_repeats, dim=-1
        )  # [384, 334]

        # Remove remove excess width caused by repeating
        repeated_first_chunk = repeated_first_chunk[..., :diff]  # [384, 334]
        chunks[-1] = torch.cat((chunks[-1], repeated_first_chunk), dim=-1)  # [384, 334]

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


def collate_fn_feature(
    examples: list[list[torch.Tensor], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Example:
    Image shape: [Chunks, Channel, ...feature]
    Label shape: [Label]
    Dataset index: [1]

    Batch is tuple (feature, label, dataset index)
    batch = [
        [[3,4], [5,6]], [1,0,0], 851
        [[2,3], [1,7]], [0,0,1], 532
        [[9,3], [2,1]], [0,1,0], 91
    ]
    """

    # Count total number of features (sum of all chunks)
    num_features = 0
    for e in examples:
        feature_chunks = e[0]
        num_features += feature_chunks.shape[0]

    example_item = examples[0]
    example_feature = example_item[0]
    example_label = example_item[1]
    feature_shape = tuple(example_feature.shape[1:])

    # Create empty matrices
    features = torch.empty((num_features, *feature_shape))
    labels = torch.empty((num_features, example_label.shape[-1]))
    file_indices = torch.empty(num_features, dtype=torch.int64)
    item_indices = torch.empty(num_features, dtype=torch.int64)

    features_passed = 0
    for unique_file_idx, item in enumerate(examples):
        feature_chunks, label, dataset_index = item
        num_chunks = feature_chunks.shape[0]

        start = features_passed
        end = features_passed + num_chunks

        features[start:end] = feature_chunks
        labels[start:end] = label
        file_indices[start:end] = torch.full((num_chunks,), unique_file_idx)
        item_indices[start:end] = torch.full((num_chunks,), int(dataset_index))

        features_passed += num_chunks

    return features, labels, file_indices, item_indices


def test_collate_fn_feature():
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

    batch = collate_fn_feature(examples)
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


def test_collate_fn_feature_rgb():
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

    batch = collate_fn_feature(examples)
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


def test_collate_fn_feature_greyscale():
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

    batch = collate_fn_feature(examples)
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


def test_collate_fn_feature_single():
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

    batch = collate_fn_feature(examples)
    images, labels, file_indices, item_indices = batch
    assert torch.all(file_indices[0:1] == 0)

    id_1 = file_indices == 0

    assert torch.all(images[id_1] == spectrogram_1)
    assert torch.all(labels[id_1] == label_1)
    assert torch.all(item_indices[id_1] == dataset_id_1)


def test_collate_fn_feature_diff_sizes():
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

    batch = collate_fn_feature(examples)
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


def test_chunk_small_image_and_height_resize():
    image = torch.rand(128, 50)
    target_image_size = (384, 384)
    repeated = chunk_image_by_width(target_image_size, image, pad_value="repeat")
    assert len(repeated) == 1
    repeated = torchvision.transforms.functional.resize(
        repeated, size=(128, repeated.shape[-1]), antialias=False
    )
    repeated = repeated[0]
    assert torch.all(torch.isclose(repeated[..., 0:50], image))
    assert torch.all(torch.isclose(repeated[..., 50:100], image))
    assert torch.all(torch.isclose(repeated[..., 150:200], image))


def test_chunk_large_image():
    image = torch.rand(384, 800)
    target_image_size = (384, 384)
    num_images = 3  # 800 / 384
    diff = 800 - (384 * (num_images - 1))  # 32
    last_chunk_width = 384 - diff

    # big image width: 384 + 384 + 32
    # ================================
    # [    384    |    384    | 32 ]
    # image1 width: 384
    # image2 width: 384
    # image3 width: 32

    fixed = chunk_image_by_width(target_image_size, image, pad_value="repeat")
    assert len(fixed) == 3

    first_image = fixed[0]
    second_image = fixed[1]
    third_image = fixed[2]

    first_image_end = 384
    second_image_end = first_image_end + 384
    third_image_end = second_image_end + diff
    original_first_image = image[..., :first_image_end]
    original_second_image = image[..., first_image_end:second_image_end]
    original_third_small_chunk = image[..., second_image_end:third_image_end]

    first_image_glued_part = image[..., 0:last_chunk_width]
    original_third_small_chunk = image[..., second_image_end:third_image_end]

    assert torch.all(torch.isclose(first_image, original_first_image))
    assert torch.all(torch.isclose(second_image, original_second_image))
    assert torch.all(torch.isclose(third_image[..., :diff], original_third_small_chunk))
    assert torch.all(torch.isclose(third_image[..., diff:], first_image_glued_part))
