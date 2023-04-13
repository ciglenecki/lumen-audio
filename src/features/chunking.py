from functools import partial
from typing import Callable

import torch

from src.config.config_defaults import ConfigDefault
from src.enums.enums import ModelInputDataType
from src.model.model import get_data_input_type


def collate_fn_spectrogram(
    examples: list[tuple[torch.Tensor], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Example:
    batch size = 3
    max_size = 2

    Batch is tuple (spectrogram image, label)
    batch = [
        [[3,4], [5,6]], [1,0,0]
        [[2,3], [1,7]], [0,0,1]
        [[9,3], [2,1]], [0,1,0]
    ]
    """

    all_audio_chunks, all_labels, file_indices = [], [], []
    for i, (audio_chunks, label) in enumerate(examples):
        for audio_chunk in audio_chunks:
            all_audio_chunks.append(audio_chunk.unsqueeze(0))
            all_labels.append(label.unsqueeze(0))
            file_indices.append(i)

    return (
        torch.cat(tuple(all_audio_chunks), dim=0),
        torch.cat(tuple(all_labels), dim=0),
        torch.tensor(file_indices),
    )


def collate_fn_audio(batch: list[tuple[torch.Tensor, torch.Tensor]], max_audio_width):
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
        chunks = list(torch.split(audio, max_audio_width))
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
            max_audio_width=config.max_audio_seconds * config.sampling_rate,
        )
    else:
        raise Exception(f"Unsupported data input type {data_input_type}")
