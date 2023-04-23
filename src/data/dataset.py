from config.config_defaults import ConfigDefault
from data.dataset_irmas import IRMASDatasetTest, IRMASDatasetTrain
from enums.enums import SupportedDatasetDirType


def get_dataset(config, dataset_enum):
    if dataset_enum == SupportedDatasetDirType.IRMAS_TRAIN:
        dataset = IRMASDatasetTrain(
            dataset_path=dataset_path,
            audio_transform=self.train_audio_transform,
            normalize_audio=self.normalize_audio,
            concat_n_samples=self.concat_n_samples,
            sum_two_samples=self.sum_two_samples,
        )
        datasets.append(dataset)
    elif dataset_enum == SupportedDatasetDirType.IRMAS_TEST:
        dataset = IRMASDatasetTest(
            dataset_path=dataset_path,
            audio_transform=self.val_audio_transform,
            normalize_audio=self.normalize_audio,
            concat_n_samples=False,
            sum_two_samples=False,
        )
        datasets.append(dataset)
    elif dataset_enum == SupportedDatasetDirType.OPENMIC:
        pass
    elif dataset_enum == SupportedDatasetDirType.OPENMIC:
        pass
    elif dataset_enum == SupportedDatasetDirType.OPENMIC:
        pass


def get_dataset_from_config(config: ConfigDefault):
    pass


def get_dataloader(config, dataset, collate_fn):
    pass
