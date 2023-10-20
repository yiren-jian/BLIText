import os
from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
# from lavis.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset
# from lavis.datasets.datasets.laion_dataset import LaionDataset
from lavis.datasets.datasets.sentence_datasets import SentenceDataset


@registry.register_builder("laion_sentence_115m")
class LaionSentence115MBuilder(BaseDatasetBuilder):
    train_dataset_cls = SentenceDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/laion/defaults_115m.yaml"
    }
