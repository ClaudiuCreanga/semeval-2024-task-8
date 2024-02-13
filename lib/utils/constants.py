import os
from enum import Enum

RANDOM_SEED = 42
DATETIME_FORMAT = "%d-%m-%Y_%H:%M:%S"

ORIGINAL_DATA_DIR = os.path.relpath("./data/original_data")
COMBINED_DATA_DIR = os.path.relpath("./data/combined_data")

MODEL_2_LABEL = {
    "human": 0,
    "chatGPT": 1,
    "cohere": 2,
    "davinci": 3,
    "bloomz": 4,
    "dolly": 5,
}


class Subtask(Enum):
    SubtaskA = "SubtaskA"
    SubtaskB = "SubtaskB"
    SubtaskC = "SubtaskC"


class Track(Enum):
    MONOLINGUAL = "monolingual"
    MULTILINGUAL = "multilingual"


class PreprocessTextLevel(Enum):
    NONE = 0
    LIGHT = 1
    HEAVY = 2


class DatasetType(Enum):
    TransformerTruncationDataset = "transformer_truncation_dataset"
    TransformerHierarchicalDataset = "transformer_hierarchical_dataset"
    LongformerTokenClassificationDataset = "longformer_token_classification_dataset"
    NeuralNetworkStatisticsDataset = "neural_network_statistics_dataset"
    TokenClassificationDataset = "token_classification_dataset"


class TruncationStrategy(Enum):
    HEAD_ONLY = "head_only"
    TAIL_ONLY = "tail_only"
    HEAD_AND_TAIL = "head_and_tail"


class MergeLayersStrategy(Enum):
    CONCATENATE = "concatenate"
    MEAN = "mean"
    MAX = "max"


class PoolingStrategy(Enum):
    MEAN = "mean"
    MAX = "max"
