import os
import pandas as pd
from typing import Callable
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader

from lib.data.preprocessing import preprocess
from lib.data.dataset import (
    MachineGeneratedTransformerTruncationDataset,
    MachineGeneratedTransformerHierarchicalDataset,
    LongformerTokenClassificationDataset,
    TokenClassificationDataset,
    MachineGeneratedNeuralNetworkStatisticsDataset,
)
from lib.data.splitting import tokenize_long_texts
from lib.data.vocabulary import CharacterVocabulary, WordVocabulary
from lib.utils.constants import (
    ORIGINAL_DATA_DIR, RANDOM_SEED,
    Subtask, Track, PreprocessTextLevel, DatasetType,
)

# 95% of the documents have less than 1500 tokens
MAX_DOCUMENT_LEN = 1500


def read_predictions_from_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".jsonl"):
        return pd_read_jsonl_file(file_path)
    else:
        raise NotImplementedError(f"No such file format for {file_path}")


def pd_read_jsonl_file(file_path: str) -> pd.DataFrame:
    return pd.read_json(file_path, lines=True)


def pd_write_jsonl_file(df: pd.DataFrame, file_path: str) -> str | None:
    return df.to_json(file_path, orient="records", lines=True)


def merge_jsonl_files(
    file_paths: [str],
    exclude_file_path: str,
    label_column: str,
    mapping_column: str,
    mapping_fn: Callable,
    debug: bool = False,
) -> pd.DataFrame:
    dfs = []
    for file_path in file_paths:
        df = pd_read_jsonl_file(file_path)
        df = df[df[label_column].notna()]
        df = df[df[mapping_column].notna()]

        df[label_column] = df[mapping_column].apply(mapping_fn)

        dfs.append(df)

    pd_result = pd.concat(dfs, ignore_index=True)
    if debug:
        print(f"Before dropping duplicates: {pd_result.shape}")

    pd_result = pd_result.drop_duplicates(subset=["text"])
    if debug:
        print(f"After dropping duplicates: {pd_result.shape}")

    pd_exclude = pd_read_jsonl_file(exclude_file_path)
    pd_result = pd_result[~pd_result.text.isin(pd_exclude.text)]
    if debug:
        print(f"After excluding texts from {exclude_file_path}: {pd_result.shape}")

    return pd_result


def load_train_dev_test_df(
    task: Subtask,
    track: Track = None,
    data_dir: str = ORIGINAL_DATA_DIR,
    label_column: str = "label",
    test_size: float = None,
    preprocess_text_level: PreprocessTextLevel = PreprocessTextLevel.NONE,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_path = os.path.join(data_dir, task.value, f"{task.value}_train.jsonl")
    dev_path = os.path.join(data_dir, task.value, f"{task.value}_dev.jsonl")
    test_path = os.path.join(data_dir, task.value, f"{task.value}_test.jsonl")

    if track is not None:
        train_path = os.path.join(
            data_dir,
            task.value,
            f"{task.value}_train_{track.value}.jsonl"
        )

    print("Loading train data...")
    print(train_path)
    df_train = pd_read_jsonl_file(train_path)
    df_train = df_train[df_train[label_column].notna()]

    # df_train = df_train[:1000]

    if test_size is None:
        dev_path = os.path.join(data_dir, task.value, f"{task.value}_dev.jsonl")
        test_path = os.path.join(data_dir, task.value, f"{task.value}_test.jsonl")

        if track is not None:
            dev_path = os.path.join(
                data_dir,
                task.value,
                f"{task.value}_dev_{track.value}.jsonl"
            )

            test_path = os.path.join(
                data_dir,
                task.value,
                f"{task.value}_test_{track.value}.jsonl"
            )

        df_dev = pd_read_jsonl_file(dev_path)
        df_test = pd_read_jsonl_file(test_path)
    else:
        print(f"Train/dev split... (df_train.shape: {df_train.shape})")

        statify = None if task == Subtask.SubtaskC else list(df_train[label_column])
        df_train, df_dev = train_test_split(
            df_train,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=statify,
        )

        test_path = os.path.join(data_dir, task.value, f"{task.value}_dev.jsonl")
        if track is not None:
            test_path = os.path.join(
                data_dir,
                task.value,
                f"{task.value}_dev_{track.value}.jsonl"
            )

        print(f"Loading test data... ---> {test_path}")
        df_test = pd_read_jsonl_file(test_path)
        # df_test = df_test[:500]

    if preprocess_text_level != PreprocessTextLevel.NONE:
        print(f"Cleaning texts with preprocess level `{preprocess_text_level}`...")
        for df in [df_train, df_dev, df_test]:
            df.text = [preprocess(t, preprocess_text_level) for t in df.text]

    return df_train, df_dev, df_test


def build_data_loader(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    max_len: int,
    batch_size: int,
    has_targets: bool = True,
    label_column: str = "label",
    shuffle: bool = False,
    dataset_type: DatasetType = DatasetType.TransformerTruncationDataset,
    dataset_type_settings: dict | None = None,
    char_vocabulary: CharacterVocabulary | None = None,
    char_max_len: int | None = None,
    word_vocabulary: WordVocabulary | None = None,
    device: str | None = None,
) -> DataLoader:
    collate_fn = None

    if dataset_type == DatasetType.TransformerTruncationDataset:
        if dataset_type_settings is None:
            dataset_type_settings = {}

        ds = MachineGeneratedTransformerTruncationDataset(
            ids=df.id.to_numpy(),
            texts=df.text.to_numpy(),
            targets=None if not has_targets else df[label_column].to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len,
            **dataset_type_settings,
        )
    elif dataset_type == DatasetType.TransformerHierarchicalDataset:
        collate_fn = MachineGeneratedTransformerHierarchicalDataset.collate_fn

        truncate_documents = True
        max_document_len = MAX_DOCUMENT_LEN

        # Remove 2 for [CLS] and [SEP]
        dataset_max_len = max_len - 2
        dataset_chunk_size = max_len - 2
        dataset_stride = max_len - 2
        dataset_min_chunk_size = max_len - 2

        if dataset_type_settings is not None:
            if "truncate_documents" in dataset_type_settings:
                truncate_documents = bool(dataset_type_settings["truncate_documents"])

            if "max_document_len" in dataset_type_settings:
                max_document_len = dataset_type_settings[
                    "max_document_len"
                ]

            if "max_len" in dataset_type_settings:
                if dataset_type_settings["max_len"] != max_len:
                    raise ValueError(
                        f"Provided data.max_len ({max_len}) and "
                        f"data.dataset_type_settings.max_len "
                        f"({dataset_type_settings['max_len']}) are not equal!"
                    )
                else:
                    dataset_max_len = dataset_type_settings["max_len"] - 2

            if "chunk_size" in dataset_type_settings:
                dataset_chunk_size = dataset_type_settings["chunk_size"] - 2

            if "stride" in dataset_type_settings:
                dataset_stride = dataset_type_settings["stride"] - 2

            if "min_chunk_size" in dataset_type_settings:
                dataset_min_chunk_size = dataset_type_settings["min_chunk_size"] - 2

        tokenize_settings = {
            "truncate_documents": truncate_documents,
            "max_document_len": max_document_len,
            "max_len": dataset_max_len,
            "chunk_size": dataset_chunk_size,
            "stride": dataset_stride,
            "min_chunk_size": dataset_min_chunk_size,
        }

        print("Will tokenize long texts (may take a while)...\n")

        tokens = tokenize_long_texts(
            df.text.to_list(),
            tokenizer,
            **tokenize_settings,
        )

        print("---- Done tokenizing long texts ----\n")

        ds = MachineGeneratedTransformerHierarchicalDataset(
            ids=df.id.to_numpy(),
            texts=df.text.to_numpy(),
            tokens=tokens,
            targets=None if not has_targets else df[label_column].to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len,
            device=device,
        )
    elif dataset_type == DatasetType.LongformerTokenClassificationDataset:
        ds = LongformerTokenClassificationDataset(
            ids=df.id.to_numpy(),
            texts=df.text.to_numpy(),
            targets=None if not has_targets else df[label_column].to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len,
            debug=False,
        )
    elif dataset_type == DatasetType.NeuralNetworkStatisticsDataset:
        ds = MachineGeneratedNeuralNetworkStatisticsDataset(
            df=df,
            label_column=label_column,
            has_targets=has_targets,
        )
    elif dataset_type == DatasetType.TokenClassificationDataset:
        if word_vocabulary is None:
            raise ValueError("Word vocabulary must be provided!")

        ds = TokenClassificationDataset(
            ids=df.id.to_numpy(),
            texts=df.text.to_numpy(),
            targets=None if not has_targets else df[label_column].to_numpy(),
            char_vocabulary=char_vocabulary,
            word_vocabulary=word_vocabulary,
            char_max_len=char_max_len,
            word_max_len=max_len,
            debug=False,
        )
    else:
        raise NotImplementedError(f"No such dataset type: {dataset_type}!")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
