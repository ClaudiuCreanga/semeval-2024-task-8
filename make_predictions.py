import os
import json
import torch
import argparse

from lib.utils import get_device
from lib.utils.constants import (
    Subtask, Track, PreprocessTextLevel, DatasetType,
    ORIGINAL_DATA_DIR,
)
from lib.data.loading import load_train_dev_test_df, build_data_loader
from lib.data.tokenizer import get_tokenizer
from lib.data.vocabulary import get_vocabulary, CharacterVocabulary, WordVocabulary
from lib.models import get_model
from lib.training.loops import make_predictions

parser = argparse.ArgumentParser(description="Machine-generated text detection tool")
parser.add_argument(
    "--best-model-dir",
    help="path to directory containing the best model",
)
parser.add_argument(
    "--config", help="path to configuration file", default="./config.json"
)

DEVICE = get_device()


def main():
    print(f"Using device: {DEVICE}")

    args = parser.parse_args()
    config = {}
    with open(args.config) as f:
        config = json.load(f)

    track = None
    if "track" in config:
        track = Track(config["track"])
    else:
        print("Warning: Track not specified in config")

    df_train, df_dev, df_test = load_train_dev_test_df(
        task=Subtask(config["task"]),
        track=track,
        data_dir=(
            ORIGINAL_DATA_DIR
            if config["data"]["data_dir"] is None
            else os.path.relpath(config["data"]["data_dir"])
        ),
        label_column=config["data"]["label_column"],
        test_size=None,
        preprocess_text_level=PreprocessTextLevel(
            config["data"]["preprocess_text_level"]
        ),
    )
    print(df_test.shape)
    print(df_test.head())
    tokenizer = get_tokenizer(**config["tokenizer"])

    dataset_type = DatasetType.TransformerTruncationDataset
    if "dataset_type" in config["data"]:
        dataset_type = DatasetType(config["data"]["dataset_type"])

    dataset_type_settings = None
    if "dataset_type_settings" in config["data"]:
        dataset_type_settings = config["data"]["dataset_type_settings"]

    char_vocabulary, word_vocabulary = None, None
    char_max_len, word_max_len = None, config["data"]["max_len"]
    if dataset_type == DatasetType.TokenClassificationDataset:
        if dataset_type_settings is not None:
            if "chars" in dataset_type_settings:
                char_vocabulary = CharacterVocabulary()
                char_vocabulary.load_vocabulary(args.best_model_dir)

                char_max_len = dataset_type_settings["chars"]["max_len"]

            if "words" in dataset_type_settings:
                word_vocabulary = WordVocabulary()
                word_vocabulary.load_vocabulary(args.best_model_dir)

                word_max_len = dataset_type_settings["words"]["max_len"]
        else:
            word_vocabulary = WordVocabulary()
            word_vocabulary.load_vocabulary(args.best_model_dir)

            word_max_len = config["data"]["max_len"]

    test_dataloader = build_data_loader(
        df_test,
        tokenizer,
        max_len=word_max_len,
        batch_size=config["data"]["batch_size"],
        label_column=config["data"]["label_column"],
        has_targets=False,
        dataset_type=dataset_type,
        dataset_type_settings=dataset_type_settings,
        char_vocabulary=char_vocabulary,
        char_max_len=char_max_len,
        word_vocabulary=word_vocabulary,
        device=DEVICE,
    )

    model = get_model(config["model"], config["model_config"]).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(args.best_model_dir, "best_model.bin"))
    )
    model.to(DEVICE)

    make_predictions(
        model,
        test_dataloader,
        DEVICE,
        args.best_model_dir,
        label_column=config["data"]["label_column"],
        file_format=config["submission_format"],
    )


if __name__ == "__main__":
    main()
