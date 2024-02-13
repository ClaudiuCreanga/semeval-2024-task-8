import os
import argparse
import json

from lib.utils import get_current_date, get_device
from lib.utils.training import EarlyStopping
from lib.utils.constants import (
    Subtask, Track, PreprocessTextLevel, DatasetType, ORIGINAL_DATA_DIR,
)
from lib.data.loading import load_train_dev_test_df, build_data_loader
from lib.data.tokenizer import get_tokenizer
from lib.data.vocabulary import get_vocabulary
from lib.models import get_model
from lib.training.loss import get_loss_fn
from lib.training.metric import get_metric
from lib.training.loops import training_loop, make_predictions

parser = argparse.ArgumentParser(description="Machine-generated text detection tool")
parser.add_argument(
    "--config", help="path to configuration file", default="./config.json"
)
parser.add_argument(
    "--save-results", help="save results to file", default=False, action="store_true"
)
parser.add_argument(
    "--print-freq", help="frequency for training print", default=10, type=int
)
parser.add_argument(
    "--debug",
    help="Debug mode - datasets are smaller", default=False, action="store_true",
)

DEVICE = get_device()


def main():
    print(f"Using device: {DEVICE}")

    args = parser.parse_args()

    config = {}
    with open(args.config) as f:
        config = json.load(f)

    task = None
    if "task" in config:
        task = Subtask(config["task"])
    else:
        raise ValueError("Task not specified in config")

    track = None
    if "track" in config:
        track = Track(config["track"])
    else:
        print(f"Warning: Track not specified in config for subtask: {task}")

    results_path = None
    if args.save_results:
        if track is None:
            results_path = (
                f"runs/{get_current_date()}-{task.value}-{config['model']}"
            )
        else:
            results_path = (
                f"runs/{get_current_date()}-"
                f"{task.value}-{track.value}-{config['model']}"
            )

        print(f"Will save results to: {results_path}")
        os.mkdir(results_path)

        with open(results_path + "/config.json", "w") as f:
            json.dump(config, f, indent=4)

    df_train, df_dev, df_test = load_train_dev_test_df(
        task=task,
        track=track,
        data_dir=(
            ORIGINAL_DATA_DIR
            if config["data"]["data_dir"] is None
            else os.path.relpath(config["data"]["data_dir"])
        ),
        label_column=config["data"]["label_column"],
        test_size=config["data"]["test_size"],
        preprocess_text_level=PreprocessTextLevel(
            config["data"]["preprocess_text_level"]
        ),
    )

    print(f"df_train.shape: {df_train.shape}")
    print(f"df_dev.shape: {df_dev.shape}")
    print(f"df_test.shape: {df_test.shape}")

    tokenizer = get_tokenizer(**config["tokenizer"])

    dataset_type = DatasetType.LongformerTokenClassificationDataset
    if "dataset_type" in config["data"]:
        dataset_type = DatasetType(config["data"]["dataset_type"])

    dataset_type_settings = None
    if "dataset_type_settings" in config["data"]:
        dataset_type_settings = config["data"]["dataset_type_settings"]

    if args.debug:
        label_column = config["data"]["label_column"]

        if task == Subtask.SubtaskA:
            # Sample 1000 random examples from each dataset with respect to label
            df_train = df_train.sample(frac=1).groupby(label_column).head(500)
            df_dev = df_dev.sample(frac=1).groupby(label_column).head(500)
            df_test = df_test.sample(frac=1).groupby(label_column).head(500)
        elif task == Subtask.SubtaskB:
            # Sample 1200 random examples from each dataset with respect to label
            df_train = df_train.sample(frac=1).groupby(label_column).head(200)
            df_dev = df_dev.sample(frac=1).groupby(label_column).head(200)
            df_test = df_test.sample(frac=1).groupby(label_column).head(200)
        elif task == Subtask.SubtaskC:
            # Sample 500 random examples from each dataset
            df_train = df_train.sample(20)
            df_dev = df_dev.sample(20)
            df_test = df_test.sample(20)
        else:
            raise ValueError(f"Unknown task: {task}")

    char_vocabulary, word_vocabulary = None, None
    char_max_len, word_max_len = None, config["data"]["max_len"]
    if dataset_type == DatasetType.TokenClassificationDataset:
        if dataset_type_settings is not None:
            if "chars" in dataset_type_settings:
                char_vocabulary = get_vocabulary("chars")
                char_vocabulary.build_vocabulary(df_train)

                char_max_len = dataset_type_settings["chars"]["max_len"]

            if "words" in dataset_type_settings:
                word_vocabulary = get_vocabulary("words")
                word_vocabulary.build_vocabulary(df_train)

                word_max_len = dataset_type_settings["words"]["max_len"]
        else:
            word_vocabulary = get_vocabulary("words")
            word_vocabulary.build_vocabulary(df_train)

            word_max_len = config["data"]["max_len"]

    if "vocab_size" in config["model_config"]:
        config["model_config"]["vocab_size"] = word_vocabulary.vocab_size()
    if "char_vocab_size" in config["model_config"]:
        config["model_config"]["char_vocab_size"] = char_vocabulary.vocab_size()
    if "word_vocab_size" in config["model_config"]:
        config["model_config"]["word_vocab_size"] = word_vocabulary.vocab_size()

    if "char_max_len" in config["model_config"]:
        config["model_config"]["char_max_len"] = char_max_len

    # Save vocab size to config
    with open(results_path + "/config.json", "w") as f:
        json.dump(config, f, indent=4)

    train_dataloader = build_data_loader(
        df_train,
        tokenizer,
        max_len=word_max_len,
        batch_size=config["data"]["batch_size"],
        label_column=config["data"]["label_column"],
        shuffle=True,
        dataset_type=dataset_type,
        dataset_type_settings=dataset_type_settings,
        char_vocabulary=char_vocabulary,
        char_max_len=char_max_len,
        word_vocabulary=word_vocabulary,
        device=DEVICE,
    )
    dev_dataloader = build_data_loader(
        df_dev,
        tokenizer,
        max_len=word_max_len,
        batch_size=config["data"]["batch_size"],
        label_column=config["data"]["label_column"],
        dataset_type=dataset_type,
        dataset_type_settings=dataset_type_settings,
        char_vocabulary=char_vocabulary,
        char_max_len=char_max_len,
        word_vocabulary=word_vocabulary,
        device=DEVICE,
    )
    test_dataloader = build_data_loader(
        df_test,
        tokenizer,
        max_len=word_max_len,
        batch_size=config["data"]["batch_size"],
        label_column=config["data"]["label_column"],
        has_targets=False if config["data"]["test_size"] is None else True,
        dataset_type=dataset_type,
        dataset_type_settings=dataset_type_settings,
        char_vocabulary=char_vocabulary,
        char_max_len=char_max_len,
        word_vocabulary=word_vocabulary,
        device=DEVICE,
    )

    num_epochs = config["training"]["num_epochs"]
    model = get_model(config["model"], config["model_config"]).to(DEVICE)
    loss_fn = get_loss_fn(config["training"]["loss"], DEVICE)
    optimizer_config = config["training"]["optimizer"]
    scheduler_config = config["training"]["scheduler"]
    metric_fn, is_better_metric_fn = get_metric(config["training"]["metric"])
    num_epochs_before_finetune = config["training"]["num_epochs_before_finetune"]

    early_stopping = None
    if "early_stopping" in config["training"]:
        early_stopping = EarlyStopping(
            path=results_path if args.save_results else None,
            verbose=True,
            **config["training"]["early_stopping"],
        )

    best_model = training_loop(
        model,
        num_epochs,
        train_dataloader,
        dev_dataloader,
        loss_fn,
        optimizer_config,
        scheduler_config,
        DEVICE,
        metric_fn,
        is_better_metric_fn,
        results_path if args.save_results else None,
        num_epochs_before_finetune,
        early_stopping=early_stopping,
        print_freq=args.print_freq,
    )

    make_predictions(
        best_model,
        test_dataloader,
        DEVICE,
        results_path if args.save_results else None,
        label_column=config["data"]["label_column"],
        file_format=config["submission_format"],
    )


if __name__ == "__main__":
    main()
