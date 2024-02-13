import os
import argparse
import json

from lib.utils import get_current_date, get_device
from lib.utils.constants import (
    Subtask, Track, PreprocessTextLevel, DatasetType, ORIGINAL_DATA_DIR,
)
from lib.data.loading import load_train_dev_test_df, build_data_loader
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

    if args.save_results:
        if config["track"] is None:
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

    # tokenizer = get_tokenizer(**config["tokenizer"])
    tokenizer = None

    # tokens_count = {}
    # for df_name, df in zip(["train", "dev", "test"], [df_train, df_dev , df_test]):
    #     print(f"Counting tokens for {df_name}...")

    #     counts = []
    #     for text in tqdm(df["text"]):
    #         counts.append(len(tokenizer.encode_plus(text)["input_ids"]))
    #     tokens_count[df_name] = counts

    # for dataset, counts in tokens_count.items():
    #     print(f"{dataset} mean: {sum(counts) / len(counts)}")
    #     print(f"{dataset} median: {sorted(counts)[len(counts) // 2]}")
    #     print(f"{dataset} max: {max(counts)}")
    #     print(f"{dataset} min: {min(counts)}")
    #     print(f"{dataset} no. > 512: {len([c for c in counts if c > 512])}")
    #     print("#" * 25)

    dataset_type = DatasetType.NeuralNetworkStatisticsDataset
    if "type" in config["data"]:
        dataset_type = DatasetType(config["data"]["type"])

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
            df_train = df_train.sample(500)
            df_dev = df_dev.sample(500)
            df_test = df_test.sample(500)
        else:
            raise ValueError(f"Unknown task: {task}")

    train_dataloader = build_data_loader(
        df_train,
        tokenizer,
        max_len=config["data"]["max_len"],
        batch_size=config["data"]["batch_size"],
        label_column=config["data"]["label_column"],
        shuffle=True,
        dataset_type=dataset_type,
        dataset_type_settings=dataset_type_settings,
        device=DEVICE,
    )
    dev_dataloader = build_data_loader(
        df_dev,
        tokenizer,
        max_len=config["data"]["max_len"],
        batch_size=config["data"]["batch_size"],
        label_column=config["data"]["label_column"],
        dataset_type=dataset_type,
        dataset_type_settings=dataset_type_settings,
        device=DEVICE,
    )
    test_dataloader = build_data_loader(
        df_test,
        tokenizer,
        max_len=config["data"]["max_len"],
        batch_size=config["data"]["batch_size"],
        label_column=config["data"]["label_column"],
        has_targets=False if config["data"]["test_size"] is None else True,
        dataset_type=dataset_type,
        dataset_type_settings=dataset_type_settings,
        device=DEVICE,
    )

    config["model_config"]["input_size"] = (
        next(iter(train_dataloader))["input_ids"].shape[1]
    )

    num_epochs = config["training"]["num_epochs"]
    model = get_model(config["model"], config["model_config"]).to(DEVICE)
    loss_fn = get_loss_fn(config["training"]["loss"], DEVICE)
    optimizer_config = config["training"]["optimizer"]
    scheduler_config = config["training"]["scheduler"]
    metric_fn, is_better_metric_fn = get_metric(config["training"]["metric"])
    num_epochs_before_finetune = config["training"]["num_epochs_before_finetune"]
    swa_config = config["training"]["swa"] if "swa" in config["training"] else None

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
        swa_config=swa_config,
        validation_freq=None,
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
