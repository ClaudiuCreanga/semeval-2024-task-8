import os
import json
import torch
import argparse
import pandas as pd

from lib.utils import get_device
from lib.utils.constants import DatasetType
from lib.data.loading import build_data_loader
from lib.data.tokenizer import get_tokenizer
from lib.models import get_model
from lib.training.loops import make_predictions

parser = argparse.ArgumentParser(description="Machine-generated text detection tool")
parser.add_argument(
    "--best-model-dir",
    help="path to directory containing the best model",
)
parser.add_argument(
    "--file",
    help="path to file containing the data to predict on",
)

DEVICE = "cpu" if get_device() == "mps" else get_device()
MAP_LOCATION = None if torch.cuda.is_available() else torch.device("cpu")


def main():
    print(f"Using device: {DEVICE}")

    args = parser.parse_args()
    config = {}
    with open(f"{args.best_model_dir}/config.json") as f:
        config = json.load(f)

    tokenizer = get_tokenizer(**config["tokenizer"])

    dataset_type = DatasetType.TransformerTruncationDataset
    if "dataset_type" in config["data"]:
        dataset_type = DatasetType(config["data"]["dataset_type"])

    dataset_type_settings = None
    if "dataset_type_settings" in config["data"]:
        dataset_type_settings = config["data"]["dataset_type_settings"]

    model = get_model(config["model"], config["model_config"]).to(DEVICE)
    model.load_state_dict(
        torch.load(
            os.path.join(args.best_model_dir, "best_model.bin"),
            map_location=MAP_LOCATION,
        )
    )

    if args.file is not None:
        texts = []
        with open(args.file) as f:
            texts = f.readlines()

        df = pd.DataFrame({
            "id": range(len(texts)),
            "text": texts,
        })

        data_loader = build_data_loader(
            df,
            tokenizer,
            max_len=config["data"]["max_len"],
            batch_size=config["data"]["batch_size"],
            label_column=config["data"]["label_column"],
            has_targets=False,
            dataset_type=dataset_type,
            dataset_type_settings=dataset_type_settings,
            device=DEVICE,
        )

        predictions = make_predictions(
            model,
            data_loader,
            DEVICE,
            args.best_model_dir,
            label_column=config["data"]["label_column"],
            file_format=config["submission_format"],
        )

        print(f"Predictions:\n\n{predictions}\n\n")
    else:
        while True:
            input_text = input("Enter text to predict on: ")
            if input_text == "exit":
                break

            df = pd.DataFrame({
                "id": [0],
                "text": [input_text],
            })

            data_loader = build_data_loader(
                df,
                tokenizer,
                max_len=config["data"]["max_len"],
                batch_size=config["data"]["batch_size"],
                label_column=config["data"]["label_column"],
                has_targets=False,
                dataset_type=dataset_type,
                dataset_type_settings=dataset_type_settings,
                device=DEVICE,
            )

            predictions = make_predictions(
                model,
                data_loader,
                DEVICE,
                args.best_model_dir,
                label_column=config["data"]["label_column"],
                file_format=config["submission_format"],
            )

            print(f"Predictions:\n\n{predictions}\n\n")


if __name__ == "__main__":
    main()
