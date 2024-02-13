import os
import json
import argparse
from lightning import Fabric

from lib.utils import get_device
from lib.utils.constants import Subtask, Track, PreprocessTextLevel, ORIGINAL_DATA_DIR
from lib.data.loading import load_train_dev_test_df
from lib.data.tokenizer import get_tokenizer
from lib.models import get_model
from lib.training.loss import get_loss_fn
from lib.utils.utilities import transformer_model_get_max_batch_size

parser = argparse.ArgumentParser(description="Machine-generated text detection tool")
parser.add_argument(
    "--config", help="path to configuration file", default="./config.json"
)
parser.add_argument(
    "--use-fabric",
    help="use PyTorch Fabric for training", default=False, action="store_true",
)
parser.add_argument(
    "--fabric-config",
    help="path to PyTorch Fabric configuration file", default="./fabric_config.json",
)
parser.add_argument(
    "--fine-tune",
    help="Fine-tune all parameters available for the model",
    default=False,
    action="store_true",
)


def main():
    device = get_device()
    print(f"Using device: {device}")

    if device == "cpu":
        raise RuntimeError("GPU is not available!")

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

    fabric = None
    if args.use_fabric:
        fabric_config = {}
        with open(args.fabric_config) as f:
            fabric_config = json.load(f)

        if "accelerator" not in fabric_config:
            fabric_config["accelerator"] = device

        fabric = Fabric(**fabric_config)
        fabric.launch()

    tokenizer = get_tokenizer(**config["tokenizer"])
    model = get_model(config["model"], config["model_config"]).to(device)
    loss_fn = get_loss_fn(config["training"]["loss"], device)

    max_allowed_batch_size = transformer_model_get_max_batch_size(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_seq_len=config["data"]["max_len"],
        out_size=config["model_config"]["out_size"],
        dataset_size=min(len(df_train), len(df_dev), len(df_test)),
        loss_fn=loss_fn,
        optimizer=[*config["training"]["optimizer"]][0],
        max_batch_size=config["data"]["batch_size"],
        num_epochs=config["training"]["num_epochs"],
        fabric=fabric,
        fine_tune=args.fine_tune,
    )

    print(f"Max allowed batch size: {max_allowed_batch_size}")


if __name__ == "__main__":
    main()
