import os
import argparse
from typing import Callable

from lib.data.loading import merge_jsonl_files, pd_write_jsonl_file
from lib.utils.utilities import map_model_to_label, map_model_label_to_binary_label

parser = argparse.ArgumentParser(description="Machine-generated text detection tool")
parser.add_argument(
    "--file1",
    help="path to first file, must be valid jsonl file",
    default="./file1.jsonl",
)
parser.add_argument(
    "--file2",
    help="path to second file, must be valid jsonl file",
    default="./file2.jsonl",
)
parser.add_argument(
    "--exclude-file",
    help="path to file containing texts to exclude from combined dataset",
    default="./exclude.jsonl",
)
parser.add_argument(
    "--output",
    help="path to output file, will be jsonl file",
    default="./output.jsonl",
)
parser.add_argument(
    "--strategy",
    help=(
        "strategy to use when combining datasets. "
        "Possible values: ['binary2multi', 'multi2binary']."
        "Defualt: 'binary2multi'"
    ),
    default="binary2multi",
)


def validate_file(file_path: str):
    if not os.path.isfile(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    if not file_path.endswith(".jsonl"):
        raise ValueError(f"File is not jsonl file: {file_path}")


def validate_output_file(file_path: str):
    if os.path.isfile(file_path):
        raise ValueError(f"Output file already exists: {file_path}")

    if not file_path.endswith(".jsonl"):
        raise ValueError(f"Output file is not jsonl file: {file_path}")


def validate_strategy(strategy: str):
    if strategy not in ["binary2multi", "multi2binary"]:
        raise ValueError(
            f"Invalid strategy {strategy}. "
            "Possible values: ['binary2multi', 'multi2binary']"
        )


def get_mappings_for_strategy(strategy: str) -> (str, Callable):
    if strategy == "binary2multi":
        return "model", map_model_to_label
    elif strategy == "multi2binary":
        return "label", map_model_label_to_binary_label
    else:
        raise ValueError(
            f"Invalid strategy {strategy}. "
            "Possible values: ['binary2multi', 'multi2binary']"
        )


def main():
    args = parser.parse_args()

    validate_file(args.file1)
    validate_file(args.file2)
    validate_file(args.exclude_file)
    validate_output_file(args.output)
    validate_strategy(args.strategy)

    label_column = "label"
    mapping_column, mapping_fn = get_mappings_for_strategy(args.strategy)
    df_combined = merge_jsonl_files(
        file_paths=[args.file1, args.file2],
        exclude_file_path=args.exclude_file,
        label_column=label_column,
        mapping_column=mapping_column,
        mapping_fn=mapping_fn,
        debug=True,
    )

    print(f"\nCombined dataset shape: {df_combined.shape}\n")

    pd_write_jsonl_file(df_combined, args.output)

    print(f"\n --- DONE: Combined datasets saved to: {args.output} ---\n")


main()
