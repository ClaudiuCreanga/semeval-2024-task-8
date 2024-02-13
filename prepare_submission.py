import os
import argparse
import numpy as np

from lib.data.loading import read_predictions_from_file, pd_write_jsonl_file

parser = argparse.ArgumentParser(description="Machine-generated text detection tool")
parser.add_argument(
    "--submission-file",
    help="path to submission file",
    default="./submission.csv"
)
parser.add_argument(
    "--subtask",
    help="the subtask for which to prepare the submission file",
    default="A"
)
parser.add_argument(
    "--track",
    help="the track of the subtask A: monolongual or multilingual",
    default=None,
)
parser.add_argument(
    "--output-dir",
    help="the output directory where to save the submission file",
    default="./",
)


def validate_submission_file(file_path: str):
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")


def validate_output_dir(output_dir: str):
    if not os.path.exists(output_dir):
        print(f"Creating output directory {output_dir}...")
        os.makedirs(output_dir)
    else:
        print(f"Output directory {output_dir} already exists")


def main():
    args = parser.parse_args()

    validate_submission_file(args.submission_file)
    validate_output_dir(args.output_dir)

    print(
        f"Reading predictions from file {args.submission_file} | "
        f"Saving submission file to {args.output_dir}..."
    )

    df = read_predictions_from_file(args.submission_file)
    df = df[["id", "label"]]

    if isinstance(df["id"][0], str):
        if args.subtask != "C":
            print("`id` is of type str, converting to int")

            df["id"] = df["id"].map(
                lambda x: int(x.split("(")[1][:-1])
            )
        else:
            print("`id` is of type str, keeping it as str for subtask C")
    elif isinstance(df["id"][0], int):
        print("`id` is already of type int")
    elif isinstance(df["id"][0], np.int64):
        print("`id` is of type np.int64, converting to int")

        df["id"] = df["id"].astype(int)
    elif isinstance(df["id"][0], float):
        print("`id` is of type float, converting to int")

        df["id"] = df["id"].astype(int)
    else:
        raise ValueError(
            f"Unknown type for `id`: {type(df['id'][0])}"
        )

    if args.subtask == "A":
        if args.track is None:
            raise ValueError("For subtask A, the track must be specified")

        selected_track = args.track.lower()
        possible_tracks = ["monolingual", "multilingual"]
        if selected_track not in possible_tracks:
            raise ValueError(
                f"Unknown track for subtask A: `{args.track}`. "
                f"Possible tracks: {possible_tracks}."
            )

        df["label"] = df["label"].astype(int)
        output_file = f"subtask_a_{selected_track}.jsonl"
    elif args.subtask == "B":
        df["label"] = df["label"].astype(int)
        output_file = "subtask_b.jsonl"
    elif args.subtask == "C":
        df["label"] = df["label"].astype(float)
        output_file = "subtask_c.jsonl"
    else:
        raise ValueError(f"Unknown subtask: {args.subtask}")

    output_file_path = f"{args.output_dir}/{output_file}"
    pd_write_jsonl_file(df, output_file_path)

    print("Done!")


if __name__ == "__main__":
    main()
