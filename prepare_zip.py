import os
import argparse
import subprocess
from glob import glob

parser = argparse.ArgumentParser(description="Machine-generated text detection tool")
parser.add_argument(
    "--submission-dir",
    help="path to submission directory containing the jsonl files for submission",
    default="./submission",
)


def validate_dir(dir_path: str):
    if not os.path.exists(dir_path):
        raise ValueError(
            f"Directory {dir_path} does not exist"
        )


def get_submission_files(submission_dir: str):
    return glob(os.path.join(submission_dir, "*.jsonl"))


def main():
    args = parser.parse_args()

    validate_dir(args.submission_dir)

    submission_files = get_submission_files(args.submission_dir)

    if len(submission_files) == 0:
        raise ValueError(
            f"No submission files found in {args.submission_dir}"
        )

    try:
        subprocess.run(
            [
                "zip",
                f"{args.submission_dir}/predictions.zip",
                *submission_files,
            ]
        )
    except Exception as ex:
        print(f"Failed to zip submission files with error: {ex}")

    print(f"Zip saved to {args.submission_dir}/predictions.zip")


main()
