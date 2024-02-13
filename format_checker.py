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
parser.add_argument(
    "--organizer-repo-dir",
    help="path to organizer repository directory",
    default="./SemEval2024-task8",
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
    validate_dir(args.organizer_repo_dir)

    submission_files = get_submission_files(args.submission_dir)
    for submission_file in submission_files:
        subtask = None
        if "subtask_a" in submission_file:
            subtask = "subtaskA"
        elif "subtask_b" in submission_file:
            subtask = "subtaskB"
        elif "subtask_c" in submission_file:
            subtask = "subtaskC"
        else:
            raise ValueError(f"Unknown subtask for file {submission_file}")

        format_checker_script_path = os.path.join(
            args.organizer_repo_dir,
            subtask,
            "format_checker",
            "format_checker.py"
        )

        print(f"Running format checker for file {submission_file}...\n")

        try:
            subprocess.run(
                [
                    "python",
                    format_checker_script_path,
                    "--pred_files_path",
                    submission_file
                ],
                check=True,
            )
        except Exception as ex:
            print(
                f"Format checker failed for file: {submission_file} "
                f"with error: {ex}"
            )

        print()
        print("-" * 20)
        print()


main()
