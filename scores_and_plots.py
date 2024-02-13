import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from lib.utils.constants import Subtask
from lib.data.loading import read_predictions_from_file

parser = argparse.ArgumentParser(description="Machine-generated text detection tool")
parser.add_argument(
    "--results-dir",
    help="path to results directory containing the results",
)


def print_scores(df: pd.DataFrame, df_type: str, results_dir: str):
    if df_type == "validation":
        true = list(df.true)
        predict = list(df.predict)
    elif df_type == "test":
        true = list(df.true)
        predict = list(df.label)

        if -1 in true:
            print("Skip scoring for test. Missing true labels...")
            return
    else:
        raise NotImplementedError(f"No such df_type: {df_type}")

    if (
        Subtask.SubtaskA.value in results_dir or Subtask.SubtaskB.value in results_dir
    ):
        average = (
            "binary" if Subtask.SubtaskA.value in results_dir else "macro"
        )

        accuracy = metrics.accuracy_score(true, predict)
        precision = metrics.precision_score(true, predict, average=average)
        recall = metrics.recall_score(true, predict, average=average)
        f1 = metrics.f1_score(true, predict, average=average)

        print(f"Results on {df_type}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1: {f1 * 100:.2f}%")
        print("-" * 20)
    elif Subtask.SubtaskC.value in results_dir:
        print(f"Results on {df_type}")
        print(f"MAE: {metrics.mean_absolute_error(true, predict):.5f}")
        print("-" * 20)
    else:
        raise NotImplementedError(f"No such subtask in {results_dir}")


def main():
    args = parser.parse_args()

    dev_predictions_path = os.path.join(
        args.results_dir, "best_model_dev_predict.csv"
    )
    test_predictions_path = os.path.join(args.results_dir, "submission.csv")

    df_dev_predictions = read_predictions_from_file(dev_predictions_path)
    df_test_predictions = read_predictions_from_file(test_predictions_path)

    is_early_stopping = os.path.exists(
        os.path.join(args.results_dir, "early_stopping_best_model.bin")
    )

    for df_type, df in zip(
        ["validation", "test"],
        [df_dev_predictions, df_test_predictions]
    ):
        print_scores(df, df_type, args.results_dir)

        if df_type == "validation":
            df_hist = pd.read_csv(os.path.join(args.results_dir, "history.csv"))

            fig, ax = plt.subplots()
            ax.plot(df_hist.train_loss, label="train")
            ax.plot(df_hist.dev_loss, label="validation")

            if is_early_stopping:
                min_dev_loss = df_hist.dev_loss.min()
                min_dev_loss_epoch = df_hist[df_hist.dev_loss == min_dev_loss].index[0]

                ax.axvline(
                    x=min_dev_loss_epoch,
                    color="r",
                    linestyle="--",
                    label="early stopping checkpoint"
                )

            ax.set_title("model loss")
            ax.set_ylabel("loss")
            ax.set_xlabel("epoch")
            plt.legend(loc="lower left")
            fig.savefig(os.path.join(args.results_dir, "loss_plot.png"))

            fig, ax = plt.subplots()
            ax.plot(df_hist.train_metric, label="train")
            ax.plot(df_hist.dev_metric, label="validation")
            ax.set_title(
                "model " + (
                    "mae" if Subtask.SubtaskC.value in args.results_dir else "accuracy"
                )
            )
            ax.set_ylabel(
                "mae" if Subtask.SubtaskC.value in args.results_dir else "accuracy"
            )
            ax.set_xlabel("epoch")
            plt.legend(loc="lower right")
            fig.savefig(os.path.join(args.results_dir, "metric_plot.png"))


if __name__ == "__main__":
    main()
