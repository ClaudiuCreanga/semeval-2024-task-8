from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
import sklearn.metrics as metrics


def get_metric(metric_name: str):
    if metric_name == "accuracy":
        return metrics.accuracy_score, lambda x, y: x > y
    elif metric_name == "torchmetrics_binary_accuracy":
        return BinaryAccuracy(), lambda x, y: x > y
    elif metric_name == "torchmetrics_multiclass_accuracy":
        return MulticlassAccuracy(num_classes=6, average="macro"), lambda x, y: x > y
    elif metric_name == "mae":
        return metrics.mean_absolute_error, lambda x, y: x < y
    else:
        raise NotImplementedError("No such metric!")
