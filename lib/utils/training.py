import torch
import numpy as np
from typing import Callable


class EarlyStopping:
    def __init__(
        self,
        patience: int = 2,
        delta: float = 0,
        path: str | None = ".",
        verbose: bool = False,
        trace_func: Callable = print,
    ):
        self.patience = patience
        self.delta = delta
        self.path = None
        self.verbose = verbose
        self.trace_func = trace_func

        if path is not None:
            self.path = f"{path}/early_stopping_best_model.bin"

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss: float, model: torch.nn.Module):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping: counter = {self.counter} out of {self.patience}"
            )

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        if self.path is None:
            if self.verbose:
                self.trace_func(
                    "Validation loss decreased "
                    f"({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                    "Skip saving model, no path provided."
                )
            return

        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} "
                f"--> {val_loss:.6f}). Saving model to {self.path}..."
            )

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
