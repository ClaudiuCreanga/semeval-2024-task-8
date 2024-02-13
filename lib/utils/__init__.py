import torch
from datetime import datetime
from lib.utils.constants import DATETIME_FORMAT


def elapsed_time(start_time: float, end_time: float) -> (float, str):
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    return (
        elapsed_time,
        f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
    )


def get_current_date(datetime_format: str = DATETIME_FORMAT) -> str:
    return datetime.now().strftime(datetime_format)


def is_mps_available() -> bool:
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device() -> str:
    return "mps" if is_mps_available() else "cuda" if is_cuda_available() else "cpu"
