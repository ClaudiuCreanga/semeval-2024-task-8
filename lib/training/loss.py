import torch.nn as nn


def get_loss_fn(loss_name: str, device: str):
    if loss_name == "bce":
        return nn.BCELoss().to(device)
    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss().to(device)
    elif loss_name == "mse":
        return nn.MSELoss().to(device)
    else:
        raise NotImplementedError("No such loss function!")
