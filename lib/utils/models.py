import torch.nn as nn
from collections import OrderedDict


def sequential_fully_connected(
    input_size: int,
    output_size: int,
    hidden: [int],
    dropout_p: float,
) -> nn.Sequential:
    hidden = [input_size] + hidden

    lst_layers = []
    for i in range(1, len(hidden)):
        lst_layers.append((f"linear_{i}", nn.Linear(hidden[i - 1], hidden[i])))
        lst_layers.append((f"layer_norm_{i}", nn.LayerNorm(hidden[i])))
        lst_layers.append((f"tanh_{i}", nn.Tanh()))
        lst_layers.append((f"dropout_{i}", nn.Dropout(dropout_p)))

    lst_layers.append(("linear_out", nn.Linear(hidden[-1], output_size)))

    od_layers = OrderedDict(lst_layers)

    return nn.Sequential(od_layers)
