import torch
import torch.nn as nn
from transformers import AlbertModel

from lib.utils.models import sequential_fully_connected


class ALBERT(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        out_size: int = 1,
        dropout_p: float = 0.5,
        fc: [int] = [],
        out_activation: str | None = None,
    ):
        super().__init__()
        self.albert = AlbertModel.from_pretrained(
            pretrained_model_name, return_dict=False
        )
        self.drop_albert = nn.Dropout(dropout_p)
        self.out = sequential_fully_connected(
            self.albert.config.hidden_size, out_size, fc, dropout_p
        )

        self.out_activation = None
        if out_activation == "sigmoid":
            self.out_activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.albert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        output = self.drop_albert(pooled_output)
        output = self.out(output)

        if self.out_activation is not None:
            output = self.out_activation(output)

        return output

    def freeze_transformer_layer(self):
        for param in self.albert.parameters():
            param.requires_grad = False

    def unfreeze_transformer_layer(self):
        for param in self.albert.parameters():
            param.requires_grad = True

    def get_predictions_from_outputs(self, outputs):
        if self.out_activation is None:
            if self.out_size == 1:
                return [int(output) for output in outputs.flatten().tolist()]
            else:
                return torch.argmax(outputs, dim=1).flatten().tolist()
        else:
            return torch.round(outputs).flatten().tolist()

    def get_hidden_layers_count(self) -> int:
        return len(self.albert.encoder.albert_layer_groups[0].albert_layers)
