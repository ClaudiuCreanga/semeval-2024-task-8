import torch
import torch.nn as nn
from enum import Enum
from transformers import RobertaModel

from lib.utils.constants import MergeLayersStrategy
from lib.utils.models import sequential_fully_connected


class RoBERTaType(Enum):
    ROBERTA = "roberta"
    ROBERTA_WITH_LAYER_SELECTION = "roberta_with_layer_selection"


class RoBERTa(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        out_size: int = 1,
        dropout_p: float = 0.5,
        selected_layers: [int] = None,
        selected_layers_merge_strategy: str | None = None,
        selected_layers_dropout_p: float | None = None,
        fc: [int] = [],
        out_activation: str | None = None,
    ):
        super().__init__()

        self.out_size = out_size
        self.selected_layers = selected_layers
        self.selected_layers_merge_strategy = (
            MergeLayersStrategy(selected_layers_merge_strategy)
            if selected_layers_merge_strategy is not None
            else None
        )
        self.selected_layers_dropout_p = selected_layers_dropout_p

        self.roberta_type = RoBERTaType.ROBERTA
        if self.selected_layers is not None and len(self.selected_layers) > 0:
            self.roberta_type = RoBERTaType.ROBERTA_WITH_LAYER_SELECTION

            self.selected_layers_dropout = None
            if selected_layers_dropout_p is not None:
                self.selected_layers_dropout = nn.Dropout(selected_layers_dropout_p)

        input_size = None
        if self.roberta_type == RoBERTaType.ROBERTA_WITH_LAYER_SELECTION:
            self.roberta = RobertaModel.from_pretrained(
                pretrained_model_name, return_dict=False, output_hidden_states=True
            )

            if self.selected_layers_merge_strategy == MergeLayersStrategy.CONCATENATE:
                input_size = (
                    len(self.selected_layers) * self.roberta.config.hidden_size
                )
            else:
                input_size = self.roberta.config.hidden_size
        else:
            self.roberta = RobertaModel.from_pretrained(
                pretrained_model_name, return_dict=False
            )
            input_size = self.roberta.config.hidden_size

        self.drop_roberta = nn.Dropout(dropout_p)
        self.out = sequential_fully_connected(input_size, out_size, fc, dropout_p)

        self.out_activation = None
        if out_activation == "sigmoid":
            self.out_activation = nn.Sigmoid()

        if self.roberta_type == RoBERTaType.ROBERTA_WITH_LAYER_SELECTION:
            self.freeze_transformer_layer()

    def forward(self, input_ids, attention_mask):
        if self.roberta_type == RoBERTaType.ROBERTA:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = outputs[0]
            pooled_output = hidden_state[:, 0]

            output = self.drop_roberta(pooled_output)
        elif self.roberta_type == RoBERTaType.ROBERTA_WITH_LAYER_SELECTION:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs[2]

            if self.selected_layers_merge_strategy == MergeLayersStrategy.CONCATENATE:
                output = torch.cat(
                    [hidden_states[i][:, 0, :] for i in self.selected_layers],
                    dim=1,
                )
            elif self.selected_layers_merge_strategy == MergeLayersStrategy.MEAN:
                output = torch.mean(
                    torch.stack([
                        hidden_states[i][:, 0, :] for i in self.selected_layers
                    ], dim=1),
                    dim=1,
                )
            elif self.selected_layers_merge_strategy == MergeLayersStrategy.MAX:
                output = torch.max(
                    torch.stack([
                        hidden_states[i][:, 0, :] for i in self.selected_layers
                    ], dim=1),
                    dim=1,
                ).values
            else:
                raise NotImplementedError(
                    f"No such selected layers merge strategy: "
                    f"{self.selected_layers_merge_strategy}"
                )

            if self.selected_layers_dropout is not None:
                output = self.selected_layers_dropout(output)
        else:
            raise NotImplementedError(f"No such RoBERTa type: {self.roberta_type}")

        output = self.out(output)

        if self.out_activation is not None:
            output = self.out_activation(output)

        return output

    def freeze_transformer_layer(self):
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_transformer_layer(self):
        if self.roberta_type == RoBERTaType.ROBERTA_WITH_LAYER_SELECTION:
            # Unfreeze the selected layers for fine-tune
            for selected_layer in self.selected_layers:
                for param in self.roberta.encoder.layer[selected_layer].parameters():
                    param.requires_grad = True

            # Unfreeze the pooler layer for fine-tune
            for param in self.roberta.pooler.parameters():
                param.requires_grad = True

            # The rest of the parameters are frozen
            return

        for param in self.roberta.parameters():
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
        return len(self.roberta.encoder.layer)
