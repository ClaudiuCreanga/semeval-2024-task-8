import torch
import torch.nn as nn
from enum import Enum
from transformers import GPT2Model

from lib.utils.constants import MergeLayersStrategy
from lib.utils.models import sequential_fully_connected


class GPT2Type(Enum):
    GPT2 = "gpt2"
    GPT2_WITH_LAYER_SELECTION = "gpt2_with_layer_selection"


class GPT2(nn.Module):
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

        self.gpt2_type = GPT2Type.GPT2
        if self.selected_layers is not None and len(self.selected_layers) > 0:
            self.gpt2_type = GPT2Type.GPT2_WITH_LAYER_SELECTION

            self.selected_layers_dropout = None
            if selected_layers_dropout_p is not None:
                self.selected_layers_dropout = nn.Dropout(selected_layers_dropout_p)

        input_size = None
        if self.gpt2_type == GPT2Type.GPT2_WITH_LAYER_SELECTION:
            self.gpt2 = GPT2Model.from_pretrained(
                pretrained_model_name, return_dict=False, output_hidden_states=True
            )
            self.gpt2.config.pad_token_id = self.gpt2.config.eos_token_id

            if self.selected_layers_merge_strategy == MergeLayersStrategy.CONCATENATE:
                input_size = len(self.selected_layers) * self.gpt2.config.n_embd
            else:
                input_size = self.gpt2.config.n_embd
        else:
            self.gpt2 = GPT2Model.from_pretrained(
                pretrained_model_name, return_dict=False
            )
            self.gpt2.config.pad_token_id = self.gpt2.config.eos_token_id

            input_size = self.gpt2.config.n_embd

        self.out = sequential_fully_connected(input_size, out_size, fc, dropout_p)

        self.out_activation = None
        if out_activation == "sigmoid":
            self.out_activation = nn.Sigmoid()

        if self.gpt2_type == GPT2Type.GPT2_WITH_LAYER_SELECTION:
            self.freeze_transformer_layer()

    def forward(self, input_ids, attention_mask):
        if self.gpt2_type == GPT2Type.GPT2:
            outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = outputs[0]

            output = self.out(hidden_state)

            batch_size, sequence_length = input_ids.shape[:2]
            sequence_lengths = (
                torch.ne(input_ids, self.gpt2.config.pad_token_id).sum(-1) - 1
            )
            output = output[range(batch_size), sequence_lengths]
        elif self.gpt2_type == GPT2Type.GPT2_WITH_LAYER_SELECTION:

            outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs[2]

            # Compute the sequence length for each sample in the batch
            # in order to extract the features from the last token
            batch_size, sequence_length = input_ids.shape[:2]
            sequence_lengths = (
                torch.ne(input_ids, self.gpt2.config.pad_token_id).sum(-1) - 1
            )

            # print(f"sequence_lengths.shape: {sequence_lengths.shape}")
            # print(f"sequence_lengths: {sequence_lengths}")

            # print(f"hidden_states[-1].shape = {hidden_states[-1].shape}")
            # print(
            #     "hidden_states[-1][:, sequence_lengths, :].shape: "
            #     f"{hidden_states[-1][:, sequence_lengths, :].shape}"
            # )

            # The equivalent of [CLS] token used by BERT
            # is the last token in the sequence for GPT2
            if self.selected_layers_merge_strategy == MergeLayersStrategy.CONCATENATE:
                output = torch.cat(
                    [
                        hidden_states[i][range(batch_size), sequence_lengths, :]
                        for i in self.selected_layers
                    ],
                    dim=1,
                )
            elif self.selected_layers_merge_strategy == MergeLayersStrategy.MEAN:
                output = torch.mean(
                    torch.stack([
                        hidden_states[i][range(batch_size), sequence_lengths, :]
                        for i in self.selected_layers
                    ], dim=1),
                    dim=1,
                )
            elif self.selected_layers_merge_strategy == MergeLayersStrategy.MAX:
                output = torch.max(
                    torch.stack([
                        hidden_states[i][range(batch_size), sequence_lengths, :]
                        for i in self.selected_layers
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

            output = self.out(output)
        else:
            raise ValueError(f"Unknown GPT2 type: {self.gpt2_type}")

        if self.out_activation is not None:
            output = self.out_activation(output)

        return output

    def freeze_transformer_layer(self):
        for param in self.gpt2.parameters():
            param.requires_grad = False

    def unfreeze_transformer_layer(self):
        if self.gpt2_type == GPT2Type.GPT2_WITH_LAYER_SELECTION:
            # Unfreeze the selected layers for fine-tune
            for selected_layer in self.selected_layers:
                for param in self.gpt2.h[selected_layer].parameters():
                    param.requires_grad = True

            # The rest of the parameters are frozen
            return

        for param in self.gpt2.parameters():
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
        return len(self.gpt2.h)
