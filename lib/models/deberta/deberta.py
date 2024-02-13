import torch
import torch.nn as nn
from enum import Enum
from transformers import DebertaV2ForSequenceClassification

from lib.utils.constants import MergeLayersStrategy, PoolingStrategy
from lib.utils.models import sequential_fully_connected


class DebertaType(Enum):
    DEBERTA = "deberta"
    DEBERTA_WITH_LAYER_SELECTION = "deberta_with_layer_selection"
    HIERARCHICAL_BERT_WITH_POOLING = "hierarchical_t5flan_with_pooling"

    
class Deberta(nn.Module):
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
        reinit_last_n_layers: int = 0,
        pooling_strategy: str | None = None,
        device: str | None = None,
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
        self.reinit_last_n_layers = reinit_last_n_layers
        self.pooling_strategy = (
            PoolingStrategy(pooling_strategy) if pooling_strategy is not None else None
        )
        self.device = device

        self.selected_layers_dropout = None
        if selected_layers_dropout_p is not None:
            self.selected_layers_dropout = nn.Dropout(selected_layers_dropout_p)

        self.deberta = DebertaV2ForSequenceClassification.from_pretrained(
            pretrained_model_name, return_dict=False, output_hidden_states=True, num_labels=self.out_size
        )

        input_size = (
                len(self.selected_layers) * self.deberta.config.hidden_size
        )

        if self.device is not None:
            self.deberta = self.deberta.to(self.device)

        self.drop_bert = nn.Dropout(dropout_p)
        self.out = sequential_fully_connected(input_size, out_size, fc, dropout_p)

        self.out_activation = None
        # if out_activation == "sigmoid":
        #     self.out_activation = nn.Sigmoid()

        if reinit_last_n_layers > 0:
            self._do_reinit()

        self.freeze_transformer_layer()

    def _do_reinit(self):
        self.deberta.pooler.dense.weight.data.normal_(
            mean=0.0, std=self.deberta.config.initializer_range
        )
        self.deberta.pooler.dense.bias.data.zero_()

        for param in self.deberta.pooler.parameters():
            param.requires_grad = True

        for n in range(self.reinit_last_n_layers):
            self.deberta.transformer.layer[-(n + 1)].apply(self._init_weights_and_biases)

    def _init_weights_and_biases(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.deberta.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):

        outputs = self.deberta(input_ids, attention_mask)
        hidden_states = outputs[1]
        output = torch.cat(
            [hidden_states[i][:, 0, :] for i in self.selected_layers],
            dim=1,
        )

        if self.selected_layers_dropout is not None:
            output = self.selected_layers_dropout(output)


        output = self.out(output)

        if self.out_activation is not None:
            output = self.out_activation(output)

        return output
    def freeze_transformer_layer(self):
        for param in self.deberta.parameters():
            param.requires_grad = False

    def unfreeze_transformer_layer(self):

        for selected_layer in self.selected_layers:
            for param in self.deberta.deberta.encoder.layer[selected_layer].parameters():
                param.requires_grad = True

        # Unfreeze the pooler layer for fine-tune
        # for param in self.t5flan.pooler.parameters():
        #     param.requires_grad = True


        # The rest of the parameters are frozen
        return

        for param in self.deberta.parameters():
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
        return len(self.deberta.deberta.encoder.layer)


