import torch
import torch.nn as nn
from enum import Enum
try:
    from transformers import T5ForSequenceClassification
except ImportError:
    print("[WARNING]: Cannot import T5ForSequenceClassification from transformers")

from lib.utils.constants import MergeLayersStrategy, PoolingStrategy
from lib.utils.models import sequential_fully_connected


class T5FlanType(Enum):
    T5FLAN = "t5flan"
    T5FLAN_WITH_LAYER_SELECTION = "t5flan_with_layer_selection"
    HIERARCHICAL_BERT_WITH_POOLING = "hierarchical_t5flan_with_pooling"

    
class T5Flan(nn.Module):
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

        self.t5flan_type = T5FlanType.T5FLAN

        if self.selected_layers is not None and len(self.selected_layers) > 0:
            self.t5flan_type = T5FlanType.T5FLAN_WITH_LAYER_SELECTION

            self.selected_layers_dropout = None
            if selected_layers_dropout_p is not None:
                self.selected_layers_dropout = nn.Dropout(selected_layers_dropout_p)
        elif self.pooling_strategy is not None:
            self.t5flan_type = T5FlanType.HIERARCHICAL_BERT_WITH_POOLING

        if self.t5flan_type == T5FlanType.T5FLAN_WITH_LAYER_SELECTION:
            self.t5flan = T5ForSequenceClassification.from_pretrained(
                pretrained_model_name, return_dict=False, output_hidden_states=True
            )

            if self.selected_layers_merge_strategy == MergeLayersStrategy.CONCATENATE:
                input_size = (
                    len(self.selected_layers) * self.t5flan.config.hidden_size
                )
            else:
                input_size = self.t5flan.config.hidden_size
        else:
            self.t5flan = T5ForSequenceClassification.from_pretrained(
                pretrained_model_name, return_dict=False
            )
            input_size = self.t5flan.config.hidden_size

        if self.device is not None:
            self.t5flan = self.t5flan.to(self.device)

        self.drop_bert = nn.Dropout(dropout_p)
        self.out = sequential_fully_connected(input_size, out_size, fc, dropout_p)

        self.out_activation = None
        if out_activation == "sigmoid":
            self.out_activation = nn.Sigmoid()

        if reinit_last_n_layers > 0:
            self._do_reinit()

        if self.t5flan_type == T5FlanType.T5FLAN_WITH_LAYER_SELECTION:
            self.freeze_transformer_layer()

    def _do_reinit(self):
        self.t5flan.pooler.dense.weight.data.normal_(
            mean=0.0, std=self.t5flan.config.initializer_range
        )
        self.t5flan.pooler.dense.bias.data.zero_()

        for param in self.t5flan.pooler.parameters():
            param.requires_grad = True

        for n in range(self.reinit_last_n_layers):
            self.t5flan.transformer.layer[-(n + 1)].apply(self._init_weights_and_biases)

    def _init_weights_and_biases(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.t5flan.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        if self.t5flan_type == T5FlanType.T5FLAN:
            _, pooled_output = self.t5flan(
                input_ids=input_ids, attention_mask=attention_mask
            )
            output = self.drop_bert(pooled_output)
        elif self.t5flan_type == T5FlanType.T5FLAN_WITH_LAYER_SELECTION:
            outputs = self.t5flan(input_ids, attention_mask)
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
        elif self.t5flan_type == T5FlanType.HIERARCHICAL_BERT_WITH_POOLING:
            # input_ids = [batch_size, number_of_chunks, max_seq_len]
            # attention_mask = [batch_size, number_of_chunks, max_seq_len]

            pooled_output = []
            for batch_sample_input_ids, batch_sample_attention_mask in zip(
                input_ids, attention_mask
            ):
                # batch_sample_input_ids = [number_of_chunks, max_seq_len]
                # batch_sample_attention_mask = [number_of_chunks, max_seq_len]

                batch_sample_pooled_output = []
                for chunk_input_ids, chunk_attention_mask in zip(
                    batch_sample_input_ids, batch_sample_attention_mask
                ):
                    # chunk_input_ids = [max_seq_len]
                    # chunk_attention_mask = [max_seq_len]

                    chunk_input_ids = chunk_input_ids.unsqueeze(0).to(self.device)
                    chunk_attention_mask = chunk_attention_mask.unsqueeze(0).to(
                        self.device
                    )

                    # chunk_input_ids = [1, max_seq_len]
                    # chunk_attention_mask = [1, max_seq_len]
                    _, chunk_pooled_output = self.t5flan(
                        input_ids=chunk_input_ids, attention_mask=chunk_attention_mask
                    )

                    # chunk_pooled_output = [1, hidden_size]
                    batch_sample_pooled_output.append(
                        chunk_pooled_output.squeeze(0)
                    )  # => [hidden_size]

                batch_sample_pooled_output = torch.stack(batch_sample_pooled_output)
                if self.pooling_strategy == PoolingStrategy.MEAN:
                    batch_sample_pooled_output = torch.mean(
                        batch_sample_pooled_output, dim=0
                    )
                elif self.pooling_strategy == PoolingStrategy.MAX:
                    batch_sample_pooled_output = torch.max(
                        batch_sample_pooled_output, dim=0
                    ).values
                else:
                    raise NotImplementedError(
                        f"No such pooling strategy: {self.pooling_strategy}"
                    )

                pooled_output.append(batch_sample_pooled_output)

            # pooled_output = [batch_size, hidden_size]
            pooled_output = torch.stack(pooled_output)

            output = self.drop_bert(pooled_output)
        else:
            raise NotImplementedError(f"No such BERT type: {self.t5flan_type}")

        output = self.out(output)

        if self.out_activation is not None:
            output = self.out_activation(output)

        return output
    def freeze_transformer_layer(self):
        for param in self.t5flan.parameters():
            param.requires_grad = False

    def unfreeze_transformer_layer(self):
        if self.t5flan_type == T5FlanType.T5FLAN_WITH_LAYER_SELECTION:
            # Unfreeze the selected layers for fine-tune
            for selected_layer in self.selected_layers:
                for param in self.t5flan.transformer.encoder.block[selected_layer].parameters():
                    param.requires_grad = True

            # Unfreeze the pooler layer for fine-tune
            # for param in self.t5flan.pooler.parameters():
            #     param.requires_grad = True

            # The rest of the parameters are frozen
            return

        for param in self.t5flan.parameters():
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
        return len(self.t5flan.transformer.encoder.block)


