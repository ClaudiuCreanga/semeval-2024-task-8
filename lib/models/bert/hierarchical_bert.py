import torch
import torch.nn as nn
from transformers import BertModel

from lib.utils.models import sequential_fully_connected
from lib.utils.constants import MergeLayersStrategy, PoolingStrategy


class HierarchicalBERT(nn.Module):
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

        self.do_layer_selection = (
            self.selected_layers is not None and len(self.selected_layers) > 0
        )

        self.bert = BertModel.from_pretrained(
            pretrained_model_name,
            return_dict=False,
            output_hidden_states=self.do_layer_selection,
        )

        input_size = self.bert.config.hidden_size
        if self.do_layer_selection:
            input_size = len(self.selected_layers) * self.bert.config.hidden_size

        self.drop_bert = nn.Dropout(dropout_p)
        if self.do_layer_selection and self.selected_layers_dropout_p is not None:
            self.selected_layers_dropout = nn.Dropout(
                self.selected_layers_dropout_p
            )

        self.out = sequential_fully_connected(input_size, out_size, fc, dropout_p)

        self.out_activation = None
        if out_activation is not None:
            self.out_activation = nn.Sigmoid()

        if self.do_layer_selection:
            self.freeze_transformer_layer()

    def forward(self, input_ids, attention_mask):
        # input_ids = [batch_size, number_of_chunks, max_seq_len]
        # attention_mask = [batch_size, number_of_chunks, max_seq_len]

        bert_output = []
        for batch_sample_input_ids, batch_sample_attention_mask in zip(
            input_ids, attention_mask
        ):
            # batch_sample_input_ids = [number_of_chunks, max_seq_len]
            # batch_sample_attention_mask = [number_of_chunks, max_seq_len]

            batch_sample_output = []
            for chunk_input_ids, chunk_attention_mask in zip(
                batch_sample_input_ids, batch_sample_attention_mask
            ):
                # chunk_input_ids = [max_seq_len]
                # chunk_attention_mask = [max_seq_len]

                chunk_input_ids = chunk_input_ids.unsqueeze(0)
                chunk_attention_mask = chunk_attention_mask.unsqueeze(0)

                # chunk_input_ids = [1, max_seq_len]
                # chunk_attention_mask = [1, max_seq_len]

                if self.do_layer_selection:
                    # Use the features from the selected layers
                    chunk_outputs = self.bert(
                        input_ids=chunk_input_ids, attention_mask=chunk_attention_mask
                    )
                    chunk_hidden_states = chunk_outputs[2]

                    if (
                        self.selected_layers_merge_strategy == MergeLayersStrategy.CONCATENATE  # noqa: E501
                    ):
                        chunk_output = torch.cat(
                            [
                                chunk_hidden_states[i][:, 0, :]
                                for i in self.selected_layers
                            ],
                            dim=1,
                        )
                    elif (
                        self.selected_layers_merge_strategy == MergeLayersStrategy.MEAN
                    ):
                        chunk_output = torch.mean(
                            torch.stack(
                                [
                                    chunk_hidden_states[i][:, 0, :]
                                    for i in self.selected_layers
                                ],
                                dim=1,
                            ),
                            dim=1,
                        )
                    elif (
                        self.selected_layers_merge_strategy == MergeLayersStrategy.MAX
                    ):
                        chunk_output = torch.max(
                            torch.stack(
                                [
                                    chunk_hidden_states[i][:, 0, :]
                                    for i in self.selected_layers
                                ],
                                dim=1,
                            ),
                            dim=1,
                        ).values
                    else:
                        raise NotImplementedError(
                            f"No such selected layers merge strategy: "
                            f"{self.selected_layers_merge_strategy}"
                        )
                else:
                    # Use the [CLS] token features from the last layer
                    _, chunk_output = self.bert(
                        input_ids=chunk_input_ids, attention_mask=chunk_attention_mask
                    )

                # chunk_output = [1, hidden_size]
                chunk_output = chunk_output.squeeze(0)
                # chunk_output = [hidden_size]

                if (
                    self.do_layer_selection
                    and self.selected_layers_dropout_p is not None  # noqa: W503
                ):
                    chunk_output = self.selected_layers_dropout(chunk_output)

                batch_sample_output.append(chunk_output)

            batch_sample_output = torch.stack(batch_sample_output)

            if self.pooling_strategy == PoolingStrategy.MEAN:
                batch_sample_output = torch.mean(batch_sample_output, dim=0)
            elif self.pooling_strategy == PoolingStrategy.MAX:
                batch_sample_output = torch.max(batch_sample_output, dim=0).values
            else:
                raise NotImplementedError(
                    f"No such pooling strategy: {self.pooling_strategy}"
                )

            bert_output.append(batch_sample_output)

        # bert_output = [batch_size, hidden_size]
        bert_output = torch.stack(bert_output)

        output = self.drop_bert(bert_output)

        output = self.out(output)

        if self.out_activation is not None:
            output = self.out_activation(output)

        return output

    def freeze_transformer_layer(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_transformer_layer(self):
        if self.do_layer_selection:
            # Fine-tune only selected layers
            for selected_layer in self.selected_layers:
                for param in self.bert.encoder.layer[selected_layer].parameters():
                    param.requires_grad = True

            # Fine-tune the pooler layer
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
        else:
            # Fine-tune the entire bert transformer
            for param in self.bert.parameters():
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
        return len(self.bert.encoder.layer)
