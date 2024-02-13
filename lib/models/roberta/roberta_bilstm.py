import torch
import torch.nn as nn
from transformers import RobertaModel

from lib.utils.models import sequential_fully_connected


class RoBERTaBiLSTM(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        out_size: int = 1,
        dropout_p: float = 0.5,
        last_layers_emb: int = 4,
        hidden_dim: int = 600,
        fc: [int] = [1024],
        out_activation: str | None = None,
        fine_tune_last_layers_emb: bool = False,
    ):
        super().__init__()
        self.out_size = out_size
        self.fine_tune_last_layers_emb = fine_tune_last_layers_emb

        self.roberta = RobertaModel.from_pretrained(
            pretrained_model_name, return_dict=False, output_hidden_states=True
        )
        embedding_dim = last_layers_emb * self.roberta.config.hidden_size
        self.last_layers_emb = last_layers_emb
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.drop_lstm = nn.Dropout(dropout_p)
        self.out = sequential_fully_connected(2 * hidden_dim, out_size, fc, dropout_p)

        self.out_activation = None
        if out_activation == "sigmoid":
            self.out_activation = nn.Sigmoid()

        self.freeze_transformer_layer()

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs[2][-self.last_layers_emb :]
        embeddings = torch.cat(embeddings, dim=2)

        lengths = attention_mask.sum(dim=1)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed_embeddings)
        hidden = self.drop_lstm(
            torch.cat(
                (hidden[0, :, :], hidden[1, :, :]),
                dim=1
            )
        )

        output = self.out(hidden)
        if self.out_activation is not None:
            output = self.out_activation(output)

        return output

    def freeze_transformer_layer(self):
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_transformer_layer(self):
        if self.fine_tune_last_layers_emb:
            last_layers = self.roberta.encoder.layer[-self.last_layers_emb :]
            for param in last_layers.parameters():
                param.requires_grad = True
        else:
            # do nothing, RoBERTa model used only for embeddings
            pass

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
