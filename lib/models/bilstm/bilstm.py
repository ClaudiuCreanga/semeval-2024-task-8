import torch
import torch.nn as nn

from lib.models.base import BaseModelForTokenClassification
from lib.utils.models import sequential_fully_connected


class BiLSTMForTokenClassification(nn.Module, BaseModelForTokenClassification):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        out_size: int,
        dropout_p: float = 0.3,
        embedding_dropout_p: float | None = None,
        n_layers: int = 1,
        hidden_dim: int = 32,
        lstm_dropout_p: float = 0,
        fc: [int] = [],
    ):
        super().__init__()

        self.out_size = out_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.embedding_dropout = None
        if embedding_dropout_p is not None:
            self.embedding_dropout = nn.Dropout(p=embedding_dropout_p)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=lstm_dropout_p,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = sequential_fully_connected(
            2 * hidden_dim, out_size, fc, dropout_p
        )

    def forward(self, input_ids, attention_mask, device, labels=None):
        embeddings = self.embedding(input_ids)

        if self.embedding_dropout is not None:
            embeddings = self.embedding_dropout(embeddings)

        lengths = attention_mask.sum(dim=1)

        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, (_, _) = self.lstm(packed_embeddings)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=embeddings.shape[1],
        )

        output = self.dropout(output)
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss().to(device)
            loss = loss_fn(logits.view(-1, self.out_size), labels.view(-1))

        return loss, logits

    def freeze_transformer_layer(self):
        pass

    def unfreeze_transformer_layer(self):
        pass

    def get_predictions_from_logits(self, logits, labels=None, corresponding_word=None):
        # logits: (batch_size, max_seq_len, out_size)
        # labels: (batch_size, max_seq_len)
        # corresponding_word: (batch_size, max_seq_len)

        # preds: (batch_size, max_seq_len)
        preds = torch.argmax(logits, dim=-1)

        if labels is not None:
            predicted_positions = []
            true_positions = []
            for p, l in zip(preds, labels):
                mask = l != -100

                clean_pred = p[mask]
                clean_label = l[mask]

                predicted_position = clean_pred.argmax(dim=-1)
                true_position = clean_label.argmax(dim=-1)

                predicted_positions.append(predicted_position.item())
                true_positions.append(true_position.item())

            return torch.Tensor(predicted_positions), torch.Tensor(true_positions)
        elif corresponding_word is not None:
            predicted_positions = []
            for p, w in zip(preds, corresponding_word):
                mask = w != -100

                clean_pred = p[mask]
                clean_corresponding_word = w[mask]

                # Get the index of the first machine text word
                index = torch.where(clean_pred == 1)[0]
                value = index[0] if index.size else len(clean_pred) - 1
                position = clean_corresponding_word[value]

                predicted_positions.append(position.item())

            return predicted_positions, None
        else:
            raise ValueError("Either labels or corresponding_word must be provided")
