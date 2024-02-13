# flake8: noqa W503

import torch
import torch.nn as nn

from lib.models.base import BaseModelForTokenClassification
from lib.utils.models import sequential_fully_connected

try:
    from torchcrf import CRF
except ImportError:
    print("Warning: CRF module not found. Install it with: pip install torchcrf")


class BiLSTMCRFForTokenClassification(nn.Module, BaseModelForTokenClassification):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        out_size,
        dropout_p=0.3,
        n_layers=1,
        hidden_dim=32,
        fc=[],
    ):
        super().__init__()

        self.out_size = out_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = sequential_fully_connected(
            2 * hidden_dim, out_size, fc, dropout_p
        )

        self.crf = CRF(out_size, batch_first=True)

    def forward(self, input_ids, attention_mask, device, labels=None):
        embeddings = self.embedding(input_ids)

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

        mask = attention_mask.bool()

        loss = None
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=mask, reduction="mean")
            logits = self.crf.decode(logits, mask=mask)

            for i in range(len(logits)):
                if len(logits[i]) < len(labels[i]):
                    logits[i] = (
                        [-100]
                        + logits[i]
                        + [-100] * (len(labels[i]) - len(logits[i]) - 1)
                    )

            loss = 0 - log_likelihood
        else:
            logits = self.crf.decode(logits, mask=mask)
        logits = torch.Tensor(logits).to(device)

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
        preds = logits.clone()

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
