import torch
import torch.nn as nn

from lib.utils.models import sequential_fully_connected
from lib.models.base import BaseModelForTokenClassification


class CharacterLevelCNNEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        kernel_size: int,
        max_len: int,
        out_size: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self._output_dim = None

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=1,
        )

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=out_size,
            kernel_size=kernel_size,
        )

        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

        self._init_embedding_weights()
        self._compute_output_dim()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input_ids, attention_mask, labels=None):
        # input_ids: (batch_size, max_seq_len, max_char_len)

        outputs = []
        for i in range(input_ids.shape[0]):
            # input_ids[i] shape: (max_seq_len, max_char_len)

            batch_output = self.embedding(input_ids[i])
            # print(f"batch_output.shape: {batch_output.shape}")

            batch_output = batch_output.permute(0, 2, 1)
            # print(f"batch_output.shape: {batch_output.shape}")

            batch_output = self.conv(batch_output)
            # print(f"batch_output.shape: {batch_output.shape}")

            batch_output = self.max_pool(batch_output)
            # print(f"batch_output.shape: {batch_output.shape}")

            batch_output = batch_output.reshape(batch_output.shape[0], -1)
            # print(f"batch_output.shape: {batch_output.shape}")

            outputs.append(batch_output)

        outputs = torch.stack(outputs)
        # print(f"outputs_out = outputs.shape: {outputs.shape}")

        # outputs.shape: (batch_size, max_seq_len, self.output_dim)

        return outputs

    def _init_embedding_weights(self):
        self.embedding.weight.data = self.embedding.weight.data.uniform_(
            -0.5, 0.5
        )

    def _compute_output_dim(self) -> int:
        x = torch.randint(low=0, high=self.vocab_size, size=(1, self.max_len))

        output = self.embedding(x)
        output = output.permute(0, 2, 1)
        output = self.conv(output)
        output = self.max_pool(output)

        # flatten
        output = output.reshape(1, -1)

        self._output_dim = output.shape[1]


class CharacterAndWordLevelEmbeddingsWithBiLSTMForTokenClassification(
    nn.Module,
    BaseModelForTokenClassification,
):
    def __init__(
        self,
        char_vocab_size: int,
        char_embedding_dim: int,
        char_kernel_size: int,
        char_max_len: int,
        char_out_size: int,
        word_vocab_size: int,
        word_embedding_dim: int,
        out_size: int,
        dropout_p: float = 0.3,
        n_layers: int = 1,
        hidden_dim: int = 32,
        fc: [int] = [],
    ):
        super().__init__()

        self.out_size = out_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.character_embeddings = CharacterLevelCNNEmbedding(
            vocab_size=char_vocab_size,
            embedding_dim=char_embedding_dim,
            kernel_size=char_kernel_size,
            max_len=char_max_len,
            out_size=char_out_size,
        )

        self.word_embeddings = nn.Embedding(
            num_embeddings=word_vocab_size,
            embedding_dim=word_embedding_dim,
            padding_idx=1,
        )

        self.lstm = nn.LSTM(
            self.character_embeddings.output_dim + word_embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = sequential_fully_connected(
            2 * hidden_dim, out_size, fc, dropout_p
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        char_input_ids,
        char_attention_mask,
        device,
        labels=None
    ):
        # input_ids: (batch_size, max_seq_len)
        # print(f"input_ids.shape: {input_ids.shape}")

        # char_embeddings.shape:
        # (batch_size, max_seq_len, self.character_embeddings.output_dim)
        char_embeddings = self.character_embeddings(
            char_input_ids, char_attention_mask,
        )
        # print(f"char_embeddings.shape: {char_embeddings.shape}")

        # word_embeddings.shape: (batch_size, max_seq_len, word_embedding_dim)
        word_embeddings = self.word_embeddings(input_ids)
        # print(f"word_embeddings.shape: {word_embeddings.shape}")

        # embeddings.shape:
        # (
        #   batch_size,
        #   max_seq_len,
        #   self.character_embeddings.output_dim + word_embedding_dim,
        # )
        embeddings = torch.cat([char_embeddings, word_embeddings], dim=2)
        # print(f"embeddings.shape: {embeddings.shape}")

        # lengths.shape: (batch_size)
        lengths = attention_mask.sum(dim=1)

        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # print(f"packed_embeddings.data.shape: {packed_embeddings.data.shape}")

        packed_output, (_, _) = self.lstm(packed_embeddings)
        # print(f"packed_output.data.shape: {packed_output.data.shape}")

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=embeddings.shape[1],
        )
        # print(f"output.shape: {output.shape}")

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

    def _init_embedding_weights(self):
        self.word_embeddings.weight.data = self.word_embeddings.weight.data.uniform_(
            -0.5, 0.5
        )
