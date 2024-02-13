# flake8: noqa W503

import torch
import torch.nn as nn
from transformers import LongformerModel

from lib.models.base import BaseModelForTokenClassification
from lib.utils.models import sequential_fully_connected

try:
    from torchcrf import CRF
except ImportError:
    print("torchcrf not installed, CRF will not be used")


class LongformerCRFForTokenClassification(nn.Module, BaseModelForTokenClassification):
    def __init__(
        self,
        pretrained_model_name: str,
        out_size: int,
        dropout_p: float = 0.5,
        fc: [int] = [],
        finetune_last_transformer_layers: int | None = None,
        crf_reduction: str = "mean",
    ):
        super().__init__()

        self.out_size = out_size
        self.finetune_last_transformer_layers = finetune_last_transformer_layers
        self.crf_reduction = crf_reduction

        self.longformer = LongformerModel.from_pretrained(
            pretrained_model_name, return_dict=False
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = sequential_fully_connected(
            self.longformer.config.hidden_size, out_size, fc, dropout_p
        )

        self.crf = CRF(out_size, batch_first=True)

        self.freeze_transformer_layer()

    def forward(self, input_ids, attention_mask, device, labels=None):
        sequence_output, _ = self.longformer(
            input_ids=input_ids, attention_mask=attention_mask
        )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        mask = attention_mask.bool()

        loss = None
        if labels is not None:
            # loss_fn = nn.CrossEntropyLoss().to(device)
            # loss = loss_fn(logits.view(-1, self.out_size), labels.view(-1))
            log_likelihood = self.crf(
                logits,
                labels,
                mask=mask,
                reduction=self.crf_reduction
            )
            logits = self.crf.decode(logits, mask=mask)

            for i in range(len(logits)):
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
        for param in self.longformer.parameters():
            param.requires_grad = False

    def unfreeze_transformer_layer(self):
        if self.finetune_last_transformer_layers is not None:
            # Fine-tune only the last selected layer
            selected_layers = self.longformer.encoder.layer[
                -self.finetune_last_transformer_layers :
            ]
            for layer in selected_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            # No fine-tuning
            pass

    def get_predictions_from_logits(self, logits, labels=None, corresponding_word=None):
        # logits: (batch_size, max_seq_len)
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
