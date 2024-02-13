import torch
import torch.nn as nn
from time import sleep
from lightning import Fabric
from transformers import PreTrainedTokenizer

from lib.data.dataset import UtilityDataset
from lib.models.base import BaseModelForTokenClassification
from lib.models.bert.bert import BERT, BertType
from lib.models.bert.hierarchical_bert import HierarchicalBERT
from lib.training.optimizer import (
    build_adamw_llrd_optimizer, build_bert_base_adamw_grouped_llrd_optimizer,
)
from lib.utils.constants import MODEL_2_LABEL


def map_model_label_to_binary_label(model_label: int) -> int:
    # Model: 0 -> Label: 0
    # Model: 1, 2, 3, 4, 5 -> Label: 1
    return int(model_label >= 1)


def map_model_to_label(model: str) -> int:
    return MODEL_2_LABEL[model]


def transformer_model_get_max_batch_size(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: str,
    max_seq_len: int,
    out_size: int,
    dataset_size: int,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | str | None = None,
    max_batch_size: int | None = None,
    num_epochs: int = 5,
    fabric: Fabric | None = None,
    fine_tune: bool = False,
) -> int:
    print(
        f"Finding max batch size for {model.__class__.__name__} | "
        f"Using fabric: {fabric is not None}...\n"
    )

    if fine_tune:
        model.unfreeze_transformer_layer()
    else:
        model.freeze_transformer_layer()

    if fabric is None:
        model.to(device)
    model.train()

    loss_fn.to(device)

    if optimizer is not None:
        if isinstance(optimizer, str):
            if optimizer == "Adam":
                optimizer = torch.optim.Adam(model.parameters())
            elif optimizer == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters())
            elif optimizer == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            elif optimizer == "AdamW_LLRD":
                optimizer = build_adamw_llrd_optimizer(model, config={}, finetune=True)
            elif optimizer == "BERT_base_AdamW_grouped_LLRD":
                optimizer = build_bert_base_adamw_grouped_llrd_optimizer(
                    model, config={}, finetune=True
                )
            else:
                raise NotImplementedError("No such optimizer!")
        elif isinstance(optimizer, torch.optim.Optimizer):
            pass
        else:
            raise ValueError("Unknown optimizer!")
    else:
        optimizer = torch.optim.AdamW(model.parameters())

    number_of_chunks = None
    if isinstance(model, BERT):
        if model.bert_type == BertType.HIERARCHICAL_BERT_WITH_POOLING:
            number_of_chunks = 10

    if isinstance(model, HierarchicalBERT):
        number_of_chunks = 10

    if fabric is not None:
        model, optimizer = fabric.setup(model, optimizer)

    batch_size = 2
    while True:
        if max_batch_size is not None and batch_size > max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = dataset_size // 2
            break
        try:
            print(f"Testing batch size {batch_size}...")

            utility_dataset = UtilityDataset(
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_len=max_seq_len,
                out_size=out_size,
                number_of_chunks=number_of_chunks,
            )
            utility_dataloader = torch.utils.data.DataLoader(
                utility_dataset, batch_size=batch_size
            )

            if fabric is not None:
                utility_dataloader = fabric.setup_dataloaders(utility_dataloader)

            for _ in range(num_epochs):
                for batch in utility_dataloader:
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    targets = batch["target"]

                    if fabric is None:
                        if isinstance(input_ids, torch.Tensor):
                            input_ids = input_ids.to(device)
                        if isinstance(attention_mask, torch.Tensor):
                            attention_mask = attention_mask.to(device)
                        targets = targets.to(device)

                    if isinstance(model, BaseModelForTokenClassification):
                        corresponding_word = batch["corresponding_word"]
                        if fabric is None:
                            corresponding_word = corresponding_word.to(device)

                        loss, _ = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            device=device,
                            labels=targets,
                        )
                    else:
                        outputs = model(input_ids, attention_mask)
                        loss = loss_fn(outputs, targets)

                    if fabric is None:
                        loss.backward()
                    else:
                        fabric.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

            batch_size *= 2
            sleep(3)
        except RuntimeError as runtime_error:
            print(f"Batch size {batch_size} failed with error:\n\n{runtime_error}")

            batch_size //= 2
            break

    del model, optimizer
    torch.cuda.empty_cache()

    return batch_size
