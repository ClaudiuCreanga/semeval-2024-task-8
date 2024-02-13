import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.optim.swa_utils import AveragedModel, update_bn

from lib.utils import elapsed_time
from lib.utils.training import EarlyStopping
from lib.training.optimizer import get_optimizer, get_scheduler, get_swalr_scheduler
from lib.models.base import BaseModelForTokenClassification


def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    metric_fn,
    swa_model=None,
    swa_scheduler=None,
    swa_step=False,
    fabric=None,
    print_freq=100,
):
    if fabric is not None:
        metric_fn = metric_fn.to(fabric.device)

    model.train()

    losses = []
    all_predictions = []
    all_true = []
    all_ids = []

    start_time = time.time()
    for i, d in enumerate(data_loader):
        ids = d["id"]
        input_ids = d["input_ids"]  # .to(device)
        attention_mask = d["attention_mask"]  # .to(device)
        targets = d["target"]  # .to(device)

        if fabric is None:
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(device)
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(device)
            targets = targets.to(device)

        if isinstance(model, BaseModelForTokenClassification):
            corresponding_word = d["corresponding_word"]
            char_input_ids = d.get("char_input_ids", None)
            char_attention_mask = d.get("char_attention_mask", None)

            if fabric is None:
                corresponding_word = corresponding_word.to(device)
                if char_input_ids is not None:
                    char_input_ids = char_input_ids.to(device)
                if char_attention_mask is not None:
                    char_attention_mask = char_attention_mask.to(device)

            if char_input_ids is None and char_attention_mask is None:
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    device=device,
                    labels=targets,
                )
            else:
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    char_input_ids=char_input_ids,
                    char_attention_mask=char_attention_mask,
                    device=device,
                    labels=targets,
                )

            predictions, true = model.get_predictions_from_logits(
                logits=logits,
                labels=targets,
                corresponding_word=corresponding_word,
            )

            predictions = predictions.flatten().tolist()
            true = true.flatten().tolist()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = model.get_predictions_from_outputs(outputs)
            true = targets.flatten().tolist()

            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss = loss_fn(outputs, targets.squeeze(1).long())
            else:
                loss = loss_fn(outputs, targets)

        all_predictions.extend(predictions)
        all_true.extend(true)
        all_ids.extend(ids)

        losses.append(loss.item())

        if fabric is not None:
            predictions = torch.Tensor(predictions).to(fabric.device)
            true = torch.Tensor(true).to(fabric.device)

        if i % print_freq == 0:
            end_time = time.time()
            elapsed_time_float, elapsed_time_str = elapsed_time(start_time, end_time)

            print(
                f"Batch=[{i + 1}/{len(data_loader)}]; Loss=[{loss.item():.5f}]; "
                f"Train Metric={metric_fn(true, predictions)}; "
                f"Elapsed time={elapsed_time_float:.2f}s [{elapsed_time_str}]",
            )

            start_time = time.time()

        if fabric is None:
            loss.backward()
        else:
            fabric.backward(loss)

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if swa_step and swa_model is not None and swa_scheduler is not None:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return np.mean(losses), (all_ids, all_true, all_predictions)


def validation_epoch(model, data_loader, loss_fn, device, metric_fn, fabric=None):
    if fabric is not None:
        metric_fn = metric_fn.to(fabric.device)

    model.eval()

    losses = []
    all_predictions = []
    all_true = []
    all_ids = []

    with torch.no_grad():
        for i, d in enumerate(tqdm(data_loader)):
            ids = d["id"]
            input_ids = d["input_ids"]  # .to(device)
            attention_mask = d["attention_mask"]  # .to(device)
            targets = d["target"]  # .to(device)

            if fabric is None:
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.to(device)
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.to(device)
                targets = targets.to(device)

            if isinstance(model, BaseModelForTokenClassification):
                corresponding_word = d["corresponding_word"]
                char_input_ids = d.get("char_input_ids", None)
                char_attention_mask = d.get("char_attention_mask", None)

                if fabric is None:
                    corresponding_word = corresponding_word.to(device)
                    if char_input_ids is not None:
                        char_input_ids = char_input_ids.to(device)
                    if char_attention_mask is not None:
                        char_attention_mask = char_attention_mask.to(device)

                if char_input_ids is None and char_attention_mask is None:
                    loss, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        device=device,
                        labels=targets,
                    )
                else:
                    loss, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        char_input_ids=char_input_ids,
                        char_attention_mask=char_attention_mask,
                        device=device,
                        labels=targets,
                    )

                predictions, true = model.get_predictions_from_logits(
                    logits=logits,
                    labels=targets,
                    corresponding_word=corresponding_word,
                )

                predictions = predictions.flatten().tolist()
                true = true.flatten().tolist()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                predictions = model.get_predictions_from_outputs(outputs)
                true = targets.flatten().tolist()

                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    loss = loss_fn(outputs, targets.squeeze(1).long())
                else:
                    loss = loss_fn(outputs, targets)

            all_predictions.extend(predictions)
            all_true.extend(true)
            all_ids.extend(ids)

            losses.append(loss.item())

    return np.mean(losses), (all_ids, all_true, all_predictions)


def train_and_validate(
    model,
    train_data_loader,
    validation_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    metric_fn,
    swa_model=None,
    swa_scheduler=None,
    swa_step=False,
    print_freq=100,
    validation_freq=3,
):
    model.train()

    losses = []
    all_predictions = []
    all_true = []
    all_ids = []

    start_time = time.time()
    for i, d in enumerate(train_data_loader):
        ids = d["id"]
        input_ids = d["input_ids"]  # .to(device)
        attention_mask = d["attention_mask"]  # .to(device)
        targets = d["target"]  # .to(device)

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(device)
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        predictions = model.get_predictions_from_outputs(outputs)
        true = targets.flatten().tolist()

        all_predictions.extend(predictions)
        all_true.extend(true)
        all_ids.extend(ids)

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss = loss_fn(outputs, targets.squeeze(1).long())
        else:
            loss = loss_fn(outputs, targets)
        losses.append(loss.item())

        if i % print_freq == 0:
            end_time = time.time()
            elapsed_time_float, elapsed_time_str = elapsed_time(start_time, end_time)

            print(
                f"Batch=[{i + 1}/{len(train_data_loader)}]; Loss=[{loss.item():.5f}]; "
                f"Acc. Metric={metric_fn(true, predictions)}; "
                f"Elapsed time={elapsed_time_float:.2f}s [{elapsed_time_str}]",
            )

            start_time = time.time()

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if swa_step and swa_model is not None and swa_scheduler is not None:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        if (i + 1) % validation_freq == 0:
            print(f"Validation at batch {i + 1}:")

            dev_loss, (dev_ids, dev_true, dev_predict) = validation_epoch(
                model, validation_data_loader, loss_fn, device, metric_fn
            )

            dev_metric = metric_fn(dev_true, dev_predict)

            print(
                f"Batch=[{i + 1}/{len(train_data_loader)}]; "
                f"Validation Loss: {dev_loss:.5f}; Validation Metric: {dev_metric:.5f}"
            )

    return np.mean(losses), (all_ids, all_true, all_predictions)


def training_loop(
    model,
    num_epochs,
    train_loader,
    dev_loader,
    loss_fn,
    optimizer_config,
    scheduler_config,
    device,
    metric_fn,
    is_better_metric_fn,
    results_dir,
    num_epochs_before_finetune,
    early_stopping: EarlyStopping = None,
    swa_config=None,
    validation_freq=None,
    fabric=None,
    print_freq: int = 100,
):
    history = defaultdict(list)
    best_metric = None
    best_model_state = None

    model.freeze_transformer_layer()

    optimizer = get_optimizer(model, optimizer_config, finetune=False)
    scheduler = None

    if fabric is not None:
        print("Setup model and optimizer with fabric")

        model, optimizer = fabric.setup(model, optimizer)
        train_loader, dev_loader = fabric.setup_dataloaders(train_loader, dev_loader)

    swa_start = num_epochs + 1
    swa_model = None
    swa_scheduler = None

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        if epoch <= num_epochs_before_finetune:
            print("Freeze transformeer")
        else:
            print("Finetune transformer")
        print("-" * 20)

        if epoch == num_epochs_before_finetune + 1:
            model.unfreeze_transformer_layer()
            optimizer = get_optimizer(model, optimizer_config, finetune=True)

            if fabric is not None:
                print("Finetune: setup model and optimizer with fabric")
                optimizer = fabric.setup_optimizers(optimizer)
                # model, optimizer = fabric.setup(model, optimizer)

            scheduler = get_scheduler(
                optimizer,
                num_training_steps=len(train_loader) * num_epochs,
                **scheduler_config,
            )

            if swa_config is not None:
                swa_start = swa_config["swa_start"]
                swa_model = AveragedModel(model).to(device)
                swa_scheduler = get_swalr_scheduler(
                    optimizer=optimizer, config=swa_config["swalr"]
                )

        if validation_freq is None:
            start_time = time.time()
            train_loss, (train_ids, train_true, train_predict) = train_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                metric_fn,
                swa_model=swa_model,
                swa_scheduler=swa_scheduler,
                swa_step=True if epoch >= swa_start else False,
                fabric=fabric,
                print_freq=print_freq,
            )
            end_time = time.time()
            elapsed_time_float, elapsed_time_str = elapsed_time(start_time, end_time)

            train_metric = None
            if fabric is not None:
                train_ids = [int(i.cpu()) for i in train_ids]
                train_true_pt = torch.Tensor(train_true)
                train_predict_pt = torch.Tensor(train_predict)
                train_metric = metric_fn(train_true_pt, train_predict_pt).cpu()
            else:
                train_metric = metric_fn(train_true, train_predict)

            print(
                f"Train Loss: {train_loss:.5f}; "
                f"Train Metric: {train_metric:.5f}; "
                f"Elapsed time={elapsed_time_float:.2f}s [{elapsed_time_str}]"
            )

            dev_loss, (dev_ids, dev_true, dev_predict) = validation_epoch(
                model, dev_loader, loss_fn, device, metric_fn
            )

            dev_metric = None
            if fabric is not None:
                dev_ids = [int(i.cpu()) for i in dev_ids]
                dev_true_pt = torch.Tensor(dev_true)
                dev_predict_pt = torch.Tensor(dev_predict)
                dev_metric = metric_fn(dev_true_pt, dev_predict_pt).cpu()
            else:
                dev_metric = metric_fn(dev_true, dev_predict)

            print(
                f"Validation Loss: {dev_loss:.5f}; Validation Metric: {dev_metric:.5f}"
            )
        else:
            print(f"\n--- Train and validate every {validation_freq} batches ---\n")

            start_time = time.time()
            train_loss, (train_ids, train_true, train_predict) = train_and_validate(
                model,
                train_loader,
                dev_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                metric_fn,
                swa_model=swa_model,
                swa_scheduler=swa_scheduler,
                swa_step=True if epoch >= swa_start else False,
                validation_freq=validation_freq,
                print_freq=print_freq,
            )
            end_time = time.time()
            elapsed_time_float, elapsed_time_str = elapsed_time(start_time, end_time)

            train_metric = metric_fn(train_true, train_predict)

            print(
                f"Train Loss: {train_loss:.5f}; "
                f"Train Metric: {train_metric:.5f}"
                f"Elapsed time={elapsed_time_float:.2f}s [{elapsed_time_str}]"
            )

            dev_loss, (dev_ids, dev_true, dev_predict) = validation_epoch(
                model, dev_loader, loss_fn, device, metric_fn
            )

            dev_metric = metric_fn(dev_true, dev_predict)

            print(
                f"Validation Loss: {dev_loss:.5f}; Validation Metric: {dev_metric:.5f}"
            )

        history["train_metric"].append(train_metric)
        history["train_loss"].append(train_loss)
        history["dev_metric"].append(dev_metric)
        history["dev_loss"].append(dev_loss)

        if early_stopping is not None:
            early_stopping(dev_loss, model)

        if best_metric is None or is_better_metric_fn(dev_metric, best_metric):
            best_metric = dev_metric
            best_model_state = model.state_dict()

            if results_dir is not None:
                torch.save(
                    best_model_state,
                    os.path.join(results_dir, "best_model.bin")
                )

                df_train_predictions = pd.DataFrame(
                    {
                        "id": train_ids,
                        "true": train_true,
                        "predict": train_predict,
                    }
                )
                df_train_predictions.to_csv(
                    os.path.join(results_dir, "best_model_train_predict.csv"),
                    index=False
                )

                df_dev_predictions = pd.DataFrame(
                    {
                        "id": dev_ids,
                        "true": dev_true,
                        "predict": dev_predict,
                    }
                )
                df_dev_predictions.to_csv(
                    os.path.join(results_dir, "best_model_dev_predict.csv"),
                    index=False
                )

        if early_stopping is not None and early_stopping.early_stop:
            print("Early stopping")
            break

    if results_dir is not None:
        df_history = pd.DataFrame(history)
        df_history.to_csv(os.path.join(results_dir, "history.csv"), index=False)

        model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.bin")))
    else:
        model.load_state_dict(best_model_state)

    if swa_config is not None:
        swa_model = AveragedModel(model).to(device)
        update_bn(train_loader, swa_model, device)

    return model


def make_predictions(
    model,
    data_loader,
    device,
    results_dir,
    label_column,
    file_format="csv",
    fabric=None,
):
    model.eval()

    all_predictions = []
    all_true = []
    all_ids = []

    with torch.no_grad():
        for i, d in enumerate(tqdm(data_loader)):
            ids = d["id"]
            input_ids = d["input_ids"]  # .to(device)
            attention_mask = d["attention_mask"]  # .to(device)
            targets = d["target"]  # .to(device)

            if fabric is None:
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.to(device)
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.to(device)
                targets = targets.to(device)

            if isinstance(model, BaseModelForTokenClassification):
                corresponding_word = d["corresponding_word"]
                char_input_ids = d.get("char_input_ids", None)
                char_attention_mask = d.get("char_attention_mask", None)

                if fabric is None:
                    corresponding_word = corresponding_word.to(device)
                    if char_input_ids is not None:
                        char_input_ids = char_input_ids.to(device)
                    if char_attention_mask is not None:
                        char_attention_mask = char_attention_mask.to(device)

                if char_input_ids is None and char_attention_mask is None:
                    _, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        device=device,
                        labels=targets,
                    )
                else:
                    _, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        char_input_ids=char_input_ids,
                        char_attention_mask=char_attention_mask,
                        device=device,
                        labels=targets,
                    )

                predictions, true = model.get_predictions_from_logits(
                    logits=logits,
                    labels=targets,
                    corresponding_word=corresponding_word,
                )

                predictions = predictions.flatten().tolist()
                true = true.flatten().tolist()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                predictions = model.get_predictions_from_outputs(outputs)
                true = targets.flatten().tolist()

            all_predictions.extend(predictions)
            all_true.extend(true)
            all_ids.extend([
                int(i.cpu()) if isinstance(i, torch.Tensor) else i for i in ids
            ])

    df_predictions = pd.DataFrame(
        {
            "id": all_ids,
            "true": all_true,
            label_column: all_predictions,
        }
    )

    if results_dir is not None:
        if file_format == "csv":
            df_predictions.to_csv(
                os.path.join(results_dir, "submission.csv"),
                index=False,
            )
        elif file_format == "jsonl":
            df_predictions.to_json(
                os.path.join(results_dir, "submission.jsonl"),
                orient="records",
                lines=True,
            )
        else:
            raise ValueError(f"Unknown file format: {file_format}")
    else:
        print("Missing results_dir, not saving predictions to file!")

    return df_predictions
