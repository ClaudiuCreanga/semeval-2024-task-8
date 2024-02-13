import copy
from torch.optim import AdamW, SGD
from torch.optim.swa_utils import SWALR
from transformers import get_linear_schedule_with_warmup


def build_adamw_llrd_optimizer(model, config, finetune):
    opt_parameters = []
    named_parameters = list(model.named_parameters())

    init_lr = config["finetune_lr"] if "finetune_lr" in config else 2e-5
    head_lr = config["head_lr"] if "head_lr" in config else 2.5e-5
    classifier_lr = config["classifier_lr"] if "classifier_lr" in config else 1e-3
    weight_decay = config["weight_decay"] if "weight_decay" in config else 0.01
    lr_decay_factor = config["lr_decay_factor"] if "lr_decay_factor" in config else 0.95

    lr = init_lr

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm_"]

    # Pooler and Classifier of transformer model

    params_0 = [
        p for n, p in named_parameters
        if ("pooler" in n or "classifier" in n) and any(nd in n for nd in no_decay)
    ]
    params_1 = [
        p for n, p in named_parameters
        if ("pooler" in n or "classifier" in n) and not any(nd in n for nd in no_decay)
    ]

    head_params = {
        "params": params_0,
        "lr": head_lr,
        "weight_decay": 0.0,
    }
    opt_parameters.append(head_params)

    head_params = {
        "params": params_1,
        "lr": head_lr,
        "weight_decay": weight_decay,
    }
    opt_parameters.append(head_params)

    # Hidden layers

    try:
        hidden_layers_count = model.get_hidden_layers_count()
    except:
        hidden_layers_count = len(model.encoder.layer)
    for layer in range(hidden_layers_count - 1, -1, -1):
        layer_name = f"encoder.layer.{layer}."
        params_0 = [
            p for n, p in named_parameters
            if layer_name in n and any(nd in n for nd in no_decay)
        ]
        params_1 = [
            p for n, p in named_parameters
            if layer_name in n and not any(nd in n for nd in no_decay)
        ]

        layer_params = {
            "params": params_0,
            "lr": lr,
            "weight_decay": 0.0,
        }
        opt_parameters.append(layer_params)

        layer_params = {
            "params": params_1,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        opt_parameters.append(layer_params)

        lr *= lr_decay_factor

    # Embeddings layer

    params_0 = [
        p for n, p in named_parameters
        if "embeddings" in n and any(nd in n for nd in no_decay)
    ]
    params_1 = [
        p for n, p in named_parameters
        if "embeddings" in n and not any(nd in n for nd in no_decay)
    ]

    embeddings_params = {
        "params": params_0,
        "lr": lr,
        "weight_decay": 0.0,
    }
    opt_parameters.append(embeddings_params)

    embeddings_params = {
        "params": params_1,
        "lr": lr,
        "weight_decay": weight_decay,
    }
    opt_parameters.append(embeddings_params)

    # Final classifier out layer

    params_0 = [
        p for n, p in named_parameters
        if (n.startswith("out.") or n.startswith("lstm."))
        and any(nd in n for nd in no_decay)
    ]
    params_1 = [
        p for n, p in named_parameters
        if (n.startswith("out.") or n.startswith("lstm."))
        and not any(nd in n for nd in no_decay)
    ]

    classifier_params = {
        "params": params_0,
        "lr": classifier_lr,
        "weight_decay": 0.0,
    }
    opt_parameters.append(classifier_params)

    classifier_params = {
        "params": params_1,
        "lr": classifier_lr,
        "weight_decay": weight_decay,
    }
    opt_parameters.append(classifier_params)

    return AdamW(opt_parameters, lr=init_lr)


def build_bert_base_adamw_grouped_llrd_optimizer(model, config, finetune):
    opt_parameters = []
    named_parameters = list(model.named_parameters())

    init_lr = config["finetune_lr"] if "finetune_lr" in config else 2e-5
    classifier_lr = config["classifier_lr"] if "classifier_lr" in config else 1e-3
    head_lr_factor = config["head_lr_factor"] if "head_lr_factor" in config else 3.6
    group_2_lr_factor = (
        config["group_2_lr_factor"] if "group_2_lr_factor" in config else 1.75
    )
    group_3_lr_factor = (
        config["group_3_lr_factor"] if "group_3_lr_factor" in config else 3.5
    )

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm_"]
    group_2 = [
        "layer.4", "layer.5", "layer.6", "layer.7",
    ]
    group_3 = [
        "layer.8", "layer.9", "layer.10", "layer.11",
    ]

    if "group_2" in config:
        group_2 = config["group_2"]
    if "group_3" in config:
        group_3 = config["group_3"]

    for i, (name, params) in enumerate(named_parameters):
        weight_decay = 0.0 if any(nd in name for nd in no_decay) else 0.01

        if name.startswith("bert.embeddings") or name.startswith("bert.encoder"):
            lr = init_lr

            lr = (
                init_lr * group_2_lr_factor if any(nd in name for nd in group_2) else lr
            )

            lr = (
                init_lr * group_3_lr_factor if any(nd in name for nd in group_3) else lr
            )

            opt_parameters.append({
                "params": params,
                "lr": lr,
                "weight_decay": weight_decay,
            })

        if name.startswith("bert.pooler") or name.startswith("bert.classifier"):
            lr = init_lr * head_lr_factor

            opt_parameters.append({
                "params": params,
                "lr": init_lr,
                "weight_decay": weight_decay,
            })

        if name.startswith("out."):
            opt_parameters.append({
                "params": params,
                "lr": classifier_lr,
                "weight_decay": weight_decay,
            })

    return AdamW(opt_parameters, lr=init_lr)


def build_adamw_optimizer(model, config, finetune):
    return AdamW(
        model.parameters(),
        lr=config["finetune_lr"] if finetune else config["freeze_lr"],
    )


def build_sgd_optimizer(model, config, finetune):
    freeze_lr = config["freeze_lr"]
    finetune_lr = config["finetune_lr"]

    params = copy.deepcopy(config)
    del params["freeze_lr"]
    del params["finetune_lr"]

    return SGD(
        model.parameters(),
        lr=finetune_lr if finetune else freeze_lr,
        **params,
    )


def get_optimizer(model, config, finetune=True):
    if "AdamW" in config:
        return build_adamw_optimizer(model, config["AdamW"], finetune)
    elif "SGD" in config:
        return build_sgd_optimizer(model, config["SGD"], finetune)
    elif "AdamW_LLRD" in config:
        return build_adamw_llrd_optimizer(model, config["AdamW_LLRD"], finetune)
    elif "BERT_base_AdamW_grouped_LLRD" in config:
        return build_bert_base_adamw_grouped_llrd_optimizer(
            model, config["BERT_base_AdamW_grouped_LLRD"], finetune
        )
    else:
        raise NotImplementedError("No such optimizer!")


def get_swalr_scheduler(optimizer, config):
    return SWALR(optimizer, **config)


def get_scheduler(optimizer, num_warmup_steps=0, num_training_steps=None):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
