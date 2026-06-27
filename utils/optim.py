from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
# from transformers.optimization import AdamW
from torch.optim import AdamW


def create_xvlm_optimizer(args, model):
    lr = args.lr
    wd = args.weight_decay
    lr_mult = getattr(args, 'lr_mult', 1)
    print("### lr_mult, ", lr_mult)

    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": wd, "lr": lr, "initial_lr": lr},
        {"params": [], "weight_decay": 0.0, "lr": lr, "initial_lr": lr},
        {"params": [], "weight_decay": wd, "lr": lr *
            lr_mult, "initial_lr": lr * lr_mult},
        {"params": [], "weight_decay": 0.0, "lr": lr *
            lr_mult, "initial_lr": lr * lr_mult}
    ]

    no_decay = {
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight"
    }

    if hasattr(model, 'init_params'):
        large_lr = model.init_params
        print("### model has 'init_params', ", len(large_lr))
    else:
        large_lr = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay):
            if name in large_lr:
                optimizer_grouped_parameters[3]['params'].append(param)
            else:
                optimizer_grouped_parameters[1]['params'].append(param)
        else:
            if name in large_lr:
                optimizer_grouped_parameters[2]['params'].append(param)
            else:
                optimizer_grouped_parameters[0]['params'].append(param)

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=lr, eps=1e-8, betas=(0.9, 0.98))
    return optimizer


def create_blip_optimizer(args, model):
    print("[Debug] utils/optim.py : create_blip_optimizer()함수 호출 : optimizer 생성")
    return torch.optim.AdamW(params=model.parameters(), lr=args['init_lr'], weight_decay=args['weight_decay'])

def create_clip_optimizer(args, model):
    print("[Debug] utils/optim.py : create_clip_optimizer()함수 호출 : optimizer 생성")
    lr = args['lr']; wd = args['weight_decay']

    no_wd_substrings = [
        "bias", "LayerNorm.weight", "LayerNorm.bias",
        "layer_norm.weight", "layer_norm.bias",
        "ln_final.weight", "ln_final.bias",
        "ln_pre.weight", "ln_pre.bias",
        "norm.weight", "norm.bias",
        "embeddings.position_embedding",
        "embeddings.class_embedding",
        "pos_embed",
        "logit_scale",
    ]
    def _no_decay(name: str) -> bool:
        return name.endswith("bias") or any(s in name for s in no_wd_substrings)

    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:   # <- 사용/동결 정리가 선행되어야 함
            continue
        (no_decay_params if _no_decay(n) else decay_params).append(p)

    param_groups = [
        {"params": decay_params,    "weight_decay": wd,  "lr": lr},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
    ]

    # 여기도 torch.optim.AdamW로 통일 권장
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-8)


def create_optimizer(args, model):
    if model.name == "blip":
        return create_blip_optimizer(args, model)
    elif model.name == "xvlm":
        return create_xvlm_optimizer(args, model)
    elif model.name in ("clip", "clipG"): 
        return create_clip_optimizer(args, model)
    elif model.name == "sd15_generator":
        return


def create_scheduler(mode, optimizer, num_warmup_steps, total_steps, last_epoch=-1):
    print("[Debug] utils/optim.py : create_scheduler()함수 호출 : scheduler 생성")
    if mode == 'linear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, last_epoch=last_epoch)
    elif mode == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, last_epoch=last_epoch)
    else:
        raise NotImplementedError
