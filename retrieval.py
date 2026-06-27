import os
import math
import json
import time
import datetime
import argparse
import numpy as np
import torch
import lightning as L
import ruamel.yaml as yaml
import torch.backends.cudnn as cudnn

import utils

from functools import partial
from pathlib import Path
from models import BLIPRetrieval, XVLMRetrieval, BertTokenizerForXVLM, CLIPRetrieval, CLIPGRetrieval, BLIP2Retrieval
from transformers import CLIPTokenizer
from utils.optim import create_optimizer, create_scheduler
from utils.loggers import init_wandb_logger
from utils.misc import millions, num_params
from utils.prune_utils import make_prunable, stats, named_masked_parameters
from utils.functions import get_unprunable_parameters
from datasets import create_dataset, create_loader
from datasets.retrieval_dataset import eval_collate_fn
from layers.linear import Linear
from evaltools import blip_itr_evaluation, xvlm_itr_evaluation, clip_itr_evaluation, clipg_itr_evaluation, blip2_itr_evaluation

from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy

import contextlib

class CUDATimer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.total_s = 0.0
        self.iters = 0

    @contextlib.contextmanager
    def measure(self):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.total_s += (t1 - t0)
        self.iters += 1

    def reset(self):
        self.total_s = 0.0
        self.iters = 0


@torch.no_grad()
def hard_disable_mask_on_fly_and_apply_once(model):
    for m in model.modules():
        if hasattr(m, "weight_pruning_mask") and hasattr(m, "weight"):
            # mask 한 번만 적용해서 weight를 실제로 0으로 고정
            m.weight.mul_(m.weight_pruning_mask)
        if hasattr(m, "bias_pruning_mask") and getattr(m, "bias", None) is not None:
            m.bias.mul_(m.bias_pruning_mask)

        # on-the-fly 강제 OFF
        if hasattr(m, "mask_on_fly"):
            m.mask_on_fly = False
            # forward가 on_the_fly로 바뀌어 있으면 원래 Linear.forward로 복구
            if isinstance(m, Linear):
                m.forward = Linear.forward.__get__(m, Linear)

def debug_mask_on_fly(model, limit=10):
    c = 0
    for n, m in model.named_modules():
        if hasattr(m, "mask_on_fly") and m.mask_on_fly:
            if c < limit:
                print("[mask_on_fly=True]", n, type(m))
            c += 1
    print("mask_on_fly True count =", c)

def train_collate(batch, tokenizer):
    images, captions, idx = zip(*batch) # Batch 내 요소들을 분리
    images = torch.stack(images, dim=0) # torch.stack -> 각 이미지들을 concatenate(가로 or 세로 방향)0하는 것이 아닌 채널(3차원으로)로 하나로 쌓아 합친다.
    indices = torch.tensor(idx, dtype=torch.long) # torch.tensor -> 각 인덱스 데이터를 tensor로 복사함
    # 텍스트를 토큰화함 -> batch 내 가장 긴 문장에 맞춰 패딩함 (결과는 pytorch 텐서 형태로 반환)
    text_input = tokenizer(captions, padding='longest', max_length=config['max_tokens'], return_tensors="pt")
    text_ids, text_atts = text_input.input_ids, text_input.attention_mask # 각 텍스트의 토큰 Id 텐서, 각 텍스트의 attention mask 텐서를 분리
    #Attention_mask는 텍스트에서 패딩과 토큰을 구분하기 위해 사용
    return images, text_ids, text_atts, indices


def determine_alpha(index, epoch, num_batches, config):
    print("<retrieval.py -> determine_alpha 함수 호출>")
    # alpha는 Contrastive learning에서 손실 함수의 가중치를 조절하는 하이퍼파라미터 -> alpha값을 조정하며 학습 속도를 조절
    if epoch>0: # 학습이 진행된 상태에서는 alpha값은 config['alpha']로 고장
        alpha = config['alpha']
    else: # 초기 학습 단계에서는 alpha값을 점진적으로 증가시키며 학습을 안정화함
        alpha = config['alpha']*min(1,index/num_batches) # 처음에는 작은 alpha값을 사용하다가 나중에 config['alpha']까지 증가
    return alpha


def train(model, data_loader, optimizer, fabric: L.Fabric, scheduler, epoch, config, debug_mode=False, **kwargs):
    print("<retrieval.py : train()함수 호출>")
    model.train() #모델을 train 모드로 변경
    optimizer.zero_grad() # gradient 초기화
    steps = 0 # 학습 step 초기화
    itm_losses, itc_losses, total_losses = [], [], [] # 각 step의 loss을 저장하는 list

    # if the model is blip, setup the momentum encoders
    if model.name == "blip": # momentum encoder 설정 -> Contrastive learning에서 과거 정보 유지를 위해 사용
        visual_encoder_m, text_encoder_m, vision_proj_m, text_proj_m = model.update_momentum(
            visual_encoder_m=kwargs.get('visual_encoder_m', None), 
            text_encoder_m=kwargs.get('text_encoder_m'), 
            vision_proj_m=kwargs.get('vision_proj_m', None), 
            text_proj_m=kwargs.get('text_proj_m', None)
        )
        kwargs['visual_encoder_m'] = visual_encoder_m
        kwargs['text_encoder_m'] = text_encoder_m
        kwargs['vision_proj_m'] = vision_proj_m
        kwargs['text_proj_m'] = text_proj_m
    
    # start training
    for i, (image, text_ids, text_atts, idx) in enumerate(data_loader): # data_loader에서 batch단위로 데이터를 로드함
        if debug_mode and i == 100: break

        # effectively synchronize only when the target batch size is met
        #분산 학습 시, 매 batch마다 동기화하면 학습 속도가 저하됨(특정 step에서만 동기화를 하여 속도를 최적화함)
        # 현재 batch가 Accumulation step의 배수이거나 마지막이면 gradient 동기화 (sync) 수행 => Accumulation step인지 (True/False)
        is_sync_step = ((i+1) % config['grad_acc_steps'] == 0) or (i+1 == len(data_loader)) #config['grad_acc_steps']는 몇 개의 batch를 누적하여 한 번만 가중치를 업데이트할 지 결정하는 값(Accumulation step)
        
        #is_sync_step = False면 동기화 건너뜀 -> gradient만 계산함
        with fabric.no_backward_sync(model, enabled=not is_sync_step):

            if model.name == "blip": # 동기화할 때 blip모델일 경우 determine_alpha()함수로 alpha값을 설정하여 학습 속도 조절(learning rate 조절)
                alpha = determine_alpha(i, epoch, len(data_loader), config)
            elif model.name == "xvlm":
                alpha = None
            elif model.name in ("clip", "clipG"):
                alpha = None

            #모델의 loss 계산 (itc = Image-text Contrastive loss(이미지와 텍스트 임베딩 유사성), itm = Image-text Matching loss(이미지와 텍스트의 올바른 매칭 여부))
            loss_itc, loss_itm = model( # blip_retrieval.py의 BLIPRetrieval클래스의 forward() 함수 호출 -> Contrastive learning 수행 후 loss 반환
                image, text_ids, text_atts, idx=idx, 
                return_losses=True, alpha=alpha,
                visual_encoder_m=kwargs.get('visual_encoder_m', None), 
                text_encoder_m=kwargs.get('text_encoder_m'), 
                vision_proj_m=kwargs.get('vision_proj_m', None), 
                text_proj_m=kwargs.get('text_proj_m', None)
            )
            loss = loss_itc / config['grad_acc_steps'] # clip용
  
            # loss = (loss_itc + loss_itm) / config['grad_acc_steps'] #여러 step 동안 gradient를 누적하면서, loss의 크기가 너무 커지지 않도록 조절
            fabric.backward(loss) # loss를 기반으로 역전파 수행하여 gradient 계산

        if is_sync_step: #is_sync_step이 True인 경우(동기화를 진행하는 step인 경우)
            optimizer.step() #가중치 업데이트
            scheduler.step() #스케줄러 업데이트
            optimizer.zero_grad() #gradient 초기화
            # 여러 step동안 누적된 gradient를 한 번에 업데이트하고, 다시 초기화함

            # move from gpu to cpu only once
            # loss값을 GPU에서 CPU로 이동시켜 메모리 절약
            loss_itc_scalar, loss_itm_scalar, loss_scalar = loss_itc.item(), loss_itm.item(), loss.item()
            fabric.log_dict({
                "loss_itc": loss_itc_scalar,
                "loss_itm": loss_itm_scalar,
                "loss": loss_scalar,
                "lr": optimizer.param_groups[0]['lr'],
                "step": steps
            })

            # keep track locally
            # 각 batch의 loss값을 list에 저장 (Epoch 단위 평균 loss를 계산하기 위함)
            itm_losses.append(loss_itm.item())
            itc_losses.append(loss_itc.item())
            total_losses.append(loss.item())
            steps += 1 #현재 학습이 진행된 step을 카운트
            print(f"[Epoch {epoch+1}] Training batch {i+1} / {len(data_loader)}\t loss itc = {loss_itc}\t loss itm = {loss_itm}")

            # if the model is blip, reupdate the momentum encoders
            #Momentum encoder의 가중치도 업데이트
            #일반 Encoder가 먼저 학습되고, 그 정보를 기반으로 Momentum Encoder가 업데이트 

            #일반 encoder가 optimizer.step()으로 빠르게 업데이트되고 일반 encoder의 최신 가중치를 기반으로 momentum encoder를 조금씩 반영함
            if model.name == "blip":
                visual_encoder_m, text_encoder_m, vision_proj_m, text_proj_m = model.update_momentum(
                    visual_encoder_m=kwargs.get('visual_encoder_m', None), 
                    text_encoder_m=kwargs.get('text_encoder_m'), 
                    vision_proj_m=kwargs.get('vision_proj_m', None), 
                    text_proj_m=kwargs.get('text_proj_m', None)
                )
                kwargs['visual_encoder_m'] = visual_encoder_m
                kwargs['text_encoder_m'] = text_encoder_m
                kwargs['vision_proj_m'] = vision_proj_m
                kwargs['text_proj_m'] = text_proj_m
    
    # finally, put on W&B epoch-aggregated data
    #loss를 저장한 리스트에 저장된 모든 step의 평균 loss값 계싼
    aggr_itm_loss = torch.tensor(itm_losses).mean().item()
    aggr_itc_loss = torch.tensor(itc_losses).mean().item()
    aggr_loss = torch.tensor(total_losses).mean().item()
    fabric.log_dict({
        "loss_itm_epoch_level": aggr_itm_loss,
        "loss_itc_epoch_level": aggr_itc_loss,
        "losses_epoch_level": aggr_loss
    })
    
    return {
        'loss_itm': aggr_itm_loss,
        'loss_itc': aggr_itc_loss,
        'loss': aggr_loss
    }


# @torch.no_grad()
# def compute_metrics(scores_i2t, scores_t2i, dataset, decimals=2):
#     #dataset -> val_datset(검증 데이터셋)
#     print("<retrieval.py : compute_metrics()함수 호출>")
#     # Images->Text 성능 평가
#     ranks = np.zeros(scores_i2t.shape[0]) #각 이미지별 정답 텍스트의 랭킹을 저장할 배열
#     for index, score in enumerate(scores_i2t):
#         inds = np.argsort(score)[::-1] #점수를 내림차순으로 정렬
#         # Score
#         rank = 1e20
#         for i in dataset.img2txt[index]: #현재 이미지와 연결된 정답 텍스트 인덱스들
#             tmp = np.where(inds == i)[0][0] #정답 이미지의 랭킹
#             if tmp < rank:
#                 rank = tmp
#         ranks[index] = rank

#     # Compute metrics
#     tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
#     tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
#     tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

#     # Text->Images 성능 평가
#     ranks = np.zeros(scores_t2i.shape[0])

#     for index, score in enumerate(scores_t2i):
#         inds = np.argsort(score)[::-1]
#         ranks[index] = np.where(inds == dataset.txt2img[index])[0][0]

#     # Compute metrics
#     ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
#     ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
#     ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

#     tr_mean = (tr1 + tr5 + tr10) / 3
#     ir_mean = (ir1 + ir5 + ir10) / 3
#     r_mean = (tr_mean + ir_mean) / 2

#     eval_result = {'txt_r1': round(tr1, decimals),
#                    'txt_r5': round(tr5, decimals),
#                    'txt_r10': round(tr10, decimals),
#                    'txt_r_mean': round(tr_mean, decimals),
#                    'img_r1': round(ir1, decimals),
#                    'img_r5': round(ir5, decimals),
#                    'img_r10': round(ir10, decimals),
#                    'img_r_mean': round(ir_mean, decimals),
#                    'r_mean': round(r_mean, decimals)}
#     return eval_result


#------------------------------
# Failuser case 저장용 compute_metrics()함수
#------------------------------
@torch.no_grad()
def compute_metrics(scores_i2t, scores_t2i, dataset, decimals=2):
    print("<retrieval.py : compute_metrics()함수 호출>")
     # Images->Text 성능 평가

    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
         # Score
        rank = 1e20
        for i in dataset.img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

     # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # --- 여기서 I→T 실패 케이스 인덱스 저장 ---
    # K=10 기준으로 GT가 Top-10 안에 없으면 실패로 간주
    ranks_i2t = ranks.copy()
    fail_i2t = np.where(ranks_i2t >= 10)[0].tolist()


     # Text->Images 성능 평가

    ranks = np.zeros(scores_t2i.shape[0])
 
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == dataset.txt2img[index])[0][0]

     # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # --- 여기서 T→I 실패 케이스 인덱스 저장 ---
    ranks_t2i = ranks.copy()
    fail_t2i = np.where(ranks_t2i >= 10)[0].tolist()
 

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {
        'txt_r1': round(tr1, decimals),
        'txt_r5': round(tr5, decimals),
        'txt_r10': round(tr10, decimals),
        'txt_r_mean': round(tr_mean, decimals),
        'img_r1': round(ir1, decimals),
        'img_r5': round(ir5, decimals),
        'img_r10': round(ir10, decimals),
        'img_r_mean': round(ir_mean, decimals),
        'r_mean': round(r_mean, decimals)
    }

    # ─── 실패 케이스 JSON 덤프 ─────────────────────────────────────
    import json, os, datetime
    # 예: 결과를 blip_no_mask_rsicd 폴더에 저장
    base_dir = os.path.join("results", "retrieval", "nwpu",
                            "epoch_10",    
                            "blip_mask_90_rsicd")
    os.makedirs(base_dir, exist_ok=True)
    fname = os.path.join(
        base_dir,
        f"rsicd_failcases_mask_90.json"
    )
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump({
            "i2t_fail_idx": fail_i2t,
            "t2i_fail_idx": fail_t2i
        }, fp, ensure_ascii=False, indent=2)
    print(f"[Fail-Dump] I→T:{len(fail_i2t)}  T→I:{len(fail_t2i)}  → {fname}")
 
    return eval_result

def _strip_once(p: str, pref: str) -> str:
    return p[len(pref):] if p.startswith(pref) else p

def _normalize_name(p: str) -> str:
    # DDP 래핑/모델 래퍼 접두사 제거
    p = _strip_once(p, "module.")
    p = _strip_once(p, "model.")
    return p

def _detect_modality_name(pname: str) -> str:
    p = _normalize_name(pname)

    # HF CLIP 계열
    if p.startswith("text_model."):
        return "text"
    if p.startswith("vision_model."):
        return "vision"

    # BLIP / XVLM 계열
    if p.startswith("text_encoder."):
        return "text"
    if p.startswith("visual_encoder."):
        return "vision"
    if p.startswith("language_model."):
        return "text"
    if p.startswith("vision_tower."):
        return "vision"
    if p.startswith("multi_modal_projector."):
        return "fusion"

    # momentum encoder는 제외
    if p.startswith("visual_encoder_m.") or p.startswith("text_encoder_m."):
        return "momentum"

    return "other"


def summarize_pruning_scope(model, exclude=(), show_samples=5, include_bias=False):
    buckets = ("text", "vision", "fusion", "other")
    counts = {k: 0 for k in buckets}
    names  = {k: [] for k in buckets}
    totals = {k: 0 for k in buckets}
    pruned = {k: 0 for k in buckets}

    for pname, mask, param in named_masked_parameters(model, bias=include_bias, exclude=exclude):
        key = _detect_modality_name(pname)
        if key == "momentum":
            continue
        if key not in counts:
            key = "other"

        counts[key] += 1
        totals[key] += mask.numel()
        pruned[key] += (mask.numel() - mask.sum().item())
        if len(names[key]) < show_samples:
            names[key].append(pname)

    print("\n[Pruning Scope Summary]")
    for k in buckets:
        rate = 0.0 if totals[k] == 0 else (pruned[k] / totals[k] * 100.0)
        print(f"- {k:6s}: params_with_masks={counts[k]:4d}, "
              f"mask_elems={totals[k]:8d}, pruned_elems={int(pruned[k]):8d} ({rate:5.2f}%)")
        for n in names[k]:
            print(f"    · {n}")
    print()
    return counts, totals, pruned

def check_mask_scope(model, expected="auto"):
    """
    expected: 'text' | 'vision' | 'both' | 'none' | 'auto'
      - auto: 그냥 비율만 출력
    """
    # get_unprunable_parameters 시그니처가 (model_name[, prune_scope]) 둘 다 있을 수 있어 방어적으로 호출
    try:
        exclude = get_unprunable_parameters(getattr(model, "name", ""), prune_scope="both")
    except TypeError:
        exclude = get_unprunable_parameters(getattr(model, "name", ""))

    _, totals, pruned = summarize_pruning_scope(model, exclude=exclude)
    rt = 0.0 if totals["text"]   == 0 else pruned["text"]   / totals["text"]
    rv = 0.0 if totals["vision"] == 0 else pruned["vision"] / totals["vision"]

    print(f"[Mask Rates] text={rt*100:.2f}%  vision={rv*100:.2f}%")

    if expected == "auto":
        return

    def _ok_text():   return rt > 0 and rv == 0
    def _ok_vision(): return rv > 0 and rt == 0
    def _ok_both():   return rt > 0 and rv > 0
    def _ok_none():   return rt == 0 and rv == 0

    ok = {
        "text": _ok_text,
        "vision": _ok_vision,
        "both": _ok_both,
        "none": _ok_none,
    }[expected]()

    status = "PASS" if ok else "FAIL"
    print(f"[Mask Scope Check] expected={expected} → {status}")



def main(args, config):
    if args.precision == '32-true':
        torch.set_float32_matmul_precision(precision="high")
    elif args.precision in ('bf16-mixed', '16-mixed'):
        torch.set_float32_matmul_precision(precision="medium")

    loggers = []
    if args.wandb:
        loggers.append(init_wandb_logger(config))

    # initialize distributed training
    #분산 학습 설정(초기화)
    # fabric = L.Fabric( # Zero-shot용
    #     accelerator="cuda",
    #     strategy="ddp",
    #     precision=args.precision,
    #     devices=args.devices,
    #     loggers=loggers
    # )

    #Fine-tuning용
    fabric = Fabric(
        accelerator="gpu",
        devices=args.devices,              # --devices 2
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="bf16-mixed",
    )

    fabric.launch()
    utils.setup_for_distributed(is_master=fabric.is_global_zero)

    # automatically infer the gradient accumulation steps
    #배치 사이즈 설정
    B = config['batch_size_train']
    E = config['batch_size_target']
    config['grad_acc_steps'] = (E // fabric.world_size) // B  #분산 학습 환경에서 GPU마다 처리할 batch 크기 결정
    
    # reproducibility settings
    L.seed_everything(args.seed)
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # initialize the datasets, distributed samplers and dataloaders
    # *데이터셋 로드*
    print("Creating retrieval dataset", flush=True)
    print("<데이터셋 생성 -> retrieval_dataset.py로 이동>")
    # config/blip/retrieval.yaml에 dataset 정보 불러와서 저장
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval', config)
    
    # tokenizer for the dataset
    if args.model == "xvlm":
        tokenizer = BertTokenizerForXVLM.from_pretrained(config['text_encoder'])
        
        # model initialization
        print("Creating model", flush=True)
        model = XVLMRetrieval(config=config)
        setattr(model, 'name', 'xvlm')

        # definition of the evaluation function
        evaluation = xvlm_itr_evaluation

        # define the momentum encoders (only for compatibility with the BLIP code, XVLM does not use them)
        visual_encoder_m = None
        text_encoder_m = None
        vision_proj_m = None
        text_proj_m = None
    
    elif args.model == "blip":
        #blip_retrieval.py의 BLIPRetrieval 클래스 객체 생성 -> init() 함수 실행
        print("<BLIPRetrieval 클래스 객체 생성 : blip_retreival.py의 BLIPRetrieval클래스 이동>")
        model = BLIPRetrieval(
            image_size=config['image_res'], 
            vit=config['vit'], 
            vit_grad_ckpt=config['vit_grad_ckpt'], 
            vit_ckpt_layer=config['vit_ckpt_layer'], 
            queue_size=config['queue_size'],  
            negative_all_rank=config['negative_all_rank']
        )
        tokenizer = model.tokenizer
        setattr(model, 'name', 'blip')

        # definition of the evaluation function
        #training 후 evaluation할 함수 지정 -> evaltools/itr_utils.py의 blip_evaluation 함수 
        evaluation = blip_itr_evaluation
    elif args.model == "clip":
        print("CLIPRetrieval 클래스 객체 생성 : clip_retrieval.py의 CLIPRetrieval클래스 이동")
        tokenizer = CLIPTokenizer.from_pretrained(config['model_name'])
        model = CLIPRetrieval(config=config)
        setattr(model, 'name', 'clip')
        evaluation = clip_itr_evaluation 

        visual_encoder_m = None
        text_encoder_m = None
        vision_proj_m = None
        text_proj_m = None
    elif args.model == "clipG":
        print("CLIPGRetrieval 클래스 객체 생성 : clip_retrieval.py의 CLIPRetrieval클래스 이동")
        tokenizer = CLIPTokenizer.from_pretrained(config['model_name'])
        model = CLIPGRetrieval(config=config)
        setattr(model, 'name', 'clipG')
        evaluation = clipg_itr_evaluation 

        visual_encoder_m = None
        text_encoder_m = None
        vision_proj_m = None
        text_proj_m = None
    elif args.model == "blip2":
        model = BLIP2Retrieval(config=config)
        tokenizer = model.tokenizer
        setattr(model, 'name', 'blip2')
        evaluation = blip2_itr_evaluation

    else:
        raise NotImplementedError(f"Model {args.model} not implemented. Please add it to the factory yourself.")
    
    # training goes in distributed mode, so it needs a sampler
    # NOTE: since the evaluation code is a bit tricky for the retrieval task, we don't use a distributed sampler
    #분산 학습을 위한 샘플러 생성하여 GPU마다 다른 데이터를 샘플링하도록 함
    #분산 학습 환경에서 모든 GPU가 동일한 데이터를 학습하면 비효율적이기 때문
    training_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=fabric.world_size, 
        rank=fabric.global_rank, 
        shuffle=True, # remember to set shuffle=False in the DataLoader
        seed=args.seed,
    )

    # define the dataloaders
    #*Dataloader 생성* -> 데이터셋을 직접 다루는 것이 아니라 배치 크기로 잘라서 GPU로 전달하는 역할
    training_collate_fn = partial(train_collate, tokenizer=tokenizer) #train_collate()함수 호출
    #train, val, test용 loader 생성함
    train_loader, val_loader, test_loader = create_loader(
        datasets=[train_dataset, val_dataset, test_dataset], 
        samplers=[training_sampler, None, None], #분산 학습 환경의 training sampler 사용(val, test는 분산 학습과 무관하기 때문에 sampler 없음)
        batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
        num_workers=[8,8,8],
        is_trains=[True,False,False],
        collate_fns=[training_collate_fn, None, None] # batch 단위로 이미지, 인덱스, 텍스트 토큰, 텍스트 어텐션 마스크를 담음
    )

    # attach pruning masks to the model and load the pretraining weights
    #pruning 적용 여부 
    #utils.prune_utils의 make_prunable 함수
    # if not args.dense: # --dense 옵션이 없으면 pruning mask 적용
    #     print("<pruning 적용 -> make_prunable 함수 호출>")
    #     make_prunable(model, pattern_lock=True, mask_on_the_fly=True)
    #     model.load_from_pruned_pretrained(args.pretraining_weights, args.mask, config, is_eval=False)
    #     try:
    #         check_mask_scope(model, expected=args.expected_mask_scope if hasattr(args, 'expected_mask_scope') else 'auto')
    #     except Exception as e:
    #         print(f"[warn] mask scope check skipped: {e}")
    # else:# --dense 옵션 있으면 pruning 없이 모델 실행
    #     print("<--dense옵션 -> pruning 없이 dense 모델 학습>")
    #     #blip/blip_retrieval.py의 BLIPRetrieval클래스 load_pretrained()함수 호출
    #     #dense한 모델의 pre-trained weight 로드하여 어떤 파라미터들이 load되고, 누락되었는지 확인
    #     model.load_pretrained(args.pretraining_weights, is_eval=False) 
    #     #일부 layer를 prunable한 구조로 변환(하지만, 실제 pruning mask를 적용하지는 않음)
    #     #dense인데 왜 prunable하게? -> Pruning할 때와 안 할 때의 모델 구조를 일관되게 통일시킴
    #     make_prunable(model, pattern_lock=False, mask_on_the_fly=False)
    #     try:
    #         check_mask_scope(model, expected='none')
    #     except Exception as e:
    #         print(f"[warn] mask scope check skipped: {e}")
    if not args.dense:
        make_prunable(model, pattern_lock=True, mask_on_the_fly=True)
        model.load_from_pruned_pretrained(args.pretraining_weights, args.mask, config, is_eval=False)
    else:
        model.load_pretrained(args.pretraining_weights, is_eval=False)
        make_prunable(model, pattern_lock=False, mask_on_the_fly=False)

    hard_disable_mask_on_fly_and_apply_once(model)
    debug_mask_on_fly(model)

    for p in model.parameters():
        p.requires_grad = False

    train_keys = ["logit_scale", "text_projection", "visual_projection", "proj"]
    for n, p in model.named_parameters():
        if any(k in n for k in train_keys):
            p.requires_grad = True

    trainable_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[Debug] trainable params =", trainable_numel)
    # if running with BLIP, setup the external momentum encoders
    #Momentum encoder는 blip의 contrastive learning에서 사용됨
    if args.model == "blip":
        # de-register the momentum parameters from the model
        #blip_retrieval.py의 init()함수에서 생성한 모델 객체 내부의 momentum encoder를 변수에 저장(복사)
        #모델 내부에서만 제거하고, 인스턴스 자체는 메모리에 남아있는 것
        visual_encoder_m = model.visual_encoder_m
        text_encoder_m = model.text_encoder_m
        vision_proj_m = model.vision_proj_m
        text_proj_m = model.text_proj_m
        #변수에 저장해놨고, 모델 내부에서는 사용하지 않을 것이기 때문에 모델에서 momentum encoder 속성 삭제함
        #모델이 evaluation 시에 불필요한 메모리 사용 X -> 메모리 절약, 모델 크기 감소
        delattr(model, 'visual_encoder_m')
        delattr(model, 'text_encoder_m')
        delattr(model, 'vision_proj_m')
        delattr(model, 'text_proj_m')
        #Training 중에는 momentum encoder가 계속 사용될 가능성이 있기에 GPU로 이동시킴
        visual_encoder_m.to(fabric.device)
        text_encoder_m.to(fabric.device)
        vision_proj_m.to(fabric.device)
        text_proj_m.to(fabric.device)
    
    # log some stats regarding the pruned parameters
    print(f"Total Params: {millions(num_params(model)):.2f}M")
    #prune.utils.py의 named_masked_parameters()함수 호출
    #전체 param에서 prunable하지 않은 parameter 제외 -> remaining param
    remaining_params, total_params = stats(named_masked_parameters(model, exclude=get_unprunable_parameters(model.name)))
    print(f"Remaining params: {millions(remaining_params, decimals=2)} / {millions(total_params, decimals=2)} ({remaining_params/total_params*100:.2f}%)")
    # when everything is ready, distribute
    model = fabric.setup_module(model) # 모델을 분산 학습 환경에 맞게 설정
    if args.zero_shot:
        # 분산용 세팅
        # keep track of time
        start_time = time.time() # 학습 시작 시간 기록
        # 학습 안 하므로 바로 평가
        model.eval()
        fabric.barrier()

        with torch.no_grad():
            # Test 
            score_test_i2t, score_test_t2i = evaluation(
                model, test_loader, tokenizer, fabric, config, debug_mode=args.debug
            )
            test_result = compute_metrics(score_test_i2t, score_test_t2i, dataset=test_dataset)
            print("[Zero-Shot] Test:", test_result)
            fabric.log_dict({f'test_{k}': v for k, v in test_result.items()})

            # 로그 파일 남기기(선택)
            if fabric.is_global_zero:
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps({"zero_shot_test": test_result}) + "\n")
                     # display the total time and exit
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Total Time for Image-Text Retrieval Finetuning {}'.format(total_time_str))
            return
    # load the configuration (they remain the same across resumes)
    arg_opt = utils.AttrDict(config['optimizer']) # optimizer 설정 저장
    optimizer = create_optimizer(arg_opt, model) # 저장된 설정에 맞는 optimizer 생성

    # once the model weights are initialized, distribute everything
    #분산 환경에서 사용할 수 있도록 함 
    optimizer = fabric.setup_optimizers(optimizer)

    # load and resume from a snapshot if provided
    #snapshot에 저장된 학습 상태를 불러옴 
    if os.path.exists(args.snapshot):
        snapshot = fabric.load(args.snapshot)
        model.load_state_dict(snapshot['model_state']) # 모델 가중치 불러옴
        optimizer.load_state_dict(snapshot['optimizer_state']) # Optimizer 상태 가져옴
        sched_state_dict = snapshot['scheduler'] # 학습률 스케줄러 상태 -> 스케줄러의 상태만 불러오는 것
        epochs_run = snapshot['epochs_run'] + 1 # 마지막으로 학습된 epoch
        best_r_mean = snapshot['best_r_mean'] # 현재까지 가장 좋은 평가 점수
        best_epoch = snapshot['best_epoch'] # 가장 좋은 평가 점수를 받은 epoch
        print(f"Resuming training from epoch {epochs_run}. \
               best_r_mean = {best_r_mean} obtained at epoch {best_epoch}")
        print(
            "IMPORTANT: You are resuming training from a snapshot.\n"
            "As per the README.md, note that while the code for resuming is given, it has not been tested.\n"
            "The authors are not responsible for any issues that may arise from resuming training.\n\n"
        )
    else: # snapshot이 없는 경우 -> 학습을 처음부터 시작
        epochs_run = 0
        best_r_mean = 0
        best_epoch = 0
        sched_state_dict = None
        print("No snapshot exists. Starting training from scratch...")

    # LR SCHEDULER initialization
    # 학습률 스케줄러를 사용하여 학습 중 학습률이 적절하게 조절되도록 함
    #snapshot이 있는 경우 스케줄러 생성 후, 상태를 주입시킴 / 없는 경우 새로 학습 시작을 위한 생성임
    
    #epoch는 데이터셋 전체를 한 번 학습하는 것/ step은 1개의 배치를 학습하는 단위
    #데이터셋 크기 : 10000, batch size : 100 -> 1 epoch는 10000개의 데이터셋 한 번 학습, 1 step은 100개의 데이터(배치)를 학습
    #1 epoch는 100 step으로 이루어져 있음 / 10 epoch -> 1000 step

    arg_sche = utils.AttrDict(config['scheduler']) #스케줄러 생성 관련 설정 가져옴
    steps_per_epoch = math.ceil(len(train_dataset) / config['batch_size_target']) #한 epoch동안 필요한 step 수 계산
    num_training_steps = steps_per_epoch * arg_sche['epochs'] # 전체 학습 step수
    num_warmup_steps = int(num_training_steps * arg_sche['num_warmup_steps']) #학습 초기에 낮은 학습률로 시작했다가 점진적으로 증가시키기 위한 warmup_step 수
    lr_scheduler = create_scheduler( # 스케줄러 생성
        mode=arg_sche['sched'], 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        total_steps=num_training_steps, 
        last_epoch=-1 if sched_state_dict is None else sched_state_dict['last_epoch']-1 
        #snapshot이 없는 경우 epoch = -1로 새로운 학습률로 생성 / 있는 경우 last_epoch로 설정하여 마지막 학습 지점에서 이어서 학습
    )
    if sched_state_dict is not None: 
        #snapshot이 있는 경우, 기존 스케줄러 상태(학습률, step.. 등 모든 정보)를 불러와서 그대로 적용(이전 학습의 학습률 스케줄을 그대로 이어감)
        lr_scheduler.load_state_dict(sched_state_dict)
    
    # Dataloader를 분산 학습 환경에 맞게 설정
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader, use_distributed_sampler=False)

    # keep track of time
    start_time = time.time() # 학습 시작 시간 기록

    # grab other training hyperparameters
    max_epoch = config['scheduler']['epochs'] #학습할 총 epoch 수를 설정
    save_freq = config['save_freq'] if 'save_freq' in config else 1 #모델을 저장하는 주기(epoch) 설정 -> snapshot저장

    # start fault-tolerant distributed training
    if epochs_run == max_epoch: #모든 epoch 학습을 마친 경우 (snapshot이 있는 경우)
        done_training = True
    else:
        done_training = False # 아닌 경우, 학습 시작
        print("Start training", flush=True)



    for epoch in range(epochs_run, max_epoch): #epochs_run부터 max_epoch까지 반복
        
        # NOTE: this is needed for data distribution across gpus since they are not in-sync 
        # before the 
        #  pass!
        train_loader.sampler.set_epoch(epoch) # 분산학습 시 각 GPU가 같은 epoch에서 같은 순서로 샘플링하도록 동기화
        # train one epoch
        # train 시작 -> 반복문 내 한 epoch 동안 학습 수행 (하나의 epoch 내 여러 step으로 training)
        train_stats = train( 
            model, train_loader, optimizer, fabric, lr_scheduler, epoch, config, args.debug, 
            visual_encoder_m=visual_encoder_m, text_encoder_m=text_encoder_m, 
            vision_proj_m=vision_proj_m, text_proj_m=text_proj_m
        )

        # evaluate on the validation set
        # 위에서 정의한 evaluation 함수로 val_loader(검증 데이터셋)을 사용해 I2T, T2I 작업 수행
        # score_val_i2t : image-to-text 검색 점수 / score_val_t2i : text-to-image 검색 점수
        # evaltools/itr_utils.py의 blip_itr_evaluation()함수 호출
        score_val_i2t, score_val_t2i = evaluation(model, val_loader, tokenizer, fabric, config, debug_mode=args.debug)
        #validation 점수 계산
        val_result = compute_metrics(score_val_i2t, score_val_t2i, dataset=val_dataset)
        print(val_result)
        
        # log validation data on wandb
        val_dict = {f'val_{k}': v for k, v in val_result.items()}
        fabric.log_dict(val_dict)

        # log train and validation data on disk
        if fabric.is_global_zero:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in val_result.items()},
                        'epoch': epoch}
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # saving best model
        if val_result['r_mean'] > best_r_mean:
            ckpt = {
                'model_state': model,
                'optimizer_state': optimizer,
                'scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epochs_run': epoch,
            }
            fabric.save(os.path.join(args.output_dir, 'checkpoint_best.pt'), ckpt)
            best_r_mean = val_result['r_mean']
            best_epoch = epoch

        # saving model every :save_freq: epochs on each node
        if (epoch+1) % save_freq == 0:
            snapshot = {
                'model_state': model,
                'optimizer_state': optimizer,
                'scheduler': lr_scheduler.state_dict(),
                'epochs_run': epoch,
                'best_r_mean': best_r_mean,
                'best_epoch': best_epoch
            }
        fabric.save(args.snapshot, snapshot)

        if epoch == max_epoch - 1: done_training = True
        if args.incremental: break


    # when training ends, write on disk the best epoch
    if fabric.global_rank == 0:
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)

    # finally, evaluate the best model on the test set
    # Training 종료 후 Inference
    if done_training: 
        print("Training completed. Start testing", flush=True)

        # make sure processes are synchronized before loading the model
        fabric.barrier()

        # NOTE: here I load with fabric since the model is one the gpu already
        snapshot = fabric.load(os.path.join(args.output_dir, 'checkpoint_best.pt')) # checkpoint 불러옴
        model.load_state_dict(snapshot['model_state']) # checkpoint 적용
        score_test_i2t, score_test_t2i = evaluation(model, test_loader, tokenizer, fabric, config, debug_mode=args.debug) # test dataset에 대해 evaluation 평가 수행
        test_result = compute_metrics(score_test_i2t, score_test_t2i, dataset=test_dataset)
        print(test_result)

        # log test data on wandb
        test_dict = {f'test_{k}': v for k, v in test_result.items()}
        fabric.log_dict(test_dict)

        # log test data on disk
        if fabric.global_rank == 0:
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(test_result) + "\n")

    # display the total time and exit
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total Time for Image-Text Retrieval Finetuning {}'.format(total_time_str))

def _pbool(x):  # "false"/"0"/"no"도 False로 파싱
    return str(x).strip().lower() in ('1','true','t','yes','y','on')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['xvlm', 'blip', 'clip', 'clipG', 'blip2'],)
    parser.add_argument('-pre', '--pretraining_weights', type=str, required=False, default="")
    parser.add_argument('-m', '--mask', type=str, required=False, 
                        help="Path to the pruning mask to load. If not provided, make sure to set --dense.")
    parser.add_argument('--dense', action='store_true', default=False, 
                        help="Train the dense model. This flag overrides any given pruning mask")
    parser.add_argument('--snapshot', type=str, required=False, default="snapshot.pt", 
                        help="Path to the snapshot to load and/or save. If not provided, the default is 'snapshot.pt'. "
                        "If the snapshot exists, the training will resume from there. If it doesn't, the training will start from scratch.")
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the .yaml configuration file of the script. For convenience, you can use "
                        "configs/xvlm/retrieval.yaml or configs/blip/retrieval.yaml.")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Path to the output directory of the script. This includes the logs and the checkpoint of the best model on the validation set.") 
    parser.add_argument('--seed', default=42, type=int, 
                        help="Seed for reproducibility. Default is 42.")
    parser.add_argument('-wdb', '--wandb', action='store_true', default=False, required=False, 
                        help='Whether or not to log data on Weights & Biases. Please remember to login first, e.g., via `wandb login`')
    parser.add_argument('-exp', '--experiment_name', type=str, default=None, required=False, help='Name of the experiment on wandb. \
                        Will override the config if given.')
    parser.add_argument('--wdb_offline', action='store_true', default=False, required=False, help='Whether or not to log data on wandb. \
                        Remember to run wandb sync at the end of the training if this flag is active.')
    parser.add_argument('--devices', type=int, default=1, required=False, 
                        help='Number of devices (i.e. gpus) to use with Lightning Fabric and DDP. Default is 1.')
    parser.add_argument('--precision', type=str, default='bf16-mixed', required=False, choices=['32-true', 'bf16-mixed', '16-mixed'], 
                        help='Precision to use for training. Default is bf16-mixed.')
    parser.add_argument('--debug', action="store_true", 
                        help="Enable debugging mode to ensure you can run the script on this machine without errors. "
                        "Will only run a few training batches as well as a few validation batches per epoch. Default is False.")
    parser.add_argument('--incremental', action="store_true", 
                        help="Enable incremental training. Will only run one epoch and then stop. Default is False. "
                        "If given, ensure to correctly set the --snapshot flag to resume training. Note that snapshot code is given, but not tested. "
                        "This also holds for every other finetuning script.")
    parser.add_argument("--zero_shot", action="store_true", help="학습 없이 사전학습 가중치로 바로 평가만 수행")
    parser.add_argument('--drop_percent', type=float, default=None,
                        help='채널 슬라이스 비율(%) — 예: 90이면 입력 채널의 90%%를 드롭(keep=10%%). 지정하지 않으면 슬라이스 비활성.')
    parser.add_argument('--rotation', type=str, default='None', choices=['axisvar_top', 'axisvar_random', 'axisvar_bottom'],
                    help='회전 후 채널 슬라이스 방법 — topk, random, bottomk 중 선택. 지정하지 않으면 기본값 사용.')
    parser.add_argument('--enabled', type=_pbool, default=None)
    parser.add_argument('--axis', type=str, default='None', choices=['dim', 'token'],)

    # load the main config
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    

    assert set(config.keys()) != set(vars(args).keys()), "Config and command line arguments must not overlap"
    
    # mixup command line arguments and config
    config.update(vars(args))

    # 1) CLI 인자 우선 적용 (None이 아니면 사용)
    rs = dict(config.get("runtime_slice", {}))

    # drop_percent
    if args.drop_percent is not None:
        dp = float(args.drop_percent)
        rs["drop_percent"] = dp/100.0 if dp > 1.0 else dp
    else:
        v = float(rs.get("drop_percent", 0.0))
        rs["drop_percent"] = v/100.0 if v > 1.0 else v
    def _pbool(x): return str(x).lower() in ('1','true','t','yes','y','on')
    # enabled
    if args.enabled is not None:
        rs["enabled"] = bool(args.enabled)
    elif args.enable is not None:  # 구버전 호환
        rs["enabled"] = _pbool(args.enable)
    rs.setdefault("enabled")

    # axis
    if args.axis is not None:
        rs["axis"] = args.axis
    rs.setdefault("axis", "dim")

    # rotation
    if args.rotation is not None:
        rs["rotation"] = args.rotation
    rs.setdefault("rotation")

    # 나머지 옵션 기본값 유지/보완
    rs.setdefault("center_input", False)
    rs.setdefault("pca_q_extra", 16)

    config["runtime_slice"] = rs

    # 확인용 로그
    print(f"[runtime_slice] enabled={rs['enabled']}  "
        f"drop_percent={rs['drop_percent']:.4f}  "
        f"keep_ratio={(1.0 - rs['drop_percent']):.4f}"
        f"  rotation={rs['rotation']}"
        f"  axis={rs['axis']}")
    # dump the config in the output folder for future reference
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    # launch the main function
    main(args, config)
