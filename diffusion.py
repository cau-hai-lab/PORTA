# diffusion.py (상단 import 보강)
import os
import json
import time
import datetime
import argparse
from pathlib import Path

import torch
import lightning as L
import ruamel.yaml as yaml
import torch.backends.cudnn as cudnn                     

import utils
from utils.loggers import init_wandb_logger

from models import SD15Generator, SDXLGenerator 
from datasets import create_dataset      
from evaltools import diffusion_evaluation

from torch.utils.data import DataLoader                    
from datasets.utils import DistributedEvalSampler                            
from utils.prune_utils import make_prunable, stats, named_masked_parameters
from utils.functions import get_unprunable_parameters
from utils.misc import millions, num_params



# diffusion.py
def diffusion_collate(batch, tokenizer, max_tokens=77):
    texts, indices, refs = zip(*batch)
    enc = tokenizer(list(texts), padding='max_length', truncation=True,
                    max_length=max_tokens, return_tensors="pt")
    return enc.input_ids, enc.attention_mask, torch.tensor(indices, dtype=torch.long), list(refs)



def main(args, config):
    if args.precision == '32-true':
        torch.set_float32_matmul_precision(precision="high")
    elif args.precision in ('bf16-mixed', '16-mixed'):
        torch.set_float32_matmul_precision(precision="medium")

    loggers = []
    if args.wandb:
        loggers.append(init_wandb_logger(config))

    fabric = L.Fabric(
        accelerator="cuda",
        strategy="ddp",
        precision=args.precision,
        devices=args.devices,
        loggers=loggers
    )
    fabric.launch()
    utils.setup_for_distributed(is_master=fabric.is_global_zero)

    # reproducibility
    L.seed_everything(args.seed)
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # ----- 데이터셋/로더 -----
    print("[Debug] Creating diffusion dataset", flush=True)
    eval_dataset = create_dataset('diffusion', config)

    print("[Debug] Diffusion 모델 생성 및 평가 준비", flush=True)

    if args.model == "sdxl":
        model = SDXLGenerator(config)
        setattr(model, "name", "sdxl_generator")
    else:
        model = SD15Generator(config)
        setattr(model, "name", "sd15_generator")
    tokenizer = model.tokenizer
    evaluation = diffusion_evaluation # 구현해야 함


    collate_fn = lambda batch: diffusion_collate(batch, tokenizer, max_tokens=config.get('max_tokens', 77))  

    # 분산 평가 샘플러 (중복 방지)
    eval_sampler = DistributedEvalSampler(
        eval_dataset,
        num_replicas=fabric.world_size,
        rank=fabric.global_rank,
        shuffle=False
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.get('batch_size_test'),
        num_workers=config.get('num_workers', 1),
        pin_memory=True,
        sampler=eval_sampler,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )

    eval_loader = fabric.setup_dataloaders(eval_loader, use_distributed_sampler=False)

    if not args.dense: # --dense 옵션이 없으면 pruning mask 적용
        if args.model == "sdxl":
            print("<SDXL 1.0 pruning 적용 -> make_prunable 함수 호출>")
            make_prunable(model.text_encoder, pattern_lock=True, mask_on_the_fly=True)
            make_prunable(model.text_encoder_2, pattern_lock=True, mask_on_the_fly=True)
            mask_dict = {}
            if args.mask:  mask_dict["l14"]  = args.mask
            if args.mask2: mask_dict["bigg"] = args.mask2

            model.load_from_pruned_pretrained(
                pretraining_weights = config.get('pretraining_weights', None),  # str 또는 {"l14":..,"bigg":..} 허용
                mask_path           = mask_dict if mask_dict else None,
                config              = config,
                is_eval             = False
            )
        else:
            print("<SD1.5 pruning 적용 -> make_prunable 함수 호출>")
            make_prunable(model.text_encoder, pattern_lock=True, mask_on_the_fly=True)
            model.load_from_pruned_pretrained(args.pretraining_weights, args.mask, config, is_eval=False)
    else:# --dense 옵션 있으면 pruning 없이 모델 실행
        if args.model == "sdxl":
            print("<--dense옵션 -> pruning 없이 dense 모델 사용>")
            #일부 layer를 prunable한 구조로 변환(하지만, 실제 pruning mask를 적용하지는 않음)
            #dense인데 왜 prunable하게? -> Pruning할 때와 안 할 때의 모델 구조를 일관되게 통일시킴
            make_prunable(model.text_encoder, pattern_lock=False, mask_on_the_fly=False)
            make_prunable(model.text_encoder_2, pattern_lock=False, mask_on_the_fly=False)
        else:
            print("<--dense옵션 -> pruning 없이 dense 모델 사용>")
            make_prunable(model.text_encoder, pattern_lock=False, mask_on_the_fly=False)

    # log some stats regarding the pruned parameters
    print(f"Total Params: {millions(num_params(model.text_encoder)):.2f}M")
    #prune.utils.py의 named_masked_parameters()함수 호출
    #전체 param에서 prunable하지 않은 parameter 제외 -> remaining param
    remaining_params, total_params = stats(named_masked_parameters(model.text_encoder, exclude=get_unprunable_parameters(model.name)))
    print(f"Encoder 1 Remaining params: {millions(remaining_params, decimals=2)} / {millions(total_params, decimals=2)} ({remaining_params/total_params*100:.2f}%)")

    remaining_params, total_params = stats(named_masked_parameters(model.text_encoder_2, exclude=get_unprunable_parameters(model.name)))
    print(f"Encoder 2 Remaining params: {millions(remaining_params, decimals=2)} / {millions(total_params, decimals=2)} ({remaining_params/total_params*100:.2f}%)")


    model = fabric.setup_module(model) # 모델을 분산 환경에 맞게 설정
    model.eval()

    # keep track of time
    start_time = time.time() # 평가 시작 시간 기록

    results = evaluation(model, eval_loader, fabric, config, debug_mode=args.debug)

    if fabric.is_global_zero:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("[Done] results saved →", outdir / "results.json")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Total Time for Diffusion Evaluation {elapsed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="sd15", choices=['sd15', 'sdxl'])
    parser.add_argument('-pre', '--pretraining_weights', type=str, default="")
    parser.add_argument('-m', '--mask', type=str, required=False,
                        help="Path to the pruning mask to load. If not provided, set --dense.")
    parser.add_argument('-m2', '--mask2', type=str, required=False,
                    help="Path to the pruning mask to load. If not provided, set --dense.")
    parser.add_argument('--dense', action='store_true', default=False,
                        help="Use dense model (no pruning). Overrides any given pruning mask.")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the .yaml configuration file for diffusion eval.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save logs/results.json")
    parser.add_argument('--snapshot', type=str, default="snapshot.pt",
                        help="(unused here) kept for interface parity.")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('-wdb', '--wandb', action='store_true', default=False)
    parser.add_argument('-exp', '--experiment_name', type=str, default=None)
    parser.add_argument('--wdb_offline', action='store_true', default=False)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='bf16-mixed',
                        choices=['32-true', 'bf16-mixed', '16-mixed'])
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(vars(args))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    assert set(config.keys()) != set(vars(args).keys()), "Config and command line arguments must not overlap"  
    config.update(vars(args))

    main(args, config)