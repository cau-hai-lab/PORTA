import argparse
import os
import math
import ruamel.yaml as yaml
import time
import datetime
import json
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import lightning as L

from models import XVLMCaptioning, BLIPCaptioning, BLIP2Captioning

import utils
from utils.misc import millions, num_params
from utils.loggers import init_wandb_logger
from utils.functions import get_unprunable_parameters
from utils.prune_utils import make_prunable, stats, named_masked_parameters
from utils.optim import create_optimizer, create_scheduler

from datasets.coco_karpathy_dataset import coco_karpathy_train_collate_fn
from datasets import create_dataset, create_sampler, create_loader
# captioning.py 상단 import 근처 추가
from torchvision.transforms.functional import to_pil_image  # 예비 폴백용
from torchvision.datasets.utils import download_url         # GT 다운로드용

from evaltools import coco_caption_eval


def train(model, data_loader, optimizer, scheduler, epoch, fabric: L.Fabric, config, debug_mode=False):
    
    model.train() # 모델을 학습 모드로 설정
    optimizer.zero_grad() # 옵티마이저의 기울기를 초기화
    
    steps = 0
    steps_per_epoch = math.ceil(len(data_loader.dataset) / (config['batch_size_target'])) # 한 epoch당 step 수
    for i, (image, caption, ids) in enumerate(data_loader): # data_loader에서 batch 단위로 데이터 불러옴

        # forward and backward under appropriate context manager
        # 현재 step에서 optimizer.step()을 호출할지 여부를 결정
        is_accumulating = not (((i+1) % config['grad_acc_steps'] == 0) or ((i+1) == len(data_loader))) 
        with fabric.no_backward_sync(model, enabled=is_accumulating): # optimizer.step()을 호출하지 않음
            loss = model(image, caption, already_tokenized=True) # blip_captioning.py의 forward() 함수 호출
            loss /= config['grad_acc_steps'] # loss를 grad_acc_steps로 나누어 기울기를 누적하며 평균화
            fabric.backward(loss) # loss에 대한 기울기를 계산함

        # optimization step
        if not is_accumulating: # optimizer.step()을 호출하는 경우
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
            # log on the web
            # fabric.log_dict()를 통해 loss와 learning rate를 기록함
            loss_scalar = loss.item()
            lr_value = optimizer.param_groups[0]["lr"]
            if (steps+1) % config['log_freq'] == 0:
                fabric.log_dict({"loss": loss_scalar, "lr": lr_value})
            
            # log locally
            print(f"[Epoch {epoch+1}] Step = {steps+1} / {steps_per_epoch}\t loss = {loss_scalar}")
            steps += 1

        # if --debug is passed to the main script
        if debug_mode and steps == 16: break



@torch.no_grad()
def evaluation(result_dir, model, data_loader, epoch, fabric: L.Fabric, config, split):
    
    assert split in ("val", "test")
    
    model.eval()
    result = []
    for batch_idx, (image, image_id) in enumerate(data_loader):

        captions = model.generate(
            image, 
            sample=False, 
            num_beams=config['num_beams'], 
            max_length=config['max_length'],
            min_length=config['min_length']
        )

        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})

        if ((batch_idx + 1) % config['log_freq'] == 0) or (batch_idx == len(data_loader)-1):
            progress = (batch_idx + 1) / len(data_loader) * 100
            print(f"[({split}) Caption Generation] Batch = {batch_idx+1} / {len(data_loader)} ({progress:.2f}%)")

    # dump on disk the json file containing the 
    # captions generated on each rank
    captions_path = os.path.join(result_dir, f"captioning_{split}_result_epoch{epoch}_rank{fabric.global_rank}.json")
    with open(captions_path, "w") as f:
        json.dump(result, f)
    return result


# captioning.py 에 유틸 함수 추가
def dedup_by_image_id(json_path: str) -> int:
    with open(json_path, "r") as f:
        items = json.load(f)
    seen = set()
    out = []
    for it in items:
        iid = int(it["image_id"])
        if iid in seen:
            continue
        seen.add(iid)
        out.append({"image_id": iid, "caption": it["caption"]})
    with open(json_path, "w") as f:
        json.dump(out, f)
    return len(out)


def merge_eval_from_ranks(result_dir, epoch, fabric, split):
    assert split in ("val", "test")
    
    result = []
    for r in range(fabric.world_size):
        path_for_this_rank = os.path.join(result_dir, f"captioning_{split}_result_epoch{epoch}_rank{r}.json")
        with open(path_for_this_rank, "r") as f:
            r_result = json.load(f)
            result += r_result
    
    result_path = os.path.join(result_dir, f"captioning_{split}_result_epoch{epoch}.json")
    with open(result_path, "w") as f:
        json.dump(result, f)
    return result_path

import glob, os, json


def cleanup_eval(result_dir, epoch, fabric, split):
    for r in range(fabric.world_size):
        path_for_this_rank = os.path.join(result_dir, f"captioning_{split}_result_epoch{epoch}_rank{r}.json")
        os.remove(path_for_this_rank)

def main(args, config):
    if "16" in args.precision:
        torch.set_float32_matmul_precision(precision="medium")
    elif "32" in args.precision:
        torch.set_float32_matmul_precision(precision="high")

    # setup the loggers before distributing
    loggers = []
    if args.wandb:
        loggers.append(init_wandb_logger(config))

    # initialize distributed mode
    # 분산 학습을 위한 Fabric 초기화
    fabric = L.Fabric(
        accelerator="cuda",
        devices=args.devices,
        strategy="ddp",
        precision=args.precision,
        loggers=loggers
    )
    fabric.launch()
    utils.setup_for_distributed(is_master=fabric.is_global_zero) # 분산 환경에서의 설정 
    
    # setup useful variables
    world_size = fabric.world_size
    global_rank = fabric.global_rank

    # define gradient accumulation steps
    # data에 따라 처리할 배치 사이즈를 조정
    E = config['batch_size_target']
    B = config['batch_size_train']
    config['grad_acc_steps'] = (E // world_size) // B # 몇 번째 step마다 optimizer.step()을 호출할지 결정

    # reproducibility settings
    L.seed_everything(args.seed)
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    # model initialization (for this script, I need to initialize the model before the dataloader)
    # since the dataloader will have a collate_fn made with a partial from model data
    print("Creating model...")
    if args.model == "xvlm":
        model = XVLMCaptioning(config=config)
        setattr(model, "name", "xvlm")
    elif args.model == "blip":
        print("[Debug] captioning.py :BLIP 모델 객체 생성 -> blip_captioning.py로 이동")
        model = BLIPCaptioning(
            image_size=config['image_res'], 
            vit=config['vit'], 
            vit_grad_ckpt=config['vit_grad_ckpt'], 
            vit_ckpt_layer=config['vit_ckpt_layer'], 
            prompt=config['prompt']
        )
        setattr(model, "name", "blip")
    elif args.model == "blip2":
        print("[Debug] captioning.py : BLIP2 모델 객체 생성 -> blip2_captioning.py로 이동")
        model = BLIP2Captioning(config=config)
        setattr(model, "name", "blip2")
    elif args.model == "flamingo":
        from models.flamingo.flamingo_captioning import FlamingoCaptioning
        print("[Debug] captioning.py : Flamingo 모델 객체 생성 -> flamingo_captioning.py로 이동")
        model = FlamingoCaptioning(config=config)
        setattr(model, "name", "flamingo")
    else:
        raise NotImplementedError(f"Model {args.model} not implemented!")


    if not args.dense: # pruned 한 모델인 경우
        print("[Debug] captioning.py : Pruned 모델")
        #utils.prune_utils의 make_prunable 함수 호출 : 모델 파라미터에 pruning mask를 적용할 수 있도록 함
        make_prunable(model, pattern_lock=True, mask_on_the_fly=True)
        # if loading from a pretrained checkpoint after the 1st stage finetuning on the 4M data, it is assumed
        # that the state dict contains the pruning masks (xvlm only)
        if args.model == 'xvlm':
            masks_for_loading = args.mask if not args.load_capt_pretrain else None
        elif args.model == 'blip':
            # pruning mask의 경로를 외부에서 지정한 경로로 사용(실행시 매개변수에 넣어줌)
            masks_for_loading = args.mask
        model.load_from_pruned_pretrained(args.pretraining_weights, args.mask, config)
    else: # dense 한 모델인 경우
        # model.load_pretrained(args.pretraining_weights, is_eval=False) 
        model.load_pretrained(args.pretraining_weights, config, load_capt_pretrain=args.load_capt_pretrain)
        # dummy op, but useful to check that the sparsity is 0% with the upcoming lines
        make_prunable(model, pattern_lock=False, mask_on_the_fly=False)

    # log some cute stuff
    print("Total Params: ", millions(num_params(model)))
    masked_params = named_masked_parameters(model, exclude=get_unprunable_parameters(model.name))
    remaining_params, prunable_params = stats(masked_params)
    print(f"Remaining Params = {millions(remaining_params, decimals=2)} / {millions(prunable_params, decimals=2)}", end=" ")
    print(f"({100*remaining_params/prunable_params:.2f}%)")
    
    # datasets initialization
    # 데이터셋 설정
    image_processor = model.image_processor
    print("Creating COCO Caption dataset...")
    print("[Debug] captioning.py : Dataset 불러옴 -> datasets/init.py create_data함수 호출")
    train_dataset, val_dataset, test_dataset = create_dataset('captioning', config, image_processor)

    # (distributed) samplers initialization    
    datasets = [train_dataset, val_dataset, test_dataset]
    samplers = create_sampler(
        datasets, 
        shuffles=[True, False, False], 
        num_replicas=world_size, 
        global_rank=global_rank, 
        is_eval=[False, True, True] # this enables DistributedEvalSampler for val and test
    )
    
    # dataloaders initialization
    # NOTE: the collate function already embeds the tokenization process
    # train_collate_fn(batch)으로 호출할 수 있도록 "단축 함수"를 정의한 것
    # Batch를 넘겨주면 모델에 넣기 편한 형태로 변환하여 줌
    train_collate_fn = partial(coco_karpathy_train_collate_fn, tokenizer=model.tokenizer, max_tokens=config['max_tokens'])
    
    train_loader, val_loader, test_loader = create_loader(
        datasets, 
        samplers,
        batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
        num_workers=[8, 8, 8], 
        is_trains=[True, False, False],
        collate_fns=[train_collate_fn, None, None]
    )

    # everything about data management is ready, so set it up with fabric
    # 이미 생성해둔 dataloader를 fabric분산 환경에 맞게 설정
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader, use_distributed_sampler=False)
    
   # -------------------------------
    # Zero-Shot 분기: 학습 생략, 평가만
    # -------------------------------
    if args.zero_shot:
        start_time = time.time()

        model = fabric.setup_module(model)
        model.eval()
        epoch = 0

        # ✅ 모든 rank가 evaluation을 수행해서 rank별 json을 만든다
        evaluation(args.result_dir, model, test_loader, epoch, fabric, config, split="test")

        # ✅ 모든 rank가 barrier에 참여해야 함 (데드락 방지)
        fabric.barrier()

        log_stats = None
        if fabric.is_global_zero:
            test_result_file = merge_eval_from_ranks(args.result_dir, epoch, fabric, split="test")
            cleanup_eval(args.result_dir, epoch, fabric, split="test")

            if not args.skip_eval:
                print("[Debug] captioning.py : 평가 결과 Scoring 수행 -> coco_caption_eval()")
                coco_test = coco_caption_eval(config['test_gt_file'], test_result_file)
                log_stats = {**{f'test_{k}': round(v*100, 4) for k, v in coco_test.eval.items()}}
                fabric.log_dict(log_stats)
                print(log_stats)

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps({"zero_shot_test": log_stats}) + "\n")

        # ✅ (선택) 마무리 barrier 한 번 더 두면 깔끔
        fabric.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Total Time for Image Captioning Evaluation {}'.format(total_time_str))
        return

    # optimizer initialization
    # 옵티마이저 생성
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    # setup model and optimizer with fabric
    # 모델과 옵티마이저를 fabric 분산 환경에 맞게 설정
    model, optimizer = fabric.setup(model, optimizer)

    # resume from a snapshot if existing
    # NOTE: the code for resuming from a snapshot is given, but it is not tested
    # during the experiments of the main paper, I have never used it
    start_epoch = 0
    start_time = time.time()
    sched_state_dict = None
    if os.path.exists(args.snapshot): # 이미 저장된 snapshot이 있는 경우
        # 마지막으로 실행된 시점의 상태를 불러옴
        restored_state = fabric.load(args.snapshot)
        model.load_state_dict(restored_state['model'])
        optimizer.load_state_dict(restored_state['optimizer'])
        sched_state_dict = restored_state['scheduler']
        start_epoch = restored_state['last_epoch'] + 1
        start_time = time.time() - restored_state['elapsed_time']
        print(f"Loaded state, resuming from epoch = {start_epoch}")
        print(
            "IMPORTANT: You are resuming training from a snapshot.\n"
            "As per the README.md, note that while the code for resuming is given, it has not been tested.\n"
            "The authors are not responsible for any issues that may arise from resuming training.\n\n"
        )
    
    # learning rate scheduler initialization
    arg_sche = utils.AttrDict(config['scheduler']) # 스케줄러 설정 불러옴
    steps_per_epoch = math.ceil(len(train_dataset) / config['batch_size_target']) # 한 epoch당 step 수
    num_training_steps = steps_per_epoch * arg_sche['epochs'] # 전체 학습 step 수
    num_warmup_steps = int(num_training_steps * arg_sche['num_warmup_steps']) # warmup step 수
    lr_scheduler = create_scheduler( # 스케줄러 생성
        mode=arg_sche['sched'], 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        total_steps=num_training_steps, 
        last_epoch=-1 if sched_state_dict is None else sched_state_dict['last_epoch']-1 
    )
    if sched_state_dict is not None: # 이미 저장된 snapshot이 있는 경우
        lr_scheduler.load_state_dict(sched_state_dict) # 스케줄러 상태 불러옴

    # start training! :) 
    max_epoch = config['scheduler']['epochs'] 
    for epoch in range(start_epoch, max_epoch):
        
        # needed to reshuffle the distribution of data among ranks epoch-wise
        # 각 데이터 loader의 sampler에 epoch을 설정하여 매 epoch마다 데이터 분포를 재조정
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)

        # one training epoch
        # 학습 시작
        train(model, train_loader, optimizer, lr_scheduler, epoch, fabric, config, debug_mode=args.debug)

        # save model checkpoint after training for an epoch
        state = {'model': model, 'optimizer': optimizer, 'scheduler': lr_scheduler.state_dict(), 
                 'last_epoch': epoch, 'elapsed_time': time.time() - start_time}
        fabric.save(path=args.snapshot, state=state) # training 후 snapshot 저장

        # evaluate the model
        if epoch >= config['start_eval']: # config['start_eval'] 이후에 평가 수행

            # generate captions for the validation set and the test sets
            evaluation(args.result_dir, model, val_loader, epoch, fabric, config, split="val") # Validation 데이터 평가(Inference)
            evaluation(args.result_dir, model, test_loader, epoch, fabric, config, split="test") # Test 데이터 평가(Inference)

            # merge the results from all ranks and clean up the json files (only on rank 0 to avoid locking issues)
            fabric.barrier()
            if fabric.is_global_zero:
                # 분산 학습 환경에서 처리된 평가 결과를 합침
                val_result_file = merge_eval_from_ranks(args.result_dir, epoch, fabric, split="val")
                cleanup_eval(args.result_dir, epoch, fabric, split="val")
                test_result_file = merge_eval_from_ranks(args.result_dir, epoch, fabric, split="test")
                cleanup_eval(args.result_dir, epoch, fabric, split="test")

                # compute scores using the PyCOCO APIs
                # NOTE: if you pass --skip_eval to the main script, evaluation will not be performed
                # This is useful if you want to evaluate the results on a different machine (my use case during development of the paper) 
                # You are responsible for evaluating these files offline, using the function `coco_caption_eval` from evaltools/ic/__init__.py
                if not args.skip_eval:
                    print("[Debug] captioning.py : 평가 결과 Scoring 수행 -> evaltools/ic/__init__.py의 coco_caption_eval()함수 호출")
                    
                    coco_val = coco_caption_eval(config['val_gt_file'], val_result_file) # Validation 데이터 평가
                    coco_test = coco_caption_eval(config['test_gt_file'], test_result_file) # Test 데이터 평가

                    log_stats = {**{f'val_{k}': round(v*100, 4) for k, v in coco_val.eval.items()},
                                **{f'test_{k}': round(v*100, 4) for k, v in coco_test.eval.items()},
                                'epoch': epoch}
                    fabric.log_dict(log_stats)
                    print(log_stats)
                else:
                    print("Skipping evaluation...")

        # synchronize all processes before the next epoch
        fabric.barrier()

    # final logging and exit
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total Finetuning Time for Image Captioning {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, choices=['xvlm', 'blip', 'blip2', 'flamingo'],)
    parser.add_argument('-pre', '--pretraining_weights', type=str, required=False)
    parser.add_argument('-m', '--mask', type=str, required=False, 
                        help="Path to the pruning mask to load. If not provided, ensure to pass the --dense flag.")
    parser.add_argument('--dense', action="store_true", default=False, 
                        help="If passed, the dense model will be trained (even if you pass a mask).")
    parser.add_argument('--config', required=True, type=str, 
                        help="Path to the .yaml file containing the configuration for the script. "
                        "For simplicity, you can use the provided .yaml files in the configs/ folder. "
                        "For example, you can use the configs/xvlm/captioning.yaml file for XVLM, and "
                        "the configs/blip/captioning.yaml file for BLIP.")
    parser.add_argument('--skip_eval', action="store_true", default=False, 
                        help="If passed, the evaluation step will be skipped. "
                        "And you fill find the files in the --output_dir folder for both val and test splits. "
                        "Have a look at the `coco_caption_eval` function in evaltools/ic/__init__.py to evaluate the results offline.")
    parser.add_argument('--output_dir', default='experiments/captioning', 
                        help="Path to the output directory where the results will be saved. "
                        "This includes a copy of the config and the json files with the generated captions.")
    parser.add_argument('--snapshot', default="snapshots/captioning.pt", 
                        help="Path to the snapshot file where the model will be saved after each epoch. " 
                        "Set this flag to an existing checkpoint to resume training from that point. "
                        "If the file does not exist, the script will start training from scratch.")
    parser.add_argument('--seed', default=42, type=int, 
                        help="Seed for reproducibility. Default is 42.")
    parser.add_argument('--load_capt_pretrain', action='store_true', 
                        help="If passed, the model will load the weights from a 1epoch pretraining on the 4M dataset. "
                        "XVLM only. Please see the XVLM paper, appendix A.2 for more details.")
    parser.add_argument('-wdb', '--wandb', action="store_true", default=False, 
                        help="Log data onWeights & Biases. Make sure to log to your account first with `wandb login`.")
    parser.add_argument('-exp', '--experiment_name', type=str, required=False, 
                        help="Name of the experiment on Weights & Biases.")
    parser.add_argument('--wdb_offline', action="store_true", 
                        help="Locally cache W&B data instead of logging it to the cloud. You can then sync the run with `wandb sync`.")
    parser.add_argument('--debug', action="store_true", default=False, 
                        help="Enables debugging mode to ensure everything works fine. Only 16 grad steps per epoch are performed.")
    parser.add_argument('--devices', type=int, default=1, 
                        help="Number of devices (i.e., gpus) to use for DistributedDataParallel with Fabric. Default is 1.")
    parser.add_argument('--precision', type=str, default='bf16-mixed', choices=['32-true', 'bf16-mixed', '16-mixed'], 
                        help="Precision strategy to use for training. Default is bf16-mixed (used in the paper).")
    # ... 기존 parser 정의 위/아래 맥락 유지 ...
    parser.add_argument('--zero_shot', action='store_true',
                        help="Zero-shot 평가만 수행합니다. 학습(파라미터 업데이트)을 건너뜁니다.")

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    os.makedirs(args.output_dir, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    
    args.result_dir = os.path.join(args.output_dir, 'result')
    os.makedirs(args.result_dir, exist_ok=True)

    # mixup command line arguments and config
    config.update(vars(args))
    main(args, config)