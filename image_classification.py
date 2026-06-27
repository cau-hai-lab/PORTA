import argparse
import os
import json
import time
import datetime
import contextlib
from typing import List

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms import InterpolationMode

import lightning as L

from transformers import CLIPModel, CLIPTokenizer
from models import CLIPClassification, CLIPGClassification
# Project utilities (expected to exist in your repo)
from datasets.vision_datasets import get_dataset
from utils.prune_utils import make_prunable


# ==========================
# Constants
# ==========================
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

BUILTIN_TEMPLATES = [
    "a photo of a {}",
    "a photo of the {}",
    "a blurry photo of a {}",
    "a close-up photo of a {}",
    "a bright photo of a {}",
    "a dark photo of a {}",
    "a cropped photo of a {}",
    "a photo of a small {}",
    "a photo of a large {}",
]

# ==========================
# CUDA timers
# ==========================
class CUDAEventTimer:
    """GPU-only timer (CUDA events)."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.total_ms = 0.0
        self.iters = 0
        self._start = torch.cuda.Event(enable_timing=True)
        self._end   = torch.cuda.Event(enable_timing=True)

    @contextlib.contextmanager
    def measure(self):
        if not self.enabled:
            yield
            return
        self._start.record()
        yield
        self._end.record()
        torch.cuda.synchronize()
        self.total_ms += self._start.elapsed_time(self._end)
        self.iters += 1

    @property
    def avg_ms(self) -> float:
        return self.total_ms / max(1, self.iters)
# ==========================
# Transforms / Data
# ==========================

def get_transforms(image_size: int = 224):
    test_tf = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    # Train TF is not needed for zero-shot, but keep a simple one for completeness
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    return train_tf, test_tf


# ==========================
# Text features
# ==========================

def build_text_features(classnames: List[str], tokenizer: CLIPTokenizer, model: CLIPModel, fabric: L.Fabric,
                        templates: List[str]):
    """Returns normalized (C, D) text feature matrix averaged over templates."""
    device = fabric.device
    all_feats = []
    with torch.no_grad(), fabric.autocast():
        for tmpl in templates:
            texts = [tmpl.format(c.replace("_", " ")) for c in classnames]
            toks = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            tfeat = model.get_text_features(**toks)  # (C, D)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            all_feats.append(tfeat)
    # average & renorm
    text_feats = torch.stack(all_feats, dim=0).mean(dim=0)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats  # (C, D)


# ==========================
# Evaluation
# ==========================

@torch.no_grad()
def evaluate_clip_zeroshot(model: CLIPModel, test_loader, classnames, fabric: L.Fabric,
                           templates: List[str]):
    model.eval()
    tokenizer = model.tokenizer

    # Precompute text features once
    text_feats = build_text_features(classnames, tokenizer, model, fabric, templates)  # (C, D)
    logit_scale = model.logit_scale.exp().to(fabric.device)

    correct = 0
    total = 0
    with fabric.autocast():
        for images, labels in test_loader:
            images, labels = fabric.to_device((images, labels))
            img_feats = model.get_image_features(pixel_values=images)  # (B, D)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = logit_scale * img_feats @ text_feats.t()  # (B, C)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    t = torch.tensor([correct, total], device=fabric.device, dtype=torch.long)
    fabric.all_reduce(t, reduce_op="sum")
    correct_g, total_g = t.tolist()
    return 100.0 * correct_g / max(1, total_g)

# ===== (2) eval_one_dataset: only_test=True + ImageNet 클래스명 치환 HOTFIX =====
def eval_one_dataset(dataset_name, fabric, model, args, templates):
    # transforms & dataset
    train_tf, test_tf = get_transforms(args.image_size)
    # (주의) 경로는 ./data/<dataset_name>
    train_ds, test_ds, classes, num_classes = get_dataset(
        dataset_name=dataset_name,
        data_dir=os.path.join(getattr(args, "data_root", "./data"), dataset_name),
        train_transform=train_tf,   # zero-shot이라 안씀
        test_transform=test_tf,
        only_test=True,             # ★ train 만들지 않음
    )

    # --- HOTFIX: ImageNet일 때 WNID를 사람이 읽는 이름으로 치환 ---
    if dataset_name == "imagenet":
        import re, pandas as pd
        if len(classes) > 0 and re.fullmatch(r"n\d{8}", str(classes[0])):  # 'n01440764' 패턴
            child_root = os.path.dirname(test_ds.root)  # "<...>/imagenet"
            meta_path  = os.path.join(child_root, "annots", "metadata.csv")
            if os.path.exists(meta_path):
                meta = pd.read_csv(meta_path)
                folder2class = dict(zip(meta["folder_name"].astype(str),
                                        meta["class_name"].astype(str)))
                classes = [folder2class.get(wnid, wnid) for wnid in classes]
            else:
                if fabric.is_global_zero:
                    print(f"[WARN] metadata.csv not found at {meta_path} — using WNID as class names (accuracy will be poor).")
    # -------------------------------------------------------------

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = fabric.setup_dataloaders(test_loader)

    # 평가
    acc = evaluate_clip_zeroshot(model, test_loader, classes, fabric, templates)
    return acc, len(test_ds), num_classes

# ===== (1) build_text_features: truncation=True 추가 =====
def build_text_features(classnames: List[str], tokenizer: CLIPTokenizer, model: CLIPModel, fabric: L.Fabric,
                        templates: List[str]):
    """Returns normalized (C, D) text feature matrix averaged over templates."""
    device = fabric.device
    all_feats = []
    with torch.no_grad(), fabric.autocast():
        for tmpl in templates:
            texts = [tmpl.format(c.replace("_", " ")) for c in classnames]
            toks = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)  # ★
            tfeat = model.get_text_features(**toks)  # (C, D)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            all_feats.append(tfeat)
    text_feats = torch.stack(all_feats, dim=0).mean(dim=0)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats  # (C, D)


def main(args):
    # MatMul precision hint
    if "32" in args.precision:
        torch.set_float32_matmul_precision("high")
    elif "16" in args.precision:
        torch.set_float32_matmul_precision("medium")

    # Fabric for DDP/precision
    fabric = L.Fabric(
        accelerator="cuda",
        strategy="ddp",
        precision=args.precision,
        devices=args.devices,
    )
    fabric.launch()

    # Reproducibility
    L.seed_everything(args.seed)
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if fabric.is_global_zero:
            os.makedirs(args.output_dir, exist_ok=True)

    # ===== 모델 준비 =====
    if args.model == "clip" :
        model = CLIPClassification()  # 하드코딩 모델ID면 그대로 사용
    else :
        model = CLIPGClassification()  # HF 모델 ID/경로 지정 가능
    if not args.dense:
        print("<pruning 적용 -> make_prunable>")
        make_prunable(model, pattern_lock=True, mask_on_the_fly=True)
        if args.mask:
            model.load_from_pruned_pretrained(mask_path=args.mask, is_eval=False)
    else:
        print("<--dense 옵션: 프루닝 없이 dense 모델>")
        # (옵션) 사전가중치 로드가 필요하면 인자 추가해서 사용
        model.load_pretrained(weights_ckpt="", is_eval=False)
        make_prunable(model, pattern_lock=False, mask_on_the_fly=False)

    # DDP 래핑
    model = fabric.setup_module(model)

    # 프롬프트 템플릿
    templates = BUILTIN_TEMPLATES if args.use_prompt_ensemble else [args.prompt_template]

    # ===== 어떤 데이터셋들을 돌릴지 결정 =====
    SUPPORTED = ["cifar10", "cifar100", "flowers102", "imagenet"]
    if args.all_datasets:
        datasets = SUPPORTED
    elif getattr(args, "datasets", None):  # ★ 안전 접근
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = [args.dataset]


    # ===== 루프 평가 =====
    results = []  # (name, acc, num_test, num_classes)
    for ds in datasets:
        if ds not in SUPPORTED:
            if fabric.is_global_zero:
                print(f"[WARN] Unsupported dataset: {ds} (skip)")
            continue

        fabric.barrier()
        start = time.time()
        acc, ntest, ncls = eval_one_dataset(ds, fabric, model, args, templates)
        elapsed = time.time() - start

        # 결과 dict 구성
        test_result = {
            "dataset": ds,
            "top1": round(acc, 4),
            "num_test": int(ntest),
            "num_classes": int(ncls),
            "time_sec": round(elapsed, 3),
            "model": args.model,
            "mask": args.mask or "",
            "batch_size": args.batch_size,
            "prompt_ensemble": bool(args.use_prompt_ensemble),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        # 콘솔 + W&B/Fabric 로그
        if fabric.is_global_zero:
            print(f"[{ds}] Zero-Shot Top-1: {acc:.2f}% | #test={ntest} | #classes={ncls} | time={int(elapsed)}s")

        # 숫자만 골라서 로깅(비수치 문자열은 로거가 싫어할 수 있음)
        numeric = {k: v for k, v in test_result.items() if isinstance(v, (int, float))}
        fabric.log_dict({f"test_{k}": v for k, v in numeric.items()})

        # JSONL(줄 단위) 로그 저장
        if fabric.is_global_zero:
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(test_result, ensure_ascii=False) + "\n")

        results.append((ds, acc, ntest, ncls))
        fabric.barrier()

    # ===== 요약 출력 =====
    if fabric.is_global_zero:
        print("\n=== Zero-Shot Summary ===")
        w = max(len(x[0]) for x in results) if results else 8
        print(f"{'dataset'.ljust(w)}  {'top1(%)':>8}  {'#test':>8}  {'#cls':>6}")
        for ds, acc, ntest, ncls in results:
            print(f"{ds.ljust(w)}  {acc:8.2f}  {ntest:8d}  {ncls:6d}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "flowers102", "imagenet"])
    ap.add_argument("--all_datasets", action="store_true", help="지원하는 모든 데이터셋을 순차 평가")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--datasets", type=str, default=None,
                    help="쉼표로 구분된 데이터셋 목록 (예: 'cifar10,imagenet')")
    ap.add_argument("--model", type=str, default="clip", choices=["clip","clipG"],
                    help="HF model id or local path for CLIP model")

    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("-p", "--precision", type=str, default="bf16-mixed",
                    choices=["32-true", "16-mixed", "bf16-mixed"])
    ap.add_argument("--seed", type=int, default=42)
    # ap.add_argument('--config', type=str, required=True, 
    #                     help="Path to the .yaml configuration file of the script. For convenience, you can use "
    #                     "configs/xvlm/retrieval.yaml or configs/blip/retrieval.yaml.")
    ap.add_argument("--mask", type=str, default=None, help="(Optional) Path to pruning mask compatible with your model")
    ap.add_argument("--dense", action="store_true", help="If set, runs dense model without pruning")
    ap.add_argument("--prompt_template", type=str, default="a photo of a {}")
    ap.add_argument("--use_prompt_ensemble", action="store_true", help="Use a built-in prompt set and average text embeddings")
    ap.add_argument("--zero_shot", action="store_true",
                    help="학습 없이 현재 가중치(및 마스크)로 test split만 바로 평가합니다.")
    ap.add_argument("--output_dir", type=str, default="results/classification/clip",
                help="로그/요약 파일을 저장할 디렉토리")
    args = ap.parse_args()
    main(args)
