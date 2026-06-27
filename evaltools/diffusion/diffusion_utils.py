# evaltools/diffusion/diffusion_utils.py
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import os, glob
# =============== FID ===============
from torch_fidelity import calculate_metrics

# =============== CLIPScore ===============
import open_clip

# =============== PickScore ===============
from transformers import AutoProcessor, AutoModel

# =============== HPSv2 ===============
_HAS_HPSV2 = True
try:
    import hpsv2  # pip install hpsv2
except Exception as e:
    print(f"[Warn] hpsv2 not available: {e}")
    _HAS_HPSV2 = False


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def preflight_and_fix(dir_path, rewrite=True, force_resize=False):
    exts = ("*.png","*.jpg","*.jpeg")
    paths = []
    for ext in exts: paths += glob.glob(os.path.join(dir_path, ext))
    bad, fixed = [], 0
    for p in sorted(paths):
        try:
            with Image.open(p) as im:
                im.verify()  # 무결성 검사
            with Image.open(p) as im:
                im = im.convert("RGB")
                if force_resize:
                    im = im.resize((299, 299))   # Inception 입력 크기
                if rewrite:
                    im.save(p, format="JPEG", quality=95, subsampling=1)
                    fixed += 1
        except Exception as e:
            bad.append((p, repr(e)))
    print(f"[preflight] {dir_path}: total={len(paths)}, fixed={fixed}, bad={len(bad)}")
    for x in bad[:10]:
        print("  BAD:", x[0], "=>", x[1])
    return bad




# ---------------- Utils ----------------
def _to_uint8_bhwc(images: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W)|(B,H,W,C), [-1,1]|[0,1] -> (B,H,W,C) uint8 CPU tensor"""
    if images.dim() != 4:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    if images.shape[1] in (1, 3):  # BCHW
        x = images
        x = (x + 1) / 2 if x.min() < 0 else x
        x = (x.clamp(0, 1) * 255.0).round().byte().permute(0, 2, 3, 1).cpu()
        return x
    elif images.shape[-1] in (1, 3):  # BHWC
        x = images
        x = (x + 1) / 2 if x.min() < 0 else x
        x = (x.clamp(0, 1) * 255.0).round().byte().cpu()
        return x
    else:
        raise ValueError(f"Unexpected channel layout: {images.shape}")


def _save_batch(images_uint8_bhwc: torch.Tensor, indices: torch.Tensor, out_dir: Path, ext: str = ".jpg"):
    out_dir.mkdir(parents=True, exist_ok=True)
    for img, idx in zip(images_uint8_bhwc, indices.tolist()):
        Image.fromarray(img.numpy()).save(out_dir / f"{idx:08d}{ext}")


def _materialize_real_dir_from_dataset(eval_dataset, out_real_dir: Path):
    """
    방법 B: dataset이 가진 ref_path로 real 디렉토리 구성(심볼릭 링크 우선, 실패시 복사).
    파일명은 전역 인덱스 8자리로 고정.
    """
    if not hasattr(eval_dataset, "_items"):
        raise RuntimeError("Dataset must expose _items = List[(caption, ref_path)]")

    refs: List[str] = [p for _, p in eval_dataset._items]

    out_real_dir.mkdir(parents=True, exist_ok=True)
    # 깨끗이 비우기
    for f in out_real_dir.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass

    for i, src in enumerate(refs):
        srcp = Path(src)
        dst = out_real_dir / f"{i:08d}.jpg"   # 확장자도 통일
        try:
            with Image.open(srcp) as im:
                im = im.convert("RGB")
                im.save(dst, format="JPEG", quality=95, subsampling=1)
        except Exception as e:
            print(f"[Warn] real copy failed {srcp} -> {dst}: {e}")


def _compute_fid_dir2dir(gen_dir: Path, real_dir: Path, use_cuda: bool, cfg: Dict[str, Any]) -> float:
    def run(batch_size, num_workers, verbose):
        return calculate_metrics(
            input1=str(gen_dir),
            input2=str(real_dir),
            cuda=use_cuda,
            fid=True, isc=False, kid=False,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
            samples_find_deep=False,
        )["frechet_inception_distance"]

    # 1차: 설정값 시도
    bs  = int(cfg.get("fid_batch_size", 64))
    nw  = int(cfg.get("fid_num_workers", 8))
    try:
        return float(run(bs, nw, True))
    except Exception as e:
        print(f"[Warn] FID failed with bs={bs}, nw={nw}: {e}")
        print("[Info] Falling back to bs=1, nw=0 to locate problematic files...")
        # 2차: 안전 모드
        return float(run(1, 0, True))



# ------------- CLIPScore -------------
class CLIPScorer:
    """open_clip ViT-L/14 (openai)로 cosine 유사도 평균"""
    def __init__(self, device: torch.device):
        print("[Info] Initializing CLIPScorer...", flush=True)
        self.device = device if device.type == "cuda" else torch.device("cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model.eval()

    @torch.no_grad()
    def score_dir(self, gen_dir: Path, prompts: List[str], batch_size: int = 64) -> float:
        print("[Info] CLIPScore scoring...", flush=True)
        paths = [gen_dir / f"{i:08d}.jpg" for i in range(len(prompts))]
        scores = []
        for i in range(0, len(prompts), batch_size):
            ps = paths[i:i+batch_size]
            ts = prompts[i:i+batch_size]

            imgs = [self.preprocess(Image.open(p).convert("RGB")) for p in ps]
            img = torch.stack(imgs, dim=0).to(self.device)
            tok = self.tokenizer(ts).to(self.device)

            img_feats = self.model.encode_image(img)
            txt_feats = self.model.encode_text(tok)
            img_feats = F.normalize(img_feats, dim=-1)
            txt_feats = F.normalize(txt_feats, dim=-1)
            sim = (img_feats * txt_feats).sum(dim=-1)  # (B,)
            scores.append(sim.detach().cpu())
            print(f"[Debug] CLIPScore batch {i//batch_size}: mean {sim.mean().item():.4f}", flush=True)

        return float(torch.cat(scores, 0).mean().item())


# ------------- PickScore -------------
class PickScorer:
    """PickScore_v1 (CLIP-H/14 backbone)로 점수 평균"""
    def __init__(self, device: torch.device):
        print("[Info] Initializing PickScorer...", flush=True)
        self.device = device if device.type == "cuda" else torch.device("cpu")
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(self.device).eval()

    @torch.no_grad()
    def score_dir(self, gen_dir: Path, prompts: List[str], batch_size: int = 64) -> float:
        print("[Info] PickScore scoring...", flush=True)
        paths = [gen_dir / f"{i:08d}.jpg" for i in range(len(prompts))]
        all_scores = []

        for i in range(0, len(prompts), batch_size):
            ps = paths[i:i+batch_size]
            ts = prompts[i:i+batch_size]

            images = [Image.open(p).convert("RGB") for p in ps]

            image_inputs = self.processor(images=images, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
            text_inputs  = self.processor(text=ts,      padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)

            img_emb = self.model.get_image_features(**image_inputs)
            txt_emb = self.model.get_text_features(**text_inputs)
            img_emb = F.normalize(img_emb, dim=-1)
            txt_emb = F.normalize(txt_emb, dim=-1)

            scores = (self.model.logit_scale.exp() * (txt_emb @ img_emb.T)).diag()  # per-pair
            print(f"[Debug] PickScore batch {i//batch_size}: mean {scores.mean().item():.4f}", flush=True)
            all_scores.append(scores.detach().cpu())

        return float(torch.cat(all_scores, 0).mean().item())


# ------------- HPSv2 -------------
class HPSv2Scorer:
    """
    hpsv2.score(imgs_path, prompt, hps_version="v2.1") 사용.
    단일 이미지-프롬프트 쌍을 반복 호출하여 평균.
    """
    def __init__(self, use_v21: bool = True):
        print("[Info] Initializing HPSv2Scorer...", flush=True)
        if not _HAS_HPSV2:
            raise RuntimeError("hpsv2 is not installed")
        self.version = "v2.1" if use_v21 else "v2"

    def score_dir(self, gen_dir: Path, prompts: List[str]) -> float:
        print("[Info] HPSv2 scoring...", flush=True)
        scores: List[float] = []
        for i, prompt in enumerate(prompts):
            img_path = gen_dir / f"{i:08d}.jpg"
            try:
                s = hpsv2.score(str(img_path), prompt, hps_version=self.version)
                # hpsv2.score는 float 또는 [float]를 반환할 수 있음 → float로 수렴
                print(f"[Debug] HPSv2 score for idx {i}: {s}", flush=True)
                if isinstance(s, (list, tuple)):
                    s = float(s[0])
                scores.append(float(s))
            except Exception as e:
                # 실패시 스킵
                print(f"[Warn] HPSv2 scoring failed on idx {i}: {e}")
        if not scores:
            return float("nan")
        return float(sum(scores) / len(scores))


# ---------------- Main Evaluation ----------------
@torch.no_grad()
def diffusion_evaluation(model, eval_loader, fabric, config, debug_mode=False) -> Dict[str, Any]:
    """
    방법 B:
      1) dataset의 ref_path로 real_dir 구성
      2) 캡션으로 이미지 생성 → gen_dir 저장 (idx 기반 파일명)
      3) FID(gen_dir vs real_dir)
      4) CLIPScore / PickScore / HPSv2 (프롬프트, gen_dir)
    """
    print("[Info] Starting diffusion evaluation...", flush=True)
    device = fabric.device
    rank   = fabric.global_rank

    out_root = Path(config["output_dir"])
    gen_dir  = out_root / "gen_images"
    real_dir = out_root / "real_images"
    save_ext = config.get("save_ext", ".jpg")

    # --------- real 디렉토리 준비 (rank0) ---------
    if fabric.is_global_zero:
        ds = getattr(eval_loader, "dataset", None)
        _materialize_real_dir_from_dataset(ds, real_dir)
        n_real = len(list(real_dir.glob("*")))
        print(f"[Info] real_dir prepared → {real_dir}  (n={n_real})")
    fabric.barrier()

    # --------- 생성 설정 ---------
    H   = int(config.get("height", 512))
    W   = int(config.get("width", 512))
    steps = int(config.get("num_inference_steps", 30))
    cfg = float(config.get("guidance_scale", 7.5))
    sampler = config.get("sampler", "ddim")
    seed = int(config.get("seed", 42))

    gen_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device=device).manual_seed(seed + rank)

    # 전체 프롬프트(지표 계산용)
    if hasattr(eval_loader.dataset, "_items"):
        all_prompts = [cap for cap, _ in eval_loader.dataset._items]
    else:
        raise RuntimeError("Dataset must expose _items with prompts")

    # --------- 생성 루프 ---------
    for batch in eval_loader:
        # collate: (text_ids, text_atts, indices, refs)
        text_ids, text_atts, indices, _ = batch
        # SD15Generator.generate는 prompts 리스트를 기대하므로, 인덱스로 원문 프롬프트를 꺼내서 전달
        batch_prompts = [all_prompts[i] for i in indices.tolist()]
        negs = config.get("negative_prompts", None)
        images = model.generate(
            prompts=batch_prompts,
            num_inference_steps=steps,
            guidance_scale=cfg,
            height=H, width=W,
            generator=g,
            negative_prompts=negs
        )  # List[PIL.Image]

        # 저장
        for img, idx in zip(images, indices.tolist()):
            outp = gen_dir / f"{idx:08d}.jpg"
            img = img.convert("RGB")
            img.save(outp, format="JPEG", quality=95, subsampling=1)

    fabric.barrier()

    # --------- rank0: 지표 계산 ---------
    results: Dict[str, Any] = {}
    if fabric.is_global_zero:
        preflight_and_fix(str(gen_dir),  rewrite=False, force_resize=False)
        preflight_and_fix(str(real_dir), rewrite=False, force_resize=False)

        # # 1) FID
        # fid = _compute_fid_dir2dir(gen_dir, real_dir, use_cuda=(device.type=="cuda"), cfg=config)

        # results["FID"] = fid
        # print(f"[Metric] FID: {fid:.4f}")

        # 2) CLIPScore
        try:
            clip_scorer = CLIPScorer(device)
            clip_score = clip_scorer.score_dir(gen_dir, all_prompts, config.get("clip_batch_size", 64))
            results["CLIPScore"] = clip_score
            print(f"[Metric] CLIPScore: {clip_score:.4f}")
        except Exception as e:
            print(f"[Warn] CLIPScore failed: {e}")
            results["CLIPScore"] = None

        # 3) PickScore
        try:
            pick_scorer = PickScorer(device)
            pick_score = pick_scorer.score_dir(gen_dir, all_prompts, config.get("pick_batch_size", 64))
            results["PickScore"] = pick_score
            print(f"[Metric] PickScore: {pick_score:.4f}")
        except Exception as e:
            print(f"[Warn] PickScore failed: {e}")
            results["PickScore"] = None

        # 4) HPSv2
        try:
            if _HAS_HPSV2:
                hps_scorer = HPSv2Scorer(use_v21=True)
                hps_score = hps_scorer.score_dir(gen_dir, all_prompts)
                results["HPSv2"] = hps_score
                print(f"[Metric] HPSv2: {hps_score:.4f}")
            else:
                results["HPSv2"] = None
        except Exception as e:
            print(f"[Warn] HPSv2 failed: {e}")
            results["HPSv2"] = None

        # 메타 정보
        results["_meta"] = {
            "height": H, "width": W, "steps": steps, "cfg_scale": cfg,
            "sampler": sampler, "seed": seed,
            "num_samples": len(all_prompts),
            "gen_dir": str(gen_dir), "real_dir": str(real_dir),
        }

    return results
