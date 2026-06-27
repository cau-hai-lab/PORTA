# --- 상단 import 정리 ---
import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from huggingface_hub import snapshot_download

# ... _CLIP_ALIASES, resolve_clip_id, normalize_clip_mask_keys 동일 ...
_CLIP_ALIASES = {
    "clipL": "openai/clip-vit-large-patch14",
    "clipL-336": "openai/clip-vit-large-patch14-336",
    "clipG": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",  # ViT-bigG/14 (LAION) 
}

def resolve_clip_id(name_or_id: str) -> str:
    """'clipG' 같은 별칭을 HF repo id로 변환. 슬래시('/')가 있으면 그대로 사용."""
    if "/" in name_or_id:
        return name_or_id
    return _CLIP_ALIASES.get(name_or_id, name_or_id)
def ensure_safetensors_index(local_dir: str):
    bin_index = os.path.join(local_dir, "pytorch_model.bin.index.json")
    safe_index = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.exists(safe_index):
        return

    if not os.path.exists(bin_index):
        raise FileNotFoundError("pytorch_model.bin.index.json not found in " + local_dir)

    with open(bin_index, "r") as f:
        data = json.load(f)

    # weight_map의 shard 파일명을 .bin -> .safetensors로 바꿔치기
    wmap = data.get("weight_map", {})
    new_map = {k: v.replace(".bin", ".safetensors") for k, v in wmap.items()}

    # 총 사이즈는 굳이 정확할 필요는 없지만 가능하면 업데이트
    shard_files = set(new_map.values())
    total_size = 0
    for s in shard_files:
        p = os.path.join(local_dir, s)
        if os.path.exists(p):
            total_size += os.path.getsize(p)

    out = {"metadata": {"total_size": total_size}, "weight_map": new_map}
    with open(safe_index, "w") as f:
        json.dump(out, f)
def normalize_clip_mask_keys(mask_sd: dict, target_keys: set):
    fixed = {}
    for k, v in mask_sd.items():
        if k.endswith(".bias_pruning_mask"):
            continue
        kk = k.replace("vision_proj.", "visual_projection.")
        kk = kk.replace("text_proj.",   "text_projection.")
        if kk.startswith(("text_model.", "vision_model.", "text_projection.", "visual_projection.")):
            kk = "model." + kk
        if kk in target_keys:
            fixed[kk] = v
    return fixed

def _maybe_build_safetensors_index(local_dir: str):
    """
    pytorch_model.bin.index.json -> model.safetensors.index.json 생성.
    파일명 매핑 규칙:
      'pytorch_model-00001-of-00008.bin' -> 'model-00001-of-00008.safetensors'
    """
    safe_index = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.exists(safe_index):
        return

    bin_index = os.path.join(local_dir, "pytorch_model.bin.index.json")
    if not os.path.exists(bin_index):
        return  # bin index 자체가 없으면 건드리지 않음

    with open(bin_index, "r") as f:
        data = json.load(f)

    wmap = data.get("weight_map", {})
    def _map_name(v: str) -> str:
        # 접두사 & 확장자 모두 치환
        v2 = v.replace("pytorch_model-", "model-").replace(".bin", ".safetensors")
        return v2

    new_map = {k: _map_name(v) for k, v in wmap.items()}

    # 존재하는 샤드 용량 합산(없어도 동작엔 문제 없음)
    shard_files = set(new_map.values())
    total_size = 0
    for s in shard_files:
        p = os.path.join(local_dir, s)
        if os.path.exists(p):
            total_size += os.path.getsize(p)

    with open(safe_index, "w") as f:
        json.dump({"metadata": {"total_size": int(total_size)}, "weight_map": new_map}, f)
    print(f"[Debug] 생성: {safe_index}")

class CLIPGRetrieval(nn.Module):
    def __init__(self, config):
        super().__init__()
        resolved_id = resolve_clip_id(config['model_name'])
        self.model_id = resolved_id
        print(f"[Debug] CLIP-ViT-bigG-14 로딩 준비: {resolved_id}")

        # --- DDP 중복 다운로드 회피 ---
        rank = int(os.environ.get("RANK", "0"))
        patterns = [
            "*.safetensors", "pytorch_model.bin.index.json",
            "config.json", "open_clip_config.json",
            "tokenizer.json","tokenizer_config.json","vocab.json","merges.txt",
            "preprocessor_config.json","special_tokens_map.json",
            "*.md","*.txt",
        ]
        if rank == 0:
            snapshot_download(resolved_id, allow_patterns=patterns)
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass

        # 캐시 재확인 (오프라인 로드)
        local_dir = snapshot_download(resolved_id, allow_patterns=patterns, local_files_only=True)

        # --- safetensors 인덱스 없으면 만들어주기(샤드형) ---
        _maybe_build_safetensors_index(local_dir)

        # --- 모델 로드: safetensors -> 단일 safetensors -> bin fallback ---
        try:
            self.model = CLIPModel.from_pretrained(
                local_dir, use_safetensors=True, low_cpu_mem_usage=True, device_map=None
            )
        except Exception as e:
            print(f"[Warn] safetensors 샤드 로드 실패: {e}")
            single_safe = os.path.join(local_dir, "open_clip_model.safetensors")
            if os.path.exists(single_safe):
                print("[Info] 단일 safetensors 파일로 시도합니다.")
                self.model = CLIPModel.from_pretrained(
                    local_dir, use_safetensors=True, low_cpu_mem_usage=True,
                    device_map=None, weights_name="open_clip_model.safetensors"
                )
            else:
                print("[Info] bin 가중치 fallback 로 시도합니다.")
                # bin 필요 시 다시 받아오기(최초 allow_patterns가 *.bin을 포함 안 했으면)
                try:
                    snapshot_download(resolved_id, allow_patterns=["*.bin", "pytorch_model.bin.index.json"])
                except Exception:
                    pass
                self.model = CLIPModel.from_pretrained(
                    resolved_id, use_safetensors=False, low_cpu_mem_usage=True
                )

        self.model = self.model.float()  # dtype은 이후 Fabric/AMP에서 조정
        self.tokenizer = CLIPTokenizer.from_pretrained(local_dir)

        self.max_tokens = 77
        self.pad_token_id = 0
        setattr(self, "name", "clipG")
        setattr(self, "is_vlm", True)
        setattr(self, "needs_tie", False)

    @property
    def logit_scale(self):
        return self.model.logit_scale

    # === L14와 동일 인터페이스 보장 ===
    def encode_image(self, images):
        feats = self.model.get_image_features(pixel_values=images)
        return F.normalize(feats, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        feats = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(feats, dim=-1)

    def base_forward(self, images, text_ids, text_atts, use_grad: bool):
        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        with ctx:
            img = self.encode_image(images)      # (B, D)
            txt = self.encode_text(text_ids, text_atts)  # (B, D)
            logit_scale = self.model.logit_scale.exp()
            logits_t = txt @ img.t() * logit_scale
            logits_i = logits_t.t()
        return logits_i, logits_t

    def forward(self, image, text_ids, text_atts,
                return_losses=False, idx=None, scores_only=False, base=False, **kwargs):
        use_grad = return_losses is True and not (scores_only or base)

        if scores_only or base:
            return self.base_forward(image, text_ids, text_atts, use_grad=False)

        logits_i, logits_t = self.base_forward(image, text_ids, text_atts, use_grad=return_losses)
        if not return_losses:
            return logits_i, logits_t

        bsz = image.size(0)
        targets = torch.arange(bsz, device=image.device)
        loss_i2t = F.cross_entropy(logits_i, targets)
        loss_t2i = F.cross_entropy(logits_t, targets)
        loss_itc = 0.5 * (loss_i2t + loss_t2i)
        loss_itm = torch.zeros((), device=image.device)
        return loss_itc, loss_itm

    # 시그니처 호환(선택적 pretraining_weights 지원)
    def load_pretrained(self, weights_ckpt: str = "", is_eval: bool = False):
        if not weights_ckpt:
            print("[CLIP.load_pretrained] Empty -> keep HF weights.")
            return
        if isinstance(weights_ckpt, str) and not os.path.exists(weights_ckpt) and "/" in weights_ckpt:
            repo = resolve_clip_id(weights_ckpt)
            print(f"[CLIP.load_pretrained] Re-loading from HF: {repo}")
            self.model = CLIPModel.from_pretrained(repo)
            self.tokenizer = CLIPTokenizer.from_pretrained(repo)
            self.model_id = repo
            return
        print(f"[CLIP.load_pretrained] Loading from file: {weights_ckpt}")
        ckpt = torch.load(weights_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        msg = self.load_state_dict(state_dict, strict=False)
        print("[CLIP.load_pretrained] missing (w/o pruning_mask):",
              [k for k in msg.missing_keys if "pruning_mask" not in k])
        print("[CLIP.load_pretrained] unexpected:", msg.unexpected_keys)

    def load_from_pruned_pretrained(self, pretraining_weights: str | None = None,
                                    mask_path: str | None = None,
                                    config=None, is_eval: bool = False):
        print("-" * 80)
        print("[CLIP] load_from_pruned_pretrained(): weights + pruning masks")

        if pretraining_weights:
            self.load_pretrained(pretraining_weights, is_eval=is_eval)

        if mask_path:
            print(f"[CLIP] mask: {mask_path}")
            mask_sd_raw = torch.load(mask_path, map_location="cpu")
            if isinstance(mask_sd_raw, dict) and ("state_dict" in mask_sd_raw or "model_state" in mask_sd_raw):
                mask_sd_raw = mask_sd_raw.get("state_dict", mask_sd_raw.get("model_state"))
            target_keys = set(self.state_dict().keys())
            mask_sd = normalize_clip_mask_keys(mask_sd_raw, target_keys)
            msg = self.load_state_dict(mask_sd, strict=False)
            miss = [k for k in msg.missing_keys if k.endswith("_pruning_mask")]
            print("[CLIP] mask missing keys:", miss[:10], "..." if len(miss) > 10 else "")
            print("[CLIP] mask unexpected keys:", [k for k in msg.unexpected_keys][:10], "...")
        print("-" * 80)
