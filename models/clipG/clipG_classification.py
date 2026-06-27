import os
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from huggingface_hub import snapshot_download
import os, json
# ---------------------------
# Alias resolver (clipL, clipG, etc.)
# ---------------------------
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

# ---------------------------
# Pruning mask key normalizer (기존 함수 그대로)
# ---------------------------
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
# ---------------------------
# Model
# ---------------------------
class CLIPGClassification(nn.Module):
    def __init__(self, model_id: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"):
        super().__init__()
        resolved_id = resolve_clip_id(model_id)
        print("[Debug] CLIP-ViT-bigG-14 모델 불러오기 (safetensors)")
        self.model_id = resolved_id

        # ---- DDP 중복 다운로드 회피 ----
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            local_dir = snapshot_download(
                resolved_id,
                allow_patterns=[
                    "*.safetensors", "pytorch_model.bin.index.json",
                    "config.json", "open_clip_config.json",
                    "tokenizer.json","tokenizer_config.json","vocab.json","merges.txt",
                    "preprocessor_config.json","special_tokens_map.json",
                    "*.md","*.txt",
                ],
            )
        # 모든 랭크 합류
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass

        # 캐시 재확인(다른 랭크도 동일 경로 얻기)
        local_dir = snapshot_download(
            resolved_id,
            allow_patterns=[
                "*.safetensors", "pytorch_model.bin.index.json",
                "config.json", "open_clip_config.json",
                "tokenizer.json","tokenizer_config.json","vocab.json","merges.txt",
                "preprocessor_config.json","special_tokens_map.json",
                "*.md","*.txt",
            ],
            local_files_only=True,
        )

        # ---- safetensors 인덱스 준비(샤드용) ----
        bin_index = os.path.join(local_dir, "pytorch_model.bin.index.json")
        safe_index = os.path.join(local_dir, "model.safetensors.index.json")
        if not os.path.exists(safe_index) and os.path.exists(bin_index):
            import json
            with open(bin_index, "r") as f:
                data = json.load(f)
            wmap = data.get("weight_map", {})
            new_map = {k: v.replace(".bin", ".safetensors") for k, v in wmap.items()}
            shard_files = set(new_map.values())
            total_size = 0
            for sfn in shard_files:
                p = os.path.join(local_dir, sfn)
                if os.path.exists(p):
                    total_size += os.path.getsize(p)
            with open(safe_index, "w") as f:
                json.dump({"metadata": {"total_size": int(total_size)}, "weight_map": new_map}, f)
            print(f"[Debug] 생성: {safe_index}")

        # ---- 모델 로드 (샤드 or 단일 파일) ----
        has_shards = os.path.exists(safe_index)
        single_safe = os.path.join(local_dir, "open_clip_model.safetensors")
        if has_shards:
            model = CLIPModel.from_pretrained(
                local_dir,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                device_map=None,
            )
        elif os.path.exists(single_safe):
            model = CLIPModel.from_pretrained(
                local_dir,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                device_map=None,
                weights_name="open_clip_model.safetensors",  # ← 단일 파일 fallback
            )
        else:
            raise FileNotFoundError(
                "safetensors 가중치를 찾지 못했습니다. (*.safetensors 또는 open_clip_model.safetensors)"
            )

        # ★★ 크리티컬: 반드시 self.model에 바인딩 ★★
        self.model = model.float()

        # 토크나이저도 같은 스냅샷에서 로드(오프라인/리비전 일치)
        self.tokenizer = CLIPTokenizer.from_pretrained(local_dir)

        # 메타
        self.max_tokens = 77
        self.pad_token_id = 0
        setattr(self, "name", "clipG")
        setattr(self, "is_vlm", True)
        setattr(self, "needs_tie", False)

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def get_text_features(self, *args, **kwargs):
        return self.model.get_text_features(*args, **kwargs)

    def load_pretrained(self, weights_ckpt: str = "", is_eval: bool = False):
        """weights_ckpt:
           - ''  : 현재 HF 가중치 유지
           - 'org/name' : HF repo에서 재로딩
           - 파일경로   : state_dict 로드
        """
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

    def load_from_pruned_pretrained(self, mask_path, is_eval: bool = False):
        print("-" * 80)
        print("[CLIP] load_from_pruned_pretrained(): applying pruning masks")
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