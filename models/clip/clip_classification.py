import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer


def normalize_clip_mask_keys(mask_sd: dict, target_keys: set):
    """Normalize pruning mask keys to this module's state_dict schema.
    - Drop bias masks
    - Map 'vision_proj.'/'text_proj.' -> 'visual_projection.'/'text_projection.'
    - Prefix 'model.' when keys refer to inner CLIP model
    - Keep only keys that exist in target state_dict
    """
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

# -------------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------------
class CLIPClassification(nn.Module):
    def __init__(self):
        super().__init__()
        print("[Debug] clip_vqa.py : CLIPVQA 클래스의 init()함수 호출")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  # HF repo id or local dir
        self.max_tokens = 77
        self.pad_token_id = 0
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # Metadata for pruning utils / outside code
        setattr(self, "name", "clip")
        setattr(self, "is_vlm", True)
        setattr(self, "needs_tie", False)

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def get_text_features(self, *args, **kwargs):
        return self.model.get_text_features(*args, **kwargs)

    def load_pretrained(self, weights_ckpt="", is_eval=False):
        """
        weights_ckpt:
          - '' 이면 HF에서 이미 로드된 상태 유지
          - 경로(.pt/.pth/.bin)면 state_dict 로드 시도
          - 허깅페이스 모델명 문자열이면 그걸로 다시 from_pretrained
        """
        if not weights_ckpt:
            print("[CLIPRetrieval.load_pretrained] Empty path -> keep HF weights as-is.")
            return

        # 허깅페이스 모델명으로 들어온 경우
        if isinstance(weights_ckpt, str) and not os.path.exists(weights_ckpt) and '/' in weights_ckpt:
            print(f"[CLIPRetrieval.load_pretrained] Re-loading from HF repo: {weights_ckpt}")
            self.model = CLIPModel.from_pretrained(weights_ckpt)
            return

        # 파일로 들어온 경우 (프루닝 전/후 가중치 등)
        print(f"[CLIPRetrieval.load_pretrained] Loading weights from: {weights_ckpt}")
        ckpt = torch.load(weights_ckpt, map_location='cpu')

        # ckpt 형태 맞춰서 유연하게 처리
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        msg = self.load_state_dict(state_dict, strict=False)
        print("[CLIPRetrieval.load_pretrained] missing keys:", [k for k in msg.missing_keys if "pruning_mask" not in k])
        print("[CLIPRetrieval.load_pretrained] unexpected keys:", msg.unexpected_keys)

    def load_from_pruned_pretrained(self, mask_path, is_eval=False):
        print("-" * 80)
        print("[CLIP] load_from_pruned_pretrained(): applying weights (if any) and pruning masks")

        # 2) 마스크
        print(f"[CLIP] Loading pruning mask from: {mask_path}")
        mask_sd_raw = torch.load(mask_path, map_location="cpu")

        # 마스크 파일이 {'state_dict': ...} 형태일 수도 있으니 보정
        if isinstance(mask_sd_raw, dict) and ("state_dict" in mask_sd_raw or "model_state" in mask_sd_raw):
            mask_sd_raw = mask_sd_raw.get("state_dict", mask_sd_raw.get("model_state"))

        target_keys = set(self.state_dict().keys())
        mask_sd = normalize_clip_mask_keys(mask_sd_raw, target_keys)

        # 마스크만 로드(strict=False): 존재하는 버퍼/파라미터의 *_pruning_mask에만 들어감
        msg = self.load_state_dict(mask_sd, strict=False)
        # 진짜 마스크만의 missing을 보고 싶다면 필터링
        miss = [k for k in msg.missing_keys if k.endswith("_pruning_mask")]
        print("[CLIP] mask missing keys:", miss[:10], "..." if len(miss) > 10 else "")
        print("[CLIP] mask unexpected keys:", [k for k in msg.unexpected_keys][:10], "...")
        print("-" * 80)