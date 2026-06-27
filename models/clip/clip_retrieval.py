# models/clip_retrieval.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer


def normalize_clip_mask_keys(mask_sd: dict, target_keys: set):
    fixed = {}
    for k, v in mask_sd.items():
        # 1) bias 마스크는 보통 사용 안 함
        if k.endswith(".bias_pruning_mask"):
            continue

        kk = k

        # 2) 프로젝트별 명칭 차이 보정
        kk = kk.replace("vision_proj.", "visual_projection.")
        kk = kk.replace("text_proj.",   "text_projection.")

        # 3) 래퍼(CLIPRetrieval) 안의 경로에 맞추기 (self.model.*)
        if kk.startswith(("text_model.", "vision_model.", "text_projection.", "visual_projection.")):
            kk = "model." + kk

        # 4) 실제 모델의 키로 존재하는 항목만 남기기
        if kk in target_keys:
            fixed[kk] = v
    return fixed

class CLIPRetrieval(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = CLIPModel.from_pretrained(config['model_name'])
        self.max_tokens = config.get('max_tokens')
        setattr(self, "name", "clip")
        setattr(self, "is_vlm", True)
        setattr(self, "needs_tie", False)

    # grad on/off를 호출부에서 제어
    def encode_image(self, images):
        feats = self.model.get_image_features(pixel_values=images)
        return F.normalize(feats, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        feats = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(feats, dim=-1)

    def base_forward(self, images, text_ids, text_atts, use_grad: bool):
        # 학습이면 grad ON, 추론이면 grad OFF
        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        with ctx:
            image_embeds = self.encode_image(images)             # (B, D)
            text_embeds  = self.encode_text(text_ids, text_atts) # (B, D)
            logit_scale  = self.model.logit_scale.exp()
            logits_per_text  = text_embeds @ image_embeds.t() * logit_scale
            logits_per_image = logits_per_text.t()
        return logits_per_image, logits_per_text

    def forward(self, image, text_ids, text_atts,
                return_losses=False, idx=None, scores_only=False, base=False, **kwargs):
        # 추론(base/scores_only) = no_grad, 학습(return_losses=True) = grad ON
        use_grad = return_losses is True and not (scores_only or base)

        if scores_only or base:
            return self.base_forward(image, text_ids, text_atts, use_grad=False)

        logits_per_image, logits_per_text = self.base_forward(image, text_ids, text_atts, use_grad=return_losses)

        if not return_losses:
            return logits_per_image, logits_per_text

        bsz = image.size(0)
        targets = torch.arange(bsz, device=image.device)
        loss_i2t = F.cross_entropy(logits_per_image, targets)
        loss_t2i = F.cross_entropy(logits_per_text,  targets)
        loss_itc = 0.5 * (loss_i2t + loss_t2i)
        loss_itm = torch.zeros((), device=image.device)
        return loss_itc, loss_itm


    def load_pretrained(self, weights_ckpt, config=None, is_eval=False):
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

    def load_from_pruned_pretrained(self, pretraining_weights, mask_path, config, is_eval=False):
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
