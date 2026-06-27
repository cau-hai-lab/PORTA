# models/flamingo/flamingo_captioning.py
# ------------------------------------------------------------
# FlamingoCaptioning (OpenFlamingo) for COCO Karpathy captioning
#  - base checkpoint loading 안정화 (원본 로드 -> 필요 시 remap fallback)
#  - pruning mask 적용 (키 normalize + bias mask 자동 스킵)
#  - generate() 안정화:
#       * image_res 자동 보정 (positional embedding length 맞춤)
#       * beam expand 시 vis_x / input_ids / attention_mask 동기화
#       * conditioning clear (OOM 누수 방지)
#       * eos/pad 확정, 깨진 문자 제거
#  - pruners.accumulators에 의존하지 않도록 필요한 유틸을 이 파일에 내장
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import math
import inspect
from typing import Any, Dict, Tuple, Optional, List
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

import unicodedata

_BAD_PATTERNS = [
    r"HD Wallpaper",
    r"background images",
    r"\bclub tagged\b",
    r"\btagged:\b",
    r"\bDimensions:\b",
    r"\boil on canvas\b",
]
def clean_caption(t: str) -> str:
    # 유니코드 정규화 + 깨진 문자 제거
    t = unicodedata.normalize("NFKC", t).replace("\ufffd", " ")

    # 제어문자 제거
    t = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", t)

    # 반복 공백 정리
    t = re.sub(r"\s+", " ", t).strip()

    # (선택) 웹-alt-text 스타일 문구 제거
    low = t.lower()
    for p in _BAD_PATTERNS:
        if re.search(p.lower(), low):
            # 통째로 비우거나, 해당 문구만 삭제하고 남은 걸 쓰는 방식 중 선택 가능
            t = re.sub(p, "", t, flags=re.IGNORECASE).strip()

    # (핵심) COCO eval 안정화: ASCII만 남기기
    t = re.sub(r"[^\x20-\x7E]", "", t)  # ASCII printable만 유지
    t = re.sub(r"\s+", " ", t).strip()

    # 너무 짧아졌으면 원본 일부라도 남기기(안전장치)
    if len(t) < 3:
        t = "a photo."
    return t
# -----------------------------
# Utils: unwrap / component finders
# -----------------------------
def _unwrap_model(m: nn.Module) -> nn.Module:
    # DDP / Fabric wrapper 대비
    if hasattr(m, "module"):
        return m.module
    return m

def _get_vision_encoder(model: nn.Module):
    m = _unwrap_model(model)
    return getattr(m, "vision_encoder", None) or getattr(getattr(m, "model", None), "vision_encoder", None)

def _get_perceiver(model: nn.Module):
    m = _unwrap_model(model)
    return (
        getattr(m, "perceiver", None)
        or getattr(m, "perceiver_resampler", None)
        or getattr(m, "resampler", None)
        or getattr(getattr(m, "model", None), "perceiver", None)
        or getattr(getattr(m, "model", None), "perceiver_resampler", None)
        or getattr(getattr(m, "model", None), "resampler", None)
    )

def _find_flamingo_lm(model: nn.Module):
    """
    OpenFlamingo는 FlamingoLM이 보통 model.lang_encoder로 붙어있고,
    그 내부가 transformers.GenerationMixin을 상속함.
    """
    m = _unwrap_model(model)

    # 1) 일반적으로 가장 확실
    lm = getattr(m, "lang_encoder", None) or getattr(getattr(m, "model", None), "lang_encoder", None)
    if lm is not None and hasattr(lm, "generate"):
        return lm

    # 2) modules를 훑어서 generate 가능한 LM 찾기
    for mm in m.modules():
        if mm is m:
            continue
        if hasattr(mm, "generate") and hasattr(mm, "forward"):
            # FlamingoLayer 같은 애가 아니라, input_ids를 받는 forward인지 검사
            try:
                sig = inspect.signature(mm.forward)
                if "input_ids" in sig.parameters:
                    return mm
            except Exception:
                # signature 못보면 generate만 믿고 리턴하되, 아래에서 한번 더 방어
                return mm
    return None

def _pick_vision_tokens(ve_out):
    """
    ve_out에서 (B,V,D) tokens 또는 (B,D) pooled를 최대한 안전하게 추출.
    우선순위: tokens(3D) > pooled(2D)
    """
    # dict
    if isinstance(ve_out, dict):
        for k in ["image_tokens", "last_hidden_state", "tokens", "x", "vision_tokens"]:
            v = ve_out.get(k, None)
            if torch.is_tensor(v):
                return v
        for k in ["image_features", "pooler_output", "pooled", "embeds"]:
            v = ve_out.get(k, None)
            if torch.is_tensor(v):
                return v
        for v in ve_out.values():
            if torch.is_tensor(v):
                return v
        raise ValueError(f"Vision encoder returned dict but no tensor found. keys={list(ve_out.keys())}")

    # tuple/list
    if isinstance(ve_out, (tuple, list)):
        ts = [x for x in ve_out if torch.is_tensor(x)]
        if not ts:
            raise ValueError("Vision encoder returned tuple/list but no tensor elements found.")
        for t in ts:
            if t.ndim == 3:
                return t
        for t in ts:
            if t.ndim == 2:
                return t
        return ts[-1]

    # tensor
    if torch.is_tensor(ve_out):
        return ve_out

    raise ValueError(f"Unsupported vision encoder output type: {type(ve_out)}")

def _infer_target_hw_from_positional_embedding(vision_encoder: nn.Module) -> Optional[int]:
    """
    CLIP ViT 계열: vision_encoder.positional_embedding 길이(= 1 + grid^2)를 보고
    기대 입력 해상도(target_hw = grid * patch_stride)를 추정.
    """
    pos = getattr(vision_encoder, "positional_embedding", None)
    conv1 = getattr(vision_encoder, "conv1", None)
    if pos is None or (not torch.is_tensor(pos)) or conv1 is None:
        return None

    n_ctx = int(pos.shape[0])  # ex) 257
    if n_ctx <= 1:
        return None
    grid = int(math.sqrt(n_ctx - 1))
    if grid * grid != (n_ctx - 1):
        return None

    # patch stride 추정 (대부분 14)
    try:
        patch = int(conv1.stride[0])
    except Exception:
        patch = 14

    return grid * patch  # ex) 16*14=224

def _resize_if_needed(images_flat: torch.Tensor, vision_encoder: nn.Module) -> torch.Tensor:
    """
    vision_encoder가 기대하는 positional_embedding length에 맞춰
    입력 이미지를 자동으로 리사이즈.
    """
    target_hw = _infer_target_hw_from_positional_embedding(vision_encoder)
    if target_hw is None:
        return images_flat

    h, w = images_flat.shape[-2], images_flat.shape[-1]
    if (h == target_hw) and (w == target_hw):
        return images_flat

    # bicubic 권장
    return F.interpolate(images_flat, size=(target_hw, target_hw), mode="bicubic", align_corners=False)

# -----------------------------
# Mask normalize for Flamingo
# -----------------------------
def normalize_flamingo_mask_keys(mask_sd: dict, target_keys: set) -> Tuple[dict, int]:
    """
    다양한 저장 규칙을 최대한 맞춰서 현재 model.state_dict() 키에 맞게 normalize.
    - module. prefix 제거
    - model. prefix 유무 보정
    - bias_pruning_mask는 스킵
    """
    fixed = {}
    matched = 0
    for k, v in mask_sd.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith(".bias_pruning_mask"):
            continue

        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]

        # 어떤 코드베이스는 model.vision_encoder... 처럼 저장
        # 어떤 코드베이스는 vision_encoder... 로 저장
        if kk not in target_keys and kk.startswith("model."):
            kk2 = kk[len("model."):]
            if kk2 in target_keys:
                kk = kk2

        if kk not in target_keys and (not kk.startswith("model.")):
            kk2 = "model." + kk
            if kk2 in target_keys:
                kk = kk2

        if kk in target_keys:
            fixed[kk] = v
            matched += 1

    return fixed, matched

def build_openflamingo_sd_with_aliases(model, sd_raw: dict):
    target = set(model.state_dict().keys())
    sd2 = {}

    # prefix 제거(필요 시)
    def strip_prefix(k: str) -> str:
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        return k

    # gated_cross_attn_layer alias: transformer.blocks.N.gated_cross_attn_layer.*  -> gated_cross_attn_layers.N.*
    pat_gated = re.compile(r"^lang_encoder\.transformer\.blocks\.(\d+)\.gated_cross_attn_layer\.(.+)$")

    # blocks alias: transformer.blocks.N.* -> old_decoder_blocks.N.*
    pat_blocks = re.compile(r"^lang_encoder\.transformer\.blocks\.(\d+)\.(.+)$")

    for k, v in sd_raw.items():
        if not torch.is_tensor(v):
            continue
        k0 = strip_prefix(k)

        # 1) 원래 키가 모델에 있으면 그대로
        if k0 in target:
            sd2[k0] = v

        # 2) gated alias 추가
        m = pat_gated.match(k0)
        if m:
            idx, rest = m.group(1), m.group(2)
            k_g = f"lang_encoder.gated_cross_attn_layers.{idx}.{rest}"
            if k_g in target:
                sd2[k_g] = v

            # old_decoder_blocks 쪽에도 gated가 같은 이름으로 노출될 수 있음(있으면 채움)
            k_old = f"lang_encoder.old_decoder_blocks.{idx}.gated_cross_attn_layer.{rest}"
            if k_old in target:
                sd2[k_old] = v

        # 3) blocks alias 추가 (LM block weight들이 ckpt에 일부만 있어도 alias 키는 채움)
        m2 = pat_blocks.match(k0)
        if m2:
            idx, rest = m2.group(1), m2.group(2)
            k_old2 = f"lang_encoder.old_decoder_blocks.{idx}.{rest}"
            if k_old2 in target:
                sd2[k_old2] = v

    return sd2
# -----------------------------
# Main module
# -----------------------------
class FlamingoCaptioning(nn.Module):
    """
    captioning.py에서 기대하는 인터페이스:
      - __call__(image, caption, already_tokenized=True) -> loss (train용)
      - generate(image, sample=False, num_beams=..., max_length=..., min_length=...) -> List[str]
      - tokenizer 속성 필요 (train collate_fn에서 사용)
      - load_pretrained / load_from_pruned_pretrained 제공
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # -------------------------
        # 1) 어떤 체크포인트를 쓸지 선택
        # -------------------------
        ckpt_repo = os.environ.get("FLAMINGO_CKPT", "openflamingo/OpenFlamingo-3B-vitl-mpt1b")

        # OpenFlamingo 공식 조합에 맞춰야 함 (여기엔 최소 세트만)
        cfg_map = {
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b": {
                "lang_encoder_path": "anas-awadalla/mpt-1b-redpajama-200b",
                "tokenizer_path":   "anas-awadalla/mpt-1b-redpajama-200b",
                "cross_attn_every_n_layers": 1,
            },
            # 필요하면 여기에 4B 등 추가
        }
        if ckpt_repo not in cfg_map:
            raise ValueError(f"Unknown FLAMINGO_CKPT='{ckpt_repo}'. Supported: {list(cfg_map.keys())}")
        cfg = cfg_map[ckpt_repo]

        # -------------------------
        # 2) 모델 생성
        # -------------------------
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=cfg["lang_encoder_path"],
            tokenizer_path=cfg["tokenizer_path"],
            cross_attn_every_n_layers=cfg["cross_attn_every_n_layers"],
        )

        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer


        # captioning.py 스타일 맞추기
        setattr(self, "name", "flamingo")
        setattr(self, "is_vlm", True)
        setattr(self, "needs_tie", False)

        # generation prompt (dataset prompt와 별개)
        # (COCO captioning에 안정적인 프롬프트)
        self.gen_prompt = config.get("gen_prompt", "<image> <|endofchunk|> A caption: ")

        # eos/pad 보정 (OpenFlamingo tokenizer는 eos==bos==0인 경우가 있음)
        # pad는 50279로 나오는 경우가 많음
        if self.tokenizer.pad_token_id is None:
            # 흔한 케이스: GPTNeoXTokenizerFast pad=50279
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        # padding side는 generate에서 left로 씀
        self.tokenizer.padding_side = "left"

        # loss용: LM에서 teacher forcing을 한다면 labels 필요
        # OpenFlamingo lang_encoder는 transformers causal LM이므로 labels 지원.
        # 이미지-conditioning은 condition_vis_x 후 LM forward로 해결.

        # 디버그 토크나이저 출력(필요 시)
        if bool(config.get("debug_tokenizer", False)):
            print("[tok] name:", self.tokenizer.__class__.__name__)
            print("[tok] bos/eos/pad/unk:", self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                  self.tokenizer.pad_token_id, self.tokenizer.unk_token_id)
            try:
                print("[tok] <image> id:", self.tokenizer.convert_tokens_to_ids("<image>"),
                      " <|endofchunk|> id:", self.tokenizer.convert_tokens_to_ids("<|endofchunk|>"))
            except Exception:
                pass
            print("[tok] '?' encode:", self.tokenizer.encode("?", add_special_tokens=False))

        # 기본 ckpt 로딩(환경변수로 끄고 싶으면)
        if not bool(config.get("skip_base_ckpt_load", False)):
            self.load_pretrained(pretraining_weights=ckpt_repo, config=config)

    # -------------------------
    # Loading
    # -------------------------

    def load_pretrained(self, pretraining_weights: str = "", config=None, load_capt_pretrain: bool = False):
        """
        1) repo_id면 hf_hub_download로 checkpoint.pt 받아서 로드
        2) file path면 torch.load 해서 로드
        """
        if not pretraining_weights:
            print("[Flamingo] load_pretrained: empty -> keep current weights.")
            return

        # repo_id인지 file path인지 구분
        is_repo = (isinstance(pretraining_weights, str) and ("/" in pretraining_weights) and (not os.path.exists(pretraining_weights)))
        ckpt_path = pretraining_weights
        if is_repo:
            ckpt_path = hf_hub_download(repo_id=pretraining_weights, filename="checkpoint.pt")

        state = torch.load(ckpt_path, map_location="cpu")
        sd_raw = state.get("state_dict", state.get("model", state))
        sd_raw = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd_raw.items()}

        sd2 = build_openflamingo_sd_with_aliases(self.model, sd_raw)


        # ---- 1차: 원본 그대로 로드 시도 (가장 정석) ----
        msg = self.model.load_state_dict(sd_raw, strict=False)

        param_names = set(dict(self.model.named_parameters()).keys())
        buffer_names = set(dict(self.model.named_buffers()).keys())
        state_keys = set(self.model.state_dict().keys())

        def is_fake_state_key(k: str) -> bool:
            # state_dict에는 있는데 실제 파라미터/버퍼로 등록이 안 된 경우
            return (k in state_keys) and (k not in param_names) and (k not in buffer_names)

        def ignorable(k: str) -> bool:
            return (
                k.startswith("vision_encoder.")
                or k.startswith("lang_encoder.old_decoder_blocks.")
                or k.startswith("lang_encoder.gated_cross_attn_layers.")
                or is_fake_state_key(k)  # ✅ 핵심
            )

        real_missing = [k for k in msg.missing_keys if not ignorable(k)]

        print("[Flamingo] missing_total:", len(msg.missing_keys))
        print("[Flamingo] real_missing:", len(real_missing))
        print("[Flamingo] real_missing head:", real_missing[:30])

        print(f"[Flamingo] base load missing: {len(msg.missing_keys)} unexpected: {len(msg.unexpected_keys)}")

        if len(msg.missing_keys) > 0:
            print("[Flamingo] missing head:", msg.missing_keys[:20])

        # missing이 너무 크면(수백~천 단위) 구조 불일치 가능성이 큼
        # (이 경우 remap 같은 복잡한 처리가 필요할 수 있는데, 우선 사용자에게 강하게 경고)
        if len(msg.missing_keys) >= 300:
            print(
                "[Warn][Flamingo] base checkpoint load has too many missing keys.\n"
                "  - ckpt_repo / lang_encoder_path / cross_attn_every_n_layers 조합이 맞는지 확인하세요.\n"
                "  - 이 상태면 캡션이 깨지거나 점수가 0으로 나올 수 있습니다."
            )

    def load_from_pruned_pretrained(self, pretraining_weights: str | None = None,
                                    mask_path: str | None = None,
                                    config=None, is_eval: bool = False):
        print("-" * 80)
        print("[Flamingo] load_from_pruned_pretrained(): weights + pruning masks")

        # base weights (선택)
        if pretraining_weights:
            self.load_pretrained(pretraining_weights=pretraining_weights, config=config)

        # pruning masks
        if mask_path:
            print(f"[Flamingo] mask: {mask_path}")
            mask_sd_raw = torch.load(mask_path, map_location="cpu")


            if isinstance(mask_sd_raw, dict) and ("state_dict" in mask_sd_raw or "model_state" in mask_sd_raw):
                mask_sd_raw = mask_sd_raw.get("state_dict", mask_sd_raw.get("model_state"))

            target_keys = set(self.model.state_dict().keys())
            mask_sd, matched = normalize_flamingo_mask_keys(mask_sd_raw, target_keys)
            print(f"[Flamingo] mask matched: {matched} keys")

            msg = self.model.load_state_dict(mask_sd, strict=False)
            miss = [k for k in msg.missing_keys if k.endswith("_pruning_mask")]
            print("[Flamingo] mask missing keys sample:", miss[:10], "..." if len(miss) > 10 else "")
            print("[Flamingo] mask unexpected keys sample:", msg.unexpected_keys[:10], "..." if len(msg.unexpected_keys) > 10 else "")

        # gate 통계(선택)
        if bool((config or {}).get("debug_gates", True)):
            attn_g, ff_g = [], []
            for n, p in self.model.named_parameters():
                if "attn_gate" in n and p.ndim == 1:
                    attn_g.append(torch.sigmoid(p.detach().float()).mean().item())
                if "ff_gate" in n and p.ndim == 1:
                    ff_g.append(torch.sigmoid(p.detach().float()).mean().item())
            if attn_g:
                print(f"[gate] attn_gate sigmoid mean={sum(attn_g)/len(attn_g):.6f} n={len(attn_g)}")
            if ff_g:
                print(f"[gate] ff_gate   sigmoid mean={sum(ff_g)/len(ff_g):.6f} n={len(ff_g)}")

        print("-" * 80)

    # -------------------------
    # Training forward (optional)
    # -------------------------
    def forward(self, image, caption, already_tokenized: bool = True):
        """
        captioning.py에서 train()이 model(image, caption, already_tokenized=True)로 호출.
        caption은 collate_fn에서 tokenizer로 이미 input_ids/attention_mask 형태로 만들었다고 가정.

        expected caption format:
          - dict with keys: input_ids, attention_mask
          - or tuple/list: (input_ids, attention_mask)
        """
        device = image.device

        # parse text
        if isinstance(caption, dict):
            input_ids = caption["input_ids"].to(device)
            attention_mask = caption["attention_mask"].to(device)
        elif isinstance(caption, (tuple, list)) and len(caption) >= 2 and torch.is_tensor(caption[0]):
            input_ids = caption[0].to(device)
            attention_mask = caption[1].to(device)
        else:
            raise ValueError("caption format unsupported. Provide tokenized dict or (ids, mask).")

        # images: (B,C,H,W) -> (B,1,1,C,H,W)
        images = image
        if images.ndim == 4:
            images = images.unsqueeze(1).unsqueeze(2)
        elif images.ndim == 5:
            images = images.unsqueeze(2)
        elif images.ndim == 6:
            pass
        else:
            raise ValueError(f"Unexpected image ndim={images.ndim}, shape={tuple(images.shape)}")

        ve = _get_vision_encoder(self.model)
        perceiver = _get_perceiver(self.model)
        lm = _find_flamingo_lm(self.model)
        if ve is None or perceiver is None or lm is None:
            raise RuntimeError("Flamingo components not found (vision/perceiver/lm)")

        B, T, Fm, C, H, W = images.shape
        images_flat = images.reshape(B * T * Fm, C, H, W)
        images_flat = _resize_if_needed(images_flat, ve)

        conditioned_modules = []
        try:
            # vision -> tokens
            out = ve(images_flat)
            vis = _pick_vision_tokens(out)
            if vis.ndim == 2:
                vis = vis.unsqueeze(1)
            V, D = vis.shape[1], vis.shape[2]
            vis = vis.reshape(B, T, Fm, V, D)

            # perceiver
            vis_x = perceiver(vis)

            # condition
            if hasattr(lm, "condition_vis_x"):
                lm.condition_vis_x(vis_x)
            elif hasattr(lm, "condition_media"):
                lm.condition_media(vis_x)
            else:
                for m in self.model.modules():
                    if hasattr(m, "condition_vis_x"):
                        m.condition_vis_x(vis_x)
                        conditioned_modules.append(m)

            # teacher forcing: labels = input_ids (shift handled in HF models)
            outputs = lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_cache=False,
                return_dict=True,
            )
            return outputs.loss

        finally:
            if hasattr(lm, "clear_conditioned_layers"):
                try:
                    lm.clear_conditioned_layers()
                except Exception:
                    pass
            for m in conditioned_modules:
                for attr in ("vis_x", "_vis_x", "conditioned_vis_x", "_conditioned_vis_x", "media", "_media"):
                    if hasattr(m, attr):
                        try:
                            setattr(m, attr, None)
                        except Exception:
                            pass

    # -------------------------
    # Generation
    # -------------------------
    import inspect


    @torch.no_grad()
    def generate(self, image, sample: bool = False, num_beams: int = 3, max_length: int = 20, min_length: int = 0):
        device = image.device
        images = image

        # (B,C,H,W) -> (B,1,1,C,H,W)  (OpenFlamingo 표준)
        if images.ndim == 4:
            images = images.unsqueeze(1).unsqueeze(2)
        elif images.ndim == 5:
            images = images.unsqueeze(2)
        elif images.ndim != 6:
            raise ValueError(f"Unexpected images.ndim={images.ndim}, shape={tuple(images.shape)}")

        B = images.size(0)

        # prompt -> lang_x
        self.tokenizer.padding_side = "left"
        prompt = getattr(self, "gen_prompt", "<image> <|endofchunk|> Describe this image.")

        p_tok = self.tokenizer(
            [prompt] * B,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        lang_x = p_tok["input_ids"].to(device)          # ✅ OpenFlamingo에선 보통 lang_x라 부름
        attention_mask = p_tok["attention_mask"].to(device)
        L = lang_x.size(1)

        # eos/pad 세팅 (MPT 계열에서 eos==0, pad 따로인 경우 많음)
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        if eos_id is None:
            eos_id = 0
        if pad_id is None:
            pad_id = eos_id

        # -------------------------
        # 1) 우선: self.model.generate (Flamingo 래퍼) 호출
        #    ✅ 이 버전은 (vision_x, lang_x) positional 요구
        # -------------------------
        if hasattr(self.model, "generate"):
            # 1) 시그니처 파싱
            try:
                sig = inspect.signature(self.model.generate)
                params = sig.parameters
            except Exception:
                sig = None
                params = {}

            # 2) **kwargs 받는지 체크 (중요)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values()
            )

            # 3) gen_kwargs 구성
            gen_kwargs = {}

            # attention_mask는 시그니처에 명시되어 있으니 항상 넣는 게 안전
            gen_kwargs["attention_mask"] = attention_mask

            if has_var_kw:
                # ✅ **kwargs가 있으면 이름 체크 하지 말고 그냥 다 넣어도 됨
                gen_kwargs["num_beams"] = int(num_beams) if num_beams is not None else 1
                gen_kwargs["do_sample"] = bool(sample)

                # 너 함수 인자 max_length는 "새로 생성할 토큰 수"로 쓰는 게 맞음
                gen_kwargs["max_new_tokens"] = int(max_length)
                if min_length and min_length > 0:
                    gen_kwargs["min_new_tokens"] = int(min_length)

                gen_kwargs["use_cache"] = False
                gen_kwargs["eos_token_id"] = int(eos_id)
                gen_kwargs["pad_token_id"] = int(pad_id)

                # (선택) 캡션 품질/반복 억제에 보통 도움
                gen_kwargs["no_repeat_ngram_size"] = 3
                gen_kwargs["length_penalty"] = 0.9
                gen_kwargs["repetition_penalty"] = 1.1
                gen_kwargs["early_stopping"] = True
            else:
                # 혹시라도 **kwargs가 없는 버전이면 기존 방식 (명시 파라미터만)
                if "num_beams" in params:
                    gen_kwargs["num_beams"] = int(num_beams) if num_beams is not None else 1
                if "do_sample" in params:
                    gen_kwargs["do_sample"] = bool(sample)

                if "max_new_tokens" in params:
                    gen_kwargs["max_new_tokens"] = int(max_length)
                elif "max_length" in params:
                    gen_kwargs["max_length"] = int(L + max_length)

                if "min_new_tokens" in params:
                    gen_kwargs["min_new_tokens"] = int(min_length)
                elif "min_length" in params:
                    gen_kwargs["min_length"] = int(L + min_length) if min_length > 0 else int(L)

                if "use_cache" in params:
                    gen_kwargs["use_cache"] = False
                if "eos_token_id" in params:
                    gen_kwargs["eos_token_id"] = int(eos_id)
                if "pad_token_id" in params:
                    gen_kwargs["pad_token_id"] = int(pad_id)

            # 4) 디버그 출력은 gen_kwargs 만든 뒤에!
            if sig is not None:
                print("[GEN] flamingo.generate sig:", sig)
            print("[GEN] has_var_kw:", has_var_kw)
            print("[GEN] gen_kwargs:", {k: (v.shape if torch.is_tensor(v) else v) for k, v in gen_kwargs.items()})

            # 5) 호출
            gen_ids = self.model.generate(images, lang_x, **gen_kwargs)

        else:
            raise RuntimeError("self.model has no generate()")


        # -------------------------
        # 2) decode: 프롬프트 이후만 잘라서 디코드
        # -------------------------
        new_ids = gen_ids[:, L:]
        texts = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        texts = [clean_caption(t) for t in texts]
        return texts
