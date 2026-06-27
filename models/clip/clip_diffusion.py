# models/SD15generator.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Dict, Any, Tuple

# --- 외부 의존성 ---

from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
    DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler,
    HeunDiscreteScheduler, DEISMultistepScheduler, UniPCMultistepScheduler,
    KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
)

import torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Optional, Dict, Any
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTextModelWithProjection
import os, torch
from typing import Union, Dict, Any, Optional
from transformers import CLIPTextModel

def _auto_device_dtype():
    # 디바이스: CUDA > MPS(Apple) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype  = torch.float16            # 보편적이고 메모리 절약
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype  = torch.float16            # MPS는 보통 fp16이 안전
    else:
        device = torch.device("cpu")
        dtype  = torch.float32            # CPU는 fp32가 무난
    return device, dtype

def _looks_text_key(k: str) -> bool:
    s = k
    # 흔한 상위 접두사 제거
    for p in ("module.", "model.", "pipe.", "net."):
        if s.startswith(p): s = s[len(p):]
    # text 도메인 단서
    if "text_model." in s or "text_encoder." in s or "cond_stage_model.transformer." in s:
        # vision/vae/unet 단어 있으면 제외
        if any(x in s for x in ("vision", "visual", "unet", "vae")):
            return False
        return True
    return False

import torch, sys, json
 # 예: "dpmpp_2m-karras" -> "dpmpp2mkarras"
def _norm_sampler(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())

def set_scheduler_from_config(pipe, config: dict):
    raw = str(config.get("sampler", ""))
    s   = _norm_sampler(raw)

    # ---- DPM-Solver++ 계열 (dpmpp, dpmpp2m, dpmsolver++, 등) ----
    if any(k in s for k in ["dpmpp", "dpmsolver", "dpmsolverpp", "dpmpp2m", "dpmpp2s"]):
        sched = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # 가능한 버전에서 알고리즘/카라스 시그마 설정
        try:  sched.algorithm_type = "dpmsolver++"
        except Exception: pass
        try:
            use_karras = config.get("use_karras_sigmas", None)
            if use_karras is None:
                use_karras = "karras" in raw.lower()
            if hasattr(sched, "use_karras_sigmas"):
                sched.use_karras_sigmas = bool(use_karras)
        except Exception: pass
        pipe.scheduler = sched
        return

    # ---- 그 외 대표 샘플러 ----
    if s in {"eulera","eulerancestral"}:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config); return
    if s in {"euler"}:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config); return
    if s in {"heun"}:
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config); return
    if s in {"ddim"}:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config); return
    if s in {"lms"}:
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config); return
    if s in {"pndm"}:
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config); return
    if s in {"deis"}:
        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config); return
    if s in {"unipc","unipcmultistep"}:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config); return
    if s in {"kdpm2"}:
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config); return
    if s in {"kdpm2a","kdpm2ancestral"}:
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config); return
    # 미지정이면 기존 스케줄러 유지
class SD15Generator(nn.Module):
    """
    Stable Diffusion 1.5 전체 파이프라인 래퍼
    - 내부에 diffusers StableDiffusionPipeline 보관 (text_encoder / tokenizer / unet / vae 접근 가능)
    - 텍스트 인코더만 pruning mask 적용 가능 (UNet/VAE는 건드리지 않음)
    - generate(): 프롬프트 리스트 → 이미지 생성 (PIL 이미지 리스트 반환)
    - encode_text(): 텍스트 임베딩 추출 (평가지표들에 활용)
    """
    def __init__(self, config: Dict[str, Any]):
        print("[Debug] models/clip/clip_diffusion.py -> SD15Generator 클래스 __init__()함수 호출")
        super().__init__()
        assert StableDiffusionPipeline is not None, \
            "diffusers가 필요합니다. `pip install diffusers accelerate transformers safetensors` 등을 확인하세요."

        self.config = config
        self.repo   = config.get("model_name", "runwayml/stable-diffusion-v1-5")
        self.device, self.dtype = _auto_device_dtype()

        # 파이프라인 로드
        print(f"[SD15Generator] Loading SD1.5 pipeline: {self.repo} (dtype={self.dtype})")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.repo,
            torch_dtype=self.dtype,
        )
        set_scheduler_from_config(pipe, config)
        pipe = pipe.to(self.device)

        # 성능 옵션
        if config.get("enable_attention_slicing", True):
            pipe.enable_attention_slicing()
        if config.get("enable_vae_slicing", False):
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        if config.get("enable_xformers", False):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                print("[SD15Generator] xFormers 메모리 최적화 실패(미설치/미지원). 계속 진행합니다.")

        # 파이프라인 구성요소 노출
        self.pipe         = pipe
        self.text_encoder = pipe.text_encoder   # CLIPTextModel
        self.tokenizer    = pipe.tokenizer      # CLIPTokenizer
        self.unet         = pipe.unet
        self.vae          = pipe.vae

        # 토크나이저 pad 토큰 안전장치 (collate에서 padding='max_length'를 쓸 수 있도록)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Retrieval 스타일 메타
        setattr(self, "name", "sd15_generator")
        setattr(self, "is_vlm", True)
        setattr(self, "needs_tie", False)

        # 임베딩 정규화 여부 (평가지표에서 코사인/로짓 등 사용할 때 편의)
        self.normalize_embeddings = config.get("normalize_embeddings", True)

        # eval 모드 기본
        self.eval()

    # -------------------------
    # 텍스트 임베딩 유틸 (평가용)
    # -------------------------
    @torch.no_grad()
    def encode_text(self, texts: List[str]):
        """
        텍스트 임베딩 추출 (pooled/CLS 아님 주의: SD는 last_hidden_state가 주로 필요하지만
        평가지표용으로는 평균 pooling 후 normalize를 기본 제공)
        반환: (N, D) torch.Tensor
        """
        # tokenizer는 pipe.tokenizer 사용
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).to(self.device)

        # SD1.5의 text_encoder는 CLIPTextModel (last_hidden_state 사용)
        outputs = self.text_encoder(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask
        )
        # (N, T, D)
        last_hidden = outputs.last_hidden_state

        # 간단히 평균 pooling (평가지표에 맞춰 후처리는 외부에서 조정 가능)
        feats = (last_hidden * enc.attention_mask.unsqueeze(-1)).sum(dim=1) / \
                enc.attention_mask.clamp(min=1).sum(dim=1, keepdim=True)
        if self.normalize_embeddings:
            feats = F.normalize(feats, dim=-1)
        return feats

    # -------------------------
    # 이미지 생성
    # -------------------------
    @torch.no_grad()
    def generate(self,
                prompts: List[str],
                num_inference_steps: int = 30,
                guidance_scale: float = 7.5,
                height: Optional[int] = None,
                width: Optional[int] = None,
                seed: Optional[int] = None,
                negative_prompts: Optional[List[str]] = None,
                **kwargs) -> List["PIL.Image.Image"]:

        # 1) seed로 생성기 만들기
        print("[Debug] models/clip/clip_diffusion.py -> SD15Generator 클래스 generate()함수 호출")
        gen_from_seed = None
        if seed is not None:
            gen_from_seed = torch.Generator(device=self.device).manual_seed(int(seed))

        # 2) 호출자가 kwargs로 준 generator가 있다면 그걸 우선
        gen_from_kwargs = kwargs.pop("generator", None)
        generator = gen_from_kwargs if gen_from_kwargs is not None else gen_from_seed

        # 3) Pipeline 호출 (generator는 한 번만 넘김)
        out = self.pipe(
            prompt=prompts,
            negative_prompt=negative_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            **kwargs
        )
        return out.images

    # -------------------------
    # forward: 토큰 입력도 지원(옵션)
    # -------------------------
    @torch.no_grad()
    def forward(self,
                text_ids: Optional[torch.Tensor] = None,
                text_atts: Optional[torch.Tensor] = None,
                prompts: Optional[List[str]] = None,
                **kwargs):
        """
        - prompts가 주어지면 pipe(prompt=...)로 이미지 생성 반환
        - text_ids/text_atts만 주어졌다면 encode_text() 임베딩을 반환 (생성 아님)
        """
        if prompts is not None:
            print("[Debug] prompts 기반 이미지 생성", flush=True)
            return self.generate(prompts=prompts, **kwargs)

        if text_ids is not None and text_atts is not None:
            # 임베딩만 반환 (평가지표 전처리/로그 등에서 사용)
            print("[Debug] text_ids/text_atts 기반 텍스트 임베딩 추출", flush=True)
            outputs = self.text_encoder(
                input_ids=text_ids.to(self.device),
                attention_mask=text_atts.to(self.device)
            )
            last_hidden = outputs.last_hidden_state
            feats = (last_hidden * text_atts.to(self.device).unsqueeze(-1)).sum(dim=1) / \
                    text_atts.clamp(min=1).sum(dim=1, keepdim=True)
            if self.normalize_embeddings:
                feats = F.normalize(feats, dim=-1)
            return feats

        raise ValueError("forward() requires either prompts or (text_ids & text_atts).")

    # -------------------------
    # 가중치/마스크 로딩
    # -------------------------
    def load_pretrained(self, weights_ckpt: str, config: Optional[Dict[str, Any]] = None, is_eval: bool = True):
        """
        SD 파이프라인 자체를 다른 리포/경로에서 다시 로딩할 경우 사용.
        - HF repo str이면 `StableDiffusionPipeline.from_pretrained`로 재로딩
        - 파일 체크포인트를 직접 주는 케이스는 거의 없으므로, 기본은 repo 재로딩만 지원
        """
        if not weights_ckpt:
            print("[SD15Generator.load_pretrained] Empty path -> keep current pipeline as-is.")
            return

        if isinstance(weights_ckpt, str) and not os.path.exists(weights_ckpt) and '/' in weights_ckpt:
            print(f"[SD15Generator.load_pretrained] Re-loading pipeline from HF repo: {weights_ckpt}")
            pipe = StableDiffusionPipeline.from_pretrained(
                weights_ckpt,
                torch_dtype=self.dtype,
                safety_checker=None if self.config.get("disable_safety_checker", True) else None,
            ).to(self.device)
            self.pipe         = pipe
            self.text_encoder = pipe.text_encoder
            self.tokenizer    = pipe.tokenizer
            self.unet         = pipe.unet
            self.vae          = pipe.vae
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if is_eval: self.eval()
            return

        # 파일 경로로 전체 파이프라인을 복원하는 사용성은 낮으므로 스킵
        print("[SD15Generator.load_pretrained] Only HF repo string is supported for full pipeline reload.")


    def load_from_pruned_pretrained(
        self,
        pretraining_weights: Optional[str],
        mask_path: Optional[str],
        config: Optional[Dict[str, Any]] = None,
        is_eval: bool = True,
    ):
        """
        SD1.5 파이프라인의 TEXT ENCODER에만 pruning mask 적용.
        - pretraining_weights: (선택) 텍스트 인코더를 HF repo에서 다시 로드
        - mask_path: *_pruning_mask 텐서들이 들어있는 .pth (dict / {'state_dict': ...} / {'model_state': ...} 모두 허용)
        사용 전 make_prunable(self.text_encoder, ...)가 선행되어야 *_pruning_mask 버퍼가 존재합니다.
        """
        import os, torch, collections
        from transformers import CLIPTextModel

        print("-" * 80)
        print("[SD15Generator] load_from_pruned_pretrained(): TEXT ENCODER weights/mask apply")

        # 0) (옵션) 텍스트 인코더 가중치 재로딩
        if pretraining_weights:
            if isinstance(pretraining_weights, str) and not os.path.exists(pretraining_weights) and "/" in pretraining_weights:
                print(f"[SD15Generator] Re-loading TEXT ENCODER from HF repo: {pretraining_weights}")
                te_new = CLIPTextModel.from_pretrained(pretraining_weights).to(self.device, dtype=self.dtype)
                self.text_encoder = te_new
                self.pipe.text_encoder = te_new
            else:
                print(f"[SD15Generator][WARN] pretraining_weights='{pretraining_weights}'은 HF repo 형식이 아님. 건너뜀.")

        if not mask_path:
            print("[SD15Generator] No mask_path provided → skip.")
            print("-" * 80)
            return

        # 1) 마스크 로드
        print(f"[SD15Generator] Loading pruning mask from: {mask_path}")
        mask_raw = torch.load(mask_path, map_location="cpu")
        if isinstance(mask_raw, dict) and ("state_dict" in mask_raw or "model_state" in mask_raw):
            mask_raw = mask_raw.get("state_dict", mask_raw.get("model_state"))

        # *_pruning_mask 항목만 필터
        mask_only = {k: v for k, v in mask_raw.items()
                    if isinstance(k, str) and k.endswith("_pruning_mask") and torch.is_tensor(v)}
        print(f"[DEBUG] mask_raw total keys: {len(mask_raw) if isinstance(mask_raw, dict) else 0}")
        print(f"[DEBUG] mask_raw *_pruning_mask keys: {len(mask_only)}")

        # 2) 대상 키셋 수집(현재 text_encoder의 state_dict 키)
        te_sd = self.text_encoder.state_dict()
        te_keys = set(te_sd.keys())
        print(f"[DEBUG] text_encoder target total keys: {len(te_keys)}")

        # 3) 키 정규화 함수(마스크 → text_encoder.* 경로 맵핑)
        def _normalize_to_te_keys(msd, target_te_keys):
            fixed = {}
            for k, v in msd.items():
                kk = k
                kk = kk.replace("text_encoder.",   "text_model.")
                kk = kk.replace("vision_encoder.", "vision_model.")
                kk = kk.replace("text_proj.",      "text_projection.")
                kk = kk.replace("vision_proj.",    "visual_projection.")
                if kk.startswith("model."):
                    kk = kk[len("model."):]
                if kk.startswith("cond_stage_model.transformer."):
                    kk = "text_model." + kk[len("cond_stage_model.transformer."):]
                if (".vision_" in kk) or (".visual_" in kk) or kk.startswith("vision_model."):
                    continue
                # ❌ 접두사 추가 금지
                # if kk.startswith(("text_model.", "text_projection.")):
                #     kk = "text_encoder." + kk

                if kk in target_te_keys:
                    fixed[kk] = v
            return fixed


        mask_norm = _normalize_to_te_keys(mask_only, te_keys)
        print(f"[DEBUG] matched text-enc *_pruning_mask keys: {len(mask_norm)}")
        if len(mask_norm) == 0:
            print("[WARN] No keys matched after normalization. Check your prefixes / source mask.")
            # 계속 진행하더라도 아무 것도 로드되지 않음
        else:
            # 4) dtype 맞춤(버퍼 dtype과 동일하게)
            #   - 일부 프루너는 bool, 일부는 float 마스크 사용 → 대상 버퍼 dtype으로 캐스팅
            sample_k = next(iter(mask_norm.keys()))
            # 버퍼가 실제로 존재하는지 확인
            missing_in_module = [k for k in mask_norm.keys() if k not in te_sd]
            if missing_in_module:
                print(f"[WARN] {len(missing_in_module)} mask keys not found in text_encoder. e.g., {missing_in_module[:3]}")

            casted = {}
            for k, v in mask_norm.items():
                if k not in te_sd:
                    continue
                buf = te_sd[k]
                want = buf.dtype
                vv = v
                if v.dtype != want:
                    # bool<->float 안전 캐스팅
                    if want.is_floating_point and v.dtype == torch.bool:
                        vv = v.float()
                    elif want == torch.bool and v.dtype.is_floating_point:
                        vv = v > 0.5
                    else:
                        vv = v.to(want)
                casted[k] = vv

            # 5) 실제 로드
            msg = self.text_encoder.load_state_dict(casted, strict=False)
            miss = [k for k in msg.missing_keys if k.endswith("_pruning_mask")]
            print("[SD15Generator] mask missing keys (text_encoder):",
                miss[:10], "..." if len(miss) > 10 else "")
            print("[SD15Generator] mask unexpected keys (text_encoder):",
                msg.unexpected_keys[:10], "..." if len(msg.unexpected_keys) > 10 else "")

            # 6) 간단 통계(마스크 평균 → 희소도 감)
            total_els = 0
            sum_vals  = 0.0
            for k, v in casted.items():
                total_els += v.numel()
                if v.dtype == torch.bool:
                    sum_vals += v.float().sum().item()
                else:
                    sum_vals += v.sum().item()
            if total_els > 0:
                print(f"[DEBUG] pruning_mask stats: total={total_els}, "
                    f"ones≈{sum_vals:.0f}, ones/total≈{(sum_vals/total_els):.6f}")

        if is_eval:
            self.text_encoder.eval()
        print("-" * 80)

class SDXLGenerator(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.repo   = config.get("model_name", "stabilityai/stable-diffusion-xl-base-1.0")
        self.device, self.dtype = _auto_device_dtype()

        extra = {}
        if config.get("disable_safety_checker", True):
            extra["safety_checker"] = None

        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.repo, torch_dtype=self.dtype, **extra
        )
        if str(config.get("sampler","")).lower() in {"dpmpp","dpm-solver++","dpmsolver"}:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        pipe = pipe.to(self.device)
        if config.get("enable_attention_slicing", True): pipe.enable_attention_slicing()
        if config.get("enable_xformers", False):
            try: pipe.enable_xformers_memory_efficient_attention()
            except Exception: pass

        self.pipe = pipe
        # ★ SDXL은 텍스트 인코더 2개
        self.text_encoder   = pipe.text_encoder      # CLIPTextModel (L/14)
        self.text_encoder_2 = pipe.text_encoder_2    # CLIPTextModelWithProjection (bigG/14)
        self.tokenizer      = pipe.tokenizer
        self.tokenizer_2    = pipe.tokenizer_2

        # 편의 플래그
        self.normalize_embeddings = config.get("normalize_embeddings", True)
        self.eval()

    # SDXLGenerator 내부
    @torch.no_grad()
    def generate(self, prompts: List[str], **kwargs):
        # 1) seed / generator 중복 방지
        seed = kwargs.pop("seed", None)
        generator = kwargs.pop("generator", None)  # <-- 여기서 꺼내서 우리가 통제

        if generator is None and seed is not None:
            generator = torch.Generator(self.device).manual_seed(int(seed))

        neg = kwargs.pop("negative_prompts", None) or kwargs.pop("negative_prompt", None)
            # 리스트로 들어오면 1개만 허용(여러 개면 배치 길이와 동일해야 함)
        if isinstance(neg, (list, tuple)):
            if len(neg) == 0:
                neg = None
            elif len(neg) == 1:
                neg = neg[0]                      # 문자열 1개로 만들어 브로드캐스트
            elif len(neg) != len(prompts):
                raise ValueError(f"`negative_prompt` length {len(neg)} "
                                f"!= batch size {len(prompts)}")

        # SDXL은 negative_prompt_2도 받습니다. 없으면 같은 걸로 채워줌(선택)
        kwargs.setdefault("negative_prompt", neg)
        kwargs.setdefault("negative_prompt_2", neg)

        return self.pipe(prompt=prompts, generator=generator, **kwargs).images


    # --- (선택) 텍스트 인코딩 유틸: SDXL은 두 인코더 모두 인코딩을 쓸 수 있음 ---
    @torch.no_grad()
    def encode_text_l14(self, texts: List[str]):
        enc = self.tokenizer(texts, padding="max_length", truncation=True,
                             max_length=self.tokenizer.model_max_length, return_tensors="pt").to(self.device)
        out = self.text_encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        last_hidden = out.last_hidden_state
        feats = (last_hidden * enc.attention_mask.unsqueeze(-1)).sum(1) / enc.attention_mask.clamp(min=1).sum(1, keepdim=True)
        return F.normalize(feats, dim=-1) if self.normalize_embeddings else feats

    @torch.no_grad()
    def encode_text_bigG(self, texts: List[str]):
        enc = self.tokenizer_2(texts, padding="max_length", truncation=True,
                               max_length=self.tokenizer_2.model_max_length, return_tensors="pt").to(self.device)
        out = self.text_encoder_2(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        last_hidden = out.last_hidden_state
        feats = (last_hidden * enc.attention_mask.unsqueeze(-1)).sum(1) / enc.attention_mask.clamp(min=1).sum(1, keepdim=True)
        return F.normalize(feats, dim=-1) if self.normalize_embeddings else feats

    # ---- 가중치/마스크 주입 (모듈 교체 X, state_dict 주입 O) ----
    def load_text_encoder_weights(self, l14_repo: Optional[str]=None, bigg_repo: Optional[str]=None):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection
        if l14_repo:
            sd = CLIPTextModel.from_pretrained(l14_repo).state_dict()
            self.text_encoder.load_state_dict(sd, strict=False)
        if bigg_repo:
            sd = CLIPTextModelWithProjection.from_pretrained(bigg_repo).state_dict()
            self.text_encoder_2.load_state_dict(sd, strict=False)

    def _normalize_to_text_model_keys(self, mask_dict, target_keys: set):
        fixed = {}
        for k, v in mask_dict.items():
            kk = k
            # 네가 쓰던 정규화 규칙 재사용
            kk = kk.replace("text_encoder.", "text_model.")
            kk = kk.replace("vision_encoder.", "vision_model.")
            kk = kk.replace("text_proj.", "text_projection.")
            kk = kk.replace("vision_proj.", "visual_projection.")
            if kk.startswith("model."): kk = kk[len("model."):]
            if kk.startswith("cond_stage_model.transformer."):
                kk = "text_model." + kk[len("cond_stage_model.transformer."):]
            # 시각 인코더 잔재 제거
            if (".vision_" in kk) or (".visual_" in kk) or kk.startswith("vision_model."):
                continue
            if kk in target_keys:
                fixed[kk] = v
        return fixed

    def load_from_pruned_pretrained(
        self,
        pretraining_weights: Optional[Union[str, Dict[str, str]]],
        mask_path: Optional[Union[str, Dict[str, str]]],
        config: Optional[Dict[str, Any]] = None,
        is_eval: bool = True,
    ):
        """
        SD1.5/SDXL 공용:
        - (옵션) 텍스트 인코더 가중치 주입(모듈 교체 X, state_dict만 주입)
        - *_pruning_mask 로드: SD1.5(L/14) 또는 SDXL(L/14 + bigG)
        사용 전 반드시 make_prunable(...)로 대상 모듈에 *_pruning_mask 버퍼가 있어야 함.
        """

        try:
            has_bigG_cls = True
        except Exception:
            has_bigG_cls = False

        print("-"*80)
        print("[SD*Generator] load_from_pruned_pretrained(): TEXT ENCODER weights/mask apply")

        is_sdxl = hasattr(self, "text_encoder_2")  # SDXL은 두 번째 인코더 존재

        # ---- 입력 정규화: 문자열(단일) 또는 dict(l14/bigg) 모두 지원 ----
        def _norm_pw(x):
            if x is None:
                return {}
            if isinstance(x, str):
                return {"l14": x}   # 과거 호환: 단일은 L/14로 간주
            if isinstance(x, dict):
                return {str(k).lower(): v for k, v in x.items()}
            raise TypeError("pretraining_weights must be str|dict|None")

        def _norm_mp(x):
            if x is None:
                return {}
            if isinstance(x, str):
                return {"l14": x}
            if isinstance(x, dict):
                return {str(k).lower(): v for k, v in x.items()}
            raise TypeError("mask_path must be str|dict|None")

        pw = _norm_pw(pretraining_weights)
        mp = _norm_mp(mask_path)

        # ---- (옵션) 가중치 주입: 모듈 교체하지 말고 state_dict만 로드 ----
        # L/14
        if "l14" in pw:
            wrepo = pw["l14"]
            if isinstance(wrepo, str) and not os.path.exists(wrepo) and "/" in wrepo:
                print(f"[SD*Generator] Loading L/14 TEXT ENCODER weights (no module replace): {wrepo}")
                te_new_sd = CLIPTextModel.from_pretrained(wrepo).state_dict()
                _msg = self.text_encoder.load_state_dict(te_new_sd, strict=False)
                if len(_msg.missing_keys) or len(_msg.unexpected_keys):
                    print("[L14] weights load missing:", _msg.missing_keys[:5], " unexpected:", _msg.unexpected_keys[:5])
            else:
                print(f"[SD*Generator][WARN] pretraining_weights['l14']='{wrepo}'는 HF repo 형식이 아님(또는 로컬 파일). 건너뜀.")

        # bigG(SDXL)
        if is_sdxl and "bigg" in pw:
            if not has_bigG_cls:
                print("[SDXL][WARN] transformers에 CLIPTextModelWithProjection 없음. bigG 가중치 주입 건너뜀.")
            else:
                wrepo = pw["bigg"]
                if isinstance(wrepo, str) and not os.path.exists(wrepo) and "/" in wrepo:
                    print(f"[SDXL] Loading bigG TEXT ENCODER weights (no module replace): {wrepo}")
                    te2_new_sd = CLIPTextModelWithProjection.from_pretrained(wrepo).state_dict()
                    _msg = self.text_encoder_2.load_state_dict(te2_new_sd, strict=False)
                    if len(_msg.missing_keys) or len(_msg.unexpected_keys):
                        print("[bigG] weights load missing:", _msg.missing_keys[:5], " unexpected:", _msg.unexpected_keys[:5])
                else:
                    print(f"[SDXL][WARN] pretraining_weights['bigg']='{wrepo}'는 HF repo 형식이 아님(또는 로컬 파일). 건너뜀.")

        # ---- 마스크 로더 헬퍼 ----
        def _normalize_to_text_keys(msd: Dict[str, torch.Tensor], target_keys: set) -> Dict[str, torch.Tensor]:
            fixed = {}
            for k, v in msd.items():
                kk = k

                # 공통 상위 접두사 제거
                for pref in ("module.", "model.", "pipe.", "net."):
                    if kk.startswith(pref):
                        kk = kk[len(pref):]

                # SD 파이프라인 내부 경로 접두사 제거(두 인코더 모두)
                for encpref in ("text_encoder.", "text_encoder_2."):
                    if kk.startswith(encpref):
                        kk = kk[len(encpref):]

                # SD1.5 스타일 → SD* 스타일 치환
                if kk.startswith("cond_stage_model.transformer."):
                    kk = "text_model." + kk[len("cond_stage_model.transformer."):]

                # 축약명 치환
                kk = kk.replace("text_proj.",   "text_projection.")
                kk = kk.replace("vision_proj.", "visual_projection.")

                # vision 관련은 제외
                if kk.startswith("vision_model.") or ".vision_" in kk or ".visual_" in kk:
                    continue

                # 최종 매칭
                if kk in target_keys:
                    fixed[kk] = v
            return fixed

        def _load_mask_into(module, tag: str, mask_file: str):
            print(f"[SD*Generator] Loading pruning mask({tag}) from: {mask_file}")
            mask_raw = torch.load(mask_file, map_location="cpu")
            if isinstance(mask_raw, dict) and ("state_dict" in mask_raw or "model_state" in mask_raw):
                mask_raw = mask_raw.get("state_dict", mask_raw.get("model_state"))
            # *_pruning_mask만
            mask_only = {k: v for k, v in mask_raw.items()
                        if isinstance(k, str) and k.endswith("_pruning_mask") and torch.is_tensor(v) and _looks_text_key(k)}
            print(f"[{tag}] total_keys={len(mask_raw) if isinstance(mask_raw, dict) else 0}  mask_keys={len(mask_only)}")

            sd  = module.state_dict()
            keys = set(sd.keys())
            norm = _normalize_to_text_keys(mask_only, keys)
            print(f"[{tag}] matched *_pruning_mask keys: {len(norm)}")

            if len(norm) == 0:
                print(f"[{tag}][WARN] No mask keys matched. make_prunable() 호출 여부/접두사 확인 필요.")
                return

            # dtype/boolean 맞추기
            casted = {}
            for k, v in norm.items():
                want = sd[k].dtype
                if want.is_floating_point and v.dtype == torch.bool:
                    vv = v.float()
                elif want == torch.bool and v.dtype.is_floating_point:
                    vv = v > 0.5
                else:
                    vv = v.to(want)
                casted[k] = vv

            msg = module.load_state_dict(casted, strict=False)
            miss = [k for k in msg.missing_keys if k.endswith("_pruning_mask")]
            if miss:
                print(f"[{tag}] missing mask buffers (first 10): {miss[:10]}")
            if msg.unexpected_keys:
                print(f"[{tag}] unexpected keys (first 10): {msg.unexpected_keys[:10]}")

            # 간단 통계
            total_els = sum(t.numel() for t in casted.values())
            ones = 0.0
            for t in casted.values():
                ones += (t.float().sum().item() if t.dtype != torch.bool else t.float().sum().item())
            if total_els > 0:
                print(f"[{tag}] pruning_mask stats: total={total_els}, ones≈{ones:.0f}, ones/total≈{ones/total_els:.6f}")

        # ---- 마스크 적용 ----
        if "l14" in mp:
            _load_mask_into(self.text_encoder, "L14", mp["l14"])
        if is_sdxl and "bigg" in mp:
            _load_mask_into(self.text_encoder_2, "bigG", mp["bigg"])

        if is_eval:
            self.text_encoder.eval()
            if is_sdxl:
                self.text_encoder_2.eval()
        print("-"*80)
