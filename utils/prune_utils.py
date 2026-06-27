import os
import numpy as np

import torch
import torch.nn as nn

from layers.linear import Linear, SupermaskLinear
from collections import OrderedDict
from transformers import Blip2ForConditionalGeneration

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

def _unwrap_model(m):
    seen = set()
    while True:
        progressed = False
        for attr in ("_forward_module", "module", "model"):
            if hasattr(m, attr):
                n = getattr(m, attr)
                if n is not None and n is not m and id(n) not in seen:
                    seen.add(id(m))
                    m = n
                    progressed = True
                    break
        if not progressed:
            break
    return m

def first_not_none(*xs):
    for x in xs:
        if x is not None:
            return x
    return None

def _get_vision_encoder(m):
    return getattr(m, "vision_encoder", None)

def _get_perceiver(m):
    return (
        getattr(m, "perceiver", None)
        or getattr(m, "perceiver_resampler", None)
        or getattr(m, "resampler", None)
    )

def _get_lang_encoder(m):
    return getattr(m, "lang_encoder", None)

def _pick_vision_tokens(out):
    # tensor
    if torch.is_tensor(out):
        return out
    # dict
    if isinstance(out, dict):
        for k in ("last_hidden_state", "image_tokens", "tokens", "x", "vision_tokens"):
            v = out.get(k, None)
            if torch.is_tensor(v):
                return v
        for v in out.values():
            if torch.is_tensor(v):
                return v
    # tuple/list
    if isinstance(out, (tuple, list)):
        ts = [x for x in out if torch.is_tensor(x)]
        for t in ts:
            if t.ndim == 3:
                return t
        for t in ts:
            if t.ndim == 2:
                return t
        if ts:
            return ts[-1]
    raise RuntimeError(f"Unsupported vision output type: {type(out)}")


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(y) for y in x]   # 변수명 충돌 방지
        elif isinstance(x, tuple):
            return tuple(_apply(y) for y in x)
        else:
            return x

    return _apply(sample)

def flamingo_loss_openflamingo(model, samples, tokenizer=None, max_text_len=64):
    import torch
    import torch.nn as nn

    base = _unwrap_model(model)

    # --- 1) components ---
    ve = getattr(base, "vision_encoder", None)
    perceiver = getattr(base, "perceiver", None) or getattr(base, "perceiver_resampler", None) or getattr(base, "resampler", None)
    lm = getattr(base, "lang_encoder", None)

    if ve is None or perceiver is None or lm is None:
        raise RuntimeError("[OpenFlamingo loss] cannot find vision_encoder / perceiver / lang_encoder")

    # --- 2) image tensor ---
    # batch 키가 환경마다 달라서 안전하게
    image = samples.get("image", None)
    if image is None:
        image = samples.get("pixel_values", None)
    if image is None:
        image = samples.get("images", None)
    if image is None:
        raise TypeError("[OpenFlamingo loss] need image in samples: ('image' or 'pixel_values' or 'images')")

    # (B,C,H,W) -> (B,1,1,C,H,W)
    if image.ndim == 4:
        images = image.unsqueeze(1).unsqueeze(2)
    elif image.ndim == 5:
        images = image.unsqueeze(2)
    elif image.ndim == 6:
        images = image
    else:
        raise ValueError(f"[OpenFlamingo loss] unexpected image.ndim={image.ndim}, shape={tuple(image.shape)}")

    B, T, Fm, C, H, W = images.shape
    images_flat = images.reshape(B * T * Fm, C, H, W)

    # --- 3) text -> input_ids/attention_mask/labels ---
    input_ids = samples.get("input_ids", samples.get("text_ids", None))
    attention_mask = samples.get("attention_mask", samples.get("text_atts", None))

    # pruning score용이면 텍스트가 없을 수도 있음 -> 더미 프롬프트로 만듦
    if input_ids is None or attention_mask is None:
        if tokenizer is None:
            tokenizer = getattr(base, "tokenizer", None)
        if tokenizer is None:
            raise TypeError("[OpenFlamingo loss] tokenizer not found and no input_ids provided.")

        prompt = "<image> <|endofchunk|> Describe this image."
        tok = tokenizer(
            [prompt] * B,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_len,
            add_special_tokens=False,
        )
        input_ids = tok["input_ids"].to(images_flat.device)
        attention_mask = tok["attention_mask"].to(images_flat.device)
    else:
        input_ids = input_ids.to(images_flat.device)
        attention_mask = attention_mask.to(images_flat.device)

    labels = input_ids.clone()
    # pad -> -100
    pad_id = None
    if tokenizer is None:
        tokenizer = getattr(base, "tokenizer", None)
    if tokenizer is not None:
        pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    # --- 4) vision -> perceiver -> vis_x ---
    conditioned_ok = False
    try:
        ve_out = ve(images_flat)
        # vision output에서 토큰 텐서 꺼내기 (CLIP ViT는 보통 [B, V, D])
        if isinstance(ve_out, (tuple, list)):
            # tensor들 중 3D 우선
            ts = [x for x in ve_out if torch.is_tensor(x)]
            vis = next((t for t in ts if t.ndim == 3), ts[-1])
        elif isinstance(ve_out, dict):
            vis = ve_out.get("last_hidden_state", None)
            if vis is None:
                # dict에서 첫 tensor fallback
                for v in ve_out.values():
                    if torch.is_tensor(v):
                        vis = v
                        break
            if vis is None:
                raise RuntimeError("[OpenFlamingo loss] vision encoder dict output has no tensor")
        else:
            vis = ve_out

        if vis.ndim == 2:
            vis = vis.unsqueeze(1)  # (B*TF, 1, D)

        V, D = vis.shape[1], vis.shape[2]
        vis = vis.reshape(B, T, Fm, V, D)
        vis_x = perceiver(vis)

        # --- 5) ✅ 핵심: LM에 conditioning 먼저 ---
        if hasattr(lm, "condition_vis_x"):
            lm.condition_vis_x(vis_x)
            conditioned_ok = True
        elif hasattr(lm, "condition_media"):
            lm.condition_media(vis_x)
            conditioned_ok = True
        else:
            # 최후: model.modules()에서 condition_vis_x 가진 모듈 찾아서 세팅
            for m in base.modules():
                if hasattr(m, "condition_vis_x"):
                    m.condition_vis_x(vis_x)
                    conditioned_ok = True

        if not conditioned_ok:
            raise RuntimeError("[OpenFlamingo loss] could not condition vis_x on language model")

        # --- 6) LM forward (teacher forcing) ---
        out = lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        if not hasattr(out, "loss") or out.loss is None:
            raise RuntimeError("[OpenFlamingo loss] LM forward did not return loss")
        return out.loss

    finally:
        # --- 7) ✅ conditioning clear ---
        if hasattr(lm, "clear_conditioned_layers"):
            try:
                lm.clear_conditioned_layers()
            except Exception:
                pass


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples

import torch
import torch.nn as nn

# def loss_vision_language(model, samples, cuda_enabled):
#     """
#     Robust loss for VLMs (e.g., CLIP/EVA-CLIP, BLIP variants).
#     - Normalizes batch to a dict
#     - Moves tensors to CUDA if requested
#     - Calls model with kwargs (prefers return_loss=True)
#     - If loss is still missing, computes CLIP-style contrastive loss
#     - Returns (loss_tensor, batch_len:int)
#     """
#     # ----- normalize samples to a dict of tensors -----
#     def to_dict(batch):
#         if isinstance(batch, dict):
#             return batch
#         elif isinstance(batch, (list, tuple)):
#             # assume [images, input_ids, attention_mask, ...] style
#             d = {}
#             if len(batch) > 0 and torch.is_tensor(batch[0]):
#                 d["pixel_values"] = batch[0]
#             if len(batch) > 1 and torch.is_tensor(batch[1]):
#                 d["input_ids"] = batch[1]
#             if len(batch) > 2 and torch.is_tensor(batch[2]):
#                 d["attention_mask"] = batch[2]
#             return d
#         elif torch.is_tensor(batch):
#             # image-only tensor
#             return {"pixel_values": batch}
#         else:
#             # unknown structure: wrap as-is (model call will likely fail, but keeps behavior explicit)
#             return {"samples": batch}

#     samples = to_dict(samples)

#     # ----- move to cuda if requested -----
#     def apply_to_sample(f, sample):
#         if len(sample) == 0:
#             return {}
#         def _apply(x):
#             if torch.is_tensor(x):
#                 return f(x)
#             elif isinstance(x, dict):
#                 return {k: _apply(v) for k, v in x.items()}
#             elif isinstance(x, list):
#                 return [_apply(v) for v in x]
#             else:
#                 return x
#         return _apply(sample)

#     if cuda_enabled:
#         samples = apply_to_sample(lambda t: t.cuda(non_blocking=True), samples)

#     # figure out batch size for logging/loop control
#     def _batch_len(samples):
#         if "pixel_values" in samples and torch.is_tensor(samples["pixel_values"]):
#             return samples["pixel_values"].shape[0]
#         if "input_ids" in samples and torch.is_tensor(samples["input_ids"]):
#             return samples["input_ids"].shape[0]
#         # fallback: try any first tensor
#         for v in samples.values():
#             if torch.is_tensor(v):
#                 return v.shape[0]
#         return 0

#     batch_len = _batch_len(samples)

#     # ----- call model, prefer return_loss=True -----
#     # not all models accept return_loss; try it first, then fallback
#     call_kwargs = dict(samples)  # shallow copy
#     call_kwargs.setdefault("return_loss", True)

#     try:
#         out = model(**call_kwargs)
#     except TypeError:
#         # remove return_loss and try again
#         call_kwargs.pop("return_loss", None)
#         out = model(**call_kwargs)

#     # ----- extract/compute loss -----
#     loss = None

#     # HF ModelOutput (attr) or plain dict (key)
#     if hasattr(out, "loss") and out.loss is not None:
#         loss = out.loss
#     elif isinstance(out, dict) and "loss" in out and out["loss"] is not None:
#         loss = out["loss"]

#     # If still no loss, try CLIP-style contrastive loss from logits
#     if loss is None:
#         # support both dict and ModelOutput attribute style
#         def get_field(obj, name):
#             if isinstance(obj, dict):
#                 return obj.get(name, None)
#             return getattr(obj, name, None)

#         logits_img = get_field(out, "logits_per_image")
#         logits_txt = get_field(out, "logits_per_text")

#         if logits_img is not None and logits_txt is not None:
#             # symmetric CE, standard CLIP loss
#             ce = nn.CrossEntropyLoss()
#             bsz_i = logits_img.size(0)
#             bsz_t = logits_txt.size(0)
#             # use min(bsz_i, bsz_t) to be safe if not perfectly aligned
#             bsz = min(bsz_i, bsz_t)
#             target = torch.arange(bsz, device=logits_img.device)
#             loss_i = ce(logits_img[:bsz, :bsz], target)
#             loss_t = ce(logits_txt[:bsz, :bsz], target)
#             loss = 0.5 * (loss_i + loss_t)

#     if loss is None:
#         raise RuntimeError("Model did not return a loss; check the model outputs.")

#     return loss, batch_len

def first_not_none(*xs):
    for x in xs:
        if x is not None:
            return x
    return None

def loss_vision_language(model, samples, cuda_enabled):
    """
    Robust loss for VLMs (CLIP/EVA-CLIP, BLIP, XVLM, HF BLIP-2) 지원.
    - HF BLIP-2:
        * Blip2ForConditionalGeneration: model(..., labels) -> loss 사용
        * Blip2Model(언어헤드 없음): forward -> logits 추출 후 shifted LM CE로 loss 직접 계산
    - BLIP/LAVIS/XVLM: forward(image, text_ids, text_atts, alpha)
    - CLIP: return_loss=True 시도, 실패 시 대칭 CE 백업
    - (loss_tensor, batch_len:int) 반환
    """
    import torch
    import torch.nn as nn

    # ---------------- Utils ----------------
    def to_dict(batch):
        if isinstance(batch, dict):
            return batch
        elif isinstance(batch, (list, tuple)):
            d = {}
            if len(batch) > 0 and torch.is_tensor(batch[0]): d["pixel_values"]   = batch[0]
            if len(batch) > 1 and torch.is_tensor(batch[1]): d["input_ids"]      = batch[1]
            if len(batch) > 2 and torch.is_tensor(batch[2]): d["attention_mask"] = batch[2]
            # alpha (옵션)
            if len(batch) > 3 and (isinstance(batch[3], (float,int)) or torch.is_tensor(batch[3])):
                d["alpha"] = batch[3]
            return d
        elif torch.is_tensor(batch):
            return {"pixel_values": batch}
        else:
            return {"samples": batch}

    def apply_to_sample(f, sample):
        if len(sample) == 0:
            return {}
        def _apply(x):
            if torch.is_tensor(x):
                return f(x)
            elif isinstance(x, dict):
                return {k: _apply(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [_apply(v) for v in x]
            else:
                return x
        return _apply(sample)

    def _batch_len(s):
        for k in ("pixel_values", "image", "input_ids"):
            v = s.get(k, None)
            if torch.is_tensor(v):
                return int(v.shape[0])
        for v in s.values():
            if torch.is_tensor(v):
                return int(v.shape[0])
        return 0

    def _unwrap_model(m):
        # Lightning Fabric(. _forward_module), DDP(.module), 커스텀(.model) 언랩
        seen = set()
        while True:
            progressed = False
            for attr in ("_forward_module", "module", "model"):
                if hasattr(m, attr):
                    n = getattr(m, attr)
                    if n is not None and n is not m and id(n) not in seen:
                        seen.add(id(m))
                        m = n
                        progressed = True
                        break
            if not progressed:
                break
        return m

    def getattr_chain(names, objs, default=None):
        for o in objs:
            if o is None: 
                continue
            for nm in (names if isinstance(names, (list, tuple)) else (names,)):
                try:
                    if hasattr(o, nm):
                        return getattr(o, nm)
                except Exception:
                    pass
        return default

    def resolve_alpha(m, s):
        a = s.get("alpha", None)
        if torch.is_tensor(a):
            a = float(a.item())
        if isinstance(a, (float, int)):
            return float(a)
        for attr in ("alpha", "alpha_base"):
            val = getattr_chain(attr, (m, base), None)
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    pass
        return 1.0

    def find_any_logits(obj):
        """
        다양한 출력 구조에서 logits를 견고하게 찾아냄:
        - obj.logits
        - dict["logits"]
        - obj.language_model_output.logits
        - dict["language_model_output"].logits
        - dict["language_model_outputs"].logits
        """
        # 직접
        if hasattr(obj, "logits"):
            return obj.logits
        if isinstance(obj, dict) and "logits" in obj:
            return obj["logits"]
        # nested 후보
        keys = ["language_model_output", "language_model_outputs", "lm_output", "decoder_outputs"]
        for k in keys:
            sub = getattr(obj, k, None) if not isinstance(obj, dict) else obj.get(k, None)
            if sub is None:
                continue
            if hasattr(sub, "logits"):
                return sub.logits
            if isinstance(sub, dict) and "logits" in sub:
                return sub["logits"]
        # 리스트/튜플 안에서 찾아보기
        if isinstance(obj, (list, tuple)):
            for it in obj:
                lg = find_any_logits(it)
                if lg is not None:
                    return lg
        # dict 값들에서도 찾아보기
        if isinstance(obj, dict):
            for it in obj.values():
                lg = find_any_logits(it)
                if lg is not None:
                    return lg
        return None

    # ---------------- Normalize ----------------
    samples = to_dict(samples)
    if cuda_enabled:
        samples = apply_to_sample(lambda t: t.cuda(non_blocking=True), samples)
    batch_len = _batch_len(samples)

    # ---------------- Detect model type ----------------
    base   = _unwrap_model(model)
    name   = (str(getattr(base, "name", "")) or str(getattr(model, "name", ""))).lower()
    clsn   = base.__class__.__name__.lower()
    modpkg = getattr(base.__class__, "__module__", "").lower()
    config = getattr_chain("config", (base, model), None)

    # HF BLIP-2 여부
    is_blip2_hf = (
        ("blip2" in clsn) or
        ("blip_2" in modpkg) or
        (getattr(config, "model_type", None) == "blip-2")
    )
        # HF LLaVA 여부
    is_llava_hf = (
        "llava" in clsn
        or "llava" in modpkg
        or "llava" in name
    )
    is_qwen_vl_hf = (
        "qwen" in clsn
        or "qwen" in modpkg
        or "qwen" in name
    )
    # ---------------- Detect OpenFlamingo (Flamingo) ----------------
    base = _unwrap_model(model)
    clsn = base.__class__.__name__.lower()

    is_openflamingo = (
        ("flamingo" in clsn) and
        hasattr(base, "vision_encoder") and
        hasattr(base, "lang_encoder")
    )

    if is_openflamingo:
        tok = getattr_chain("tokenizer", (base, model), None)
        loss = flamingo_loss_openflamingo(model, samples, tokenizer=tok, max_text_len=64)
        return loss, batch_len

    # BLIP/LAVIS/XVLM 여부 (괄호 강조)
    is_blip_like = (
        (("blip" in name) or ("blip" in clsn) or ("xvlm" in name) or ("xvlm" in clsn))
    ) and (not is_blip2_hf)

    # ---------------- Forward per family ----------------
    out = None
    loss = None

    if is_blip2_hf:
        # pixel_values
        pixel_values = samples.get("pixel_values", None)
        if pixel_values is None:
            # 가능하면 processor 변환 (DataLoader에서 미리 만드는 것을 권장)
            img  = samples.get("image", None)
            proc = getattr_chain("processor", (base, model), None)
            if img is not None and proc is not None:
                try:
                    px = proc(images=img, return_tensors="pt")["pixel_values"]
                    pixel_values = px.cuda(non_blocking=True) if cuda_enabled else px
                except Exception:
                    pass
        if pixel_values is None:
            raise TypeError("BLIP-2(HF) 손실 계산에는 'pixel_values'가 필요합니다. "
                            "DataLoader에서 AutoProcessor로 미리 만들어 넣어주세요.")

        # ids / mask / labels
        input_ids      = samples.get("input_ids", samples.get("text_ids", None))
        attention_mask = samples.get("attention_mask", samples.get("text_atts", None))
        labels = samples.get("labels", None)
        if labels is None and input_ids is not None:
            labels = input_ids.clone()

        # pad -> -100
        if labels is not None:
            pad_id = None
            tok = getattr_chain("tokenizer", (base, model), None)
            if tok is not None:
                try:
                    pad_id = tok.pad_token_id
                except Exception:
                    pad_id = None
            if pad_id is None and config is not None:
                try:
                    pad_id = getattr(getattr(config, "text_config", None), "pad_token_id", None)
                except Exception:
                    pad_id = None
            if pad_id is not None:
                labels = labels.clone()
                labels[labels == pad_id] = -100

        if cuda_enabled:
            if input_ids is not None:      input_ids = input_ids.cuda(non_blocking=True)
            if attention_mask is not None: attention_mask = attention_mask.cuda(non_blocking=True)
            if labels is not None:         labels = labels.cuda(non_blocking=True)

        # ForConditionalGeneration이면 모델이 loss 제공
        is_for_cg = ("forconditionalgeneration" in clsn)
        if is_for_cg:
            out = model(pixel_values=pixel_values, labels=labels, return_dict=True)
            if hasattr(out, "loss") and out.loss is not None:
                loss = out.loss
        else:
            # Blip2Model: logits를 찾아서 shifted LM CE 직접 계산
            if input_ids is None:
                raise TypeError("Blip2Model 경로에서는 LM loss 계산을 위해 'input_ids'가 필요합니다.")
            # forward 호출 (labels는 전달하지 않음)
            out = model(pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True)
            logits = find_any_logits(out)
            if logits is None:
                raise TypeError(
                    "Blip2Model 출력에서 logits를 찾지 못했습니다. "
                    "가능한 키: .logits / ['logits'] / ['language_model_output'].logits 등"
                )
            if labels is None:
                # 그래도 labels가 없다면 input_ids로 대체
                labels = input_ids

            # Shifted LM CE
            # logits: (B, T, V), labels: (B, T)
            if logits.dim() != 3:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)} (expected 3D: BxTxV)")
            if labels.dim() != 2:
                raise RuntimeError(f"Unexpected labels shape: {tuple(labels.shape)} (expected 2D: BxT)")

            # 길이 최소 2 필요 (shift)
            if logits.size(1) < 2 or labels.size(1) < 2:
                raise RuntimeError("Sequence length must be >=2 for shifted LM loss.")

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:,  1: ].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

    elif is_blip_like:
        # LAVIS BLIP / XVLM: forward(image, text_ids, text_atts, alpha)
        image     = samples.get("image", samples.get("pixel_values", None))
        text_ids  = samples.get("text_ids", samples.get("input_ids", None))
        text_atts = samples.get("text_atts", samples.get("attention_mask", None))
        alpha     = resolve_alpha(model, samples)

        if image is None or text_ids is None or text_atts is None:
            missing = [k for k,v in [("image",image),("text_ids",text_ids),("text_atts",text_atts)] if v is None]
            raise TypeError(f"BLIP forward requires {missing}; got keys={list(samples.keys())}")

        out = model(image=image, text_ids=text_ids, text_atts=text_atts, alpha=alpha)

    elif is_qwen_vl_hf:
        input_ids = samples.get("input_ids", None)
        attention_mask = samples.get("attention_mask", None)
        labels = samples.get("labels", None)

        if input_ids is None:
            raise TypeError("Qwen-VL loss 계산에는 'input_ids'가 필요합니다.")
        if labels is None:
            labels = input_ids.clone()

        pad_id = None
        if config is not None:
            try:
                pad_id = getattr(config, "pad_token_id", None)
            except Exception:
                pad_id = None
        if pad_id is not None:
            labels = labels.clone()
            labels[labels == pad_id] = -100

        top = getattr(model, "_forward_module", model)
        top = getattr(top, "module", top)

        try:
            out = top(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            loss = getattr(out, "loss", None)
        except Exception:
            qwen_text = getattr(top, "model", top)
            out = qwen_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None and isinstance(out, (list, tuple)) and out:
                hidden = out[0]
            if hidden is None:
                raise
            loss = hidden.float().pow(2).mean()

    elif is_llava_hf:
        # ----- LLaVA(HF): language_model(Llama)만 써서 LM loss 계산 -----
        # samples 에서 토큰 꺼내기
        input_ids      = samples.get("input_ids", None)
        attention_mask = samples.get("attention_mask", None)
        labels         = samples.get("labels", None)

        if input_ids is None:
            raise TypeError("LLaVA loss 계산에는 'input_ids'가 필요합니다.")

        # labels 없으면 input_ids 로 재사용
        if labels is None:
            labels = input_ids.clone()

        # pad 토큰을 -100 으로 마스킹
        pad_id = None
        if config is not None:
            # LlavaConfig 에 pad_token_id 가 있을 수도 있고
            try:
                pad_id = getattr(config, "pad_token_id", None)
            except Exception:
                pad_id = None
            # 없으면 text_config 쪽에서 가져오기
            if pad_id is None:
                try:
                    pad_id = getattr(getattr(config, "text_config", None), "pad_token_id", None)
                except Exception:
                    pad_id = None

        if pad_id is not None:
            labels = labels.clone()
            labels[labels == pad_id] = -100

        # 실제 LM 모듈 꺼내기 (LlavaForConditionalGeneration.language_model)
        llava_base = base   # _unwrap_model(model) 로 위에서 만든 애
        lm = getattr(llava_base, "language_model", None)
        if lm is None:
            raise TypeError("LLaVA base model 에 'language_model' 속성이 없습니다.")

        # 텍스트-only LM loss 계산
        out = lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        loss = out.loss

    else:
        # CLIP 계열
        call_kwargs = {}
        if "pixel_values" in samples:   call_kwargs["pixel_values"] = samples["pixel_values"]
        if "input_ids" in samples:      call_kwargs["input_ids"] = samples["input_ids"]
        if "attention_mask" in samples: call_kwargs["attention_mask"] = samples["attention_mask"]
        call_kwargs["return_loss"] = True
        try:
            out = model(**call_kwargs)
        except TypeError:
            call_kwargs.pop("return_loss", None)
            out = model(**call_kwargs)

    # ---------------- loss 추출/백업 ----------------
    if loss is None:
        # 1) 표준 위치
        if hasattr(out, "loss") and out.loss is not None:
            loss = out.loss
        elif isinstance(out, dict) and out.get("loss", None) is not None:
            loss = out["loss"]

    if loss is None and isinstance(out, dict):
        # 2) BLIP/LAVIS 분해 로스 합
        acc = 0.0
        found = False
        for k in ("loss_itc", "loss_itm", "loss_lm", "loss_ita", "loss"):
            v = out.get(k, None)
            if torch.is_tensor(v):
                acc = acc + v
                found = True
        if found:
            loss = acc

    if loss is None:
        # 3) CLIP-style CE from logits
        def get_field(obj, name):
            if isinstance(obj, dict):
                return obj.get(name, None)
            return getattr(obj, name, None)
        logits_img = get_field(out, "logits_per_image")
        logits_txt = get_field(out, "logits_per_text")
        if logits_img is not None and logits_txt is not None:
            ce = nn.CrossEntropyLoss()
            bsz = min(logits_img.size(0), logits_txt.size(0))
            target = torch.arange(bsz, device=logits_img.device)
            loss_i = ce(logits_img[:bsz, :bsz], target)
            loss_t = ce(logits_txt[:bsz, :bsz], target)
            loss = 0.5 * (loss_i + loss_t)

    if loss is None:
        # 4) 마지막 안전장치: (loss_tensor, ...) 형태
        if isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
            loss = out[0]

    if loss is None:
        raise RuntimeError("Model did not return a loss; HF BLIP-2/BLIP/XVLM/CLIP 시그니처 모두 시도했지만 실패했습니다.")

    return loss, batch_len


def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """

    # this will work for Linear classes
    for name, buf in module.named_buffers():
        if "pruning_mask" in name:
            yield buf
        
    # this will work for SupermaskLinear classes
    for name, param in module.named_parameters():
        if "pruning_mask" in name:
            yield param


def named_masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    # this will work for Linear classes
    for name, buf in module.named_buffers():
        if "pruning_mask" in name:
            yield name.replace('_forward_module.', ''), buf
    
    # this will work for SupermaskLinear classes
    for name, param in module.named_parameters():
        if "pruning_mask" in name:
            yield name.replace('_forward_module.', ''), param


def prunable(module):
    return isinstance(module, (Linear, SupermaskLinear))


def parameters(model):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    for module in model.modules():
        for param in module.parameters(recurse=False):
            yield param


def masked_parameters(model, bias=False):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in filter(lambda p: prunable(p), model.modules()):
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
            if param is not module.bias or bias is True:
                yield mask, param

        
def recursive_getattr(obj, attr):
    attrs = attr.split('.')
    if len(attrs) == 1:
        return getattr(obj, attrs[0])
    
    direct_child = getattr(obj, attrs[0])
    rest_of_attrs = '.'.join(attrs[1:])
    return recursive_getattr(direct_child, rest_of_attrs)


def named_masked_parameters(model, bias=False, exclude=[]):
    for pname, param in model.named_parameters(): #모델의 모든 학습 가능한 파라미터 반환
        if any(e in pname for e in exclude): continue
        if '.bias' in pname and not bias: continue
        
        parent_module = '.'.join(pname.split('.')[:-1])#각 파라미터의 상위 모듈 경로 추출
        if parent_module == '': continue
        
        parent_module = recursive_getattr(model, parent_module)#recursive_getattr()로 실제 객체로 변환
        if not prunable(parent_module): continue
        
        if pname.endswith('pruning_mask'):
            continue
        else:
            yield pname, recursive_getattr(model, pname + '_pruning_mask'), param



# def _make_prunable(model: nn.Module, mask_dtype=torch.bool, pattern_lock=True, mask_on_the_fly=True, store_input=False, store_output=False) -> nn.Module: 
#     # replace every module children with their prunable counterpart
#     for name, module in model.named_children(): # 모듈의 하위 모듈을 순회
#         #기본 레이어에서는 가중치와 편향만 관리하는데, 마스크 값(0)을 저장할 수 있는 변수를 갖게 함
#         # replacing linear layers
#         if isinstance(module, nn.Linear): # 모듈이 nn.Linear 타입일 경우 prunable하게 함
#             prunable_linear = Linear.from_pretrained( # pruning 가능한 Linear layer로 변환함
#                 module, 
#                 mask_dtype=mask_dtype, 
#                 pattern_lock=pattern_lock, 
#                 mask_on_the_fly=mask_on_the_fly, 
#                 store_input=store_input, 
#                 store_output=store_output
#             )
#             setattr(model, name, prunable_linear) # 기존 모듈을 prunable_linear로 교체
        
        # NOTE: implement and substitute your custom layers here :) 

def _make_prunable(model, mask_dtype=torch.bool, pattern_lock=True, mask_on_the_fly=True, store_input=False, store_output=False):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            prunable_linear = Linear.from_pretrained(
                module,
                mask_dtype=mask_dtype,
                pattern_lock=pattern_lock,
                mask_on_the_fly=mask_on_the_fly,
                store_input=store_input,
                store_output=store_output
            )
            setattr(model, name, prunable_linear)

    

def make_prunable(model: nn.Module, mask_dtype=torch.bool, pattern_lock=True, mask_on_the_fly=True, store_input=False, store_output=False) -> nn.Module:
    """
    Makes an nn.Module prunable by replacing native torch
    layers with custom prunable layers containing masks.
    """
    if hasattr(model, 'prunable') and model.prunable:  #이미 prunable한 모델이면 돌아감
        return 
    print("[Debug] utils.prune_utils.py : make_prunable() 함수 실행 -> 모델의 모든 하위 module 순회")
    for module in model.modules(): # 모델의 모든 하위 모듈을 순회
        _make_prunable(
            module, 
            mask_dtype=mask_dtype, 
            pattern_lock=pattern_lock, 
            mask_on_the_fly=mask_on_the_fly, 
            store_input=store_input, 
            store_output=store_output
        )
    
    setattr(model, 'prunable', True) # 모델에 prunable 속성 추가하여, 이 모델이 prunable해짐을 표시


def _make_searchable(model: nn.Module, exclude: list[str] = []) -> nn.Module: 

    # replace every module children with their searchable counterpart
    for name, module in model.named_children():
        # skip excluded modules, e.g. matching head if provided
        if any(e in name for e in exclude): continue

        # replacing linear layers
        if isinstance(module, nn.Linear):
            searchable_linear = SupermaskLinear.from_pretrained(module)
            setattr(model, name, searchable_linear)



def make_searchable(model: nn.Module, exclude: list[str] = []) -> nn.Module:
    """
    Makes an nn.Module searchable by replacing native torch
    `nn.Linear` layers with custom `SupermaskLinear` layers.
    """   
    # replace every module children with their searchable counterpart
    if hasattr(model, 'searchable') and model.searchable: 
        return
    [_make_searchable(module, exclude=exclude) for name, module in model.named_modules() \
     if not any(e in name for e in exclude)]
    return model


def stats(named_masked_parameters):
    remaining_params, total_params = 0, 0 
    for _, mask, _ in named_masked_parameters:
        remaining_params += mask.detach().cpu().numpy().sum()
        total_params += mask.numel()
    return remaining_params, total_params


def dumpable_named_masks(model: nn.Module):
    return OrderedDict(named_masks(model))


def state_dict_without_masks(model: nn.Module):
    return OrderedDict(
        filter(
            lambda tpl: not tpl[0].endswith('pruning_mask'), 
            {k.replace('_forward_module', ''): v for k, v in model.state_dict().items()}.items()
        )
    )


def disentangle_path(path):
    name, ext = os.path.splitext(path)
    params_path = name + '_params' + ext
    masks_path = name + '_pruning_masks' + ext
    return params_path, masks_path


def save_prunable_model(model, path, mask_only=False, **kwargs):
    params_path, masks_path = disentangle_path(path)
    if mask_only:
        torch.save(dumpable_named_masks(model), masks_path, **kwargs)
    else:
        torch.save(dumpable_named_masks(model), masks_path, **kwargs)
        torch.save(state_dict_without_masks(model), params_path, **kwargs)
    return params_path, masks_path


def check_gradients(model: nn.Module):
    assert all(
        [module.check_gradients() for module in filter(lambda m: isinstance(m, Linear), model.modules())]
    ), "Gradients at masked positions are not 0. Please check your backward hooks are set correctly."


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def remaining_params(model: nn.Module) -> int:
    remaining_params = 0
    for mask in masks(model):
        remaining_params += torch.sum(mask)
    return remaining_params.item()

def prunable_params(model: nn.Module) -> int:
    prunable_params = 0
    for mask in masks(model):
        prunable_params += mask.numel()
    return prunable_params


def generate_mesh(num_stages, base_level, sparsity_level, mesh_type):
    repeat=1
    if num_stages == 1:
        return [sparsity_level]
    if mesh_type == 'exp':
        sparsity_multiplier = (sparsity_level - base_level)*np.power(2, num_stages-1)/(np.power(2, num_stages-1) - 1)
        l = [base_level + sparsity_multiplier*((np.power(2, stage) - 1)/np.power(2, stage)) for stage in range(num_stages)]
        return [x for x in l for _ in range(repeat)]
    elif mesh_type == 'poly':
        l = [sparsity_level + (base_level-sparsity_level)*np.power(1 - (stage/(num_stages-1)), 3) for stage in range(num_stages)]
        return [x for x in l for _ in range(repeat)]
    elif mesh_type == 'const':
        return [sparsity_level for stage in range(num_stages)]
    elif mesh_type == 'linear':
        return [base_level + stage*(sparsity_level - base_level)/(num_stages-1) for stage in range(num_stages)]
    elif mesh_type == 'MFAC':
        sparsity_multiplier = ((1. - sparsity_level) / (1. - base_level)) ** (1./num_stages)
        return [1. - ((1. - base_level) * (sparsity_multiplier**(stage+1))) for stage in range(num_stages)]
    

def generate_schedule(num_stages, sparsity_level, schedule):
    if schedule == 'linear':
        return [sparsity_level + (1.0 - sparsity_level)*((stage + 1) / num_stages) for stage in range(num_stages)]
    elif schedule == 'exp':
        return [1.0 - (1.0 - sparsity_level)**((stage + 1) / num_stages) for stage in range(num_stages)]
    elif schedule == 'const':
        return [sparsity_level for _ in range(num_stages)]
    else:
        raise NotImplementedError(f"Schedule {schedule} not implemented.")


def tie_blip_params_for_pruner(named_masked_parameters):
    new_named_masked_parameters = []
    for name, mask, param in named_masked_parameters:
        vision_or_text_encoder = name.startswith("visual_encoder") or name.startswith("text_encoder")
        causal_self_attn_in_decoder = name.startswith("text_decoder") and ".attention." in name
        if vision_or_text_encoder or causal_self_attn_in_decoder:
            new_named_masked_parameters.append((name, mask, param))
    return new_named_masked_parameters


def is_blip_text_encoder_weight_tied(name):
    # all the weights of both the text-enc and the text-dec are tied to 
    # each other except for the SA / Causal SA layers
    return name.startswith("text_encoder") and ".attention." not in name

def is_blip_text_decoder_weight_tied(name):
    # all the weights of both the text-enc and the text-dec are tied to 
    # each other except for the SA / Causal SA layers
    return name.startswith("text_decoder") and ".attention." not in name


def tie_blip_gradients_for_pruner(named_masked_parameters, blip_model):
    # organize blip model's gradients as a dict
    text_dec_grads_from_model = {k: v.grad for k, v in blip_model.named_parameters() if k.startswith("text_decoder")}
    for name, mask, param in named_masked_parameters:
        if is_blip_text_encoder_weight_tied(name):
            name_in_decoder = name.replace("text_encoder", "text_decoder.bert")
            param.grad += text_dec_grads_from_model[name_in_decoder]
    return named_masked_parameters


def inherit_encoder_decoder_params(named_masked_parameters, blip_model):
    text_dec_masked = {}
    for name, _, param in named_masked_parameters:
        if is_blip_text_encoder_weight_tied(name):
            name_in_decoder = name.replace("text_encoder", "text_decoder.bert")
            text_dec_masked[name_in_decoder] = param
    
    blip_model.load_state_dict(text_dec_masked, strict=False)
    return blip_model


def check_blip_tie(named_masked_parameters, blip_model):
    # arrange blip's weights as a dict
    text_dec_weights_from_model = {k: v for k, v in blip_model.named_parameters() if k.startswith("text_decoder")}
    for name, _, param in named_masked_parameters:
        if is_blip_text_encoder_weight_tied(name):
            name_in_decoder = name.replace("text_encoder", "text_decoder.bert")
            assert torch.allclose(param, text_dec_weights_from_model[name_in_decoder], atol=1e-6), f"BLIP tie check failed for {name}" 


def check_blip_gradients_tie(named_masked_parameters, blip_model):
    # arrange blip's gradients as a dict
    text_dec_grads_from_model = {k: v.grad for k, v in blip_model.named_parameters() if k.startswith("text_decoder")}
    for name, _, param in named_masked_parameters:
        if is_blip_text_encoder_weight_tied(name):
            name_in_decoder = name.replace("text_encoder", "text_decoder.bert")
            assert torch.allclose(param.grad, text_dec_grads_from_model[name_in_decoder], atol=1e-6), f"BLIP gradients tie check failed for {name}"
    

def inherit_encoder_decoder_masks(mask_dict):
    # for every encoder mask, if the weights is tied to the decoder, inherit the mask
    enc_dec_dict = {k: v for k, v in mask_dict.items()}
    for name, mask in mask_dict.items():
        if is_blip_text_encoder_weight_tied(name) and not name.startswith("text_encoder_m"):
            name_in_decoder = name.replace("text_encoder", "text_decoder.bert")
            enc_dec_dict[name_in_decoder] = mask
    return enc_dec_dict


def inherit_encoder_momentum_masks(mask_dict):
    # for every encoder mask, duplicate it for the respective momentum encoder
    enc_momentum_dict = {k: v for k, v in mask_dict.items()}
    for name, mask in mask_dict.items():
        if name.startswith("visual_encoder") and not name.startswith("visual_encoder_m"):
            name_in_momentum = name.replace("visual_encoder", "visual_encoder_m")
            enc_momentum_dict[name_in_momentum] = mask
        elif name.startswith("text_encoder") and not name.startswith("text_encoder_m"):
            name_in_momentum = name.replace("text_encoder", "text_encoder_m")
            enc_momentum_dict[name_in_momentum] = mask
    return enc_momentum_dict


def grab_encoder_from_decoder_module(blip_model, decoder_name):
    name_in_encoder = decoder_name.replace("text_decoder.bert", "text_encoder")
    for name, module in blip_model.named_modules():
        if name == name_in_encoder:
            return module


def check_blip_state_dict(state_dict):
    blip = {k: v for k, v in state_dict.items() if 'visual_encoder' not in k and 'text_encoder_m' not in k}

    for k, v in blip.items():
        if is_blip_text_encoder_weight_tied(k) and 'layer.' in k:
            v_in_decoder = blip[k.replace('text_encoder', 'text_decoder.bert')]
            assert torch.allclose(v, v_in_decoder)
