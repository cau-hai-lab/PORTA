import torch
from utils.prune_utils import (
    recursive_getattr
)
def _find_conditioner(obj):
    """
    OpenFlamingo에서 vis_x를 주입(condition)할 대상(lang_encoder 등)을 찾아 반환.
    우선순위:
      1) obj.lang_encoder (가장 흔함)
      2) obj.model.lang_encoder / obj.module.lang_encoder 등 래핑
      3) obj 내부 모듈들 중 condition_vis_x/condition_media 메서드 가진 것
    """
    candidates = []

    # 1) 흔한 경로들 먼저
    for path in [
        "lang_encoder",
        "model.lang_encoder",
        "module.lang_encoder",
        "model.module.lang_encoder",
    ]:
        cur = obj
        ok = True
        for p in path.split("."):
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok:
            candidates.append(cur)

    # 2) 혹시 language_model로 붙어있는 케이스도 방어
    for path in [
        "language_model",
        "model.language_model",
        "module.language_model",
    ]:
        cur = obj
        ok = True
        for p in path.split("."):
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok:
            candidates.append(cur)

    # 3) 최후: 전체 서브모듈 스캔
    try:
        for m in obj.modules():
            if hasattr(m, "condition_vis_x") or hasattr(m, "condition_media"):
                candidates.append(m)
    except Exception:
        pass

    # 후보 중에서 실제 메서드 있는 것 반환
    for m in candidates:
        if hasattr(m, "condition_vis_x") or hasattr(m, "condition_media"):
            return m

    return None


def _apply_conditioner(cond_module, vis_x):
    if hasattr(cond_module, "condition_vis_x"):
        cond_module.condition_vis_x(vis_x)
        return "condition_vis_x"
    if hasattr(cond_module, "condition_media"):
        cond_module.condition_media(vis_x)
        return "condition_media"
    raise ValueError("no condition method")

def _unwrap_model(m):
    # DDP/DataParallel
    if hasattr(m, "module"):
        m = m.module
    # (환경에 따라) 다른 wrapper도 방어
    if hasattr(m, "_fsdp_wrapped_module"):
        m = m._fsdp_wrapped_module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m

import inspect
import torch.nn as nn

def _looks_like_lm_forward(mod: nn.Module) -> bool:
    """input_ids 기반 forward를 받는 '언어모델'인지 대충 판별"""
    if mod is None or not hasattr(mod, "forward"):
        return False
    try:
        sig = inspect.signature(mod.forward)
        params = sig.parameters
        return ("input_ids" in params) or ("input_ids" in sig.parameters.keys())
    except Exception:
        # signature 못 뽑는 경우도 있으니, 최소한 호출 키워드 후보로 판별
        # (너 케이스에선 거의 signature가 뽑힘)
        return hasattr(mod, "generate") or hasattr(mod, "transformer")

def _find_flamingo_lm(model) -> nn.Module | None:
    """
    ✅ FlamingoLayer 같은 '레이어'를 절대 반환하지 않고,
    ✅ 실제 LM(lang_encoder)을 우선 반환하도록 강제.
    """
    # 1) 가장 확실: model.lang_encoder
    lm = getattr(model, "lang_encoder", None)
    if lm is not None:
        # FlamingoLayer는 보통 forward에 input_ids가 없음
        if _looks_like_lm_forward(lm):
            return lm

    # 2) 혹시 다른 이름으로 붙어있으면 후보 탐색
    # (model.language_model, model.lm 등)
    for attr in ("language_model", "lm", "text_model", "decoder"):
        cand = getattr(model, attr, None)
        if cand is not None and _looks_like_lm_forward(cand):
            return cand

    # 3) 최후: 모듈 탐색 (단, FlamingoLayer 제외)
    for m in model.modules():
        if m is None:
            continue
        # FlamingoLayer(크로스어텐션 레이어)류는 input_ids forward가 아니므로 걸러짐
        if _looks_like_lm_forward(m) and hasattr(m, "condition_vis_x"):
            return m

    return None


def _get_vision_encoder(model):
    model = _unwrap_model(model)
    return getattr(model, "vision_encoder", None) or getattr(getattr(model, "model", None), "vision_encoder", None)

def _get_perceiver(model):
    model = _unwrap_model(model)
    # 코드베이스마다 이름이 섞일 수 있어서 여러 후보
    return (
        getattr(model, "perceiver", None)
        or getattr(model, "perceiver_resampler", None)
        or getattr(model, "resampler", None)
        or getattr(getattr(model, "model", None), "perceiver", None)
    )

def _pick_vision_tokens(ve_out):
    """
    open_clip / hf-clip / 커스텀 VE가 반환하는 다양한 형태에서
    'tokens' 텐서(B, V, D) 또는 (B, D)를 최대한 안전하게 뽑아낸다.
    우선순위: tokens(3D) > pooled(2D)
    """
    # 1) dict 형태
    if isinstance(ve_out, dict):
        # 가능한 키 후보들
        for k in ["image_tokens", "last_hidden_state", "tokens", "x", "vision_tokens"]:
            v = ve_out.get(k, None)
            if torch.is_tensor(v):
                return v
        for k in ["image_features", "pooler_output", "pooled", "embeds"]:
            v = ve_out.get(k, None)
            if torch.is_tensor(v):
                return v
        # 못 찾으면 dict 안의 첫 텐서
        for v in ve_out.values():
            if torch.is_tensor(v):
                return v
        raise ValueError(f"Vision encoder returned dict but no tensor found. keys={list(ve_out.keys())}")

    # 2) tuple/list 형태
    if isinstance(ve_out, (tuple, list)):
        # 텐서만 모아서
        ts = [x for x in ve_out if torch.is_tensor(x)]
        if not ts:
            raise ValueError("Vision encoder returned tuple/list but no tensor elements found.")
        # 3D(tokens) 우선
        for t in ts:
            if t.ndim == 3:
                return t
        # 그 다음 2D(pooled)
        for t in ts:
            if t.ndim == 2:
                return t
        # 마지막 fallback
        return ts[-1]

    # 3) 그냥 텐서면 그대로
    if torch.is_tensor(ve_out):
        return ve_out

    raise ValueError(f"Unsupported vision encoder output type: {type(ve_out)}")

def xvlm_general_forward(model, batch, device, custom_temp=None):
    # code taken from XVLM-pretraining
    images, batch = batch[0].to(device, non_blocking=True), [t.to(device) if t is not None else None for t in batch[1:]]
    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = batch
    loss_itc, loss_itm, loss_mlm = model(
        images, text_ids, text_atts, 
        text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids, custom_temp=custom_temp
    )
    loss = loss_itc + loss_itm + loss_mlm
    return loss


def blip_general_forward(model, batch, device, **kwargs):
    image, text_ids, text_atts = batch
    loss_ita, loss_itm, loss_lm = model(image.to(device), text_ids.to(device), text_atts.to(device), alpha = 0.)  
    loss = loss_ita + loss_itm + loss_lm  
    return loss


def vit_general_forward(model, batch, device, **kwargs):
    ce_loss = torch.nn.CrossEntropyLoss().to(device)
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    preds = model(images).logits
    loss = ce_loss(preds, labels)
    return loss


def general_forward(model_name, model, batch, device, **kwargs):
    if model_name == 'xvlm':
        return xvlm_general_forward(model, batch, device, **kwargs)
    elif model_name == 'blip':
        return blip_general_forward(model, batch, device, **kwargs)
    elif model_name in ('vit-b', 'dino'):
        return vit_general_forward(model, batch, device, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported.")


def xvlm_forward_output(model, batch, device, modality="vision"):
    images, batch = batch[0].to(device, non_blocking=True), [t.to(device) if t is not None else None for t in batch[1:]]
    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = batch
    image_embeds, image_atts = model.get_vision_embeds(images)
    if modality == "vision":
        return image_embeds
    text_embeds = model.get_text_embeds(text_ids, text_atts)
    if modality == "text":
        return text_embeds
    output_cls_token = model.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds, text_atts=text_atts)
    return output_cls_token

# def blip_forward_output(model, batch, device, **kwargs):
#     print("[Debug] pruners/accumulators.py -> blip_forward_output()함수 호출 : pruning을 위한 순전파 수행")

#     use_wx_hooks = kwargs.get("use_wx_hooks", False)
#     plot_spec    = kwargs.get("plot_spec", {"vision":[0,5,11], "text_enc":[0,5,11], "text_dec":[]})
#     run_tag      = kwargs.get("run_tag", "run")
#     base_plot_dir= kwargs.get("base_plot_dir", "/data/hai_kms/multiflow/plot/smoothflow")
#     downsample   = kwargs.get("downsample", None)

#     image, text_ids, text_atts = batch

#     # [HOOK] 등록
#     plotter = None
#     if use_wx_hooks:
#         plotter = ForwardWXPlotter(
#             model, run_tag=run_tag, base_dir=str(base_plot_dir),
#             plot_spec=plot_spec, downsample=downsample
#         )
#         plotter.register_all()

#     try:
#         # ===== Vision encoder =====
#         print("[Debug] ... Vision encoder forward")
#         image_embeds = model.visual_encoder(image)
#         image_atts   = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

#         # ===== Text encoder =====
#         encoder_input_ids = text_ids.clone()
#         encoder_input_ids[:, 0] = model.tokenizer.enc_token_id
#         print("[Debug] ... Image-grounded text encoder forward")
#         output_pos = model.text_encoder(
#             encoder_input_ids,
#             attention_mask=text_atts,
#             encoder_hidden_states=image_embeds,
#             encoder_attention_mask=image_atts,
#             return_dict=True,
#         )

#         # ===== Text decoder =====
#         decoder_input_ids = text_ids.clone()
#         decoder_input_ids[:, 0] = model.tokenizer.bos_token_id
#         print("[Debug] ... Image-grounded text decoder forward")
#         decoder_output = model.text_decoder(
#             decoder_input_ids,
#             attention_mask=text_atts,
#             encoder_hidden_states=image_embeds,
#             encoder_attention_mask=image_atts,
#             return_dict=True,
#         )

#     finally:
#         # [HOOK] 해제
#         if plotter is not None:
#             plotter.unregister_all()

#     return {
#         "image_embeds": image_embeds,
#         "encoder_output": output_pos,
#         "decoder_output": decoder_output
#     }


def blip_forward_output(model, batch, device, **kwargs):
    print("[Debug] pruners/accumulators.py -> blip_forward_output()함수 호출 : pruning을 위한 순전파 수행")

    # === X-Heatmap 훅 옵션 ===
    use_x_heatmap = kwargs.get("use_x_heatmap", False)
    run_tag       = kwargs.get("run_tag", "run")
    base_plot_dir = kwargs.get("base_plot_dir", "./plots/smoothflow/x_heatmaps")
    plot_spec     = kwargs.get("plot_spec", {"vision":[0,5,11], "text_enc":[0,5,11], "text_dec":[0,5,11]})

    xhook = None

    image, text_ids, text_atts = batch

    try:
        # [HOOK] 등록 (배치 0에서만 호출되도록 바깥에서 use_x_heatmap 제어)
        if use_x_heatmap:
            xhook = XHeatmapHook(model, run_tag=run_tag, base_dir=base_plot_dir, plan=plot_spec, remove_pad=True)
            xhook.set_text_attention(text_atts)   # 패딩 제거용
            xhook.register_all()

        # ===== Vision encoder =====
        print("[Debug] ... Vision encoder forward")
        image_embeds = model.visual_encoder(image.to(device))
        image_atts   = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        # ===== Text encoder =====
        encoder_input_ids = text_ids.clone()
        encoder_input_ids[:, 0] = model.tokenizer.enc_token_id
        print("[Debug] ... Image-grounded text encoder forward")
        output_pos = model.text_encoder(
            encoder_input_ids.to(device),
            attention_mask=text_atts.to(device),
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # ===== Text decoder =====
        decoder_input_ids = text_ids.clone()
        decoder_input_ids[:, 0] = model.tokenizer.bos_token_id
        print("[Debug] ... Image-grounded text decoder forward")
        decoder_output = model.text_decoder(
            decoder_input_ids.to(device),
            attention_mask=text_atts.to(device),
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

    finally:
        # [HOOK] 해제
        if xhook is not None:
            xhook.unregister_all()

    # (원래 반환값 유지 필요하면 그대로 리턴)
    # 여기선 기존 코드가 반환을 안 하던 버전이라 생략


# def blip_forward_output(model, batch, device, **kwargs):
#     print("[Debug] pruners/accumulators.py -> blip_forward_output()함수 호출 : pruning을 위한 순전파 수행")

#     image, text_ids, text_atts = batch

#     # forward the images through the vision encoder
#     print("[Debug] pruners/accumulators.py -> blip_forward_output()함수 : Vision encoder forward")
#     image_embeds = model.visual_encoder(image) 
#     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) 

#     # forward the text through the image-grounded text encoder of the med
#     encoder_input_ids = text_ids.clone()
#     encoder_input_ids[:,0] = model.tokenizer.enc_token_id
#     print("[Debug] pruners/accumulators.py -> blip_forward_output()함수 : Image-grounded text encoder forward")
#     output_pos = model.text_encoder(
#         encoder_input_ids,
#         attention_mask = text_atts,
#         encoder_hidden_states = image_embeds,
#         encoder_attention_mask = image_atts,      
#         return_dict = True,
#     )      

#     # forward the text through the image-grounded text decoder of the med (to store activations of the causal self-attention)
#     decoder_input_ids = text_ids.clone()      
#     decoder_input_ids[:,0] = model.tokenizer.bos_token_id
#     print("[Debug] pruners/accumulators.py -> blip_forward_output()함수 : Image-grounded text decoder forward")
#     decoder_output = model.text_decoder(
#         decoder_input_ids, 
#         attention_mask = text_atts, 
#         encoder_hidden_states = image_embeds,
#         encoder_attention_mask = image_atts,                  
#         return_dict = True,   
#     )


def vit_forward_output(model, batch, device, **kwargs):
    images, _ = batch
    images = images.to(device)
    preds = model(images).logits
    return preds

# def clip_forward_output(model, batch, device, **kwargs):
#     print("[Debug] pruners/accumulators.py -> clip_forward_output()함수 호출 : pruning을 위한 순전파 수행")

#     image, text_ids, text_atts = batch
#     image = image.to(device)
#     text_ids = text_ids.to(device)
#     text_atts = text_atts.to(device)

#     out = model(
#         pixel_values=image,
#         input_ids=text_ids,
#         attention_mask=text_atts,
#         return_dict=True,
#     )
#     return out.logits_per_image.diag()

def clip_forward_output(model, batch, device, **kwargs):
    print("[Debug] pruners/accumulators.py -> clip_forward_output() 호출")

    image = text_ids = text_atts = None
    if isinstance(batch, dict):
        image     = batch.get("pixel_values") or batch.get("image") or batch.get("images")
        text_ids  = batch.get("input_ids")    or batch.get("text_ids")
        text_atts = batch.get("attention_mask") or batch.get("text_atts")
    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 3 and torch.is_tensor(batch[1]):
            image, text_ids, text_atts = batch[0], batch[1], batch[2]
        else:
            image = batch[0]

    if image is not None:     image = image.to(device)
    if text_ids is not None:  text_ids = text_ids.to(device)
    if text_atts is not None: text_atts = text_atts.to(device)

    if (text_ids is not None) and (text_atts is not None):
        out = model(pixel_values=image, input_ids=text_ids, attention_mask=text_atts, return_dict=True)
        return out.logits_per_image.diag()

    # 텍스트가 없으면 비전만 태워서 hook 기록 유도
    try:
        _ = model.get_image_features(pixel_values=image)
    except Exception:
        _ = model(pixel_values=image, return_dict=True)
    return None




# def clip_forward_output(model, batch, device, **kwargs):
#     print("[Debug] pruners/accumulators.py -> clip_forward_output() 함수 호출 : pruning을 위한 순전파 수행")

#     # --- WX 플로터 옵션 ---
#     use_wx_hooks = kwargs.get("use_wx_hooks", False)
#     run_tag       = kwargs.get("run_tag", "run")
#     base_plot_dir = kwargs.get("base_plot_dir", "./plots/smoothflow/wx")
#     plot_spec     = kwargs.get("plot_spec", {"vision":[0,6,11], "text":[0,6,11]})  # 원하는 레이어 지정 or 빈 리스트면 전부
#     downsample    = kwargs.get("downsample", None)
#     remove_pad    = kwargs.get("remove_pad", True)

#     image, text_ids, text_atts = batch
#     image     = image.to(device)
#     text_ids  = text_ids.to(device)
#     text_atts = text_atts.to(device) if text_atts is not None else None

#     plotter = None
#     try:
#         if use_wx_hooks:
#             plotter = ForwardWXPlotterCLIP(
#                 model,
#                 run_tag=run_tag,
#                 base_dir=str(base_plot_dir),
#                 plot_spec=plot_spec,
#                 downsample=downsample,
#                 remove_pad=remove_pad,
#             )
#             if text_atts is not None:
#                 plotter.set_text_attention(text_atts)  # 텍스트 패딩 제거용
#             plotter.register_all()

#         # HF CLIP: 한 번의 호출로 비전/텍스트 모두 forward
#         out = model(
#             pixel_values=image,
#             input_ids=text_ids,
#             attention_mask=text_atts,
#             return_dict=True,
#         )
#     finally:
#         if plotter is not None:
#             plotter.unregister_all()

#     return out.logits_per_image.diag()

# def clip_forward_output(model, batch, device, **kwargs):
#     print("[Debug] pruners/accumulators.py -> clip_forward_output() 함수 호출 : pruning을 위한 순전파 수행")


#     use_wx_hooks = kwargs.get("use_wx_hooks", False)        # ★ 추가: WX 바차트 플로터
#     plot_spec    = kwargs.get("plot_spec", {"vision": [], "text_enc": [], "text_dec": []})
#     run_tag      = kwargs.get("run_tag", "run")
#     base_plot_dir= kwargs.get("base_plot_dir", "./plots/smoothflow/wx_forward")
#     downsample   = kwargs.get("downsample", None)

#     # --- XW heatmap 옵션 ---
#     use_x_heatmap = kwargs.get("use_x_heatmap", False)
#     run_tag       = kwargs.get("run_tag", "run")
#     base_plot_dir = kwargs.get("base_plot_dir", "./plots/smoothflow/xw_heatmaps")

#     image, text_ids, text_atts = batch
#     image     = image.to(device)
#     text_ids  = text_ids.to(device)
#     text_atts = text_atts.to(device) if text_atts is not None else None

#     xhook = None
#     try:
#         if use_x_heatmap:
#             # BLIP/CLIP 공통 자동탐색 버전의 XHeatmapHook (이전에 준 자동 처음/중간/끝 선택 코드)
#             xhook = XHeatmapHook(
#                 model,
#                 run_tag=run_tag,
#                 base_dir=base_plot_dir,
#                 plan=None,              # 입력 없이 자동으로 처음/중간/끝 선택
#                 remove_pad=True
#             )
#             if text_atts is not None:
#                 xhook.set_text_attention(text_atts)  # 텍스트 패딩 제거용
#             xhook.register_all()

#         # HF CLIP: 한 번의 호출로 비전/텍스트 모두 forward → 훅 트리거됨
#         out = model(
#             pixel_values=image,
#             input_ids=text_ids,
#             attention_mask=text_atts,
#             return_dict=True,
#         )

#     finally:
#         if xhook is not None:
#             xhook.unregister_all()

#     # 원래 반환 유지
#     return out.logits_per_image.diag()
import torch
from torch import nn

@torch.no_grad()
def blip2_forward_output(model, batch, device, **kwargs):
    print("[Debug] pruners/accumulators.py -> blip2_forward_output()함수 호출")
    images, text_ids, text_atts = batch
    pixel_values = images.to(device)
    input_ids    = text_ids.to(device)
    attn_ids     = text_atts.to(device)

    # 1) Vision encoder
    vision_out   = model.vision_model(pixel_values=pixel_values, return_dict=True)
    image_embeds = vision_out.last_hidden_state                            # (B, N_vision, D_vision)
    image_atts   = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

    # 2) Q-Former with query tokens (정상 BLIP2 패턴)
    if hasattr(model, "query_tokens"):
        query_embeds = model.query_tokens.expand(pixel_values.size(0), -1, -1)   # (B, Nq, D_q)
    elif hasattr(model, "qformer") and hasattr(model.qformer, "query_tokens"):
        query_embeds = model.qformer.query_tokens.expand(pixel_values.size(0), -1, -1)
    else:
        raise AttributeError("BLIP2 model has no attribute 'query_tokens'")

    query_atts = torch.ones(query_embeds.size()[:-1], dtype=torch.long, device=device)

    # ⚠️ 중요: positional 인자 X, 이름 있는 인자로 전달
    q_out = model.qformer(
        query_embeds=query_embeds,
        attention_mask=query_atts,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    query_out = q_out.last_hidden_state                                      # (B, Nq, D_q)

    # 3) Language prefix 만들기: Q-Former out → language_projection → LM inputs_embeds prefix
    if not hasattr(model, "language_projection"):
        raise AttributeError("BLIP2 model missing `language_projection` for LM prefix projection.")
    prefix_embeds = model.language_projection(query_out)                      # (B, Nq, D_lm)

    # 텍스트 임베딩
    tok_emb = model.language_model.get_input_embeddings()                     # nn.Embedding
    text_embeds = tok_emb(input_ids)                                          # (B, T, D_lm)

    # prefix + text 연결
    inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)            # (B, Nq+T, D_lm)
    prefix_atts   = torch.ones(prefix_embeds.size()[:-1], dtype=attn_ids.dtype, device=device)
    lm_atts       = torch.cat([prefix_atts, attn_ids], dim=1)                 # (B, Nq+T)

    # 4) LM 전방 (OPT는 encoder_hidden_states를 받지 않음!)
    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=lm_atts,
        return_dict=True,
    )

    return {
        "vision_embeds": image_embeds,
        "query_embeds":  query_out,
        "lm_output":     outputs,
    }
@torch.no_grad()
def llava_forward_output(model, batch, device, **kwargs):
    """
    LLaVA-v1.5-7B용 forward.
    - LLaVA의 멀티모달 로직(이미지 토큰 위치 체크)을 피해,
      vision_tower와 language_model을 분리해서 직접 호출한다.
    """
    print("[Debug] pruners/accumulators.py -> llava_forward_output() 호출")

    def _first_present(mapping, keys):
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None

    images = text_ids = text_atts = None

    # ---- 배치 파싱 (dict / tuple 모두 지원) ----
    if isinstance(batch, dict):
        images    = _first_present(batch, ("pixel_values", "image", "images"))
        text_ids  = _first_present(batch, ("input_ids", "text_ids"))
        text_atts = _first_present(batch, ("attention_mask", "text_atts"))
    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 3 and torch.is_tensor(batch[1]):
            images, text_ids, text_atts = batch[0], batch[1], batch[2]
        else:
            images = batch[0]

    if images is not None:    images    = images.to(device)
    if text_ids is not None:  text_ids  = text_ids.to(device)
    if text_atts is not None: text_atts = text_atts.to(device)

    # ---- Fabric/DDP 래핑 고려: core 모델 꺼내기 ----
    # LlavaForConditionalGeneration면 .model이 LlavaModel
    core = getattr(model, "model", model)

    # ===== 1) Vision tower forward (이미지 있으면) =====
    if images is not None:
        vision_tower = getattr(core, "vision_tower", None)
        vision_out = None
        if vision_tower is not None:
            try:
                vision_out = vision_tower(
                    pixel_values=images,
                    output_hidden_states=True,
                    return_dict=True,
                )
            except TypeError:
                # 일부 구현은 pixel_values 대신 입력 이름이 다를 수 있어 방어 코드
                vision_out = vision_tower(images)
        else:
            # 혹시 구조가 다르면 그냥 core에 pixel_values만 넣어보기
            try:
                vision_out = core(pixel_values=images, use_cache=False, return_dict=True)
            except Exception:
                pass  # 어차피 hook만 필요하니까 실패해도 크게 상관 없음

        projector = getattr(core, "multi_modal_projector", None)
        if projector is not None and vision_out is not None:
            try:
                hidden_states = getattr(vision_out, "hidden_states", None)
                if hidden_states is not None:
                    layer = getattr(getattr(core, "config", None), "vision_feature_layer", -2)
                    image_features = hidden_states[layer]
                else:
                    image_features = getattr(vision_out, "last_hidden_state", None)

                strategy = getattr(getattr(core, "config", None), "vision_feature_select_strategy", "default")
                if image_features is not None and strategy == "default" and image_features.ndim == 3:
                    image_features = image_features[:, 1:]
                if image_features is not None:
                    _ = projector(image_features)
            except Exception as exc:
                print(f"[Warn] llava_forward_output: projector forward skipped: {type(exc).__name__}: {exc}")

    # ===== 2) Language model forward (텍스트 있으면) =====
    lm_out = None
    if (text_ids is not None) and (text_atts is not None):
        language_model = getattr(core, "language_model", None)
        if language_model is not None:
            lm_out = language_model(
                input_ids=text_ids,
                attention_mask=text_atts,
                use_cache=False,
                return_dict=True,
            )
        else:
            # fallback: 전체 model에 텍스트만 넣기 (pixel_values는 전달하지 않음 → image tokens 에러 회피)
            try:
                lm_out = model(
                    input_ids=text_ids,
                    attention_mask=text_atts,
                    use_cache=False,
                    return_dict=True,
                )
            except Exception:
                lm_out = None

    # 프루닝 관점에선 forward 동안 hook만 잘 돌면 되므로,
    # 여기서는 logits 또는 None 정도만 돌려주면 충분
    if lm_out is not None and hasattr(lm_out, "logits"):
        return lm_out.logits  # (B, T, vocab)
    return None


@torch.no_grad()
def qwen_vl_forward_output(model, batch, device, **kwargs):
    """
    Qwen2-VL calibration forward.
    The pruning dataloader provides CLIP-normalized image-caption batches.
    Run the visual tower explicitly so visual-layer activation hooks receive
    data, then run the text backbone for language-layer hooks.
    """
    print("[Debug] pruners/accumulators.py -> qwen_vl_forward_output() 호출")

    images = None
    text_ids = text_atts = None
    if isinstance(batch, dict):
        images = batch.get("pixel_values", None)
        if images is None:
            images = batch.get("image", None)
        if images is None:
            images = batch.get("images", None)
        text_ids = batch.get("input_ids", None)
        if text_ids is None:
            text_ids = batch.get("text_ids", None)
        text_atts = batch.get("attention_mask", None)
        if text_atts is None:
            text_atts = batch.get("text_atts", None)
    elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
        images = batch[0]
        text_ids = batch[1]
        text_atts = batch[2]

    if text_ids is None:
        raise TypeError("Qwen-VL calibration forward requires input_ids/text_ids.")

    text_ids = text_ids.to(device, non_blocking=True)
    if text_atts is not None:
        text_atts = text_atts.to(device, non_blocking=True)

    base = getattr(model, "_forward_module", model)
    base = getattr(base, "module", base)

    if images is not None:
        processor = getattr(base, "processor", None)
        if processor is None:
            from transformers import AutoProcessor
            import os
            hf_id = os.environ.get("QWEN_VL_ID", "Qwen/Qwen2-VL-2B-Instruct")
            processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
            setattr(base, "processor", processor)
            if hasattr(processor, "tokenizer"):
                setattr(base, "tokenizer", processor.tokenizer)

        from torchvision.transforms.functional import to_pil_image

        image_batch = images
        if torch.is_tensor(image_batch):
            if image_batch.ndim == 5:
                image_batch = image_batch[:, 0]
            mean = torch.tensor(
                (0.48145466, 0.4578275, 0.40821073),
                dtype=image_batch.dtype,
                device=image_batch.device,
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                (0.26862954, 0.26130258, 0.27577711),
                dtype=image_batch.dtype,
                device=image_batch.device,
            ).view(1, 3, 1, 1)
            image_batch = (image_batch.float() * std.float() + mean.float()).clamp(0, 1)
            pil_images = [to_pil_image(img.cpu()) for img in image_batch]
        else:
            pil_images = image_batch

        image_processor = getattr(processor, "image_processor", processor)
        proc_inputs = image_processor(images=pil_images, return_tensors="pt")
        pixel_values = proc_inputs.get("pixel_values", None)
        image_grid_thw = proc_inputs.get("image_grid_thw", None)
        visual = getattr(base, "visual", None)
        if visual is not None and pixel_values is not None and image_grid_thw is not None:
            pixel_values = pixel_values.to(device=device, dtype=next(visual.parameters()).dtype)
            image_grid_thw = image_grid_thw.to(device)
            _ = visual(pixel_values, grid_thw=image_grid_thw)

    text_model = getattr(base, "model", None)
    if text_model is None:
        raise TypeError("Qwen-VL model has no `.model` text backbone.")

    out = text_model(
        input_ids=text_ids,
        attention_mask=text_atts,
        use_cache=False,
        return_dict=True,
    )
    return getattr(out, "logits", None)


import torch
import math

@torch.inference_mode()
def flamingo_forward_output(model, batch, device, **kwargs):
    """
    OpenFlamingo forward for pruning-calibration (activation hooks 목적)
    - 이미지 shape 표준화: (B,C,H,W) / (B,T,C,H,W) / (B,T,F,C,H,W) 모두 지원
    - vision -> perceiver(resampler) -> vis_x 생성
    - (중요) conditioning(vis_x)을 try/finally로 "항상" 해제해서 OOM 누수 방지
    - (권장) text 길이 truncate 옵션 제공: max_text_len (default 64)
    - (권장) autocast fp16 옵션: amp_fp16 (default True)
    """

    print("[Debug] pruners/accumulators.py -> flamingo_forward_output() called")

    # -------------------------
    # 0) batch 파싱
    # -------------------------
    images = text_ids = text_atts = None

    if isinstance(batch, dict):
        images    = batch.get("pixel_values") or batch.get("images") or batch.get("image")
        text_ids  = batch.get("input_ids") or batch.get("text_ids")
        text_atts = batch.get("attention_mask") or batch.get("text_atts")
    elif isinstance(batch, (list, tuple)):
        # (images, input_ids, attention_mask) 형태를 기대
        if len(batch) >= 3 and torch.is_tensor(batch[1]):
            images, text_ids, text_atts = batch[0], batch[1], batch[2]
        else:
            images = batch[0]

    if images is not None:
        images = images.to(device, non_blocking=True)
        print("[Debug] images shape:", tuple(images.shape))

    if text_ids is not None:
        text_ids = text_ids.to(device, non_blocking=True)
    if text_atts is not None:
        text_atts = text_atts.to(device, non_blocking=True)

    # -------------------------
    # (옵션) text truncate (OOM 완화에 매우 효과적)
    # -------------------------
    max_text_len = int(kwargs.get("max_text_len", 64))
    if text_ids is not None and text_ids.ndim == 2 and text_ids.size(1) > max_text_len:
        text_ids = text_ids[:, :max_text_len].contiguous()
        if text_atts is not None:
            text_atts = text_atts[:, :max_text_len].contiguous()

    # -------------------------
    # 1) image shape -> (B, T, F, C, H, W)
    #    너 배치: (B, 1, 3, 224, 224) => (B, 1, 1, 3, 224, 224)
    # -------------------------
    if images is not None:
        if images.ndim == 4:         # (B,C,H,W)
            images = images.unsqueeze(1).unsqueeze(2)
        elif images.ndim == 5:       # (B,T,C,H,W)
            images = images.unsqueeze(2)
        elif images.ndim == 6:       # (B,T,F,C,H,W)
            pass
        else:
            raise ValueError(f"Unexpected images.ndim={images.ndim}, shape={tuple(images.shape)}")

    # -------------------------
    # 2) vision encoder -> vis tokens
    # -------------------------
    ve = _get_vision_encoder(model)
    if ve is None:
        raise ValueError("Flamingo: vision_encoder not found")

    B, T, F, C, H, W = images.shape
    images_flat = images.reshape(B * T * F, C, H, W)  # (BTF,C,H,W)

    # -------------------------
    # 3) perceiver(resampler) -> vis_x
    # -------------------------
    perceiver = _get_perceiver(model)
    if perceiver is None:
        raise ValueError("Flamingo: perceiver/resampler not found")

    # -------------------------
    # 4) LM(FlamingoLMMixin) 찾기
    # -------------------------
    lm = _find_flamingo_lm(model)
    if lm is None:
        raise ValueError("Flamingo: no FlamingoLM found for conditioning")

    # ✅ 방어: FlamingoLayer(레이어)는 input_ids forward가 아니다
    import inspect
    try:
        if "input_ids" not in inspect.signature(lm.forward).parameters:
            raise TypeError(f"Found wrong lm object: {lm.__class__.__name__} (no input_ids in forward)")
    except Exception:
        # signature 추출 실패 시에도, 최소한 forward 호출 형태로 방어하고 싶으면 여기서 더 체크 가능
        pass
    # -------------------------
    # 5) conditioning을 “항상” 해제하기 위한 준비
    #    - open_flamingo 표준: lm.clear_conditioned_layers()
    #    - 너 로그처럼 "24 modules"에 직접 condition_vis_x 하는 경우도 방어
    # -------------------------
    conditioned_modules = []
    used = None
    amp_fp16 = bool(kwargs.get("amp_fp16", True))

    # autocast 컨텍스트
    if device.type == "cuda" and amp_fp16:
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        # CPU거나 fp32로 돌릴 때
        class _NullCtx:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        autocast_ctx = _NullCtx()

    try:
        # ---------- vision forward ----------
        # open_clip VisionTransformer는 return_all_tokens를 안 받는 경우가 많아서 시도하지 않음
        with autocast_ctx:
            out = ve(images_flat)

        vis = _pick_vision_tokens(out)

        # vis shape 정규화: (BTF,V,D)
        if vis.ndim == 2:
            vis = vis.unsqueeze(1)  # (BTF, 1, D)
        elif vis.ndim != 3:
            raise ValueError(f"Unexpected vision tokens shape: {tuple(vis.shape)}")

        V, D = vis.shape[1], vis.shape[2]
        vis = vis.reshape(B, T, F, V, D)  # (B,T,F,V,D)

        with autocast_ctx:
            vis_x = perceiver(vis)  # open_flamingo perceiver는 (B,T,F,V,D) 입력 기대

        # ---------- condition ----------
        # (A) 표준 LM 메서드로 conditioning
        if hasattr(lm, "condition_vis_x"):
            lm.condition_vis_x(vis_x)
            used = "lm.condition_vis_x"
        elif hasattr(lm, "condition_media"):
            lm.condition_media(vis_x)
            used = "lm.condition_media"
        else:
            # (B) 너처럼 레이어(FlamingoLayer)들에 직접 condition_vis_x를 심는 구조면,
            # model.modules()에서 condition_vis_x 가진 모듈들을 찾아 호출
            for m in model.modules():
                if hasattr(m, "condition_vis_x"):
                    m.condition_vis_x(vis_x)
                    conditioned_modules.append(m)
            if len(conditioned_modules) == 0:
                raise ValueError("Flamingo: LM/modules have no condition_* method")
            used = f"modules.condition_vis_x ({len(conditioned_modules)})"

        print(f"[Debug] conditioned vis_x via {used}")

        # ---------- LM forward ----------
        lm_out = None
        if text_ids is not None and text_atts is not None:
            with autocast_ctx:
                lm_out = lm(
                    input_ids=text_ids,
                    attention_mask=text_atts,
                    use_cache=False,     # KV cache로 누수/피크 늘어나는 것 방지
                    return_dict=True,
                )

        return getattr(lm_out, "logits", None) if lm_out is not None else None

    finally:
        # ✅ 어떤 예외가 나도 conditioning 해제 (OOM 누수 방지의 핵심)
        # 1) open_flamingo 표준 클리어
        if hasattr(lm, "clear_conditioned_layers"):
            try:
                lm.clear_conditioned_layers()
            except Exception:
                pass

        # 2) 레이어에 직접 condition한 경우: 내부 참조를 끊어줌(구현차 방어)
        if conditioned_modules:
            for m in conditioned_modules:
                # 가능한 속성 후보를 폭넓게 None 처리
                for attr in ("vis_x", "_vis_x", "conditioned_vis_x", "_conditioned_vis_x", "media", "_media"):
                    if hasattr(m, attr):
                        try:
                            setattr(m, attr, None)
                        except Exception:
                            pass

        # 3) 큰 텐서 참조 끊기
        try:
            del out
        except Exception:
            pass
        try:
            del vis
        except Exception:
            pass
        try:
            del vis_x
        except Exception:
            pass

        # 4) (선택) 캐시 비우기: 디버깅 단계에서만 켜는 걸 권장
        # if device.type == "cuda" and kwargs.get("empty_cache", False):
        #     torch.cuda.empty_cache()



def forward_output(model_name, model, batch, device, **kwargs):
    if model_name == 'xvlm':
        return xvlm_forward_output(model, batch, device, **kwargs)
    elif model_name == 'blip':
        return blip_forward_output(model, batch, device, **kwargs)
    elif model_name in ('vit-b', 'dino'):
        return vit_forward_output(model, batch, device, **kwargs)
    elif model_name == 'clip':
        return clip_forward_output(model, batch, device, **kwargs)
    elif model_name == 'clipG':
        return clip_forward_output(model, batch, device, **kwargs)
    elif model_name == 'blip2':
        return blip2_forward_output(model, batch, device, **kwargs)
    elif model_name == 'llava':
        return llava_forward_output(model, batch, device, **kwargs)
    elif model_name == 'qwen_vl':
        return qwen_vl_forward_output(model, batch, device, **kwargs)
    elif model_name == 'flamingo':
        return flamingo_forward_output(model, batch, device, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported.")


def xvlm_region_forward(model, region_batch, device, **kwargs):
    calc_image_bbox_loss = kwargs.get('calc_image_bbox_loss', False)
    custom_temp = kwargs.get('custom_temp', None)
    return_bbox_loss = kwargs.get('return_bbox_loss', False)

    # code taken from XVLM-pretraining
    images = region_batch[0].to(device, non_blocking=True)
    region_batch = [t.to(device) if t is not None else None for t in region_batch[1:]]

    # unroll region data
    idx_to_group_img, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
        image_atts, target_bbox, is_image = region_batch

    # no idea what this does, should check pretraining forward
    if calc_image_bbox_loss:
        is_image = None

    # forward pass with region data
    loss_itc, loss_itm, loss_mlm, loss_bbox, loss_giou = \
        model(images, text_ids, text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
            image_atts=image_atts, idx_to_group_img=idx_to_group_img, target_bbox=target_bbox, is_image=is_image, ret_bbox_loss=True, custom_temp=custom_temp)

    # compute gradients
    if return_bbox_loss:
        loss = loss_itc + loss_itm + loss_mlm + loss_bbox + loss_giou
    else:
        loss = loss_itc + loss_itm + loss_mlm
    return loss




def blip_region_forward(model, region_batch, device, **kwargs):
    return blip_general_forward(model, region_batch, device, **kwargs)


def region_forward(model_name, model, batch, device, **kwargs):
    if model_name == 'xvlm':
        return xvlm_region_forward(model, batch, device, **kwargs)
    elif model_name == 'blip':
        return blip_region_forward(model, batch, device, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported.")


def blip_region_forward_output(model, batch, device, **kwargs):
    return blip_forward_output(model, batch, device, **kwargs)


def xvlm_region_forward_output(model, region_batch, device, calc_image_bbox_loss=False):
    # code taken from XVLM-pretraining
    images = region_batch[0].to(device, non_blocking=True)
    region_batch = [t.to(device) if t is not None else None for t in region_batch[1:]]

    # unroll region data
    idx_to_group_img, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
        image_atts, target_bbox, is_image = region_batch

    # no idea what this does, should check pretraining forward
    if calc_image_bbox_loss:
        is_image = None

    # forward pass with region data
    image_embeds, image_atts, image_embeds_fullatts = model.get_vision_embeds(
        images, image_atts=image_atts, idx_to_group_img=idx_to_group_img
    )
    text_embeds = model.get_text_embeds(text_ids, text_atts)
    output_cls_token = model.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds, text_atts=text_atts)
    return output_cls_token
    
def clip_region_forward_output(model, batch, device, **kwargs):
    return clip_forward_output(model, batch, device, **kwargs)

def blip2_region_forward_output(model, batch, device, **kwargs):
    return blip2_forward_output(model, batch, device, **kwargs)

def region_forward_output(model_name, model, batch, device, **kwargs):
    if model_name == 'xvlm':
        return xvlm_region_forward_output(model, batch, device, **kwargs)
    elif model_name == 'blip':
        return blip_region_forward_output(model, batch, device, **kwargs)
    elif model_name == 'clip':
        return clip_region_forward_output(model, batch, device, **kwargs)
    elif model_name == 'clipG':
        return clip_region_forward_output(model, batch, device, **kwargs)
    elif model_name == 'blip2':
        return blip2_region_forward_output(model, batch, device, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported.")


# ======== WX Hook Plotter (붙여넣기) ========
class ForwardWXPlotterCLIP:
    """
    HF CLIP 전용:
      - vision: model.vision_model.encoder.layers[i]
      - text  : model.text_model.encoder.layers[i]
    각 Linear에서 X/W/WX를 저장. (옵션) remove_pad=True면 text에서 attention_mask로 패딩 제거.
    """
    def __init__(self, model, *, run_tag, base_dir, plot_spec=None, downsample=None, remove_pad=True):
        self.model = model
        self.run_tag = run_tag
        self.base_dir = base_dir
        self.plot_spec = plot_spec or {"vision": [], "text": []}  # CLIP은 text_enc만 존재
        self.downsample = downsample
        self.remove_pad = remove_pad
        self.text_att = None          # (B, L) attention_mask
        self.handles, self._pre_inputs = [], {}
        self._order = 0

    def set_text_attention(self, att):  # (B, L)
        self.text_att = att

    def _want(self, key, li):
        arr = self.plot_spec.get(key, [])
        return (not arr) or (li in set(int(x) for x in arr))

    def register_all(self):
        # ---- vision ----
        v_layers = getattr(getattr(self.model.vision_model, "encoder", None), "layers", [])
        for li, layer in enumerate(v_layers):
            if self._want("vision", li):
                self._register_linear_under(layer, modality="vision", layer_idx=li, root=f"clip.vision.layers.{li}")
        # ---- text ----
        t_layers = getattr(getattr(self.model.text_model, "encoder", None), "layers", [])
        for li, layer in enumerate(t_layers):
            if self._want("text", li):
                self._register_linear_under(layer, modality="text", layer_idx=li, root=f"clip.text.layers.{li}")

    def unregister_all(self):
        for h in self.handles:
            try: h.remove()
            except: pass
        self.handles.clear(); self._pre_inputs.clear()

    def _register_linear_under(self, module, *, modality, layer_idx, root):
        for name, sub in module.named_modules():
            if isinstance(sub, torch.nn.Linear):
                full = f"{root}.{name}" if name else root
                self._attach(sub, modality, layer_idx, full)

    def _attach(self, lin: torch.nn.Linear, modality: str, layer_idx: int, mod_path: str):
        pid = id(lin)
        def pre_hook(mod, inp):
            if inp and torch.is_tensor(inp[0]):
                self._pre_inputs[pid] = inp[0].detach()

        def fwd_hook(mod, inp, out):
            try:
                x = self._pre_inputs.get(pid, None)
                if x is None: return

                # 패딩 제거 (text만)
                if self.remove_pad and modality == "text" and self.text_att is not None:
                    att = self.text_att
                    if x.ndim == 3 and att is not None and att.shape[:2] == x.shape[:2]:
                        mask = att.to(x.device).unsqueeze(-1).to(x.dtype)  # (B,L,1)
                        x = x * mask
                        flat = x.reshape(-1, x.size(-1))
                        keep = flat.abs().sum(dim=1) > 0
                        x = flat[keep]                                      # (tokens, D)

                # 배치 0만(정책 유지) + 2D 정리
                if x.ndim >= 3:
                    Xb0 = x[0]
                else:
                    Xb0 = x[:1, :]
                X2 = Xb0.reshape(-1, Xb0.shape[-1]).to(torch.float32).cpu()  # (L, Din)

                W = lin.weight
                if W is None or W.ndim != 2: return
                W2 = W.detach().to(torch.float32).cpu()                      # (Dout, Din)

                Y  = X2 @ W2.T                                               # (L, Dout)

                self._order += 1
                pref   = f"{self._order:05d}_{self.run_tag}_L{layer_idx:02d}"
                outdir = os.path.join(self.base_dir, f"wx_forward/{modality}/layer{layer_idx:02d}/{_sanitize(mod_path)}")

                _bar3d_matrix(W2, title=f"{modality} | {mod_path} | W",
                              outdir=outdir, stem=pref+"_W",  xlabel="D_in",  ylabel="D_out",
                              downsample=self.downsample)
                _bar3d_matrix(X2, title=f"{modality} | {mod_path} | X(pad-removed)" if (self.remove_pad and modality=="text") else f"{modality} | {mod_path} | X(b=0)",
                              outdir=outdir, stem=pref+"_X",  xlabel="D_in",  ylabel="L",
                              downsample=self.downsample)
                _bar3d_matrix(Y,  title=f"{modality} | {mod_path} | WX",
                              outdir=outdir, stem=pref+"_WX", xlabel="D_out", ylabel="L",
                              downsample=self.downsample)
            except Exception as e:
                print(f"[WX-HOOK][{modality}][{mod_path}] {e}")

        self.handles.append(lin.register_forward_pre_hook(pre_hook))
        self.handles.append(lin.register_forward_hook(fwd_hook))


import os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import torch

def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

def _save(fig, outdir: str, stem: str):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{stem}.png"), dpi=500)
    plt.close(fig)

def _styled_bar3d(ax, X, Y, Z, *, cmap="coolwarm", alpha=0.9, width=0.6):
    vals = np.abs(Z).ravel()
    cmap_fn = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    vmin = float(vals.min()); vmax = float(max(vals.max(), 1e-8))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    facecol = cmap_fn(norm(vals))
    Xr = X.ravel() + 0.5 - width/2
    Yr = Y.ravel() + 0.5 - width/2
    Z0 = np.zeros_like(vals)
    ax.bar3d(Xr, Yr, Z0, width, width, vals, color=facecol, shade=False,
             edgecolor="none", linewidth=0.0, alpha=alpha)

def _bar3d_matrix(M: torch.Tensor, *, title, outdir, stem, xlabel, ylabel, downsample=None):
    Z = M.detach().float().cpu().numpy()
    if downsample:
        rs, cs = downsample
        Z = Z[::max(rs,1), ::max(cs,1)]
    R, C = Z.shape
    Xg, Yg = np.meshgrid(np.arange(C), np.arange(R))
    fig = plt.figure(figsize=(18,10))
    ax  = fig.add_subplot(111, projection="3d")
    _styled_bar3d(ax, Xg, Yg, Z.T)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, zlabel="|val|")
    _save(fig, outdir, stem)

class ForwardWXPlotter:
    """
    - forward 순서대로 nn.Linear에서 X/W/WX 저장
    - vision/text encoder/text decoder 각각 레이어 필터링 (plot_spec)
    """
    def __init__(self, model, *, run_tag, base_dir, plot_spec, downsample=None):
        self.model = model
        self.run_tag = run_tag
        self.base_dir = base_dir
        self.plot_spec = plot_spec or {"vision": [], "text_enc": [], "text_dec": []}
        self.downsample = downsample
        self.handles = []
        self._pre_inputs = {}
        self._order = 0

    def register_all(self):
        # ===== 1) CLIP 구조 먼저 감지 =====
        is_clip = hasattr(self.model, "vision_model") and hasattr(self.model, "text_model")
        if is_clip:
            # Vision (CLIP)
            v_enc = getattr(getattr(self.model.vision_model, "encoder", None), "layers", [])
            for li, blk in enumerate(v_enc):
                if self._want("vision", li):
                    self._register_linear_under(blk, modality="vision", layer_idx=li,
                                                root=f"clip.vision.layers.{li}")

            # Text (CLIP)
            t_enc = getattr(getattr(self.model.text_model, "encoder", None), "layers", [])
            for li, blk in enumerate(t_enc):
                if self._want("text_enc", li):
                    self._register_linear_under(blk, modality="text_enc", layer_idx=li,
                                                root=f"clip.text.layers.{li}")

            # CLIP에는 decoder 없음
            return

        # ===== 2) BLIP fallback =====
        vit_blocks = getattr(self.model.visual_encoder, "blocks", [])
        for li, blk in enumerate(vit_blocks):
            if self._want("vision", li):
                self._register_linear_under(blk, modality="vision", layer_idx=li,
                                            root=f"vision.blocks.{li}")

        enc = getattr(self.model.text_encoder, "encoder", None)
        layers = getattr(enc, "layer", []) if enc is not None else []
        for li, layer in enumerate(layers):
            if self._want("text_enc", li):
                self._register_linear_under(layer, modality="text_enc", layer_idx=li,
                                            root=f"text_encoder.encoder.layer.{li}")

        dec_bert = getattr(self.model.text_decoder, "bert", None)
        dec_enc  = getattr(dec_bert, "encoder", None) if dec_bert is not None else None
        d_layers = getattr(dec_enc, "layer", []) if dec_enc is not None else []
        for li, layer in enumerate(d_layers):
            if self._want("text_dec", li):
                self._register_linear_under(layer, modality="text_dec", layer_idx=li,
                                            root=f"text_decoder.bert.encoder.layer.{li}")

    def unregister_all(self):
        for h in self.handles:
            try: h.remove()
            except: pass
        self.handles.clear()
        self._pre_inputs.clear()

    def _want(self, key, li):
        arr = self.plot_spec.get(key, [])
        return (not arr) or (li in set(int(x) for x in arr))

    def _register_linear_under(self, module, *, modality, layer_idx, root):
        for name, sub in module.named_modules():
            if isinstance(sub, torch.nn.Linear):
                full = f"{root}.{name}" if name else root
                self._attach(sub, modality, layer_idx, full)

    def _attach(self, lin: torch.nn.Linear, modality: str, layer_idx: int, mod_path: str):
        pid = id(lin)

        def pre_hook(mod, inp):
            if not inp: return
            x = inp[0]
            if torch.is_tensor(x):
                self._pre_inputs[pid] = x.detach()

        def fwd_hook(mod, inp, out):
            try:
                x = self._pre_inputs.get(pid, None)
                if x is None: return
                # 배치 0만
                if x.ndim >= 3:
                    Xb0 = x[0]         # (L, Din) or (tokens, Din)
                elif x.ndim == 2:
                    Xb0 = x[:1, :]     # (1, Din)
                else:
                    return

                W = mod.weight
                if W is None or W.ndim != 2: return

                X2 = Xb0.reshape(-1, Xb0.shape[-1]).to(torch.float32).cpu()  # (L, Din)
                W2 = W.detach().to(torch.float32).cpu()                      # (Dout, Din)
                Y  = X2 @ W2.T                                               # (L, Dout)

                # 순서 프리픽스
                self._order += 1
                pref = f"{self._order:05d}_{self.run_tag}_L{layer_idx:02d}"
                outdir = os.path.join(self.base_dir, f"wx_forward/{modality}/layer{layer_idx:02d}/{_sanitize(mod_path)}")

                _bar3d_matrix(W2, title=f"{modality} | {mod_path} | W",
                                outdir=outdir, stem=pref+"_W", xlabel="D_in", ylabel="D_out",
                                downsample=self.downsample)
                _bar3d_matrix(X2, title=f"{modality} | {mod_path} | X(b=0)",
                                outdir=outdir, stem=pref+"_X", xlabel="D_in", ylabel="L",
                                downsample=self.downsample)
                _bar3d_matrix(Y,  title=f"{modality} | {mod_path} | WX",
                                outdir=outdir, stem=pref+"_WX", xlabel="D_out", ylabel="L",
                                downsample=self.downsample)
            except Exception as e:
                print(f"[WX-HOOK][{modality}][{mod_path}] {e}")

        self.handles.append(lin.register_forward_pre_hook(pre_hook))
        self.handles.append(lin.register_forward_hook(fwd_hook))


# ======== X Heatmap Hook (값/Gram 모두 저장, off-diagonal sum 캡션) ========
import os, re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def _xhm_sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

def _xhm_prep_tokens(X: torch.Tensor) -> torch.Tensor:
    # (B,T,D) or (T,D) -> (tokens, D), float cpu
    x = X.detach().float().cpu()
    if x.ndim == 3:
        x = x[0]
    return x

def _xhm_save_gram_heatmap(X: torch.Tensor, title: str, out_path: str, center=True, l2norm=True):
    x = _xhm_prep_tokens(X)
    eps = 1e-6
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    if l2norm:
        x = x / (x.norm(dim=1, keepdim=True) + eps)
    G = x @ x.t()                        # cosine Gram
    offsum = (G.sum() - G.diag().sum()).item()
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(G, vmin=-1.0, vmax=1.0, aspect='equal', interpolation='nearest')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('token idx'); ax.set_ylabel('token idx')
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.01, f"off-diagonal sum = {offsum:.6f}", ha='center', va='bottom', fontsize=10)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return offsum

def _xhm_save_dim_gram_heatmap(Y: torch.Tensor, title: str, out_path: str, center=True, l2norm=True):
    # Y: (tokens, D_out) 또는 (B,T,D_out) → (tokens, D_out)로 정리
    y = _xhm_prep_tokens(Y)            # (tokens, D_out)
    eps = 1e-6
    if center:
        y = y - y.mean(dim=0, keepdim=True)           # 각 "열(특징)"별 평균 제거
    if l2norm:
        y = y / (y.norm(dim=0, keepdim=True) + eps)   # 각 "열(특징)"별 L2 정규화
    G = y.t() @ y                                     # (D_out, D_out) = feature×feature cosine Gram
    offsum = (G.sum() - G.diag().sum()).item()

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(G, vmin=-1.0, vmax=1.0, aspect='equal', interpolation='nearest')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('feature idx'); ax.set_ylabel('feature idx')
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.01, f"off-diagonal sum = {offsum:.6f}", ha='center', va='bottom', fontsize=10)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return offsum

class XHeatmapHook:
    """
    - nn.Linear의 forward_pre_hook에서 입력 X를 받아
      1) 값 히트맵 (tokens x hidden)
      2) Gram(코사인) 히트맵 (tokens x tokens)
      를 모두 저장.
    - 타이틀: "{modality} | blocks.{idx}.{sub} | {shape}"
    - 하단 캡션: off-diagonal sum 값(값 히트맵에도 같은 수치 표기)
    - 대상: vision/text_enc/text_dec의 첫/중/마지막 블록 내 Linear들 (attn.qkv/proj, mlp.fc1/fc2 등)
    """
    def __init__(self, model, *, run_tag, base_dir, plan, remove_pad=True):
        self.model = model
        self.run_tag = run_tag or "run"
        self.base_dir = str(base_dir)
        self.plan = plan or {"vision": [], "text_enc": [], "text_dec": []}
        self.remove_pad = remove_pad
        self.handles = []
        self.text_att = None  # (B,L)
        self._target_subs = (
            "attn.q", "attn.k", "attn.v",   # ← 추가
            "attn.proj",
            "mlp.fc1", "mlp.fc2",
            "attn.qkv"                      # ← 호환성 유지용(합쳐진 qkv도 여전히 잡음)
        )

    def set_text_attention(self, att):  # (B,L)
        self.text_att = att

    # ---- 모듈 트리 순회: BLIP 구조 기준 ----
    def register_all(self):
        # 1) 모달리티별 레이어 리스트 수집 (BLIP/CLIP 모두 지원)
        mod2layers = self._gather_modal_layers()

        # 2) 각 모달리티에서 처음/중간/끝 인덱스만 선택
        for modality, layers in mod2layers.items():
            if not layers:
                continue
            idxs = self._pick_three_indices(len(layers))
            for i in idxs:
                layer, root = layers[i]  # (module, root_str)
                self._register_under_layer(layer, modality=modality, layer_idx=i, root=root)

    def unregister_all(self):
        for h in self.handles:
            try: h.remove()
            except: pass
        self.handles.clear()

     # --------- 유틸: 처음/중간/끝 인덱스 뽑기 ---------
    def _pick_three_indices(self, n: int):
        if n <= 1:
            return [0]
        if n == 2:
            return [0, 1]
        mid = n // 2
        return sorted({0, mid, n - 1})

    # --------- BLIP/CLIP 레이어 자동 수집 ---------
    def _gather_modal_layers(self):
        """
        return:
          {
            "vision": [(module, "vision.blocks.{i}"), ...],
            "text_enc": [(module, "text_enc.blocks.{i}"), ...],
            "text_dec": [(module, "text_dec.blocks.{i}") or empty]
          }
        BLIP이면 vision/text_enc/text_dec,
        CLIP이면 vision/text_enc 만 채움.
        """
        out = {"vision": [], "text_enc": [], "text_dec": []}

        # ---- CLIP (HF) 탐지 ----
        if hasattr(self.model, "vision_model") and hasattr(self.model, "text_model"):
            v_layers = getattr(getattr(self.model.vision_model, "encoder", None), "layers", [])
            t_layers = getattr(getattr(self.model.text_model, "encoder", None), "layers", [])

            for i, m in enumerate(v_layers):
                out["vision"].append((m, f"clip.vision.layers.{i}"))
            for i, m in enumerate(t_layers):
                out["text_enc"].append((m, f"clip.text.layers.{i}"))

            # decoder 없음
            return out

        # ---- BLIP 탐지 ----
        # Vision
        vit_blocks = getattr(self.model, "visual_encoder", None)
        vit_blocks = getattr(vit_blocks, "blocks", []) if vit_blocks is not None else []
        for i, m in enumerate(vit_blocks):
            out["vision"].append((m, f"vision.blocks.{i}"))

        # Text encoder (BERT encoder)
        enc = getattr(self.model, "text_encoder", None)
        bert_enc = getattr(enc, "encoder", None) if enc is not None else None
        te_layers = getattr(bert_enc, "layer", []) if bert_enc is not None else []
        for i, m in enumerate(te_layers):
            out["text_enc"].append((m, f"text_enc.blocks.{i}"))

        # Text decoder (optional)
        dec_bert = getattr(self.model, "text_decoder", None)
        dec_bert = getattr(dec_bert, "bert", None) if dec_bert is not None else None
        dec_enc  = getattr(dec_bert, "encoder", None) if dec_bert is not None else None
        td_layers = getattr(dec_enc, "layer", []) if dec_enc is not None else []
        for i, m in enumerate(td_layers):
            out["text_dec"].append((m, f"text_dec.blocks.{i}"))

        return out

    # --------- 서브모듈 매칭(이전과 동일) ---------
    def _match_sub(self, mod_path: str) -> str:
        p = mod_path.lower()

        # Attention 계열
        if ("attn" in p) or ("attention" in p):
            # q/k/v 분리 네이밍(q_proj, k_proj, v_proj 등)도 한데 묶어 qkv로 표기
            if any(s in p for s in ["qkv", "q_proj", "k_proj", "v_proj", "query", "key", "value"]):
                return "attn.qkv"
            if any(s in p for s in ["proj", "out_proj", "output.dense"]):
                return "attn.proj"

        # MLP 계열
        if ("mlp" in p) or ("intermediate" in p) or ("output.dense" in p):
            if "fc1" in p or "intermediate" in p:
                return "mlp.fc1"
            if "fc2" in p or "output.dense" in p:
                return "mlp.fc2"

        return ""

    def _register_under_layer(self, module, *, modality, layer_idx, root):
        for name, sub in module.named_modules():
            if isinstance(sub, torch.nn.Linear):
                full = f"{root}.{name}" if name else root
                subname = self._match_sub(full)
                if subname in self._target_subs:
                    self._attach(sub, modality, layer_idx, full, subname)

    def _outpath(self, modality, layer_idx, subname, mod_path, suffix):
        subdir = os.path.join(self.base_dir, modality, f"layer{layer_idx:02d}")
        fname = f"{self.run_tag}_{_xhm_sanitize(mod_path)}_{suffix}.png"
        return os.path.join(subdir, fname)
        
    

    def _attach(self, lin: torch.nn.Linear, modality: str, layer_idx: int, mod_path: str, subname: str):
        @torch.no_grad()
        def pre_hook(mod, inp):
            if not inp: 
                return
            X = inp[0]                     # (B,T,D_in) or (T,D_in)
            Xv = X

            # (옵션) 텍스트 패딩 제거 그대로 유지
            if self.remove_pad and modality.startswith("text") and self.text_att is not None:
                att = self.text_att.to(X.device)
                if X.dim() == 3 and att.shape[:2] == X.shape[:2]:
                    mask = att.unsqueeze(-1).to(X.dtype)   # (B,L,1)
                    Xv = X * mask
                    flat = Xv.reshape(-1, Xv.size(-1))
                    keep = flat.abs().sum(dim=1) > 0
                    Xv = flat[keep]                        # (tokens, D_in)

            # 배치 차원 정리: (tokens, D_in)
            if Xv.ndim == 3:
                Xmat = Xv[0]               # 첫 배치만 쓰는 현재 정책 유지
            else:
                Xmat = Xv
            Xmat = Xmat.to(torch.float32)

            # 선형층 가중치
            W = mod.weight.detach().to(Xmat.dtype).to(Xmat.device)   # (D_out, D_in)

            # ★ 핵심: Y = XW^T 계산 ★
            Y = Xmat @ W.t()               # (tokens, D_out)

            title = f"{modality} | blocks.{layer_idx}.{subname} | XW shape {tuple(Y.shape)}"

            # 이제 Gram은 Y 기반으로 계산/저장
            out_gram = self._outpath(modality, layer_idx, subname, mod_path, "xw_gram")
            _xhm_save_gram_heatmap(Y, title, out_gram, center=True, l2norm=True)

            out_feat = self._outpath(modality, layer_idx, subname, mod_path, "xw_feat_gram")
            _xhm_save_dim_gram_heatmap(Y, title + " | feature-gram", out_feat, center=True, l2norm=True)

        self.handles.append(lin.register_forward_pre_hook(pre_hook))
