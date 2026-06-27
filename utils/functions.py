import torch


def percentile(scores, sparsity):
    k = 1 + round(sparsity * scores.numel())
    threshold, _ = torch.kthvalue(scores.view(-1), k)
    return threshold


class GetTopSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, threshold=None):
        return torch.where(scores <= threshold, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None
    
    
class GetBottomSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        # this inverts the behaviour of GetTopSubnet, i.e. 
        # if GetTopSubnet returns the mask of the top (1-sparsity) scores, GetBottomSubnet 
        # applies a logical not to the mask of the bottom (1-sparsity) scores
        kth_val = percentile(scores, sparsity)
        return torch.where(scores > kth_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return -g, None, None, None


def xvlm_detect_modality_fn(param_name, return_layer_idx=False):
    if param_name.startswith('vision_encoder'):
        modality = 'vision'
        layer = int(param_name.split('.')[3])
    elif param_name.startswith('text_encoder'):
        layer = int(param_name.split('.')[4])
        if layer > 5:
            modality = 'fusion'
        else:
            modality = 'text'
    
    if return_layer_idx:
        return modality, layer
    return modality



def blip_detect_modality_fn(param_name, return_layer_idx=False):
    # 파라미터 이름을 분석해 해당 파라미터가 vision용인지 text용인지 판별
    if param_name.startswith('visual_encoder'):
        modality = 'vision'
        layer_idx = int(param_name.split('.')[2])
    elif param_name.startswith('text'):
        modality = 'text'
        if param_name.startswith('text_encoder'):
            layer_idx = int(param_name.split('.')[3])
        elif param_name.startswith('text_decoder'):
            layer_idx = int(param_name.split('.')[4])
    
    if return_layer_idx:
        return modality, layer_idx
    return modality


def vit_detect_modality_fn(param_name, return_layer_idx=False):
    modality = 'vision'
    if return_layer_idx:
        if "patch_embeddings.projection" not in param_name:
            layer_idx = int(param_name.split('.')[3])
        else:
            layer_idx = -1
        return modality, layer_idx
    return modality
        
def clip_detect_modality_fn(param_name, return_layer_idx=False):
    if param_name.startswith('vision_model'):
        modality = 'vision'
        key = 'vision_model.encoder.layers.'
        if key in param_name:
            try:
                layer_idx = int(param_name.split(key, 1)[1].split('.', 1)[0])
            except Exception:
                layer_idx = -1

    elif param_name.startswith('text_model'):
        modality = 'text'
        key = 'text_model.encoder.layers.'
        if key in param_name:
            try:
                layer_idx = int(param_name.split(key, 1)[1].split('.', 1)[0])
            except Exception:
                layer_idx = -1
    elif param_name.startswith('encoder.layers.'):
        try:
            layer_idx = int(param_name.split('encoder.layers.', 1)[1].split('.', 1)[0])
        except Exception:
            layer_idx = -1

    return (modality, layer_idx) if return_layer_idx else modality


def blip2_detect_modality_fn(param_name, return_layer_idx=False):
    """
    BLIP2 파라미터 이름을 분석해 vision / text / fusion 중 어느 모달리티에 속하는지 판별.
    구조: vision_model (CLIP ViT) + qformer (BERT-like) + language_model (OPT)
    """
    modality = 'fusion'
    layer_idx = -1

    # ---- Vision Encoder (CLIP ViT) ----
    if param_name.startswith('vision_model'):
        modality = 'vision'
        key = 'vision_model.encoder.layers.'
        if return_layer_idx:
            try:
                layer_idx = int(param_name.split(key, 1)[1].split('.', 1)[0])
            except Exception:
                layer_idx = -1

    # ---- Q-Former (Cross-modal Transformer) ----
    elif param_name.startswith('qformer'):
        modality = 'fusion'
        if return_layer_idx:
            try:
                layer_idx = int(param_name.split('qformer.encoder.layer.', 1)[1].split('.', 1)[0])
            except Exception:
                layer_idx = -1

    # ---- Language Model (OPT / T5 등) ----
    elif param_name.startswith('language_model'):
        modality = 'text'
        # OPT: language_model.model.decoder.layers.N
        # T5:  language_model.encoder.block.N
        if return_layer_idx:
            if 'decoder.layers.' in param_name:
                try:
                    layer_idx = int(param_name.split('decoder.layers.', 1)[1].split('.', 1)[0])
                except Exception:
                    layer_idx = -1
            elif 'encoder.block.' in param_name:
                try:
                    layer_idx = int(param_name.split('encoder.block.', 1)[1].split('.', 1)[0])
                except Exception:
                    layer_idx = -1

    return (modality, layer_idx) if return_layer_idx else modality

def llava_detect_modality_fn(param_name, return_layer_idx=False):
    """
    LLaVA 파라미터 이름을 vision / text / fusion으로 분리.
    구조:
      - vision_tower.vision_model.encoder.layers.N...  -> vision
      - multi_modal_projector.*                       -> fusion
      - language_model.model.layers.N...              -> text
    """
    modality  = 'fusion'
    layer_idx = -1

    # lightning/Fabric, LlavaForConditionalGeneration에서 오는 경우 "model." prefix 있을 수 있음
    name = param_name
    if name.startswith("model."):
        name = name[len("model."):]

    # ---- Vision tower (CLIP ViT) ----
    if name.startswith("vision_tower"):
        modality = "vision"
        if return_layer_idx:
            key = "vision_tower.vision_model.encoder.layers."
            if key in name:
                try:
                    layer_idx = int(name.split(key, 1)[1].split(".", 1)[0])
                except Exception:
                    layer_idx = -1

    # ---- Language model (LLaMA/Vicuna) ----
    elif name.startswith("language_model") or ".language_model." in name:
        modality = "text"
        if return_layer_idx:
            # 일반적인 LLaMA 스타일: language_model.model.layers.N
            key = "language_model.model.layers."
            if key in name:
                try:
                    layer_idx = int(name.split(key, 1)[1].split(".", 1)[0])
                except Exception:
                    layer_idx = -1

    # ---- Multi-modal projector (이미지→텍스트 브릿지) ----
    elif "multi_modal_projector" in name:
        modality  = "fusion"
        layer_idx = -1  # 보통 별도 레이어 인덱스 안 씀

    return (modality, layer_idx) if return_layer_idx else modality

def qwen_vl_detect_modality_fn(param_name, return_layer_idx=False):
    """
    Qwen2-VL 파라미터 이름을 vision / text / fusion으로 분리.
    구조:
      - visual.blocks.N...  -> vision
      - model.layers.N...   -> text
      - visual.merger...    -> fusion
    """
    modality = "fusion"
    layer_idx = -1

    name = param_name
    if name.startswith("module."):
        name = name[len("module."):]

    if name.startswith("visual."):
        modality = "fusion" if name.startswith("visual.merger") else "vision"
        if return_layer_idx and "visual.blocks." in name:
            try:
                layer_idx = int(name.split("visual.blocks.", 1)[1].split(".", 1)[0])
            except Exception:
                layer_idx = -1
    elif name.startswith("model."):
        modality = "text"
        if return_layer_idx and "model.layers." in name:
            try:
                layer_idx = int(name.split("model.layers.", 1)[1].split(".", 1)[0])
            except Exception:
                layer_idx = -1
    elif name.startswith("lm_head"):
        modality = "text"

    return (modality, layer_idx) if return_layer_idx else modality

import re

def flamingo_detect_modality_fn(param_name, return_layer_idx=False):
    modality, layer_idx = "fusion", -1

    name = param_name
    if name.startswith("model."):
        name = name[len("model."):]

    # vision
    if name.startswith("vision_encoder"):
        modality = "vision"
        if return_layer_idx and "vision_encoder.transformer.resblocks." in name:
            try:
                layer_idx = int(name.split("vision_encoder.transformer.resblocks.", 1)[1].split(".", 1)[0])
            except Exception:
                layer_idx = -1
        return (modality, layer_idx) if return_layer_idx else modality

    # language
    if name.startswith("lang_encoder"):
        modality = "text"
        if return_layer_idx and "lang_encoder.transformer.blocks." in name:
            try:
                layer_idx = int(name.split("lang_encoder.transformer.blocks.", 1)[1].split(".", 1)[0])
            except Exception:
                layer_idx = -1
        # gated cross-attn은 fusion으로 보자 (block 안에 섞임)
        if "gated_cross" in name or "cross_attn" in name or "attn_gate" in name or "ff_gate" in name:
            modality = "fusion"
        return (modality, layer_idx) if return_layer_idx else modality

    # perceiver/resampler = fusion
    if name.startswith("perceiver") or name.startswith("resampler") or "perceiver_resampler" in name:
        modality = "fusion"
        return (modality, layer_idx) if return_layer_idx else modality

    # 기타 gated
    if "gated_cross" in name or "cross_attn" in name or "attn_gate" in name or "ff_gate" in name:
        modality = "fusion"

    return (modality, layer_idx) if return_layer_idx else modality


def detect_modality_fn(model_name, param_name, return_layer_idx=False):
    if model_name == 'xvlm':
        return xvlm_detect_modality_fn(param_name, return_layer_idx)
    elif model_name == 'blip':
        return blip_detect_modality_fn(param_name, return_layer_idx)
    elif model_name == 'dino':
        return vit_detect_modality_fn(param_name, return_layer_idx)
    elif model_name == 'clip':
        return clip_detect_modality_fn(param_name, return_layer_idx)
    elif model_name == 'clipG':
        return clip_detect_modality_fn(param_name, return_layer_idx)
    elif model_name == 'blip2':
        return blip2_detect_modality_fn(param_name, return_layer_idx)
    elif model_name == 'llava':
        return llava_detect_modality_fn(param_name, return_layer_idx)
    elif model_name == 'qwen_vl':
        return qwen_vl_detect_modality_fn(param_name, return_layer_idx)
    elif model_name == 'flamingo':
        return flamingo_detect_modality_fn(param_name, return_layer_idx)
    else:
        raise NotImplementedError(f"Modality detection for {model_name} not implemented")


def get_unprunable_parameters(model_name, prune_scope='both'):
    base=[]
    print(f"[Debug] Pruning 대상에서 제외할 파라미터 목록 반환: {model_name}")
    if model_name == "xvlm":
        return ['vision_proj', 'text_proj', 'itm_head', 'bbox_head', 'text_encoder.cls']
    elif model_name == "blip":
        return ['vision_proj', 'text_proj', 'itm_head', 'visual_encoder_m', 'text_encoder_m', 'cls.predictions']
    elif model_name == "dino":
        return ['classifier']
    elif model_name == "clip":
        base = ['logit_scale',
        'text_projection',
        'visual_projection',
        'text_model.embeddings.token_embedding',
        'text_model.embeddings.position_embedding',
        'vision_model.embeddings.position_embedding',
        'vision_model.embeddings.class_embedding',
        'vision_model.embeddings.patch_embedding',  
        'final_layer_norm',
        'pre_layrnorm',
        'post_layernorm',
        'layer_norm',      
        'LayerNorm']
        if prune_scope == 'text':
            # 비전 타워 전체를 제외 → 텍스트만 프루닝
            base += ['vision_model.']
        elif prune_scope == 'vision':
            # 텍스트 타워 전체를 제외 → 비전만 프루닝
            base += ['text_model.']
        return base
    elif model_name == "sd15_generator":
        return [
            'text_model.embeddings.token_embedding',     # 임베딩
            'text_model.embeddings.position_embedding',
            'text_model.final_layer_norm',               # 최종 LN
            'layer_norm',                                # 각 블록의 layer_norm1/2 를 포괄
            'LayerNorm'
        ]
    elif model_name in ("clipG"):
        return [
            'logit_scale',
            'text_projection',
            'visual_projection',
            'text_model.embeddings.token_embedding',
            'text_model.embeddings.position_embedding',
            'text_model.embeddings.positional_embedding',
            'vision_model.embeddings.position_embedding',
            'vision_model.embeddings.positional_embedding',
            'vision_model.embeddings.class_embedding',
            'vision_model.embeddings.patch_embedding',
            'final_layer_norm', 'pre_layrnorm', 'post_layernorm', 'layer_norm', 'LayerNorm'
        ]
    elif model_name == "sdxl_generator":
        return [
            'text_model.embeddings.token_embedding',
            'text_model.embeddings.position_embedding',
            'text_model.embeddings.positional_embedding',  # 일부 가중치 덤프에서 쓰이는 이름 대비
            'text_model.final_layer_norm',
            'text_projection',  # CLIPTextModelWithProjection(bigG)에서 projection은 보통 제외
            'layer_norm', 'LayerNorm', 'pre_layrnorm', 'post_layernorm'
        ]
    elif model_name == "blip2":
        return [
            # ===== Vision Encoder (CLIP-ViT 기반) =====
            'vision_model.embeddings.patch_embedding.weight',
            'vision_model.embeddings.position_embedding.weight',
            'vision_model.embeddings.class_embedding.weight',
            'vision_model.pre_layrnorm.weight',         # CLIP-style pre LN
            'vision_model.post_layernorm.weight',       # CLIP-style post LN
            'vision_model.layer_norm.weight', 'vision_model.LayerNorm.weight',

            # ===== Q-Former (BERT-like Transformer) =====
            'qformer',

            # ===== Language Model (OPTForCausalLM) =====
            "language_model.model.embed_tokens.weight",
            "language_model.model.embed_positions.weight",
            "language_model.model.final_layer_norm.weight",
            "language_model.lm_head.weight",            # LM output head
            "language_projection",  
            # ===== Projection Heads =====
            'vision_proj', 'text_proj',

            # ===== Generic Normalization Layers =====
            'layer_norm', 'LayerNorm', 'final_layer_norm',
        ]
    elif model_name == "llava":
        base = [
            # ===== Vision tower (CLIP-ViT) 임베딩 & Norm =====
            "vision_tower.vision_model.embeddings.patch_embedding",
            "vision_tower.vision_model.embeddings.position_embedding",
            "vision_tower.vision_model.embeddings.class_embedding",
            "vision_tower.vision_model.pre_layrnorm",
            "vision_tower.vision_model.post_layernorm",
            "vision_tower.vision_model.layernorm",
            "vision_tower.vision_model.LayerNorm",

            # ===== Multimodal projector =====
            # connector는 vision-language alignment에 민감하므로 기본 보호한다.
            "multi_modal_projector",

            # ===== LLaMA/Vicuna 언어 모델 임베딩 & Norm & head =====
            "language_model.model.embed_tokens",
            "language_model.model.embed_positions",
            "language_model.model.norm",     # LLaMA final RMSNorm
            "language_model.lm_head",

            # ===== Generic Norm =====
            "layer_norm",
            "LayerNorm",
            "final_layer_norm",
        ]

        # --- prune_scope 적용 ---
        if prune_scope == "text":
            # 텍스트만 프루닝 => 비전 타워 전체 제외
            base += ["vision_tower."]

            # projector는 base에서 이미 보호된다.

        elif prune_scope == "vision":
            # 비전만 프루닝 => 언어모델 전체 제외
            base += ["language_model."]

            # projector는 base에서 이미 보호된다.

        elif prune_scope in ("both", None):
            # 둘 다 프루닝 (기본) => 아무 것도 추가하지 않음
            pass
        else:
            raise ValueError(f"Unknown prune_scope: {prune_scope}")

        return base
    elif model_name == "qwen_vl":
        base = [
            # ===== Vision input / position / bridge =====
            "visual.patch_embed",
            "visual.rotary_pos_emb",
            "visual.merger",

            # ===== Language embedding / final norm / head =====
            "model.embed_tokens",
            "model.norm",
            "lm_head",

            # ===== Norm 계열 =====
            "layer_norm",
            "LayerNorm",
            "RMSNorm",
            "input_layernorm",
            "post_attention_layernorm",
            "norm1",
            "norm2",
            "ln_q",
        ]

        # Qwen에서는 generation 품질 보호를 위해 attention을 먼저 제외한다.
        # language self-attn + vision attn 모두 제외
        base += [
            "self_attn",
            ".attn.",
        ]

        if prune_scope == "text":
            base += ["visual."]
        elif prune_scope == "vision":
            base += ["model.", "lm_head"]
        elif prune_scope in ("both", None):
            pass
        else:
            raise ValueError(f"Unknown prune_scope: {prune_scope}")

        return base
    
    elif model_name == "flamingo":
        base = [
            # ----- vision: embedding/pos/conv/proj/ln 정도만 보호 -----
            "vision_encoder.class_embedding",
            "vision_encoder.positional_embedding",
            "vision_encoder.position_embedding",
            "vision_encoder.conv1",
            "vision_encoder.proj",
            "vision_encoder.ln_pre",
            "vision_encoder.ln_post",# 이걸 정확히 제외
            ".attn.out_proj",    

            # ----- language: token/pos embedding + final norm + lm head 보호 -----
            "lang_encoder.transformer.wte",
            "lang_encoder.transformer.wpe",
            "wte", "wpe",                 # fallback (이름 변형 대비)
            "embed_tokens", "embed_positions",
            "norm_f", "final_layer_norm", "final_ln",   # final norm 후보들
            "lm_head",

            # ----- fusion: perceiver/resampler + gated cross-attn + gate 보호 -----
            "perceiver", "resampler", "perceiver_resampler",
            "gated_cross",             # OpenFlamingo에서 가장 확실한 키
            "attn_gate", "ff_gate",
        ]

        # ⚠️ 범용 패턴은 과보호라 제거/축소 추천
        # ".norm", ".ln_" 같은 거 넣고 싶으면 아래처럼 'final' 계열에만 걸리게 좁혀라:
        # base += ["norm_f", "final_layer_norm", "final_ln"]

        if prune_scope == "text":
            base += ["vision_encoder.", "perceiver", "resampler", "gated_cross"]
        elif prune_scope == "vision":
            base += ["lang_encoder.", "perceiver", "resampler", "gated_cross", "lm_head"]
        elif prune_scope in ("both", None):
            pass
        else:
            raise ValueError(f"Unknown prune_scope: {prune_scope}")

        return base




    else:
        raise NotImplementedError(f"Unprunable parameters for {model_name} not implemented")
