import torch
import ruamel.yaml as yaml


from models import BLIPPretrain, XVLMPretrain
from huggingface_hub import snapshot_download
from utils.misc import update_config_for_vit

available_models = ["xvlm", "blip", "dino", "clip", "clipG", "blip2", "llava", "qwen_vl", 'flamingo']
import os, json

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

def migrate_concat_qkv_to_split(state_dict: dict) -> dict:
    """
    기존 ckpt의 ...attn.qkv.{weight,bias}를
    ...attn.{q_proj,k_proj,v_proj}.{weight,bias}로 분리해서 리맵.
    visual_encoder / visual_encoder_m 모두 자동 처리.
    """
    new_sd = state_dict.copy()
    to_delete = []

    for k, v in state_dict.items():
        if k.endswith("attn.qkv.weight"):
            # ex) k = 'visual_encoder.blocks.0.attn.qkv.weight'
            prefix = k[:-len("qkv.weight")]  # '...attn.'
            W = v  # shape: (3*D_out, D_in)
            D3, Din = W.shape
            assert D3 % 3 == 0, f"unexpected qkv weight shape: {W.shape}"
            D = D3 // 3
            Wq, Wk, Wv = W[:D, :].contiguous(), W[D:2*D, :].contiguous(), W[2*D:, :].contiguous()
            new_sd[prefix + "q_proj.weight"] = Wq
            new_sd[prefix + "k_proj.weight"] = Wk
            new_sd[prefix + "v_proj.weight"] = Wv
            to_delete.append(k)

            # bias가 같이 있으면 동일하게 분리
            bkey = prefix + "qkv.bias"
            if bkey in state_dict:
                b = state_dict[bkey]  # (3*D,)
                assert b.numel() == 3 * D, f"unexpected qkv bias shape: {b.shape}"
                new_sd[prefix + "q_proj.bias"] = b[:D].contiguous()
                new_sd[prefix + "k_proj.bias"] = b[D:2*D].contiguous()
                new_sd[prefix + "v_proj.bias"] = b[2*D:].contiguous()
                to_delete.append(bkey)

    # 사용 안 할 키들 제거
    for k in to_delete:
        new_sd.pop(k, None)

    # BLIP 계열에서 자주 뜨는 예외 키 정리(필요 시 무시해도 됨)
    # vocab 사이즈 불일치 등으로 종종 등장
    new_sd.pop("text_decoder.cls.predictions.bias", None)

    return new_sd

import re
from collections import OrderedDict
from typing import Dict, Any


def remap_flamingo_keys(sd_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenFlamingo 체크포인트(state_dict) 키를 현재 코드베이스 모델 키 스키마에 맞게
    '안전하게' 보정한다.

    안전 원칙:
    - 기존 키는 절대 삭제/대체하지 않는다.
    - alias(추가)만 만든다.
    - 특히 gated_cross 관련 키는 여기서 절대 건드리지 않는다.
      (gated alias는 add_blocks_to_list_alias 같은 별도 함수에서 처리)

    반환:
    - sd2: 원본 키 + alias 키가 함께 들어있는 dict
    """
    # 원본 보존(그대로 남기고, alias만 추가)
    sd2 = OrderedDict(sd_raw)

    def add_alias(old_key: str, new_key: str):
        # old_key가 원본에 있고, 새 키가 아직 없으면 alias 추가
        if old_key in sd_raw and new_key not in sd2:
            sd2[new_key] = sd_raw[old_key]

    # ------------------------------------------------------------
    # 0) module./model. prefix 제거는 보통 호출부에서 처리하는 걸 권장
    #    (여기서도 하고 싶으면 주석 해제해서 쓰면 됨)
    # ------------------------------------------------------------
    # tmp = OrderedDict()
    # for k, v in sd2.items():
    #     nk = k
    #     if nk.startswith("module."):
    #         nk = nk[len("module."):]
    #     if nk.startswith("model."):
    #         nk = nk[len("model."):]
    #     tmp[nk] = v
    # sd_raw = dict(tmp)
    # sd2 = OrderedDict(tmp)

    # ------------------------------------------------------------
    # 1) Perceiver/Resampler 이름 차이 보정
    #    - 어떤 ckpt는 perceiver.*, 어떤 ckpt는 resampler.*를 사용
    #    - 둘 중 하나만 있을 때 반대쪽 alias를 만들어줌
    # ------------------------------------------------------------
    for k in list(sd_raw.keys()):
        if k.startswith("resampler."):
            add_alias(k, "perceiver." + k[len("resampler."):])

    # ------------------------------------------------------------
    # 2) lang encoder 내부 블록 이름 차이 보정
    #    - 버전별로 old_decoder_blocks vs transformer.blocks.{i}.decoder_layer
    #    - 여기서는 '추가'만 한다(삭제/이동 X)
    # ------------------------------------------------------------
    # old_decoder_blocks.N.*  -> transformer.blocks.N.decoder_layer.*
    pat_old = re.compile(r"^lang_encoder\.old_decoder_blocks\.(\d+)\.(.+)$")
    for k in list(sd_raw.keys()):
        m = pat_old.match(k)
        if m:
            idx, rest = m.group(1), m.group(2)
            add_alias(k, f"lang_encoder.transformer.blocks.{idx}.decoder_layer.{rest}")

    # transformer.blocks.N.decoder_layer.* -> old_decoder_blocks.N.*
    pat_new = re.compile(r"^lang_encoder\.transformer\.blocks\.(\d+)\.decoder_layer\.(.+)$")
    for k in list(sd_raw.keys()):
        m = pat_new.match(k)
        if m:
            idx, rest = m.group(1), m.group(2)
            add_alias(k, f"lang_encoder.old_decoder_blocks.{idx}.{rest}")

    # ------------------------------------------------------------
    # 3) gated_cross_attn은 여기서 remap/alias 하지 않는다.
    #    - blocks<->list 변환은 add_blocks_to_list_alias(sd2, model) 같은
    #      '모델 구조를 보고' 만드는 별도 함수에서 처리하는 게 안전.
    # ------------------------------------------------------------

    return sd2




def remap_gated_cross_attn_blocks_to_list(sd, model=None):
    """
    ckpt: lang_encoder.transformer.blocks.{i}.gated_cross_attn_layer.*
      -> model expected: lang_encoder.gated_cross_attn_layers.{j}.*
    여기서 j는 'gated_cross_attn_layer가 존재하는 block 순서'로 매핑.
    cross_attn_every_n_layers != 1인 경우도 안전하게 처리.
    """

    # 0) module. prefix 제거
    def strip_module(k: str):
        return k[len("module."):] if k.startswith("module.") else k

    # 1) 어떤 block들이 gated_cross_attn_layer를 갖는지 추정
    #    model이 있으면 model 기준이 제일 정확함.
    if model is not None:
        blocks = model.lang_encoder.transformer.blocks
        gca_block_ids = []
        for i, b in enumerate(blocks):
            if hasattr(b, "gated_cross_attn_layer") and getattr(b, "gated_cross_attn_layer") is not None:
                gca_block_ids.append(i)
        # block_id -> j (등장순서) 역매핑
        block_to_j = {bid: j for j, bid in enumerate(gca_block_ids)}
    else:
        # model이 없으면 ckpt에서 등장하는 block id를 정렬해 순서대로 매핑
        block_ids = set()
        pat_tmp = re.compile(r"^lang_encoder\.transformer\.blocks\.(\d+)\.gated_cross_attn_layer\..+$")
        for k in sd.keys():
            k = strip_module(k)
            m = pat_tmp.match(k)
            if m:
                block_ids.add(int(m.group(1)))
        gca_block_ids = sorted(block_ids)
        block_to_j = {bid: j for j, bid in enumerate(gca_block_ids)}

    # 2) 실제 리맵
    pat = re.compile(r"^lang_encoder\.transformer\.blocks\.(\d+)\.gated_cross_attn_layer\.(.+)$")
    out = {}
    for k, v in sd.items():
        k = strip_module(k)
        m = pat.match(k)
        if m:
            block_id = int(m.group(1))
            suffix = m.group(2)
            if block_id in block_to_j:
                j = block_to_j[block_id]
                k = f"lang_encoder.gated_cross_attn_layers.{j}.{suffix}"
        out[k] = v

    return out

def add_blocks_to_list_alias(sd, model):
    out = dict(sd)  # 원본 유지(중요!)

    # 모델에서 gated_cross_attn_layer가 실제로 달린 block index만 뽑기
    blocks = model.lang_encoder.transformer.blocks
    gca_block_ids = []
    for i, b in enumerate(blocks):
        if hasattr(b, "gated_cross_attn_layer") and getattr(b, "gated_cross_attn_layer") is not None:
            gca_block_ids.append(i)
    block_to_j = {bid: j for j, bid in enumerate(gca_block_ids)}

    pat = re.compile(r"^lang_encoder\.transformer\.blocks\.(\d+)\.gated_cross_attn_layer\.(.+)$")
    for k, v in sd.items():
        m = pat.match(k)
        if not m:
            continue
        block_id = int(m.group(1))
        suffix = m.group(2)
        if block_id in block_to_j:
            j = block_to_j[block_id]
            out[f"lang_encoder.gated_cross_attn_layers.{j}.{suffix}"] = v

    return out

def model_factory(model_name: str) -> torch.nn.Module:
    r"""Shared access interface for all models. The `model_name` argument switches between models.
    By the time of code release, the function supports 3 values for `model_name`: "xvlm", "blip" and "dino".  
    - "xvlm" initializes an XVLM with a CLIP-ViT-B/16 as the vision encoder. The text encoder is a 6-layer BERT Encoder, while 
    the fusion module is a 6-layer BERT Decoder which cross-attends visual and text latents from both unimodal encoders; 
    By default, the model loads pretraining weights from the 4M Pretraining Dataset for VLMs.
    - "blip" initializes a BLIP-Base model, with a ViT-B/16 and a 12-layer BERT Encoder and a 12-layer BERT Decoder which comprise the 
    MED (Multimodal Mixture of Encoder-Decoder networks). The weights of this model results from 14M Pretraining;  
    - "dino" initializes a ViT-B/16 pretrained with the DINO method from Meta AI.
    """
    import os
    import torch

    if model_name == "xvlm":
        # initialize the config for XVLM
        config_path = 'configs/xvlm/pretrain_xvlm_base_4m.yaml'
        config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

        # the default config would initialize an XVLM-Swin, while the repo always uses XVLM-ClipViT
        config = update_config_for_vit(config)
        pretraining_weights = 'weights/4m-xvlm-vit-bert.pth'

        # load weights from the 4M pretraining
        model = XVLMPretrain(config)
        pretraining_weights = torch.load(pretraining_weights, map_location="cpu")['model']
        model.load_state_dict(pretraining_weights, strict=False)
        
        # FIXME: define a 'property' within the model in-place of this manual attribute assignment
        setattr(model, 'dtype', torch.float32)
        setattr(model, 'is_vlm', True)
        setattr(model, 'needs_tie', False)
    
    elif model_name == "blip":
        config_path = 'configs/blip/pretrain_blip_base_14m.yaml'
        config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        model = BLIPPretrain( #blip_pretrained.py의 BLIPPretrain클래스의 객체 생성 -> init()함수 실행행
            image_size=config['image_res'], 
            vit=config['vit'], 
            vit_grad_ckpt=config['vit_grad_ckpt'], 
            vit_ckpt_layer=config['vit_ckpt_layer'], 
            queue_size=config['queue_size']
        )

        #Pre-trained된 가중치 가져옴
        pretraining_weights = 'weights/14m-blip-vitB.pth'
        #가중치를 model에 적용함
        pretraining_weights = torch.load(pretraining_weights, map_location="cpu")['model']
        pretraining_weights = migrate_concat_qkv_to_split(pretraining_weights)
        msg = model.load_state_dict(pretraining_weights, strict=False)
        print(f"{'='*50} Loaded BLIP ViT-B weights {'='*50}")
        print("missing keys:", msg.missing_keys)
        print("unexpected keys:", msg.unexpected_keys, end="\n\n")
        
        # FIXME: define a 'property' within the model in-place of this manual attribute assignment
        #속성 추가가
        setattr(model, 'dtype', torch.float32)
        setattr(model, 'is_vlm', True)
        setattr(model, 'needs_tie', True)

    elif model_name == "dino":
        # some models can easily be initialized from the HF hub :)
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained("facebook/dino-vitb16", num_labels=1000)
        setattr(model, 'is_vlm', False)
        setattr(model, 'needs_tie', False)

    elif model_name == "clip":
        print("[Debug] CLIP 모델 불러오기")
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        setattr(model, 'is_vlm', True)
        setattr(model, 'needs_tie', False)
    elif model_name == "clipG":
        print("[Debug] CLIP-ViT-bigG-14 모델 불러오기 (safetensors)")
        from transformers import CLIPModel
        from huggingface_hub import snapshot_download
        import os, json

        HF_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

        local_dir = snapshot_download(
            HF_ID,
            allow_patterns=[
                # 가중치(sharded safetensors) + bin 인덱스(변환에 필요)
                "*.safetensors", "pytorch_model.bin.index.json",
                # 구성/토크나이저
                "config.json", "open_clip_config.json",
                "tokenizer.json","tokenizer_config.json","vocab.json","merges.txt",
                "preprocessor_config.json","special_tokens_map.json",
                # 문서
                "*.md","*.txt",
            ],
        )
        

        # --- 여기가 핵심: .bin 인덱스를 safetensors 인덱스로 변환 ---
        bin_index = os.path.join(local_dir, "pytorch_model.bin.index.json")
        safe_index = os.path.join(local_dir, "model.safetensors.index.json")
        if not os.path.exists(safe_index):
            if not os.path.exists(bin_index):
                raise FileNotFoundError(f"[ERR] {bin_index} 가 없습니다. allow_patterns를 확인하세요.")
            with open(bin_index, "r") as f:
                data = json.load(f)
            wmap = data.get("weight_map", {})
            # .bin → .safetensors 로 파일명 치환
            new_map = {k: v.replace(".bin", ".safetensors") for k, v in wmap.items()}
            # 총 용량(meta) 계산(선택)
            shard_files = set(new_map.values())
            total_size = 0
            for sfn in shard_files:
                p = os.path.join(local_dir, sfn)
                if os.path.exists(p):
                    total_size += os.path.getsize(p)
            safe_payload = {"metadata": {"total_size": int(total_size)}, "weight_map": new_map}
            with open(safe_index, "w") as f:
                json.dump(safe_payload, f)
            print(f"[Debug] 생성: {safe_index}")

        # 이제 safetensors 샤드 + 인덱스로 로드 가능
        model = CLIPModel.from_pretrained(
            local_dir,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            device_map=None,   # Fabric이 이후 장치로 올림
        )
        model = model.float()  # pruning은 fp32
        setattr(model, "is_vlm", True)
        setattr(model, "needs_tie", False)
    elif model_name == "blip2":
        print("[Debug] BLIP-2 모델 불러오기")
        from transformers import Blip2Model
        from transformers import Blip2ForConditionalGeneration
        HF_ID = "Salesforce/blip2-opt-2.7b"
        model = Blip2Model.from_pretrained(
                HF_ID,
                torch_dtype=torch.float32,  # pruning 호환성 위해 fp32
                low_cpu_mem_usage=True
            )
        setattr(model, "is_vlm", True)
        setattr(model, "needs_tie", False)

        print(f"[Debug] Loaded {HF_ID}") 
    elif model_name == "llava":
        print("[Debug] LLaVA-v1.5-7B 모델 불러오기")
        from transformers import LlavaForConditionalGeneration

        # HF에 올라와 있는 LLaVA-v1.5-7B 체크포인트
        HF_ID = "llava-hf/llava-1.5-7b-hf"

        # 1) 로드는 fp16 + low_cpu_mem_usage로 가볍게
        model = LlavaForConditionalGeneration.from_pretrained(
            HF_ID,
            low_cpu_mem_usage=True,
            device_map=None,            # Fabric이 나중에 알아서 올리도록 CPU에 둠
        )

        # 2) 프루닝 파이프라인이 fp32 기준이면 여기서 올려주기
        model = model.float()

        # 3) 다른 모델들과 맞추는 메타 정보
        setattr(model, "is_vlm", True)     # vision + language 모델
        setattr(model, "needs_tie", False) # LLaVA는 별도의 tie 작업 필요 없음
    elif model_name == "qwen_vl":
        print("[Debug] Qwen2-VL 모델 불러오기")
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        hf_id = os.environ.get("QWEN_VL_ID", "Qwen/Qwen2-VL-2B-Instruct")
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None,
            trust_remote_code=True,
        )
        model = model.float()
        setattr(model, "is_vlm", True)
        setattr(model, "needs_tie", False)
        setattr(model, "processor", processor)
        setattr(model, "tokenizer", processor.tokenizer)
    elif model_name == "flamingo":
        print("[Debug] Flamingo(OpenFlamingo) 모델 불러오기")

        import os
        import torch
        from huggingface_hub import hf_hub_download
        from open_flamingo import create_model_and_transforms  # pip install open-flamingo

        # -------------------------
        # 1) 어떤 체크포인트를 쓸지 선택
        #    - 환경변수로 바꿀 수 있게 해두면 편함
        # -------------------------
        # 예) export FLAMINGO_CKPT="openflamingo/OpenFlamingo-4B-vitl-rpj3b"
        ckpt_repo = os.environ.get("FLAMINGO_CKPT", "openflamingo/OpenFlamingo-3B-vitl-mpt1b")

        # OpenFlamingo 공식 README에 나온 조합 기준 매핑 :contentReference[oaicite:1]{index=1}
        # (ckpt_repo별로 LM/토크나이저/크로스어텐션 주기가 다름)
        cfg_map = {
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b": {
                "lang_encoder_path": "anas-awadalla/mpt-1b-redpajama-200b",
                "tokenizer_path":   "anas-awadalla/mpt-1b-redpajama-200b",
                "cross_attn_every_n_layers": 1,
            },
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct": {
                "lang_encoder_path": "anas-awadalla/mpt-1b-redpajama-200b-dolly",
                "tokenizer_path":   "anas-awadalla/mpt-1b-redpajama-200b-dolly",
                "cross_attn_every_n_layers": 1,
            },
            "openflamingo/OpenFlamingo-4B-vitl-rpj3b": {
                "lang_encoder_path": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
                "tokenizer_path":   "togethercomputer/RedPajama-INCITE-Base-3B-v1",
                "cross_attn_every_n_layers": 2,
            },
            "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct": {
                "lang_encoder_path": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
                "tokenizer_path":   "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
                "cross_attn_every_n_layers": 2,
            },
            "openflamingo/OpenFlamingo-9B-vitl-mpt7b": {
                "lang_encoder_path": "anas-awadalla/mpt-7b",
                "tokenizer_path":   "anas-awadalla/mpt-7b",
                "cross_attn_every_n_layers": 4,
            },
        }

        if ckpt_repo not in cfg_map:
            raise ValueError(
                f"Unknown FLAMINGO_CKPT='{ckpt_repo}'. "
                f"Supported: {list(cfg_map.keys())}"
            )

        cfg = cfg_map[ckpt_repo]

        # -------------------------
        # 2) 모델 뼈대(vision encoder + LM + cross-attn) 구성
        # -------------------------
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=cfg["lang_encoder_path"],
            tokenizer_path=cfg["tokenizer_path"],
            cross_attn_every_n_layers=cfg["cross_attn_every_n_layers"],
        )

        checkpoint_path = hf_hub_download(repo_id=ckpt_repo, filename="checkpoint.pt")
        state = torch.load(checkpoint_path, map_location="cpu")

        sd_raw = state.get("state_dict", state.get("model", state))
        sd_raw = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd_raw.items()}

        # 1) remap 결과(sd2)를 만들되,
        sd2 = remap_flamingo_keys(sd_raw)

        # 2) remap이 gated를 "바꿔치기" 했을 가능성이 있으니,
        #    gated 관련 키는 sd_raw(원본)로 강제 복구해서 blocks를 살린다.
        for k in list(sd2.keys()):
            if "gated_cross" in k:
                sd2.pop(k)  # remap이 만든 gated 키 제거(대체 방지)

        for k, v in sd_raw.items():
            if "gated_cross" in k:
                sd2[k] = v  # 원본 blocks 스타일 gated 복구

        # 3) 그리고 blocks -> list alias를 "추가" (blocks는 그대로 유지)
        sd2 = add_blocks_to_list_alias(sd2, model)

        msg = model.load_state_dict(sd2, strict=False)

        print("missing gated:", sum("gated_cross" in k for k in msg.missing_keys))

        missing_g = [k for k in msg.missing_keys if "gated_cross" in k]
        print("missing gated sample:", missing_g[:20])

        print("missing total:", len(msg.missing_keys))
        print("unexpected total:", len(msg.unexpected_keys))
        print("missing head:", msg.missing_keys[:30])

        print("unexpected head:", msg.unexpected_keys[:50])
        print("unexpected tail:", msg.unexpected_keys[-20:])
        ve = model.vision_encoder
        w = ve.conv1.weight
        print("vision conv1 mean/std:", w.float().mean().item(), w.float().std().item())
        print("has perceiver?", hasattr(model, "perceiver"))
        print("has resampler?", hasattr(model, "resampler"))


        # -------------------------
        # 4) factory가 Module만 반환하니까,
        #    processor/tokenizer는 모델에 붙여서 쓰자
        # -------------------------
        setattr(model, "image_processor", image_processor)
        setattr(model, "tokenizer", tokenizer)

        # 기존 코드 스타일 맞추기
        setattr(model, "dtype", torch.float32)
        setattr(model, "is_vlm", True)
        setattr(model, "needs_tie", False)

    
    else:
        raise NotImplementedError(
            f"Model {model_name} not implemented. Please add it to the factory yourself."
        )
    

    # IMPORTANT: do NOT remove this line! It injects a "name" attribute into each Module, which is then used by all pruners to switch 
    # between various function implementations.  
    #모델 이름 속성 추가 후 모델 객체 반환
    setattr(model, 'name', model_name)
    return model
