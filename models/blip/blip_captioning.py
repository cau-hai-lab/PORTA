'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

from models.blip.vit import VisionTransformer, interpolate_pos_embed
from models.blip.med import BertConfig, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from models.blip.vit import interpolate_pos_embed
from utils.prune_utils import inherit_encoder_decoder_masks


class BLIPCaptioning(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/blip/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        print("[Debug] blip_captioning.py : BLIPCaptioning 클래스 init() 함수 호출")
        # Vision encoder 생성
        print("[Debug] blip_captioning.py : Vision encoder 생성")
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer() # 디코더 입력에 사용될 초기 프롬프트를 위한 토크나이저 생성   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width # Text decoder가 vision encoder의 이미지 임베딩을 입력으로 받기 위해 차원을 맞춰줌
        # Text decoder 생성
        print("[Debug] blip_captioning.py : Text decoder 생성")
        self.text_decoder = BertLMHeadModel(config=med_config)    
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1 # 프롬프트를 구분하기 위해 패딩을 추가하여 토크나이징함

        
    def forward(self, image, caption, already_tokenized=False, **kwargs):
        print("[Debug] blip_captioning.py : BLIPCaptioning 클래스 forward() 함수 호출")
        
        # 1) 혹시 PEFT/어댑터 경유로 top-level에 input_ids/attention_mask가 넘어올 경우 처리
        if caption is None and ('input_ids' in kwargs or 'attention_mask' in kwargs):
            q_ids = kwargs.get('input_ids', None)
            q_att = kwargs.get('attention_mask', None)
            if q_ids is not None:
                if q_att is None:
                    q_att = torch.ones_like(q_ids)
                caption = SimpleNamespace(input_ids=q_ids, attention_mask=q_att)

        # 2) question/answer가 dict(BatchEncoding)로 들어와도 동작
        if isinstance(caption, dict):
            caption = SimpleNamespace(**caption)
        
        image_embeds = self.visual_encoder(image) # 비전 인코더를 통해 이미지 임베딩 생성
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device) # 이미지 어텐션 마스크 생성
        
        if not already_tokenized: # caption이 이미 토크나이징 되어있지 않은 경우
            # caption을 토크나이징하여 텐서로 변환
            text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device)
        else: # caption이 이미 토크나이징 되어있는 경우
            text = caption.to(image.device)
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id # 디코더 입력의 첫 번째 토큰을 [DEC]로 설정
        
        # 정답 레이블 생성
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100) # 패딩 토큰을 -100으로 설정하여 손실 계산에서 제외     
        decoder_targets[:,:self.prompt_length] = -100 # 프롬프트 부분을 -100으로 설정하여 손실 계산에서 제외
     
        decoder_output = self.text_decoder(text.input_ids, # 입력된 캡션의 일부를 기반으로 다음 토큰을 예측하여 loss 계산
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss
        return loss_lm
        

    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image) # 비전 인코더를 통해 이미지 임베딩 생성

        # if not sample:
        #     image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        # 이미지 어텐션 마스크 생성     
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        # 디코더 호출 시 전달할 준비된 인자들
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        prompt = [self.prompt] * image.size(0) # Batch에 있는 각 이미지를 위해 동일한 프롬프트 준비
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) # 프롬프트를 토크나이징하여 텐서로 변환
        input_ids[:,0] = self.tokenizer.bos_token_id # 첫 번째 토큰을 [BOS]로 설정
        input_ids = input_ids[:, :-1] # 마지막 토근 [SEP]은 제거하여 디코더가 프롬프트 뒤에 새 토큰을 생성하도록 함

        if sample: # sample = True인 경우
            #nucleus sampling : 모델이 예측한 확률 상위 p의 후보만 남긴 뒤, 그 중 랜덤으로 한 토큰을 선택하여 문장 생성
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search : 여러 beam 경로를 탐색하며 가장 높은 확률의 문장을 찾는다.
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        captions = []    
        for output in outputs: # 디코더가 생성한 토큰 시퀀스를 디코딩하여 문장으로 변환
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions
    
    def load_from_pruned_pretrained(self, pretraining_weights, mask, config, load_capt_pretrain=False):
        print("[Debug] blip_captioning.py : load_from_pruned_pretrained() 함수 호출 -> pruning mask 적용")
        self.load_pretrained(pretraining_weights, config, load_capt_pretrain)

        print(f"Loading from mask at: {mask}")
        mask = torch.load(mask, map_location="cpu")
        mask = inherit_encoder_decoder_masks(mask) # 가중치를 공유하고 있는 encoder와 decoder 사이에 mask를 공유
        msg = self.load_state_dict(mask, strict=False) # pruninig mask 적용
        print("missing keys:")
        print([k for k in msg.missing_keys if "bias" not in k and "layernorm" not in k.lower() and "pruning_mask" in k])
        
        keys_to_exclude = ["bias", "layernorm", "pruning_mask", "text_encoder", "text_encoder_m", "visual_encoder_m"]
        print("unexpected keys:")
        print([k for k in msg.unexpected_keys if not any([x in k.lower() for x in keys_to_exclude])])
    
    # def load_pretrained(self, weights_ckpt, *args, **kwargs):
    #     print("[Debug] blip_captioning.py : load_pretrained() 함수 호출 -> pre-trained된 가중치 정보 출력")
    #     print("Loaded params from: ", weights_ckpt)
    #     _, msg = load_checkpoint(self, weights_ckpt)
    #     print("missing keys:")
    #     print([k for k in msg.missing_keys if "pruning_mask" not in k])

    #     # the checkpoint also contains the weights of the momentum encoders, which are not to be loaded
    #     # as well as the 'text_encoder' weights, which are not used in captioning 
    #     print("unexpected keys:")
    #     keys_to_exclude = ["visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m", "text_decoder", "text_encoder"]
    #     print([k for k in msg.unexpected_keys if not any([x in k for x in keys_to_exclude])])
    
    from models.blip.vit import interpolate_pos_embed  # 이미 추가했다고 가정

    def load_pretrained(self, weights_ckpt, *args, **kwargs):
        print("[Debug] blip_captioning.py : load_pretrained()")
        print("Loaded params from: ", weights_ckpt)

        import torch
        ckpt = torch.load(weights_ckpt, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)  # ckpt 구조에 따라 조정

        new_state_dict = {}

        for k, v in state_dict.items():
            # 1) ViT 블록 qkv → q_proj / k_proj / v_proj로 분해
            if "visual_encoder.blocks" in k and ".attn.qkv." in k:
                # k 예시: "visual_encoder.blocks.0.attn.qkv.weight"
                prefix, suffix = k.split(".attn.qkv.")
                # prefix = "visual_encoder.blocks.0"
                # suffix = "weight" 또는 "bias"
                base = prefix + ".attn."  # "visual_encoder.blocks.0.attn."

                if suffix == "weight":
                    # v.shape = (3*dim, dim)
                    dim3, d_in = v.shape
                    dim = dim3 // 3
                    new_state_dict[base + "q_proj.weight"] = v[0:dim, :]
                    new_state_dict[base + "k_proj.weight"] = v[dim:2*dim, :]
                    new_state_dict[base + "v_proj.weight"] = v[2*dim:3*dim, :]
                elif suffix == "bias":
                    # v.shape = (3*dim,)
                    dim3 = v.shape[0]
                    dim = dim3 // 3
                    new_state_dict[base + "q_proj.bias"] = v[0:dim]
                    new_state_dict[base + "k_proj.bias"] = v[dim:2*dim]
                    new_state_dict[base + "v_proj.bias"] = v[2*dim:3*dim]
                else:
                    # 혹시 모를 이상한 suffix 방어용 (거의 안 타겠지만)
                    print("[Warn] unexpected suffix for attn.qkv:", k)
            else:
                new_state_dict[k] = v

        # 2) pos_embed 크기 안 맞으면 interpolate 해서 맞춰주기
        pe_key = "visual_encoder.pos_embed"
        if pe_key in new_state_dict:
            ckpt_pe = new_state_dict[pe_key]
            model_pe = self.visual_encoder.pos_embed
            if ckpt_pe.shape != model_pe.shape:
                print(f"[Debug] interpolate pos_embed: ckpt {ckpt_pe.shape} -> model {model_pe.shape}")
                new_state_dict[pe_key] = interpolate_pos_embed(ckpt_pe, self.visual_encoder)

        # 3) 실제 로딩
        msg = self.load_state_dict(new_state_dict, strict=False)

        print("missing keys:")
        print([k for k in msg.missing_keys if "pruning_mask" not in k])

        keys_to_exclude = ["visual_encoder_m", "text_encoder_m", "vision_proj_m",
                        "text_proj_m", "text_decoder", "text_encoder"]
        print("unexpected keys:")
        print([k for k in msg.unexpected_keys if not any([x in k for x in keys_to_exclude])])




def blip_captioning(pretrained='',**kwargs):
    model = BLIPCaptioning(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model    
 

def init_tokenizer():
    print("[Debug] blip_captioning.py : init_tokenizer()함수 호출 -> Tokenizer 생성")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    print("[Debug] blip_captioning.py : create_vit()함수 호출 -> vision encoder 생성")
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model, url_or_filename):
    print("[Debug] blip_captioning.py -> load_checkpoint()함수 호출")
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    
