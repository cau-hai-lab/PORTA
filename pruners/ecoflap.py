import torch
import torch.nn as nn
import numpy as np
import os
import time
from copy import deepcopy
from functools import partial
import re
import datetime
import torch
from functools import partial
import math
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from pruners.base import Pruner
from pruners.accumulators import forward_output, region_forward_output
from utils.prune_utils import (
    make_prunable, check_blip_state_dict, recursive_getattr,
    loss_vision_language, prepare_sample
)
from utils.functions import detect_modality_fn
import os
import torch.nn.functional as F
from utils.functions import detect_modality_fn
from pathlib import Path
class ECoFLaP(Pruner):
    def __init__(self, model, *args, **kwargs):
        print("[Debug] multiflow/pruners/ecoflap.py - ECoFLaP 클래스의 __init__ 함수 호출")
        self.model = model
        make_prunable(model, mask_dtype=torch.bool, pattern_lock=True, mask_on_the_fly=True, store_input=True)
        self._toggle_input_capture(False, only_prunable=False)
        super().__init__(model, keys_to_exclude=kwargs.get('keys_to_exclude', []))

        # zeroth-order 전용(고정)
        self.score_method = kwargs.get("score_method", "MEZO-GradMagSquare_asum")
        assert self.score_method.startswith("MEZO")
        self.score_compute, self.score_aggregate = self.score_method.split("_")
        self.num_noise = 1
        self.noise_eps = 1e-3
        self.output_dir = output_dir=str(Path(kwargs.get("output_dir", getattr(self, "output_dir", "."))).resolve())
        # 지연 세팅될 것들
        self.data_loader = None
        self.loss_func = None
        self.num_samples = None
        self.original_sparsity = None
        self.max_sparsity_per_layer = 0.6
        self.layer_to_group_mapping = {} 
        self.importance_measure = {}
        self.prune_per_model = False
        self.per_model_group = []
        self.name = 'ecoflap'
        self.is_one_shot = True 
        self.modifies_weights = False 
        self.actn_norms = {id(p): torch.ones(p.size()[1], dtype=torch.float32) for _, _, p in self.named_masked_parameters}
        self.actn_M2  = {id(p): torch.zeros(p.size(1), dtype=torch.float32) for _, _, p in self.named_masked_parameters}
        self.actn_cnt = {id(p): 0 for _, _, p in self.named_masked_parameters}
        #각 layer에서 수집한 샘플 수를 저장(초기 값은 0으로 설정)
        self.nsamples = {id(p): 0 for _, _, p in self.named_masked_parameters}
        # 모달리티 감지 함수(모델의 이름에 따라 각 layer가 text, vision중 어디에 속하는지 감지)
        self.detect_modality_fn = partial(detect_modality_fn, self.model.name) 
        self.forward_output = partial(forward_output, self.model.name)

    def _names_to_pids(self, name_sparsity: dict):
        """{param_name: s} -> {id(param): s} (compute_mask가 pid dict를 기대)"""
        by_pid = {}
        for name, _, p in self.named_masked_parameters:
            if name in name_sparsity:
                by_pid[id(p)] = float(name_sparsity[name])
        return by_pid

    # def _build_mapping_from_prunable(self):
    #     """self.named_masked_parameters 기준으로 그룹 매핑 생성"""
    #     mapping = {}
    #     for name, _, _ in self.named_masked_parameters:
    #         toks = name.split(".")
    #         if "vision_model" in toks and "layers" in toks:
    #             gi = toks.index("layers")
    #             group = ".".join(toks[:gi+2])   # ...layers.<i>
    #         elif ("text" in name or "language" in name or "bert" in name) and ("layer" in name or "layers" in name):
    #             if "layers" in toks:
    #                 gi = toks.index("layers")
    #                 group = ".".join(toks[:gi+2])
    #             elif "layer" in toks:
    #                 gi = toks.index("layer")
    #                 group = ".".join(toks[:gi+2])
    #             else:
    #                 group = toks[0]
    #         else:
    #             group = toks[0]
    #         mapping[name] = group
    #     return mapping



    def get_mask(self, importance_scores, p, max_sparsity_per_layer):
        print("[Debug] multiflow/pruners/ecoflap.py - get_mask 함수 호출")
        # Set top (1 - max_sparsity)% of parameters to be very large value to avoid 
        # them being pruned
        
        for k, v in importance_scores.items():
            num_to_set = int(importance_scores[k].numel() * (1 - max_sparsity_per_layer))
            
            if num_to_set > 0:
                threshold, _ = torch.topk(importance_scores[k].flatten(), num_to_set, largest=True)
                threshold = threshold[-1] # take the last value

                importance_scores[k][torch.where(v >= threshold)] = torch.finfo(v.dtype).max
        
        # Flatten all tensors and concatenate them
        all_scores = torch.cat([t.flatten() for t in importance_scores.values()])
        
        # Sort and find the threshold
        num_to_zero_out = int(p * all_scores.numel())
        threshold, _ = torch.topk(all_scores, num_to_zero_out, largest=False)
        threshold = threshold[-1]
        
        # Create mask based on threshold
        masks = {}
        for k, v in importance_scores.items():
            masks[k] = (v > threshold).type(v.dtype)
        
        return masks
    
    def get_layerwise_mask(self, importance_scores, p):
        print("[Debug] multiflow/pruners/ecoflap.py - get_layerwise_mask 함수 호출")
        # Set top (1 - max_sparsity)% of parameters to be very large value to avoid 
        # them being pruned
        
        masks = {}
        for k, v in importance_scores.items():
            all_scores = importance_scores[k].flatten().cuda()
            num_to_zero_out = int(p * all_scores.numel())
            threshold, _ = torch.topk(all_scores, num_to_zero_out, largest=False)
            threshold = threshold[-1].cpu()

            masks[k] = (v > threshold).type(v.dtype)

        return masks
        
    def global_iterative_pruning(self, target_sparsity, dict_layers_to_prune, iteratation=1, max_sparsity_per_layer=1.0):
        print("[Debug] multiflow/pruners/ecoflap.py - global_iterative_pruning 함수 호출")
        weight_copy = {}
        total_parameters = 0
        names = []
        params = []
        for k, v in self.model.named_parameters():  
            if k in dict_layers_to_prune:
                names.append(k)
                params.append(v)
                weight_copy[k] = torch.clone(v).cpu()

        masks = None
        for i in range(1, iteratation+1):
            p_i = target_sparsity ** (iteratation / i) # Compute modified sparsity for the i^th iteration
            
            importance_measure = self.compute_importance_scores_mezo(
                dict_layers_to_prune
            )
            
            importance_measure = {k: v for k, v in importance_measure.items() if k in dict_layers_to_prune}
            
            if masks is not None:
                # Apply mask to importance scores (this step is to simulate pruning in iterations)
                for k in importance_measure:
                    importance_measure[k] *= masks[k]

            masks = self.get_mask(importance_measure, p_i, max_sparsity_per_layer)

            # prune the model
            for k, v in self.model.named_parameters():
                if k in masks:
                    v.data *= masks[k].type(v.dtype).to(v.device)
                    
            print(f"Step {i}, target sparsity: {p_i:.4f}")
        
        sparsity_dict = {}
        for k, v in self.model.named_parameters():
            sparsity_dict[k] = ((v == 0).float().sum() / v.numel()).item()
            
        for k, p in zip(names, params):
            # use current_batch_index rather than self.num_samples because sometimes
            # the batch size might not be 1, and the loss is already normalized by 
            # batch size, now when only have to normalize it by num_batches now
            p.data = weight_copy[k].to(p.device)
        
        return sparsity_dict
    
    def compute_the_sparsity_per_group(self, total_parameters_to_keep, group_scores, group_num_parameters, max_sparsity_per_layer=0.8):
        print("[Debug] multiflow/pruners/ecoflap.py - compute_the_sparsity_per_group 함수 호출")
        scores = torch.FloatTensor(list(group_scores.values()))
        num_parameters = torch.LongTensor(list(group_num_parameters.values()))
        
        parameters_to_keep_per_group = torch.zeros_like(scores, dtype=int)
        
        parameters_to_keep_per_group += torch.ceil(num_parameters * (1 - max_sparsity_per_layer)).int() # to gaurantee the max_sparsity
        
        while parameters_to_keep_per_group.sum() < total_parameters_to_keep:
            total_ratio = torch.sum(scores)
            
            rest_total_parameters_to_keep = total_parameters_to_keep - parameters_to_keep_per_group.sum()
            
            parameters_to_add = torch.ceil((scores / total_ratio) * rest_total_parameters_to_keep)
            
            parameters_to_keep_per_group = parameters_to_keep_per_group + parameters_to_add
            
            scores[parameters_to_keep_per_group >= num_parameters] = 0 # make sure they are not going to add more parameters
            
            parameters_to_keep_per_group = torch.clamp(parameters_to_keep_per_group, max=num_parameters) # remove the extra parameters

            # they are to make sure the sum of parameters_to_keep_per_group is EXACTLY the same as total_parameters_to_keep
            if parameters_to_add.sum() == 0: # for some reason the algo cannot add more parameters
                # the algo stuck
                current_sum = parameters_to_keep_per_group.sum()
                if current_sum < total_parameters_to_keep:
                    num_need_to_add = total_parameters_to_keep - current_sum
                    
                    while num_need_to_add > 0:
                        # distributed the parameters to the rest of groups
                        for index in torch.where(scores > 0)[0]:
                            parameters_can_add = min(
                                num_need_to_add, num_parameters[index] - parameters_to_keep_per_group[index]
                            )
                            parameters_to_keep_per_group[index] += parameters_can_add
                            
                            num_need_to_add -= parameters_can_add
                            
                            if num_need_to_add == 0:
                                break
                            
            if parameters_to_keep_per_group.sum() > total_parameters_to_keep: # for some reason the algo cannot add more parameters
                # the algo stuck
                current_sum = parameters_to_keep_per_group.sum()

                num_need_to_remove = current_sum - total_parameters_to_keep
                
                while num_need_to_remove > 0:
                    # remove the parameters from full groups
                    for index in torch.argsort(parameters_to_keep_per_group, descending=True, stable=True):
                        parameters_can_remove = min(
                            num_need_to_remove, 
                            parameters_to_keep_per_group[index] - (num_parameters[index] * (1 - max_sparsity_per_layer)).int() # extra parameters
                        )
                        parameters_to_keep_per_group[index] -= parameters_can_remove
                        
                        num_need_to_remove -= parameters_can_remove
                        
                        if num_need_to_remove == 0:
                            break
                        
        # convert the group parameters to keep to sparsity    
        group_sparsity = {}
        
        for k, param_to_keep, group_max_param in zip(group_num_parameters.keys(), parameters_to_keep_per_group, num_parameters):
            group_sparsity[k] = torch.clamp(1 - param_to_keep / group_max_param, min=0, max=1).item()
            
        return group_sparsity
    
    def return_sparsity(self):
        print("[Debug] multiflow/pruners/ecoflap.py - return_sparsity 함수 호출")
        original_sparsity = self.original_sparsity
        layer_to_group_mapping = self.layer_to_group_mapping
        
        if self.score_compute.startswith("Real"):
            # get the layer sparsity perform the real global pruning
            return self.global_iterative_pruning(
                original_sparsity, layer_to_group_mapping, iteratation=3, max_sparsity_per_layer=1.0
            )

        if layer_to_group_mapping is None or len(layer_to_group_mapping) == 0:
            class uniform_sparsity_module:
                def __getitem__(self, key):
                    return original_sparsity
            return uniform_sparsity_module()

        # compute the global information
        if len(self.importance_measure) == 0:
            self.importance_measure = self.compute_importance_scores_mezo(layer_to_group_mapping)

        # create the layer list that for each group
        group_to_layer_mapping = {}
        for k, v in layer_to_group_mapping.items():
            if v not in group_to_layer_mapping:
                group_to_layer_mapping[v] = []

            group_to_layer_mapping[v].append(k)
        
        # store the num of parameters for each group and the total paramters
        # 수정 (항상 일관된 이름/객체 보장)
        num_parameters_dict = {}
        total_parameters = 0
        for name, _, p in self.named_masked_parameters:
            if name in layer_to_group_mapping:
                num_parameters_dict[name] = p.numel()
                total_parameters += p.numel()

        # total params to keep
        total_parameters_to_keep = int(total_parameters * (1 - original_sparsity))
        
        # store the importance per parameter for each group
        group_scores = {}
        group_num_parameters = {}
        for group_name, layers in group_to_layer_mapping.items():
            if group_name not in group_scores:
                group_scores[group_name] = 0
            
            num_params = 0
            for l in layers:
                group_scores[group_name] += float(self.importance_measure[l].sum().detach().cpu())

                
                num_params += num_parameters_dict[l]
            
            if self.score_aggregate == "avg":
                group_scores[group_name] /= num_params # normalization
            
            group_num_parameters[group_name] = num_params

        if self.prune_per_model:
            group_sparsity = {}
            for submodel_prefix in self.per_model_group:
                print(submodel_prefix)
                submodel_group_scores = {k: v for k, v in group_scores.items() if k.startswith(submodel_prefix)}
                submodel_group_num_parameters = {k: v for k, v in group_num_parameters.items() if k.startswith(submodel_prefix)}
                
                submodel_total_parameters_to_keep = int(sum(list(submodel_group_num_parameters.values())) * (1 - original_sparsity))
                submodel_group_sparsity = self.compute_the_sparsity_per_group(
                    submodel_total_parameters_to_keep, 
                    submodel_group_scores, 
                    submodel_group_num_parameters, 
                    max_sparsity_per_layer=self.max_sparsity_per_layer,
                )
                group_sparsity.update(submodel_group_sparsity)
        else:
            group_sparsity = self.compute_the_sparsity_per_group(
                total_parameters_to_keep, 
                group_scores, 
                group_num_parameters, 
                max_sparsity_per_layer=self.max_sparsity_per_layer,
            )
        
        compute_total_keep_parameters = 0
        for k in group_num_parameters:
            compute_total_keep_parameters += (1 - group_sparsity[k]) * group_num_parameters[k]

        # sanity check
        print(compute_total_keep_parameters, total_parameters_to_keep)
        
        layer_sparsity = {
            k: group_sparsity[v]
            for k, v in layer_to_group_mapping.items()
        }
        
        return layer_sparsity
    
    def zo_perturb_parameters(self, params, random_seed=1, scaling_factor=1, zo_eps=1e-3):
        print("[Debug] multiflow/pruners/ecoflap.py - zo_perturb_parameters 함수 호출")
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed)
        
        for param in params:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * zo_eps
    
    def _collect_params_by_group(self, layer_to_group_mapping):
        g2ps = {}
        name2param = {n: p for n, _, p in self.named_masked_parameters}
        for n, g in layer_to_group_mapping.items():
            g2ps.setdefault(g, []).append(name2param[n])
        return g2ps

    @torch.no_grad()
    def compute_importance_scores_mezo(self, layer_to_group_mapping):
        model, dl, loss_func = self.model, self.data_loader, self.loss_func
        model.eval()

        g2ps = self._collect_params_by_group(layer_to_group_mapping)
        zo_eps, n_mezo = float(self.noise_eps), int(self.num_noise)

        group_grad = {g: 0.0 for g in g2ps}
        accum = 0
        for batch in dl:
            if accum >= self.num_samples: break
            for g, params in g2ps.items():
                val = 0.0
                for _ in range(n_mezo):
                    seed = np.random.randint(1_000_000_000)
                    self.zo_perturb_parameters(params, random_seed=seed, scaling_factor=1,  zo_eps=zo_eps)
                    loss1, blen = loss_func(model, batch, True)
                    self.zo_perturb_parameters(params, random_seed=seed, scaling_factor=-2, zo_eps=zo_eps)
                    loss2, _     = loss_func(model, batch, True)
                    self.zo_perturb_parameters(params, random_seed=seed, scaling_factor=1,  zo_eps=zo_eps)
                    val += abs((float(loss1) - float(loss2)) / (2 * zo_eps))
                group_grad[g] += val / max(n_mezo, 1)
            accum += blen

        # 그룹 스칼라 → 레이어 텐서로 브로드캐스트
        importance = {}
        for name, _, p in self.named_masked_parameters:
            if name not in layer_to_group_mapping: continue
            g = layer_to_group_mapping[name]
            if   self.score_compute == "MEZO-GradOnly":
                importance[name] = torch.full_like(p, group_grad[g], dtype=torch.float32, device="cpu")
            elif self.score_compute == "MEZO-GradMagAbs":
                importance[name] = p.detach().abs().to("cpu", torch.float32) * group_grad[g]
            elif self.score_compute == "MEZO-GradMagSquare":
                importance[name] = (p.detach().to("cpu", torch.float32)**2) * (group_grad[g]**2)
            else:
                raise ValueError(self.score_compute)
        return importance


    def _offload_actns(self, text_atts_history): #text_atts_history -> Text data의 Attention_mask를 저장한 리스트
        print("[Debug] Multiflow.py -> _offload_actns 함수 호출 : Prunable한 파라미터별 저장된 활성화값의 Norm 계산")
        for name, _, param in self.named_masked_parameters:
            mname = ".".join(name.split(".")[:-1])#현재 파라미터가 속한 파라미터 이름에서 모듈 이름만 추출
            module = recursive_getattr(self.model, mname)#모델의 해당 모듈을 동적으로 참조한다
            modality = self.detect_modality_fn(name)#현재 파라미터가 어떤 모달리티에 속하는지 확인
            
            # if the current layer is a textual one, we must make sure not to include the [PAD] tokens 
            # in the computation of the score
            num_samples_to_add = 0
            if modality in ("text", "fusion"): # 현재 처리중인 파라미터가 text, fusion 일 때(text관련 모달리티인 경우)
                #text 모달리티에 저장된 활성화 값들은 패딩 토큰을 포함하고  있기 때문에 각 임베딩 별로 어텐션 마스크를 이용하여 패딩값을 제거한 후
                #활성화 값만 추출하여 계산하기 위한 절차이다

                # 각 batch의 어텐션 마스크(의미 있는 토큰 : 1/ 패딩 값 : 0)와 모델이 입력으로 받은 토큰 임베딩을 저장한 리스트의 길이가 같아야 한다
                # ex)text_atts_history=[어텐션1, 어텐션2, 어텐션3], module.input_history=[입력1, 입력2, 패딩1]
                assert len(text_atts_history) == len(module.input_history), \
                    "The number of text attentions and the number of input histories must be the same." \
                    "Instead got {} attentions and {} input histories.".format(len(text_atts_history), len(module.input_history))
                
                #어텐션 마크스와 임력 임베딩을 한 쌍으로 묶어서 하나씩 처리한다
                for index_in_history, (text_att, input_sample) in enumerate(zip(text_atts_history, module.input_history)):
                    # text_att is a tensor of size [B, L] telling which tokens are relevant for each sample of the current batch
                    # input_sample is a tensor of size [B, L, embed_size] containing the embedding of each token of each batch
                    # our goal is to remove from the second dimension of :input_sample: the tokens that are not relevant
                    #어텐션 마스크의 Batch크기와 입력 임베딩의 Batch 크기가 같아야 한다
                    #ex)text_att=[[1,1,0],[1,0,0]] => 배치 크기 : 2 , 시퀀스 길이 : 3
                    #   input_sample=[ => 배치 크기 : 2, 시퀀스 길이 : 3
                    #                   [[0.1,0.3],[0.2,0.3],[0.0,0.0]],
                    #                   [[0.2,0.4],[0.0,0.0],[0.0,0.0]]]
                    assert text_att.size()[0] == input_sample.size()[0], \
                        f"The text attentions and the input history must have the same batch size. Instead got {text_att.size()[0]} and {input_sample.size()[0]}." \
                        f"Please check your implementation."
                    
                    num_samples_to_add += input_sample.size()[0]
                    
                    # in cross attention layers inputs will have a shape defined by the number of image patches, so we must
                    # make sure to avoid filtering out those
                    # L_att = text_att.size()[-1] # 어텐션 마스크 시퀀스 길이
                    # L_seq = input_sample.size()[1] # 임력 임베딩 시퀀스 길이
                    # if L_att == L_seq:#시퀀스 길이가 일치할 때만 패딩값 제거
                    #     #text_att == 1인 토큰만 남기고, 0인 토큰은 제거
                    #     input_sample = input_sample[text_att.to(input_sample.device).squeeze() == 1, :] 

                    L_att = text_att.size(-1)            # [B, L]에서 L
                    L_seq = input_sample.size(1)         # [B, L, D]에서 L
                    B_in, _, D = input_sample.size()

                    # att / input_sample 디바이스 맞춰주고, batch 차원 유지
                    att = text_att.to(input_sample.device)
                    if att.dim() == 1:        # [L]인 경우는 [1, L]로 맞춰줌
                        att = att.unsqueeze(0)

                    assert att.size(0) == B_in, \
                        f"Batch mismatch: att={att.size(0)}, inp={B_in}"

                    if L_att == L_seq:
                        # att: [B, L] -> mask: [B, L]
                        mask = (att == 1)             # [B, L] bool
                        # boolean mask를 앞 두 차원에 적용하면 [N, D]로 flatten돼서 나옴
                        # (PyTorch가 (B, L, D)를 (B*L, D)처럼 보고 mask 적용)
                        input_sample = input_sample[mask]   # -> [N, D]
                    else:
                        # cross-attn처럼 길이가 다르면 그냥 flatten만
                        input_sample = input_sample.view(-1, D)

                    # modify the input sample s.t. it has shape (L, embed_size), where L is now the number of relevant tokens after filtering
                    #기존 3차원의 input_sample텐서를 패딩 제거 후 남은 (유효한 토큰의 개수)와 (임베딩 크기)로 이루어진 2차원의 텐서로 변환한다
                    module.input_history[index_in_history] = input_sample.view(-1, input_sample.size()[-1])
                
            # if the current layer is a vision one, there is no concept of image attention and we simply
            # reshape each input sample to have shape (L, embed_size), where L is the number image patches
            elif modality == "vision":# 현재 파라미터가 VISION 모달리티일 때
                for index_in_history, input_sample in enumerate(module.input_history):
                    # each vision input sample will have shape [B, P, embed_size], where P is the number of patches
                    #B : 배치 크기(이미지 개수), P : 패치 수(이미지 1개를 몇 개의 패치로 분할했는지), embed_size : 임베딩 크기(각 패치를 몇 차원 벡터로 표현됐는지)
                    num_samples_to_add += input_sample.size()[0]
                    #input_sample의 모양을 BxP, embed_size의 2차원 텐서 형태로 변환 -> 이미지 패치를 일렬로 나열된 벡터로 처리
                    #ex) (2,4,3): 2개 이미지를 각 4개의 패치로 분할함, 각 패치는 3차원 임베딩 -> (8,3) : 2개 이미지 x 4개의 패치, 각 패치는 3차원 임베딩
                    module.input_history[index_in_history] = input_sample.view(-1, input_sample.size()[-1])
            else:
                raise NotImplementedError("Modality {} not supported.".format(modality))

            # rescale the offloaded activations and update the number of samples to include in the reduction
            #활성화 norm 재조정 - 이미 계산된 Norm을 샘플 수에 대한 비율로 스케일 다운하여 샘플이 증가함에 따른 Norm값의 기여도 조정

            #self.actn_norms[id(param)] : 현재 활성화 norm / self.nsamples[id(param)] : 각 파라미터의 누적 활성화값 / 지금까지 누적된 데이터 샘플 수
            self.actn_norms[id(param)] *= self.nsamples[id(param)] / (self.nsamples[id(param)]+num_samples_to_add)
            #누적 샘플 수 업데이트 -> 기존 샘플 수에 새로 처리한 샘플 수를 더한다
            self.nsamples[id(param)] += num_samples_to_add

            # update the running norm vector
            #활성화 Norm이란 활성화 값의 크기를 나타내는 개념(절댓값, 제곱합..등)

            #파라미터별 입력 데이터가 저장된 리스트(패딩값 제외된 임베딩)를 하나의 큰 텐서로 병합하여 활성화 norm을 계산할 수 있게 한다
            X = torch.cat(module.input_history, dim=0).type(torch.float32)
            #torch.norm(X, p=2, dim=0) : L2 Norm을 사용하여 입력 데이터 X의 크기를 계산한다
            #/ self.nsamples[id(param)] : 총 샘플 수로 나누어 평균 활성화 norm을 계산한다
            #최종적으로 이전 활성화norm에 새로 계산된 norm을 더하여 최종 활성화 norm을 업데이트한다
            self.actn_M2[id(param)]  = self.actn_M2[id(param)].to(X.device) + (X * X).sum(dim=0)   # Σ x^2
            self.actn_cnt[id(param)] = int(self.actn_cnt[id(param)]) + int(X.size(0))  

            # when activation norms are computed, release the input history
            del module.input_history; del X # 파라미터별 입력임베딩 값을 저장한 X텐서 삭제하여 메모리에서 해제
            module.input_history = [] # 파라미터별 입력 임베딩을 저장한 input_history 리스트로 초기화(forward pass 시 새로운 활성화 값들을 다시 쌓을 수 있도록 준비)

    def _toggle_input_capture(self, enabled: bool, only_prunable: bool):
        def _ensure(module):
            # 속성/리스트 없으면 만들어 줌 (후킹 시 안전)
            if not hasattr(module, "store_input_flag"):
                setattr(module, "store_input_flag", False)
            if not hasattr(module, "input_history"):
                module.input_history = []
            module.store_input_flag = bool(enabled)
            print(f"[Debug] Setting store_input_flag of module {module} to {module.store_input_flag}")
        if only_prunable:
            print("[Debug] multiflow/pruners/ecoflap.py - _toggle_input_capture 함수 호출 : Prunable한 파라미터에 대해서만 활성화 값 수집 토글")
            for name, _, _ in self.named_masked_parameters:
                module = recursive_getattr(self.model, ".".join(name.split(".")[:-1]))
                _ensure(module)
        else:
            print("[Debug] multiflow/pruners/ecoflap.py - _toggle_input_capture 함수 호출 : 모델의 모든 모듈에 대해 활성화 값 수집 토글")
            for module in self.model.modules():
                _ensure(module)



    def wanda_S(self, param):
        pid  = id(param)
        cnt  = max(int(self.actn_cnt[pid]), 1)  # 0 division 방지
        a_in = torch.sqrt(self.actn_M2[pid] / cnt).to(param.device, dtype=param.dtype)  # (D_in,)
        return param.abs() * a_in.view(1, -1)  # (D_out, D_in)

    
    def score(self):
        print("[Debug] smooth.py -> score() : WANDA(S=|W|·sqrt(E[x^2])) 기반 점수 계산")
        for _, _, param in self.named_masked_parameters:
            pid = id(param)
            score = self.wanda_S(param)   #출력 뉴런과 입력 뉴런의 Norm값의 외적 결과에 가중치 행렬의 절댓값을 곱하여 최종 중요도 점수를 계산한다.

            self.scores[id(param)] = torch.clone(score).detach().cpu()
    

    def _parse_role(self, name: str) -> str:
        n = name.lower()
        if re.search(r"\.q(_proj)?\.", n): return "attn_q"
        if re.search(r"\.k(_proj)?\.", n): return "attn_k"
        if re.search(r"\.v(_proj)?\.", n): return "attn_v"
        if re.search(r"attn.*out(_proj)?", n): return "attn_out"
        if re.search(r"(fc1|mlp.*fc1)", n): return "mlp_fc1"
        if re.search(r"(fc2|mlp.*fc2)", n): return "mlp_fc2"
        return "other"

    def _build_mapping_from_prunable(self):
        mapping = {}
        for name, _, _ in self.named_masked_parameters:
            toks = name.split(".")
            # 블록 식별 (layers.<i> or layer.<i>)
            if "layers" in toks:
                bidx = toks.index("layers"); block = ".".join(toks[:bidx+2])  # ...layers.<i>
            elif "layer" in toks:
                bidx = toks.index("layer");  block = ".".join(toks[:bidx+2])  # ...layer.<i>
            else:
                block = toks[0]


            group = f"{block}.block_group"
            mapping[name] = group
        return mapping


    @torch.no_grad()
    def prune(self, target_sparsity, model, dataloader, device, fabric, num_batches_per_step, **kwargs):
        time_in = time.time()
        print("[Debug] ecoflap.py -> Prune() 실행 : pruning 시작 부분")
        # 0) 런타임 바인딩
        self.model = model
        self.data_loader = dataloader
        self.original_sparsity = float(target_sparsity)
        self.num_samples = int(kwargs.get('num_batches_per_step', num_batches_per_step))
        self._toggle_input_capture(False, only_prunable=False)
        # 손실 선택 (VLM 기준)
        self.loss_func = loss_vision_language


        # 1) 레이어→그룹 매핑(없으면 생성)
        self.layer_to_group_mapping = self._build_mapping_from_prunable()

        # 2) MeZO 중요도 (이름→tensor)
        self.importance_measure = self.compute_importance_scores_mezo(self.layer_to_group_mapping)

        # 3) 레이어별 sparsity 분배 (이름→s)
        name_sparsity = self.return_sparsity()  # 내부에서 self.importance_measure 사용
        # prune() 안에서 name_sparsity 만든 뒤:
        for name, _, _ in self.named_masked_parameters:
            if name not in name_sparsity:
                name_sparsity[name] = self.original_sparsity

        # 4) BasePruner의 compute_mask가 쓰는 self.scores 채우기 (pid→tensor)
    # # -------------------- Fine 시작 --------------------
    #     # 활성화 수집 준비
        self._toggle_input_capture(True, only_prunable=True)

        text_atts_history = []
        is_combined = ('region_loader' in kwargs) and (kwargs['region_loader'] is not None)
        loader = (CombinedLoader((dataloader, kwargs['region_loader']), mode="min_size")
                if is_combined else dataloader)

        for batch_idx, batch in enumerate(loader):
            if is_combined:
                general_batch, region_batch = batch
            else:
                general_batch = batch
                region_batch = None

            # VLM이면 text attn 마스크도 모아둠(패딩 영향 제거용)
            if hasattr(model, "is_vlm") and model.is_vlm:
                def _get_attn(b):
                    if isinstance(b, dict): return b.get('attention_mask', None)
                    if isinstance(b, (list, tuple)) and len(b) > 2: return b[2]
                    return None
                ta = _get_attn(general_batch)
                if ta is not None: text_atts_history.append(ta)

            # 일반(Out-of-domain)
            _ = self.forward_output(model, general_batch, device, modality="fusion")


            # 누적된 활성화의 norm을 집계하고 캐시 해제
            self._offload_actns(text_atts_history)
            text_atts_history = []

            # 배치 제한 있으면 끊기
            if (batch_idx + 1) % num_batches_per_step == 0:
                break

        # 정보-흐름 점수 계산 → self.scores 채워짐
        self.score()
        # -------------------- Fine 끝 --------------------

        # 4) 레이어별 sparsity를 pid로 매핑해서 마스크 생성(정보-흐름 self.scores 사용)
        pid_sparsity = self._names_to_pids(name_sparsity)
        self.compute_mask(pid_sparsity, scope="local")

        # 마스크 기반으로 최종 sparsity 계산
        name_final_sparsity = {}
        pid_final_sparsity  = {}
        log_lines = []
        tot_zeros, tot_elems = 0, 0

        for name, mask, p in self.named_masked_parameters:
            elems = mask.numel()
            kept  = int(mask.sum().item())   # 1=keep
            zeros = elems - kept             # 0=prune
            sp    = zeros / elems if elems else 0.0

            name_final_sparsity[name]  = float(sp)
            pid_final_sparsity[id(p)]  = float(sp)

            tot_zeros += zeros
            tot_elems += elems
            log_lines.append(
                f"layer={name:<80} pid={id(p)}  elements={elems}  kept={kept}  sparsity={sp:.6f}"
            )

        global_sparsity = (tot_zeros / tot_elems) if tot_elems else 0.0
        log_lines.append(
            f"\n[GLOBAL] effective_sparsity={global_sparsity:.6f}  (zeros={tot_zeros} / total={tot_elems})"
        )

        # 저장 경로
        output_dir = output_dir=str(Path(kwargs.get("output_dir", getattr(self, "output_dir", "."))).resolve())
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, "sparsity")


        with open(base, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")

        self._toggle_input_capture(False, only_prunable=False)       
        self.scoring_time = int(time.time() - time_in)
        print(f"Total pruning time (hh:mm:ss) = {datetime.timedelta(seconds=self.scoring_time)}")

    def reset(self):
        for _, mask, param in self.named_masked_parameters:
            mask.fill_(1)
            if mask.grad is not None:  mask.grad.zero_()
            if param.grad is not None: param.grad.zero_()
        self.scores = {}
        self.importance_measure = {}
        self.actn_M2  = {id(p): torch.zeros(p.size(1), dtype=torch.float32) for _, _, p in self.named_masked_parameters}
        self.actn_cnt = {id(p): 0 for _, _, p in self.named_masked_parameters}
