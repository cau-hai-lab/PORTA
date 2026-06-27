import time
import datetime
import torch
from functools import partial
from pathlib import Path
import os
import torch.nn.functional as F
import torch.nn as nn

from lightning.pytorch.utilities.combined_loader import CombinedLoader

# === multiflow 프레임워크 내 유틸/베이스 ===
from pruners.base import Pruner
from pruners.accumulators import forward_output, region_forward_output
from utils.prune_utils import make_prunable, check_blip_state_dict, recursive_getattr
from utils.functions import detect_modality_fn


def _sanitize(name: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

import re

_LAYER_PATTS = [
    re.compile(r'text_model\.encoder\.layers\.(\d+)\.'),   # HF CLIP 계열
    re.compile(r'transformer\.resblocks\.(\d+)\.'),        # OpenAI CLIP Big/G 계열
]
def _get_layer_idx_from_name(name: str):
    for p in _LAYER_PATTS:
        m = p.search(name)
        if m:
            return int(m.group(1))
    return None

def _make_mask(att, B, L, device, dtype):
    """att를 (B,L,1) float 마스크로 정규화. 실패 시 all-ones."""
    if att is None:
        return torch.ones(B, L, 1, device=device, dtype=dtype)
    # (B,L) or (B,L,1)만 허용
    if att.dim() == 2 and att.shape == (B, L):
        att = att.unsqueeze(-1)
    if att.dim() == 3 and att.shape[0] == B and att.shape[1] == L:
        return att.to(device=device, dtype=dtype, non_blocking=True)
    return torch.ones(B, L, 1, device=device, dtype=dtype)

def _select_att_for_module(name: str, text_atts_history):
    """모듈 이름의 레이어 idx에 맞는 att 1개만 선택(없으면 마지막/None)."""
    if text_atts_history is None:
        return None
    idx = _get_layer_idx_from_name(name)

    # dict: {layer_idx: att}
    if isinstance(text_atts_history, dict):
        if idx in text_atts_history:
            return text_atts_history[idx]
        # 폴백: 키 중 마지막
        if len(text_atts_history) > 0:
            return text_atts_history[sorted(text_atts_history.keys())[-1]]
        return None

    # list/tuple: [att_0, att_1, ...] 혹은 2스트림이면 [stream0, stream1] 같은 구조일 수 있음
    if isinstance(text_atts_history, (list, tuple)) and len(text_atts_history) > 0:
        # 2스트림 구조라면 첫 스트림 우선 (원하면 결합 로직으로 바꿔도 됨)
        cand = text_atts_history
        if len(cand) == 2 and isinstance(cand[0], (list, tuple, dict)):
            cand = cand[0]
        if isinstance(cand, dict):
            if idx in cand:
                return cand[idx]
            if len(cand) > 0:
                return cand[sorted(cand.keys())[-1]]
            return None
        if isinstance(cand, (list, tuple)):
            if idx is not None and 0 <= idx < len(cand):
                return cand[idx]
            return cand[-1]
    return None
class Varmin(Pruner):
    """
    심플/유니폼 버전:
      - score(): row_col_std_norm 기반 (행/열 std 통계로 만든 중요도 맵)
      - prune(): 전 레이어 동일 target_sparsity 적용 (레이어별 분배 없음)
      - BarlowTwins/Gram/분배 로직 모두 제거
    """
    def __init__(self, model, *args, **kwargs):
        print("<SmoothFlow(Simple-Uniform) :: __init__>")
        make_prunable(
            model,
            mask_dtype=torch.bool,
            pattern_lock=True,
            mask_on_the_fly=True,
            store_input=True,        # 입력 활성화 수집용
        )
        super().__init__(model, *args, **kwargs)

        # 바인딩: 외부 프레임워크 함수
        self.detect_modality_fn   = partial(detect_modality_fn, self.model.name)
        self.forward_output       = partial(forward_output, self.model.name)
        self.region_forward_output= partial(region_forward_output, self.model.name)

        # 메타
        self.name = "varmin"
        self.output_dir = str(Path(kwargs.get("output_dir", getattr(self, "output_dir", "."))).resolve())

        # 스코어 버퍼
        self.scores = {}
        self.scores_computed = False
        self.is_one_shot = True
        self.modifies_weights = False

        # 활성화 통계 버퍼 (열/행)
        # # 열(입력 채널, D_in) : 누적 카운트/평균/M2(Welford)
        # self.col_act_cnt  = {id(p): 0 for _, _, p in self.named_masked_parameters}
        # self.col_act_mean = {id(p): torch.zeros(p.size(1), dtype=torch.float32) for _, _, p in self.named_masked_parameters}
        # self.col_act_M2   = {id(p): torch.zeros(p.size(1), dtype=torch.float32) for _, _, p in self.named_masked_parameters}

        # # 행(토큰/패치 위치, L) : 가변 길이 → dict에 텐서로 저장 (필요 시 확장)
        # self.row_act_cnt  = {}
        # self.row_act_mean = {}
        # self.row_act_M2   = {}
        self.col_act_cnt = {}   # {pid: scalar float32 on GPU}      ┐ 열(채널) 측 가중 샘플 수
        self.col_ss      = {}   # {pid: (D,) float32 on GPU}         ┘ 열(채널) 측 제곱합

        self.row_act_cnt = {}   # {pid: (L,) float32 on GPU}         ┐ 행(위치) 측 가중 샘플 수
        self.row_ss      = {}  
        # 기타
        self.init_weights = {id(p): torch.clone(p.data).detach().cpu() for _, _, p in self.named_masked_parameters}
        self.actn_norms   = {id(p): torch.ones(p.size(1), dtype=torch.float32) for _, _, p in self.named_masked_parameters}
        self.prune_modalities = set(
            kwargs.get("prune_modalities", ("vision", "text", "fusion"))
        )

    # --------------------------
    # 통계 버퍼 보조
    # --------------------------
    def _ensure_row_buffers(self, pid, length, device, dtype=torch.float32):
        """
        행(위치 L) 통계 버퍼를 (필요 시) 확장.
        """
        if pid not in self.row_act_cnt:
            self.row_act_cnt[pid]  = torch.zeros(length, dtype=torch.long,  device=device)
            self.row_act_mean[pid] = torch.zeros(length, dtype=dtype,       device=device)
            self.row_act_M2[pid]   = torch.zeros(length, dtype=dtype,       device=device)
        else:
            cur = self.row_act_mean[pid].numel()
            if length > cur:
                pad = (0, length - cur)
                self.row_act_cnt[pid]  = F.pad(self.row_act_cnt[pid],  pad).to(device)
                self.row_act_mean[pid] = F.pad(self.row_act_mean[pid], pad).to(device)
                self.row_act_M2[pid]   = F.pad(self.row_act_M2[pid],   pad).to(device)

    # --------------------------
    # 입력 저장 플래그 세팅
    # --------------------------
    def _flag_norms(self):
        print("[Debug] _flag_norms(): only prunable layers store input")
        for module in self.model.modules():
            if hasattr(module, "store_input_flag"):
                module.store_input_flag = False
        for name, _, _ in self.named_masked_parameters:
            module_name = ".".join(name.split(".")[:-1])
            module = recursive_getattr(self.model, module_name)
            if hasattr(module, "store_input_flag"):
                module.store_input_flag = True

    # --------------------------
    # 활성화 통계 누적
    # --------------------------
    # @torch.no_grad()
    # def _offload_actns(self, text_atts_history):
    #     """
    #     prunable 파라미터별로 module.input_history에 쌓인 입력 x를 꺼내
    #     - 열(D_in) 방향 통계: (B, L/P, D)에서 (0,1)축 합쳐 (D,) Welford 병합
    #     - 행(위치) 방향 통계: 채널 RMS 스칼라로 (B, L/P) 만들고 위치별 Welford 병합
    #     끝나면 input_history 메모리 해제
    #     """
    #     print("[Debug] _offload_actns: update row/col stats")
    #     for name, _, param in self.named_masked_parameters:
    #         pid      = id(param)
    #         mname    = ".".join(name.split(".")[:-1])
    #         module   = recursive_getattr(self.model, mname)
    #         modality = self.detect_modality_fn(name)

    #         inputs = getattr(module, "input_history", None)
    #         if not inputs:
    #             continue

    #         if modality in ("text", "fusion"):
    #             assert len(text_atts_history) == len(inputs), \
    #                 f"Mismatch: {len(text_atts_history)} attentions vs {len(inputs)} input histories"

    #             for idx, (att, x) in enumerate(zip(text_atts_history, inputs)):
    #                 if x is None:
    #                     continue

    #                 # x: (B, L, D)
    #                 x = x.detach().to(torch.float32)
    #                 B, L, D = x.shape

    #                 # attention mask → (B, L, 1) (불일치면 ones)
    #                 if att.size(0) == B and att.size(-1) == L:
    #                     m = att.to(x.device).unsqueeze(-1).to(x.dtype)  # (B,L,1)
    #                 else:
    #                     m = torch.ones(B, L, 1, device=x.device, dtype=x.dtype)

    #                 # ---- 열(D) 방향 병합 ----
    #                 sum1_d  = (x * m).sum(dim=(0, 1))      # (D,)
    #                 sumsq_d = (x * x * m).sum(dim=(0, 1))  # (D,)
    #                 T       = int(m.sum().item())          # scalar (#유효 토큰)

    #                 if T > 0:
    #                     col_n    = self.col_act_cnt[pid]
    #                     col_mean = self.col_act_mean[pid].to(x.device)
    #                     col_M2   = self.col_act_M2[pid].to(x.device)

    #                     b_mean_d = sum1_d / T
    #                     b_M2_d   = sumsq_d - T * (b_mean_d * b_mean_d)

    #                     delta    = b_mean_d - col_mean
    #                     new_n    = col_n + T
    #                     col_mean = col_mean + delta * (T / max(new_n, 1))
    #                     col_M2   = col_M2 + b_M2_d + (delta * delta) * (col_n * T / max(new_n, 1))

    #                     self.col_act_cnt[pid]  = int(new_n)
    #                     self.col_act_mean[pid] = col_mean.detach().to("cpu")
    #                     self.col_act_M2[pid]   = col_M2.detach().to("cpu")

    #                 # ---- 행(L) 방향 병합: 채널 RMS ----
    #                 pos_val   = x.pow(2).mean(dim=2).sqrt()    # (B,L)
    #                 m2        = m.squeeze(-1)                  # (B,L)
    #                 sum1_pos  = (pos_val * m2).sum(dim=0)      # (L,)
    #                 sumsq_pos = ((pos_val ** 2) * m2).sum(dim=0)# (L,)
    #                 T_pos     = m2.sum(dim=0).to(device=x.device, dtype=torch.long)  # (L,)

    #                 self._ensure_row_buffers(pid, length=L, device=x.device)

    #                 row_n    = self.row_act_cnt[pid].to(x.device)     # (L,)
    #                 row_mean = self.row_act_mean[pid].to(x.device)     # (L,)
    #                 row_M2   = self.row_act_M2[pid].to(x.device)       # (L,)

    #                 nz = T_pos > 0
    #                 if nz.any():
    #                     b_mean_pos = torch.zeros_like(row_mean)
    #                     b_M2_pos   = torch.zeros_like(row_M2)
    #                     b_mean_pos[nz] = sum1_pos[nz] / T_pos[nz].to(torch.float32)
    #                     b_M2_pos[nz]   = sumsq_pos[nz].to(torch.float32) - T_pos[nz].to(torch.float32) * (b_mean_pos[nz] ** 2)

    #                     delta_pos = b_mean_pos - row_mean
    #                     new_n_pos = row_n + T_pos

    #                     row_mean = row_mean + delta_pos * (T_pos.to(torch.float32) / new_n_pos.clamp_min(1).to(torch.float32))
    #                     row_M2   = row_M2   + b_M2_pos + (delta_pos * delta_pos) * (
    #                         row_n.to(torch.float32) * T_pos.to(torch.float32) / new_n_pos.clamp_min(1).to(torch.float32)
    #                     )
    #                     row_n    = new_n_pos

    #                     self.row_act_cnt[pid]  = row_n.detach().to("cpu")
    #                     self.row_act_mean[pid] = row_mean.detach().to("cpu")
    #                     self.row_act_M2[pid]   = row_M2.detach().to("cpu")

    #                 inputs[idx] = None

    #         elif modality == "vision":
    #             for idx, x in enumerate(inputs):
    #                 if x is None:
    #                     continue

    #                 # x: (B, P, D)
    #                 x = x.detach().to(torch.float32)
    #                 B, P, D = x.shape
    #                 m = torch.ones(B, P, 1, device=x.device, dtype=x.dtype)

    #                 # ---- 열(D) 방향 병합 ----
    #                 sum1_d  = (x * m).sum(dim=(0, 1))      # (D,)
    #                 sumsq_d = (x * x * m).sum(dim=(0, 1))  # (D,)
    #                 T       = int(m.sum().item())

    #                 if T > 0:
    #                     col_n    = self.col_act_cnt[pid]
    #                     col_mean = self.col_act_mean[pid].to(x.device)
    #                     col_M2   = self.col_act_M2[pid].to(x.device)

    #                     b_mean_d = sum1_d / T
    #                     b_M2_d   = sumsq_d - T * (b_mean_d * b_mean_d)

    #                     delta    = b_mean_d - col_mean
    #                     new_n    = col_n + T
    #                     col_mean = col_mean + delta * (T / max(new_n, 1))
    #                     col_M2   = col_M2 + b_M2_d + (delta * delta) * (col_n * T / max(new_n, 1))

    #                     self.col_act_cnt[pid]  = int(new_n)
    #                     self.col_act_mean[pid] = col_mean.detach().to("cpu")
    #                     self.col_act_M2[pid]   = col_M2.detach().to("cpu")

    #                 # ---- 행(P) 방향 병합: 채널 RMS ----
    #                 pos_val   = x.pow(2).mean(dim=2).sqrt()  # (B,P)
    #                 m2        = torch.ones_like(pos_val)     # (B,P)
    #                 sum1_pos  = (pos_val * m2).sum(dim=0)    # (P,)
    #                 sumsq_pos = ((pos_val ** 2) * m2).sum(dim=0)
    #                 T_pos     = m2.sum(dim=0).to(device=x.device, dtype=torch.long)

    #                 self._ensure_row_buffers(pid, length=P, device=x.device)

    #                 row_n    = self.row_act_cnt[pid].to(x.device)
    #                 row_mean = self.row_act_mean[pid].to(x.device)
    #                 row_M2   = self.row_act_M2[pid].to(x.device)

    #                 b_mean_pos = sum1_pos / T_pos.to(torch.float32)
    #                 b_M2_pos   = sumsq_pos.to(torch.float32) - T_pos.to(torch.float32) * (b_mean_pos ** 2)

    #                 delta_pos = b_mean_pos - row_mean
    #                 new_n_pos = row_n + T_pos

    #                 row_mean = row_mean + delta_pos * (T_pos.to(torch.float32) / new_n_pos.clamp_min(1).to(torch.float32))
    #                 row_M2   = row_M2   + b_M2_pos + (delta_pos * delta_pos) * (
    #                     row_n.to(torch.float32) * T_pos.to(torch.float32) / new_n_pos.clamp_min(1).to(torch.float32)
    #                 )
    #                 row_n    = new_n_pos

    #                 self.row_act_cnt[pid]  = row_n.detach().to("cpu")
    #                 self.row_act_mean[pid] = row_mean.detach().to("cpu")
    #                 self.row_act_M2[pid]   = row_M2.detach().to("cpu")

    #                 inputs[idx] = None

    #         else:
    #             raise NotImplementedError(f"Modality {modality} not supported.")

    #         # 메모리 해제
    #         del module.input_history
    #         module.input_history = []


    def _get_layer_idx_from_name(self, name: str):
            for p in _LAYER_PATTS:
                m = p.search(name)
                if m:
                    return int(m.group(1))
            return None

    def _select_att_for_module(self, name: str, text_atts_history):
        """모듈 이름에서 레이어 idx를 뽑아 해당 attention을 선택.
        - dict면 dict[idx]
        - list면 list[idx], 없으면 마지막
        - 비어있으면 None
        """
        if text_atts_history is None:
            return None
        idx = self._get_layer_idx_from_name(name)
        if isinstance(text_atts_history, dict):
            return text_atts_history.get(idx, None)
        if isinstance(text_atts_history, (list, tuple)):
            if idx is not None and 0 <= idx < len(text_atts_history):
                return text_atts_history[idx]
            return text_atts_history[-1] if len(text_atts_history) > 0 else None
        return None


    # @torch.inference_mode()
    # def _offload_actns(self, text_atts_history):
    #     """
    #     - 통계 버퍼를 GPU에 상주시킴 (왕복 제거)
    #     - .item() 사용 금지 (동기화 제거)
    #     - 텐서 카운터/분기 최소화
    #     """
    #     print("[Debug] _offload_actns(GPU-only): col(D_in) & row(L/P) stats update")

    #     def _ensure_col(pid, D, device):
    #         # GPU 상주 보장
    #         if pid not in self.col_act_cnt:
    #             self.col_act_cnt[pid]  = torch.zeros((), dtype=torch.long, device=device)
    #             self.col_act_mean[pid] = torch.zeros(D, dtype=torch.float32, device=device)
    #             self.col_act_M2[pid]   = torch.zeros(D, dtype=torch.float32, device=device)
    #         else:
    #             # 한 번만 GPU로 올려두고 유지
    #             cnt = self.col_act_cnt[pid]
    #             if not torch.is_tensor(cnt):
    #                 cnt = torch.tensor(int(cnt), dtype=torch.long, device=device)
    #             else:
    #                 cnt = cnt.to(device, non_blocking=True)
    #             self.col_act_cnt[pid]  = cnt
    #             self.col_act_mean[pid] = self.col_act_mean[pid].to(device, non_blocking=True)
    #             self.col_act_M2[pid]   = self.col_act_M2[pid].to(device, non_blocking=True)

    #     def _ensure_row(pid, L, device):
    #         # 네가 쓰는 self._ensure_row_buffers를 그대로 호출하되 device를 GPU로
    #         self._ensure_row_buffers(pid, length=L, device=device)
    #         # 보장: row 버퍼도 GPU에 상주
    #         self.row_act_cnt[pid]  = self.row_act_cnt[pid].to(device, non_blocking=True)
    #         self.row_act_mean[pid] = self.row_act_mean[pid].to(device, non_blocking=True)
    #         self.row_act_M2[pid]   = self.row_act_M2[pid].to(device, non_blocking=True)

    #     for name, _, param in self.named_masked_parameters:
    #         pid      = id(param)
    #         mname    = ".".join(name.split(".")[:-1])
    #         module   = recursive_getattr(self.model, mname)
    #         modality = self.detect_modality_fn(name)

    #         inputs = getattr(module, "input_history", None)
    #         if not inputs:
    #             continue

    #         if modality in ("text", "fusion"):
    #             att_for_module = _select_att_for_module(name, text_atts_history)

    #             for idx, x in enumerate(inputs):
    #                 if x is None:
    #                     continue

    #                 x = x.detach().to(torch.float32)   # (B, L, D)
    #                 B, L, D = x.shape
    #                 device = x.device

    #                 # GPU 상주 버퍼 보장 (너가 만든 함수/로직 그대로)
    #                 _ensure_col(pid, D, device)
    #                 _ensure_row(pid, L, device)

    #                 # (B,L,1) 마스크 생성 (모양 안 맞으면 all-ones)
    #                 m = _make_mask(att_for_module, B, L, device, torch.float32)

    #                 # ------- 열(채널 D) : Welford -------
    #                 sum1_d  = (x * m).sum(dim=(0, 1))            # (D,)
    #                 sumsq_d = (x * x * m).sum(dim=(0, 1))        # (D,)
    #                 T       = m.sum().to(dtype=torch.long)       # ()

    #                 col_n    = self.col_act_cnt[pid]             # ()
    #                 col_mean = self.col_act_mean[pid]            # (D,)
    #                 col_M2   = self.col_act_M2[pid]              # (D,)

    #                 T_f      = T.to(torch.float32)
    #                 new_n    = col_n + T
    #                 new_n_f  = new_n.clamp_min(1).to(torch.float32)

    #                 b_mean_d = sum1_d / T_f.clamp_min(1)
    #                 b_M2_d   = sumsq_d - T_f * (b_mean_d * b_mean_d)

    #                 delta    = b_mean_d - col_mean
    #                 col_mean = col_mean + delta * (T_f / new_n_f)
    #                 col_M2   = col_M2 + b_M2_d + (delta * delta) * (col_n.to(torch.float32) * T_f / new_n_f)

    #                 self.col_act_cnt[pid]  = new_n
    #                 self.col_act_mean[pid] = col_mean
    #                 self.col_act_M2[pid]   = col_M2

    #                 # ------- 행(위치 L) : 채널 RMS 스칼라 Welford -------
    #                 pos_val   = x.pow(2).mean(dim=2).sqrt()      # (B,L)  # 더 빠르게 하고 싶으면 sqrt 지연
    #                 m2        = m.squeeze(-1)                    # (B,L)
    #                 sum1_pos  = (pos_val * m2).sum(dim=0)        # (L,)
    #                 sumsq_pos = ((pos_val ** 2) * m2).sum(dim=0) # (L,)
    #                 T_pos     = m2.sum(dim=0).to(dtype=torch.long)

    #                 row_n    = self.row_act_cnt[pid]
    #                 row_mean = self.row_act_mean[pid]
    #                 row_M2   = self.row_act_M2[pid]

    #                 T_pos_f    = T_pos.to(torch.float32)
    #                 new_n_pos  = row_n + T_pos
    #                 new_n_posf = new_n_pos.clamp_min(1).to(torch.float32)

    #                 b_mean_pos = sum1_pos / T_pos_f.clamp_min(1)
    #                 b_M2_pos   = sumsq_pos - T_pos_f * (b_mean_pos ** 2)

    #                 delta_pos = b_mean_pos - row_mean
    #                 row_mean  = row_mean + delta_pos * (T_pos_f / new_n_posf)
    #                 row_M2    = row_M2 + b_M2_pos + (delta_pos * delta_pos) * (
    #                                 row_n.to(torch.float32) * T_pos_f / new_n_posf
    #                             )
    #                 row_n     = new_n_pos

    #                 self.row_act_cnt[pid]  = row_n
    #                 self.row_act_mean[pid] = row_mean
    #                 self.row_act_M2[pid]   = row_M2

    #                 # 즉시 해제
    #                 inputs[idx] = None

    #         elif modality == "vision":
    #             for idx, x in enumerate(inputs):
    #                 if x is None:
    #                     continue

    #                 # x: (B, P, D)
    #                 x = x.detach().to(torch.float32)
    #                 device = x.device
    #                 B, P, D = x.shape

    #                 _ensure_col(pid, D, device)
    #                 _ensure_row(pid, P, device)

    #                 m = torch.ones(B, P, 1, device=device, dtype=torch.float32)

    #                 # ------- 열(채널 D) -------
    #                 sum1_d  = (x * m).sum(dim=(0, 1))              # (D,)
    #                 sumsq_d = (x * x * m).sum(dim=(0, 1))          # (D,)
    #                 T       = m.sum().to(dtype=torch.long)         # scalar tensor

    #                 col_n    = self.col_act_cnt[pid]
    #                 col_mean = self.col_act_mean[pid]
    #                 col_M2   = self.col_act_M2[pid]

    #                 T_f   = T.to(torch.float32)
    #                 new_n = col_n + T
    #                 new_n_f = new_n.clamp_min(1).to(torch.float32)

    #                 b_mean_d = sum1_d / T_f.clamp_min(1)
    #                 b_M2_d   = sumsq_d - T_f * (b_mean_d * b_mean_d)

    #                 delta    = b_mean_d - col_mean
    #                 col_mean = col_mean + delta * (T_f / new_n_f)
    #                 col_M2   = col_M2 + b_M2_d + (delta * delta) * (col_n.to(torch.float32) * T_f / new_n_f)

    #                 self.col_act_cnt[pid]  = new_n
    #                 self.col_act_mean[pid] = col_mean
    #                 self.col_act_M2[pid]   = col_M2

    #                 # ------- 행(위치 P) — 채널 RMS -------
    #                 pos_val   = x.pow(2).mean(dim=2).sqrt()        # (B,P)
    #                 m2        = torch.ones_like(pos_val)
    #                 sum1_pos  = (pos_val * m2).sum(dim=0)          # (P,)
    #                 sumsq_pos = ((pos_val ** 2) * m2).sum(dim=0)   # (P,)
    #                 T_pos     = m2.sum(dim=0).to(dtype=torch.long) # (P,)

    #                 row_n    = self.row_act_cnt[pid]
    #                 row_mean = self.row_act_mean[pid]
    #                 row_M2   = self.row_act_M2[pid]

    #                 T_pos_f    = T_pos.to(torch.float32)
    #                 new_n_pos  = row_n + T_pos
    #                 new_n_posf = new_n_pos.clamp_min(1).to(torch.float32)

    #                 b_mean_pos = sum1_pos / T_pos_f.clamp_min(1)
    #                 b_M2_pos   = sumsq_pos - T_pos_f * (b_mean_pos ** 2)

    #                 delta_pos = b_mean_pos - row_mean
    #                 row_mean  = row_mean + delta_pos * (T_pos_f / new_n_posf)
    #                 row_M2    = row_M2 + b_M2_pos + (delta_pos * delta_pos) * (
    #                                 row_n.to(torch.float32) * T_pos_f / new_n_posf
    #                             )
    #                 row_n     = new_n_pos

    #                 self.row_act_cnt[pid]  = row_n
    #                 self.row_act_mean[pid] = row_mean
    #                 self.row_act_M2[pid]   = row_M2

    #                 inputs[idx] = None

    #         else:
    #             raise NotImplementedError(f"Modality {modality} not supported.")

    #         # 모듈 입력 히스토리 비우기(메모리 회수)
    #         del module.input_history
    #         module.input_history = []

# ===== 2) mean=0 근사 전용 _offload_actns =====
    @torch.inference_mode()
    def _offload_actns(self, text_atts_history):
        """
        mean=0 근사: 분산 ≈ E[x^2]
        - 열(채널 D): col_ss(D), col_act_cnt(scalar) 만 누적
        - 행(위치 L/P): row_ss(L/P), row_act_cnt(L/P) 만 누적
        - 마스크가 연속 가중(0..1)이어도 정확히 반영 (정수화 X)
        - CPU 동기화 유발 .item()/int() 미사용
        """
        print("[Debug] _offload_actns(mean=0): col(D_in) & row(L/P) stats (GPU-only)")

        def _ensure_col(pid, D, device):
            # 열 버퍼 준비 (shape 변화 시 안전 갱신)
            if pid not in self.col_act_cnt:
                self.col_act_cnt[pid] = torch.zeros((), dtype=torch.float32, device=device)
                self.col_ss[pid]      = torch.zeros(D,  dtype=torch.float32, device=device)
            else:
                self.col_act_cnt[pid] = self.col_act_cnt[pid].to(device, dtype=torch.float32)
                if self.col_ss[pid].device != device or self.col_ss[pid].dtype != torch.float32:
                    self.col_ss[pid] = self.col_ss[pid].to(device, dtype=torch.float32)
                if self.col_ss[pid].numel() != D:
                    # D 변경 시 보존 가능한 부분만 복사
                    old = self.col_ss[pid]
                    new = torch.zeros(D, dtype=torch.float32, device=device)
                    keep = min(old.numel(), D)
                    if keep > 0:
                        new[:keep].copy_(old[:keep])
                    self.col_ss[pid] = new

        def _ensure_row(pid, L, device):
            # 행 버퍼 준비 (길이 변동에 안전)
            if pid not in self.row_act_cnt:
                self.row_act_cnt[pid] = torch.zeros(L, dtype=torch.float32, device=device)
                self.row_ss[pid]      = torch.zeros(L, dtype=torch.float32, device=device)
            else:
                self.row_act_cnt[pid] = self.row_act_cnt[pid].to(device, dtype=torch.float32)
                self.row_ss[pid]      = self.row_ss[pid].to(device, dtype=torch.float32)
                if self.row_act_cnt[pid].numel() != L:
                    old_cnt = self.row_act_cnt[pid]; old_ss = self.row_ss[pid]
                    new_cnt = torch.zeros(L, dtype=torch.float32, device=device)
                    new_ss  = torch.zeros(L, dtype=torch.float32, device=device)
                    keep = min(old_cnt.numel(), L)
                    if keep > 0:
                        new_cnt[:keep].copy_(old_cnt[:keep])
                        new_ss[:keep].copy_(old_ss[:keep])
                    self.row_act_cnt[pid] = new_cnt
                    self.row_ss[pid]      = new_ss

        for name, _, param in self.named_masked_parameters:
            pid      = id(param)
            mname    = ".".join(name.split(".")[:-1])
            module   = recursive_getattr(self.model, mname)
            modality = self.detect_modality_fn(name)

            inputs = getattr(module, "input_history", None)
            if not inputs:
                continue

            if modality in ("text", "fusion"):
                att_for_module = _select_att_for_module(name, text_atts_history)

                for idx, x in enumerate(inputs):
                    if x is None:
                        continue

                    x = x.detach().to(torch.float32)  # (B, L, D) or (B, D)
                    if x.ndim == 3:
                        B, L, D = x.shape
                    elif x.ndim == 2:
                        B, D = x.shape
                        L = 1
                        x = x.unsqueeze(1)
                    else:
                        inputs[idx] = None
                        continue
                    device  = x.device

                    _ensure_col(pid, D, device)
                    _ensure_row(pid, L, device)

                    # (B,L,1) 마스크(연속 가중 가능). 없으면 ones
                    m  = _make_mask(att_for_module, B, L, device, torch.float32)  # (B,L,1)
                    m2 = m.squeeze(-1)                                            # (B,L)

                    # ---- 열(채널 D): 제곱합 & 가중 카운트(스칼라) ----
                    x2 = x * x                                                    # (B,L,D)
                    sumsq_d = (x2 * m).sum(dim=(0, 1))                            # (D,)
                    T_f     = m.sum().to(torch.float32)                           # scalar

                    self.col_ss[pid]      = self.col_ss[pid] + sumsq_d
                    self.col_act_cnt[pid] = self.col_act_cnt[pid] + T_f

                    # ---- 행(위치 L): mean(x^2) 누적(=pos_val^2) ----
                    ms       = x2.mean(dim=2)                                     # (B,L)  == (RMS)^2
                    sum_ms   = (ms * m2).sum(dim=0)                               # (L,)
                    T_pos_f  = m2.sum(dim=0).to(torch.float32)                    # (L,)

                    self.row_ss[pid]      = self.row_ss[pid] + sum_ms
                    self.row_act_cnt[pid] = self.row_act_cnt[pid] + T_pos_f

                    # 즉시 해제
                    inputs[idx] = None

            elif modality == "vision":
                for idx, x in enumerate(inputs):
                    if x is None:
                        continue

                    # x: (B, P, D) or (B, D)
                    x = x.detach().to(torch.float32)
                    device = x.device
                    if x.ndim == 3:
                        B, P, D = x.shape
                    elif x.ndim == 2:
                        B, D = x.shape
                        P = 1
                        x = x.unsqueeze(1)
                    else:
                        inputs[idx] = None
                        continue

                    _ensure_col(pid, D, device)
                    _ensure_row(pid, P, device)

                    # 비전은 마스크 전체 1
                    m  = torch.ones(B, P, 1, device=device, dtype=torch.float32)  # (B,P,1)
                    m2 = m.squeeze(-1)                                            # (B,P)

                    x2 = x * x
                    sumsq_d = (x2 * m).sum(dim=(0, 1))                            # (D,)
                    T_f     = m.sum().to(torch.float32)                           # scalar

                    self.col_ss[pid]      = self.col_ss[pid] + sumsq_d
                    self.col_act_cnt[pid] = self.col_act_cnt[pid] + T_f

                    ms      = x2.mean(dim=2)                                      # (B,P)
                    sum_ms  = (ms * m2).sum(dim=0)                                 # (P,)
                    T_pos_f = m2.sum(dim=0).to(torch.float32)                      # (P,)

                    self.row_ss[pid]      = self.row_ss[pid] + sum_ms
                    self.row_act_cnt[pid] = self.row_act_cnt[pid] + T_pos_f

                    inputs[idx] = None
            else:
                raise NotImplementedError(f"Modality {modality} not supported.")

            # 모듈 입력 히스토리 정리
            del module.input_history
            module.input_history = []


    # --------------------------
    # 스코어: row_col_std_norm
    # --------------------------
    # def row_col_std_norm(self, param, eps: float = 1e-5):
    #     """
    #     논리:
    #       - 행(위치) std(L,) → 위치별 std를 가중 평균해 스칼라 Rp (위치 스케일)
    #       - 열(입력채널) std(D,) → 열 std 벡터
    #       - std_norm_vec(D,) = Rp * std_in(D,)
    #       - score(D_out, D_in) = |W| * std_norm_vec (열방향 브로드캐스트)
    #     """
    #     Wabs32 = param.detach().to(torch.float32).abs()
    #     pid    = id(param)
    #     device = Wabs32.device

    #     # 행 std (L,)
    #     row_M2 = self.row_act_M2.get(pid, None)
    #     row_n  = self.row_act_cnt.get(pid, None)
    #     if row_M2 is None or row_n is None:
    #         # 통계 없으면 기본 |W| 반환
    #         return Wabs32.to(param.dtype)

    #     row_M2 = row_M2.to(device).to(torch.float32)
    #     row_n  = row_n.to(device)
    #     row_var = (row_M2 / row_n.clamp_min(1).to(torch.float32)).clamp_min(0.0)
    #     row_std = torch.sqrt(row_var + eps)  # (L,)

    #     # 위치 가중치: 유효 토큰 비율
    #     w = (row_n.to(torch.float32) / row_n.sum().clamp_min(1).to(torch.float32))  # (L,)
    #     Rp = torch.sqrt((w * (row_std ** 2)).sum().clamp_min(0.0))                  # scalar

    #     # 열 std (D,)
    #     col_M2 = self.col_act_M2.get(pid, None)
    #     col_n  = int(self.col_act_cnt.get(pid, 0))
    #     if col_M2 is None or col_n <= 0:
    #         return Wabs32.to(param.dtype)

    #     col_M2  = col_M2.to(device).to(torch.float32)
    #     col_var = (col_M2 / float(col_n)).clamp_min(0.0)
    #     col_std = torch.sqrt(col_var + eps)  # (D,)

    #     std_norm_vec = (Rp * col_std).to(param.dtype)  # (D,)
    #     return (Wabs32 * std_norm_vec).to(param.dtype) # (D_out, D_in)

    # ===== 3) mean=0 근사 전용 row_col_std_norm =====
    def row_col_std_norm(self, param, eps: float = 1e-5):
        """
        score = |W| * (Rp * col_std)
        - col_std(D) = sqrt( col_ss / col_cnt + eps )
        - row_std(L) = sqrt( row_ss / row_cnt + eps )
        - w(L) = row_cnt / sum(row_cnt)
        - Rp = sqrt( sum( w * row_std^2 ) )
        """
        Wabs32 = param.detach().to(torch.float32).abs()
        pid    = id(param)
        device = Wabs32.device

        # ---- 행(위치) 통계 (필수) ----
        row_ss  = self.row_ss.get(pid, None)
        row_cnt = self.row_act_cnt.get(pid, None)
        if row_ss is None or row_cnt is None:
            return Wabs32.to(param.dtype)

        row_ss  = row_ss.to(device, dtype=torch.float32)
        row_cnt = row_cnt.to(device, dtype=torch.float32)
        row_var = (row_ss / row_cnt.clamp_min(1.0)).clamp_min(0.0)     # (L,)
        row_std = torch.sqrt(row_var + eps)                             # (L,)

        w   = row_cnt / row_cnt.sum().clamp_min(1.0)                    # (L,)
        Rp  = torch.sqrt((w * (row_std ** 2)).sum().clamp_min(0.0))     # scalar

        # ---- 열(채널) 통계 (필수) ----
        col_ss  = self.col_ss.get(pid, None)
        col_cnt = self.col_act_cnt.get(pid, None)
        if col_ss is None or col_cnt is None:
            return Wabs32.to(param.dtype)

        col_ss  = col_ss.to(device, dtype=torch.float32)                # (D,)
        col_cnt = col_cnt.to(device, dtype=torch.float32)               # scalar
        col_var = (col_ss / col_cnt.clamp_min(1.0)).clamp_min(0.0)      # (D,)
        col_std = torch.sqrt(col_var + eps)                             # (D,)

        std_norm_vec = (Rp * col_std).to(param.dtype)                   # (D,)
        return (Wabs32 * std_norm_vec).to(param.dtype)                  # (D_out, D_in)


    @torch.no_grad()
    def score(self, eps: float = 1e-5):
        """
        row_col_std_norm(param) → (D_out, D_in) 스코어 행렬을 그대로 사용.
        """
        print("[Debug] score(): row_col_std_norm")
        for name, _, param in self.named_masked_parameters:
            pid = id(param)
            if param.ndim != 2:
                self.scores[pid] = torch.zeros_like(param, dtype=torch.float32, device='cpu')
                continue
            Wb = self.row_col_std_norm(param, eps=eps)     # (D_out, D_in)
            score = Wb.abs().to(torch.float32).detach().cpu()
            self.scores[pid] = score



    # --------------------------
    # prune(): 유니폼 분배
    # --------------------------
    @torch.no_grad()
    def prune(self, target_sparsity, model, dataloader, device, fabric,
              num_batches_per_step, **kwargs):
        """
        - 배치별로 입력 캡처/오프로딩으로 통계 누적
        - 마지막 스텝에 score() 호출 → 모든 pid에 동일 sparsity 적용
        """
        t0 = time.time()
        print("[Debug] prune(): Simple-Uniform start")

        # region_loader 포함여부
        is_combined = ('region_loader' in kwargs) and (kwargs['region_loader'] is not None)
        if is_combined:
            dataloader = CombinedLoader((dataloader, kwargs['region_loader']), mode="min_size")

        # 입력 저장 플래그
        self._flag_norms()

        print(f"[Debug] DataLoader length = {len(dataloader)}")
        if len(dataloader) == 0:
            raise RuntimeError("Empty dataloader")

        # (선택) 캡처 스텝: 통계를 충분히 모으기 위한 체크포인트 배치
        CAPTURE_STEPS = {0, max(0, num_batches_per_step // 2), max(0, num_batches_per_step - 1)}

        for batch_idx, batch in enumerate(dataloader):
            print(f"[{self.name.upper()}] Processing batch {batch_idx%num_batches_per_step+1}/{num_batches_per_step}",
                  end="\r", flush=True)

            # 배치 언팩
            if is_combined:
                general_batch, region_batch = batch
            else:
                general_batch = batch

            # 텍스트 마스크 보관(있는 경우)
            text_atts_history = []
            if hasattr(model, "is_vlm") and model.is_vlm:
                text_att = general_batch['attention_mask'] if isinstance(general_batch, dict) and 'attention_mask' in general_batch \
                           else (general_batch[2].clone() if not isinstance(general_batch, dict) and len(general_batch) > 2 else None)
                if text_att is not None:
                    text_atts_history.append(text_att)

            # forward (입력 히스토리 저장)
            _ = self.forward_output(model, general_batch, device, modality="fusion")

            # if is_combined:
            #     if hasattr(model, "is_vlm") and model.is_vlm:
            #         text_att2 = region_batch['attention_mask'] if isinstance(region_batch, dict) and 'attention_mask' in region_batch \
            #                     else (region_batch[2].clone() if not isinstance(region_batch, dict) and len(region_batch) > 2 else None)
            #         if text_att2 is not None:
            #             text_atts_history.append(text_att2)
            #     _ = self.region_forward_output(model, region_batch, device)

            # 활성화 통계 합치기 + 입력 히스토리 해제
            self._offload_actns(text_atts_history)
            del text_atts_history; text_atts_history = []

            # 스텝 종료시 마스킹
            is_end_step = ((batch_idx + 1) % num_batches_per_step == 0) or (batch_idx == len(dataloader) - 1)
            if is_end_step:
                
                # 1) 스코어 계산 (row_col_std_norm)
                self.score()

                # ---- [NEW] prune_scope -> target modalities 결정 ----
                prune_scope = kwargs.get("prune_scope", "text")  # "text" | "vision" | "both"
                prune_scope = (prune_scope or "both").lower()

                if prune_scope == "text":
                    TARGET_MODALITIES = {"text"}
                elif prune_scope == "vision":
                    TARGET_MODALITIES = {"vision"}
                elif prune_scope == "both":
                    TARGET_MODALITIES = {"vision", "text"}
                else:
                    raise ValueError(f"[prune] Unknown prune_scope={prune_scope} (use text/vision/both)")

                # ---- [NEW] 모달리티 판별 fallback (detect_modality_fn이 llava를 other로 뱉는 경우 대비) ----
                def _modality_with_fallback(param_name: str):
                    m = self.detect_modality_fn(param_name)

                    # detect_modality_fn이 llava 네이밍을 못 잡는 케이스 보정
                    if m not in ("vision", "text", "fusion"):
                        if param_name.startswith("vision_tower."):
                            m = "vision"
                        elif param_name.startswith("language_model."):
                            m = "text"
                        elif "multi_modal_projector" in param_name:
                            m = "fusion"
                    return m

                # ---- [NEW] distribution 생성 ----
                distribution = {}
                for name, _, p in self.named_masked_parameters:
                    pid = id(p)
                    modality = _modality_with_fallback(name)

                    distribution[pid] = float(target_sparsity) if modality in TARGET_MODALITIES else 0.0

                # 디버깅 출력(한 번만 봐도 바로 감 잡힘)
                from collections import Counter
                mods = Counter(_modality_with_fallback(n) for n,_,_ in self.named_masked_parameters)
                print(f"[Debug] prune_scope={prune_scope} TARGET_MODALITIES={TARGET_MODALITIES}")
                print(f"[Debug] modality counts: {mods}")
                nonzero = sum(1 for v in distribution.values() if v > 0)
                print(f"[Debug] nonzero prune entries: {nonzero}/{len(distribution)}")


                # 디버깅 출력
                nonzero = [(pid, s) for pid, s in distribution.items() if s > 0]
                print(f"[Debug] nonzero prune entries: {len(nonzero)} (예: {nonzero[:5]})")

                # 2) 전 레이어 동일 sparsity (uniform) 분배
                # uniform_distribution = {id(p): float(target_sparsity) for _, _, p in self.named_masked_parameters}

                # 3) 마스크 계산
                # self.compute_mask(uniform_distribution, scope="local")
                self.compute_mask(distribution, scope="local")
                plan_lines = []
                tot_elems_p = 0
                tot_zeros_p = 0

                for name, _, p in self.named_masked_parameters:
                    elems = int(p.numel()) if hasattr(p, "numel") else 0
                    # ★ pid 기반 분포에서 sparsity 가져오기 (없으면 target_sparsity로 대체)
                    sp = float(distribution.get(id(p), float(target_sparsity)))
                    sp = max(0.0, min(1.0, sp))  # 안전 클램프

                    kept = int(round(elems * (1.0 - sp))) if elems else 0
                    kept = max(0, min(elems, kept))       # 안전 클램프

                    tot_elems_p += elems
                    tot_zeros_p += (elems - kept)

                    # ★ 결과 로그와 동일 포맷 ★
                    plan_lines.append(
                        f"layer={name:<80} pid={id(p)}  elements={elems}  kept={kept}  sparsity={sp:.6f}"
                    )

                global_plan = (tot_zeros_p / tot_elems_p) if tot_elems_p else 0.0
                plan_lines.append(
                    f"\n[GLOBAL] effective_sparsity={global_plan:.6f}  (zeros={tot_zeros_p} / total={tot_elems_p})"
                )

                # 3) 저장 (결과 파일과 분리: sparsity.plan)
                output_dir = str(Path(kwargs.get('output_dir', getattr(self, 'output_dir', '.'))).resolve())
                os.makedirs(output_dir, exist_ok=True)
                base = os.path.join(output_dir, "sparsity")
                with open(base + ".plan", "w", encoding="utf-8") as f:
                    f.write("\n".join(plan_lines) + "\n")

                # one-shot이면 다음 스텝에서 추가 진행 없이 종료(일반적으로 외부 루프가 종료)
                # 필요 시 break 가능
                break


        self.scoring_time = int(time.time() - t0)
        print(f"Total pruning time (hh:mm:ss) = {datetime.timedelta(seconds=self.scoring_time)}")

    # --------------------------
    # 리셋
    # --------------------------
    def hard_reset(self):
        self.reset()

    def reset(self):
        print("<Reset>")
        for _, mask, param in self.named_masked_parameters:
            mask.fill_(1)
            if mask.grad is not None:
                mask.grad.data.zero_()
            if param.grad is not None:
                param.grad.data.zero_()
            self.scores[id(param)] = torch.zeros(param.size(), dtype=torch.float32)

            # 열 통계 초기화
            self.col_act_cnt[id(param)]  = 0
            self.col_act_mean[id(param)] = torch.zeros(param.size(1), dtype=torch.float32)
            self.col_act_M2[id(param)]   = torch.zeros(param.size(1), dtype=torch.float32)

            # 행 통계 제거(가변 길이)
            self.row_act_cnt.pop(id(param),  None)
            self.row_act_mean.pop(id(param), None)
            self.row_act_M2.pop(id(param),   None)

        # BLIP 가중치 tie 복원 필요 시
        if getattr(self.model, "name", "") == "blip":
            check_blip_state_dict(self.model.state_dict())
