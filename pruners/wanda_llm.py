# pruners/wanda_sparse.py
import math, time, datetime, torch, torch.nn as nn
from functools import partial
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from pruners.base import Pruner
from utils.prune_utils import make_prunable, recursive_getattr
from pruners.accumulators import forward_output, region_forward_output
from utils.functions import detect_modality_fn

# WANDA_TARGET_MODALITIES     = {"text"}
# SPARSEGPT_TARGET_MODALITIES = {"text"}

def _enable_store_input_only_for_prunable(model, named_masked_parameters):
    # 전체 끄고
    for m in model.modules():
        if hasattr(m, "store_input_flag"):
            m.store_input_flag = False
    # 프루닝 대상만 켜기
    for name, _, _ in named_masked_parameters:
        mname = ".".join(name.split(".")[:-1])
        module = recursive_getattr(model, mname)
        if hasattr(module, "store_input_flag"):
            module.store_input_flag = True

@torch.no_grad()
def _gather_inputs_matrix(module):
    """
    module.input_history 리스트를 (N, d)로 평탄화해서 합치고,
    메모리 해제를 위해 즉시 비운다.
    """
    mats = []
    for x in getattr(module, "input_history", []):
        if x is None:
            continue
        # [B, L, d] 또는 [B, P, d] → (B*L, d)
        if x.dim() == 3:
            mats.append(x.reshape(-1, x.size(-1)).float())
        # [L, d] 또는 [B, d] → 그대로(또는 reshape 없이)
        elif x.dim() == 2:
            mats.append(x.float())
        else:
            d = x.size(-1)
            mats.append(x.reshape(-1, d).float())
    # history 비우기
    module.input_history = []
    if not mats:
        return None
    return torch.cat(mats, dim=0)  # (N, d)


# ---------- SparseGPT per-module ----------
def _sparsegpt_prune_module(weight: torch.Tensor, H: torch.Tensor, sparsity: float,
                            prune_n=0, prune_m=0, blocksize=128, percdamp=0.01):
    # H: (in,in)  (이미 sqrt(2/N) 스케일 반영하여 누적해둔 공분산 근사)
    W = weight.data.clone().float()  # (out, in)
    d = W.size(1)
    dev = W.device

    diag_idx = torch.arange(d, device=dev)
    dead = (torch.diag(H) == 0)
    H = H.clone()
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    damp = percdamp * torch.mean(torch.diag(H))
    H[diag_idx, diag_idx] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    for i1 in range(0, d, blocksize):
        i2 = min(i1 + blocksize, d)
        cnt = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if prune_n and prune_m:
            mask1 = torch.zeros_like(W1, dtype=torch.bool)
        else:
            tmp = W1.pow(2) / (torch.diag(Hinv1).view(1, -1)).pow(2)
            thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
            mask1 = tmp <= thresh

        for i in range(cnt):
            w = W1[:, i]
            d_i = Hinv1[i, i]

            if prune_n and prune_m and (i % prune_m == 0):
                tmp = W1[:, i:(i+prune_m)].pow(2) / (torch.diag(Hinv1)[i:(i+prune_m)].view(1, -1)).pow(2)
                k = min(prune_n, tmp.size(1))
                idx = torch.topk(tmp, k, dim=1, largest=False)[1]
                mask1.scatter_(1, i + idx, True)

            q = w.clone()
            q[mask1[:, i]] = 0
            Q1[:, i] = q

            err1 = (w - q) / d_i
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        if i2 < d:
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    weight.data.copy_(W.type_as(weight.data))


# ---------- 베이스: 공통 초기화/캘리브레이션 루프 ----------
class _BaseCalibPruner(Pruner):
    is_one_shot = False
    modifies_weights = True  # 실제 가중치 0으로 만듦 (마스크 저장 아님)

    def __init__(self, model, *args, **kwargs):
        # ★ 입력 저장 불필요하므로 store_input=False (OOM 방지)
        make_prunable(model, mask_dtype=torch.bool, pattern_lock=True,
                      mask_on_the_fly=True, store_input=True)
        super().__init__(model, *args, **kwargs)
        self.detect_modality_fn = partial(detect_modality_fn, self.model.name)
        self.forward_output = partial(forward_output, self.model.name)
        self.region_forward_output = partial(region_forward_output, self.model.name)
        self.name = self.__class__.__name__.lower()
        self._hooks = []
    def _sync_masks_with_weights(self):
        for _, mask, param in self.named_masked_parameters:
            new_mask = param.data.ne(0).to(dtype=mask.dtype, device=mask.device)
            mask.copy_(new_mask)
    def _clear_hooks(self):
        for h in self._hooks:
            try: h.remove()
            except: pass
        self._hooks = []

    def reset(self):
        self._clear_hooks()
        self.scores = {}
        if hasattr(self, "_row_sums"):
            self._row_sums = {}
        if hasattr(self, "_nsamples"):
            self._nsamples = {}
        for name, _, _ in self.named_masked_parameters:
            try:
                module = recursive_getattr(self.model, ".".join(name.split(".")[:-1]))
                if hasattr(module, "input_history"):
                    module.input_history = []
                if hasattr(module, "store_input_flag"):
                    module.store_input_flag = False
            except Exception:
                continue

    @torch.no_grad()
    def _run_calibration_forward(self, model, dataloader, device, fabric, num_batches_per_step, region_loader=None):
        # region_loader 존재하면 CombinedLoader로 묶기 (MultiFlow와 동일)
        if region_loader is not None:
            dataloader = CombinedLoader((dataloader, region_loader), mode="min_size")

        for bidx, batch in enumerate(dataloader):
            if region_loader is not None:
                general_batch, region_batch = batch
            else:
                general_batch = batch

            _ = self.forward_output(model, general_batch, device, modality="fusion")
            # if region_loader is not None:
            #     _ = self.region_forward_output(model, region_batch, device)

            if (bidx + 1) % num_batches_per_step == 0:
                break

class _BaseCalibPruner_for_sgpt(Pruner):
    is_one_shot = False
    modifies_weights = True  # 실제 가중치 0으로 만듦 (마스크 저장 아님)

    def __init__(self, model, *args, **kwargs):
        # ★ 입력 저장 불필요하므로 store_input=False (OOM 방지)
        make_prunable(model, mask_dtype=torch.bool, pattern_lock=True,
                      mask_on_the_fly=True, store_input=False)
        super().__init__(model, *args, **kwargs)
        self.detect_modality_fn = partial(detect_modality_fn, self.model.name)
        self.forward_output = partial(forward_output, self.model.name)
        self.region_forward_output = partial(region_forward_output, self.model.name)
        self.name = self.__class__.__name__.lower()
        self._hooks = []
    def _sync_masks_with_weights(self):
        for _, mask, param in self.named_masked_parameters:
            new_mask = param.data.ne(0).to(dtype=mask.dtype, device=mask.device)
            mask.copy_(new_mask)
    def _clear_hooks(self):
        for h in self._hooks:
            try: h.remove()
            except: pass
        self._hooks = []

    def reset(self):
        self._clear_hooks()
        self.scores = {}
        if hasattr(self, "_H"):
            self._H = {}
        if hasattr(self, "_nsamples"):
            self._nsamples = {}
        for name, _, _ in self.named_masked_parameters:
            try:
                module = recursive_getattr(self.model, ".".join(name.split(".")[:-1]))
                if hasattr(module, "input_history"):
                    module.input_history = []
                if hasattr(module, "store_input_flag"):
                    module.store_input_flag = False
            except Exception:
                continue

    @torch.no_grad()
    def _run_calibration_forward(self, model, dataloader, device, fabric, num_batches_per_step, region_loader=None):
        # region_loader 존재하면 CombinedLoader로 묶기 (MultiFlow와 동일)
        if region_loader is not None:
            dataloader = CombinedLoader((dataloader, region_loader), mode="min_size")

        for bidx, batch in enumerate(dataloader):
            if region_loader is not None:
                general_batch, region_batch = batch
            else:
                general_batch = batch

            _ = self.forward_output(model, general_batch, device, modality="fusion")
            # if region_loader is not None:
            #     _ = self.region_forward_output(model, region_batch, device)

            if (bidx + 1) % num_batches_per_step == 0:
                break

# ---------- Wanda ----------
# --- 클래스 상단에 상태 변수 준비 ---
class WandaPruner(_BaseCalibPruner):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.name = "wanda"
        # pid -> 누적 열별 제곱합, 샘플 수
        self._row_sums = {}
        self._nsamples = {}

    def _compute_abs_scores(self): # 모델 Parameter의 절댓값을 기반으로 Saliency 점수를 계산
        print("[Debug] Multiflow.py -> _compute_abs_scores() 함수 호출 : 파라미터 절댓값 계산")
        for _, _, param in self.named_masked_parameters: # Pruning이 가능한 (파라미터 이름, 마스크, 파라미터) 튜플 리스트 중 파라미터에 대해 반복 수행
            #param.data : 파라미터의 실제 데이터(가중치)
            #데이터 clone 후, 그래프와 연결을 끊어 역전파 시 영향을 받지 않도록 설정한 후, 파라미터 절댓값 계산
            self.scores[id(param)] = torch.clone(param.data).detach().abs_() #파라미터(가중치 행렬, 편향)의 고유 Id를 key로 하여 행렬 내 개별 원소에 대하여 절댓값을 저장하는 딕셔너리(tensor)
            # Linear layer는 scores[파라미터][파라미터 내 개별 원소 절댓값] 형태로 저장, Conv layer는 4차원 tensor

    def _modal_mask(self, target_sparsity): # 모달리티 별로 중요도 점수를 기준으로 target sparsity에 따라 중요도가 낮은 파라미터를 0으로 만들어 mask 업데이트
        print("[Debug] Multiflow.py -> modal_mask() 함수 호출 : Modality, Layer 별 sparsity ratio 계산을 위한 Masking")
        #파라미터 이름 n을 바탕으로 파라미터가 속한 모달리티를 탐지한다(Vision, Text)
        different_modalities = set([self.detect_modality_fn(n) for n, _, _ in self.named_masked_parameters]) #set을 사용하여 모달리티들을 중복 제거하여 수집
        for modality in different_modalities: # 모달리티 별 점수 수집(Vision or Text)

            # get the scores of the tensors of this modality
            #현재 모달리티에 속하는 파라미터들만 필터링하여 각 파라미터의 중요도 점수를 딕셔너리로 저장한다.
            scores_for_this_modality = {
                #_compute_abs_scores에서 구해놓은 파라미터 내 절댓값 점수를 기반으로 현재 파라미터에 해당하는 부분만 딕셔너리로 만듦
                id(p): self.scores[id(p)] for n, _, p in self.named_masked_parameters 
                if self.detect_modality_fn(n) == modality #Init함수에서 선언한 모달리티 감지 함수
            }
            #현재 반복중인 모달리티와 일치하는 모달리티 내 모든 파라미터 중요도 점수를 1차원 텐서로 병합한다.
            modal_scores = torch.cat([torch.flatten(v) for v in scores_for_this_modality.values()])

            # get the modality threshold
            k = int(modal_scores.numel() * target_sparsity) #  modal_scores에 있는 파라미터 개수 * 목표 sparsity = Pruning할 파라미터 개수를 계산한다.
            threshold, _ = torch.kthvalue(modal_scores, k=k) # k번째로 작은 값을 기준으로 임계값 설정(임계값으로 설정된 파라미터의 절댓값보다 작은 파라미터는 0으로 취급되어 pruning 목표 개수에 반영X)

            # compute the mask for the parameters of this modality
            for name, mask, param in self.named_masked_parameters:
                if self.detect_modality_fn(name) != modality: continue # 현재 반복 중인 파라미터가 현재 모달리티에 속하는지 확인
                score = self.scores[id(param)]# 파라미터 절댓값이 담긴 딕셔너리
                zero = torch.tensor([0], dtype=torch.bool).to(mask.device)#0
                one = torch.tensor([1], dtype=torch.bool).to(mask.device)#1
                #절댓값이 계산된 score딕셔너리에 대해 임계값보다 작으면 0, 그렇지 않으면 1로 설정된 Mask 딕셔너리에 복사하여 생성
                mask.copy_(torch.where(score.to(mask.device) <= threshold, zero, one))

    #모달리티 별로 target sparsity에 따른 pruning 분포를 계산한다.
    def multimodal_distribution(self, target_sparsity):
        print("[Debug] Multiflow.py -> multimodal_distribution() 함수 호출 : 모달리티별 pruning 분포 계산")
        self._compute_abs_scores()#파라미터의 절댓값 기반 Saliency 점수 계산(self.scores에 저장됨)
        self._modal_mask(target_sparsity)#모달리티 별로 절댓값 점수를 기준으로 Mask 딕셔너리 생성

        distribution = {}
        # grab the sparsity distribution for each param, rewind the masks and the scores
        #각 파라미터의 Mask에서 1인 요소들의 개수를 Layer별 Sparsity로 정함
        for name, mask, param in self.named_masked_parameters:
            #mask.sum().item() : 현재 파라미터에서 Mask값이 1인 요소의 개수
            #mask.numel(): 총 파라미터 개수
            sparsity = 1 - mask.sum().item() / mask.numel()#파라미터 100개중 70개가 Masking되었으면 1 - 30% = 희소성은 70%

            distribution[id(param)] = sparsity #각 파라미터의 ID를 key로 하여 Sparsity 비율 저장
            #파라미터별로 희소성을 저장하는 이유는 파라미터(가중치)마다 크기나 중요도가 다를 수 있기 때문이다.
            mask.fill_(1)#모든 마스크를 1로 초기화 (pruning할 때 다시 사용해야 하기 때문)

            print(f"[Debug] Layer: {name}, pid: {id(param)}, shape: {tuple(param.shape)}, sparsity: {sparsity:.4f}")#각 파라미터의 이름, 모양, 희소성 출력

            self.scores[id(param)] = torch.zeros_like(self.scores[id(param)])#self.scores를 0으로 초기화(pruning할 때 다시 사용해야 하기 때문)
        try:
            import os
            out = {}
            for name, _, param in self.named_masked_parameters:
                pid  = id(param)
                mod  = self._mod_for_plot(name)
                blk  = self._extract_block_smart(name)
                try:
                    mname  = ".".join(name.split(".")[:-1])
                    module = recursive_getattr(self.model, mname)
                except Exception:
                    module = None
                layer = self._parse_role_smart(name, module, param)
                it = {
                    "mod": mod,
                    "blk": blk,
                    "layer": layer,
                    "pid": pid,
                    # score 키를 재사용해 sparsity 값을 넣어 동일 포맷으로 출력
                    "score": float(distribution.get(pid, 0.0)),
                    "w": float(param.numel()),
                }
                out.setdefault((mod, blk), []).append(it)

            log_lines = []
            for (mod, blk), items in out.items():
                scores  = torch.tensor([float(it["score"]) for it in items], dtype=torch.float32)
                weights = torch.tensor([float(it.get("w", 1)) for it in items], dtype=torch.float32)
                wsum    = weights.sum().clamp_min(1.0)
                mean_w  = (weights * scores).sum() / wsum   # \bar R_weighted

                header = f"[DIST] {mod} block {blk} | target_sparsity={target_sparsity} | mean_score={mean_w:.6f}"
                print(header)
                log_lines.append(header)

                for it in items:
                    pid   = it["pid"]
                    sval  = float(it["score"])  # 여기선 score == sparsity
                    mod_i = it.get("mod", mod)
                    blk_i = it.get("blk", blk)
                    line = (f"   - [{mod_i} blk {blk_i:>2}] layer={it['layer']:<12} pid={pid} "
                            f"score={float(it['score']):.6f} -> sparsity={sval:.6f}")
                    print(line)
                    log_lines.append(line)

            # ---- 파일 저장 ----
            if log_lines:
                output_dir = getattr(self, "output_dir", ".")
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, "multiflow_sparsity")  # 파일명 정확히 'sparsity'
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(log_lines) + "\n")
                print(f"[DIST] saved to: {save_path}")
        except Exception as e:
            print(f"[DIST] save failed: {e}")
        
        
        return distribution #각 파라미터의 희소성 분포를 반환하여 pruning에 사용(파라미터별(Layer별)로 sparsity저장한 딕셔너리)


    @torch.no_grad()
    def _accumulate_from_modules(self, model):
        """현재 배치에서 prunable 모듈들이 저장한 input_history를
        (N,d)로 펴서 열별 제곱합을 누적하고, history는 즉시 비운다."""
        for name, _, p in self.named_masked_parameters:
            mname = ".".join(name.split(".")[:-1])
            module = recursive_getattr(model, mname)

            # # --- ✅ 모달리티 하드코딩 필터 ---
            # modality = self.detect_modality_fn(name)  # "vision", "text", "fusion", ...
            # if modality not in WANDA_TARGET_MODALITIES:
            #     # 프루닝 대상이 아니면 통계 안 쌓고 history만 정리
            #     if hasattr(module, "input_history"):
            #         module.input_history = []
            #     continue
            # # --- ✅ 모달리티 하드코딩 필터 ---


            if not isinstance(module, nn.Linear):
                # Wanda 원본도 Linear/Conv1D 대상. 여기선 Linear만.
                continue

            # (N,d)로 수집하면서 module.input_history를 비움
            X = _gather_inputs_matrix(module)  # 위에 네가 둔 helper
            if X is None or X.numel() == 0:
                continue

            pid = id(p)
            # 열별 제곱합 누적
            rs = X.pow(2).sum(dim=0)  # (d,)
            self._row_sums[pid] = self._row_sums.get(pid, torch.zeros_like(rs))
            self._row_sums[pid] += rs
            # 샘플 수 누적
            self._nsamples[pid] = self._nsamples.get(pid, 0) + X.size(0)

    @torch.no_grad()
    def prune(self, target_sparsity, model, dataloader, device, fabric, num_batches_per_step, **kwargs):
        t0 = time.time()
        region_loader = kwargs.get('region_loader', None)
        cfg = kwargs.get("config", {})
        prune_n = int(cfg.get("prune_n", 0))
        prune_m = int(cfg.get("prune_m", 0))


        # 1) 캘리브레이션: 배치마다 forward → 즉시 통계 누적 → history 비움
        _enable_store_input_only_for_prunable(model, self.named_masked_parameters)

        if region_loader is not None:
            from lightning.pytorch.utilities.combined_loader import CombinedLoader
            dataloader = CombinedLoader((dataloader, region_loader), mode="min_size")

        for bidx, batch in enumerate(dataloader):
            if region_loader is not None:
                general_batch, region_batch = batch
            else:
                general_batch = batch

            _ = self.forward_output(model, general_batch, device, modality="fusion")
            # if region_loader is not None:
            #     _ = self.region_forward_output(model, region_batch, device)

            # ★ 여기서 바로 누적하고 비움 → OOM 방지
            self._accumulate_from_modules(model)

            if (bidx + 1) % num_batches_per_step == 0:
                break

        # 2) 누적 통계로 scaler_row 만들고 Wanda 마스크 적용
        for name, _, p in self.named_masked_parameters:
            mname = ".".join(name.split(".")[:-1])
            module = recursive_getattr(model, mname)

            # # --- ✅ 모달리티 하드코딩 필터 ---
            # modality = self.detect_modality_fn(name)
            # if modality not in WANDA_TARGET_MODALITIES:
            #     continue
            # # ---------------------------------

            if not isinstance(module, nn.Linear):
                continue

            pid = id(p)
            if pid not in self._row_sums or self._nsamples.get(pid, 0) == 0:
                # 해당 모듈에 입력이 없었으면 스킵(혹은 magnitude 폴백 원하면 ones로 대체 가능)
                continue

            scaler_row = (self._row_sums[pid] / max(self._nsamples[pid], 1)).to(p.device)  # (in,)
            # Wanda metric = |W| * sqrt(scaler_row)
            W = p.data
            W_metric = W.abs() * torch.sqrt(scaler_row.view(1, -1))

            # N:M 또는 unstructured
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            if prune_n and prune_m:
                cols = W_metric.size(1)
                for ii in range(0, cols, prune_m):
                    block = W_metric[:, ii:ii+prune_m].float()
                    k = min(prune_n, block.size(1))
                    if k > 0:
                        idx = torch.topk(block, k, dim=1, largest=False)[1]
                        W_mask.scatter_(1, ii + idx, True)
            else:
                cols = W_metric.size(1)
                k = int(cols * target_sparsity)
                k = max(0, min(k, cols))
                if k > 0:
                    idx = torch.argsort(W_metric, dim=1, stable=True)[:, :k]
                    W_mask.scatter_(1, idx, True)

            p.data[W_mask.to(p.device)] = 0
            self._sync_masks_with_weights()
        print(f"[WANDA] Total pruning time (hh:mm:ss) = {datetime.timedelta(seconds=int(time.time()-t0))}")



# ---------- SparseGPT ----------
class SparseGPTPruner(_BaseCalibPruner_for_sgpt):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.name = "sparsegpt"
        # pid -> (in,in) 공분산 근사
        self._H = {}
        self._nsamples = {}

    # def _register_sparsegpt_hooks(self):
    #     for name, _, p in self.named_masked_parameters:
    #         mname = ".".join(name.split(".")[:-1])
    #         module = recursive_getattr(self.model, mname)

    #         # # --- ✅ 모달리티 하드코딩 필터 ---
    #         # modality = self.detect_modality_fn(name)
    #         # if modality not in SPARSEGPT_TARGET_MODALITIES:
    #         #     continue
    #         # # ---------------------------------

    #         if not isinstance(module, nn.Linear):
    #             continue
    #         pid = id(p)
    #         d = p.size(1)
    #         dev = p.device
    #         self._H[pid] = torch.zeros((d, d), dtype=torch.float32, device=dev)
    #         self._nsamples[pid] = 0

    #         @torch.no_grad()
    #         def hook(mod, inputs, output, pid_local=pid, dev_local=dev):
    #             x = inputs[0]
    #             if x.dim() == 3:
    #                 x = x.reshape(-1, x.size(-1))
    #             elif x.dim() == 2:
    #                 pass
    #             else:
    #                 return
    #             x = x.to(dtype=torch.float32, device=dev_local)
    #             # 누적 스케일: sqrt(2/N) * X
    #             n_prev = self._nsamples[pid_local]
    #             n_new  = n_prev + x.size(0)
    #             scale = math.sqrt(2.0 / max(n_new, 1))
    #             # 이전 H를 평균화해서 감쇠시키고, 현재 배치를 새 스케일로 반영
    #             if n_prev > 0:
    #                 self._H[pid_local].mul_(n_prev / n_new)
    #             self._H[pid_local].addmm_(x.t(), x, alpha=scale*scale)
    #             self._nsamples[pid_local] = n_new

    #         self._hooks.append(module.register_forward_hook(hook))
    def _register_sparsegpt_hooks(self):
        for name, _, p in self.named_masked_parameters:
            mname = ".".join(name.split(".")[:-1])
            module = recursive_getattr(self.model, mname)

            if not isinstance(module, nn.Linear):
                continue

            pid = id(p)
            d = p.size(1)

            # 🔥 H는 무조건 CPU에 둔다 (GPU OOM 방지)
            h_device = torch.device("cpu")
            self._H[pid] = torch.zeros((d, d), dtype=torch.float32, device=h_device)
            self._nsamples[pid] = 0

            @torch.no_grad()
            def hook(mod, inputs, output, pid_local=pid, h_dev=h_device):
                x = inputs[0]
                if x.dim() == 3:
                    x = x.reshape(-1, x.size(-1))
                elif x.dim() == 2:
                    pass
                else:
                    return

                # 🔥 x도 CPU로 모아서 H를 CPU에서 업데이트
                x = x.detach().to(dtype=torch.float32, device=h_dev)

                n_prev = self._nsamples[pid_local]
                n_new  = n_prev + x.size(0)
                scale = math.sqrt(2.0 / max(n_new, 1))

                if n_prev > 0:
                    self._H[pid_local].mul_(n_prev / n_new)

                self._H[pid_local].addmm_(x.t(), x, alpha=scale * scale)
                self._nsamples[pid_local] = n_new

            self._hooks.append(module.register_forward_hook(hook))


    @torch.no_grad()
    def prune(self, target_sparsity, model, dataloader, device, fabric, num_batches_per_step, **kwargs):
        t0 = time.time()
        cfg = kwargs.get("config", {})
        prune_n = int(cfg.get("prune_n", 0))
        prune_m = int(cfg.get("prune_m", 0))
        blocksize = int(cfg.get("blocksize", 128))
        percdamp  = float(cfg.get("percdamp", 0.1))

        # 1) 훅 등록 → 캘리브레이션 → 훅 제거
        self._register_sparsegpt_hooks()
        self._run_calibration_forward(model, dataloader, device, fabric, num_batches_per_step, kwargs.get('region_loader', None))
        self._clear_hooks()

        # 2) 레이어별 SparseGPT 적용
        # for name, _, p in self.named_masked_parameters:
        #     mname = ".".join(name.split(".")[:-1])
        #     module = recursive_getattr(model, mname)

        #     # # --- ✅ 모달리티 하드코딩 필터 ---
        #     # modality = self.detect_modality_fn(name)
        #     # if modality not in SPARSEGPT_TARGET_MODALITIES:
        #     #     continue
        #     # # ---------------------------------

        #     if not isinstance(module, nn.Linear):
        #         continue
        #     pid = id(p)

        #     H = self._H[pid]
        #     _sparsegpt_prune_module(p, H, target_sparsity,
        #                             prune_n=prune_n, prune_m=prune_m,
        #                             blocksize=blocksize, percdamp=percdamp)
        for name, _, p in self.named_masked_parameters:
            mname = ".".join(name.split(".")[:-1])
            module = recursive_getattr(model, mname)

            if not isinstance(module, nn.Linear):
                continue
            pid = id(p)

            if pid not in self._H or self._nsamples.get(pid, 0) == 0:
                # 이 레이어는 캘리브레이션에서 입력이 없었으면 스킵
                continue

            # 🔥 H는 CPU → 현재 레이어 weight와 같은 디바이스로 잠깐 올림
            H_cpu = self._H[pid]
            H = H_cpu.to(p.device, non_blocking=True)

            _sparsegpt_prune_module(
                p, H, target_sparsity,
                prune_n=prune_n, prune_m=prune_m,
                blocksize=blocksize, percdamp=percdamp,
            )

            # 사용 끝난 H는 GPU에서 바로 제거
            del H

        self._sync_masks_with_weights()
        print(f"[SparseGPT] Total pruning time (hh:mm:ss) = {datetime.timedelta(seconds=int(time.time()-t0))}")
