# Taken and adapted from: https://github.com/YiteWang/NTK-SAP/blob/main/Pruners/pruners.py
import torch
from collections import OrderedDict
from utils.prune_utils import stats, named_masked_parameters
import math

class Pruner:
    def __init__(self, model, keys_to_exclude=[]):
        print("**Pruner base.py 파일 init 함수**")
        self.name = 'pruner'
        self.model = model
        self.requires_training = False

        self.scores = {}
        self.keys_to_exclude = keys_to_exclude
        
        if hasattr(self.model, "needs_tie") and self.model.needs_tie:
            if not hasattr(model, "tie_fn"):
                raise ValueError("You have passed a module with needs_tie=True to a pruner, but the model you are trying to prune does not have a tie_fn method. "
                                 "Please implement a tie_fn method in your model and try again.")
            if not callable(model.tie_fn):
                raise ValueError("The tie_fn method in your model is not callable. Please make sure that the tie_fn method in your model is callable.")
            
            print("Pruner was initialized with a model that needs to be tied: {}".format(self.model.name))
            print("<Pruner base.py 파일 init 함수 - blip_pretrain.py ->tie_fn() 호출>")
            self.model.tie_fn()
            print("<Pruner base.py 파일 init 함수 - blip_pretrain.py ->tie_fn() 호출 종료>")
            
        if len(self.keys_to_exclude) > 0:
            print("Excluding keys: ", self.keys_to_exclude)
            self.named_masked_parameters = []
            for name, mask, param in named_masked_parameters(model):
                if not any(key in name for key in self.keys_to_exclude):
                    self.named_masked_parameters.append((name, mask, param))
        else:
            self.named_masked_parameters = list(named_masked_parameters(model))

        # NOTE: when wrapping with Lightning Fabric, a '_forward_module.' prefix is added to 
        # the names of the parameters; this is a hack to remove it
        self.named_masked_parameters = [
            (name.replace('_forward_module.', ''), mask, param) 
            for name, mask, param in self.named_masked_parameters
        ]
        

    def __str__(self):
        return self.name
    

    def __repr__(self):
        return self.name

        
    def score(self, *args, **kwargs):
        raise NotImplementedError


    @torch.no_grad()
    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """

        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int(sparsity * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for name, mask, param in self.named_masked_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0], dtype=torch.bool).to(mask.device)
                one = torch.tensor([1], dtype=torch.bool).to(mask.device)
                mask.copy_(torch.where(score.to(mask.device) <= threshold, zero, one))
            
    # @torch.no_grad()
    # def _local_mask(self, sparsity):
    #     print("[Debug] pruners/base.py -> _local_mask()함수 호출 : pruning을 위한 mask 생성")
    #     r"""Updates masks of model with scores by sparsity level parameter-wise.
    #     """
    #     for _, mask, param in self.named_masked_parameters:
    #         score = self.scores[id(param)]
    #         # k = int((1.0 - sparsity) * score.numel())
            
    #         if isinstance(sparsity, dict):
    #             # 파라미터별 sparsity 비율 적용
    #             sparsity_for_this_param = sparsity[id(param)]
    #         else:
    #             sparsity_for_this_param = sparsity
    #         k = int(sparsity_for_this_param * score.numel())
    #         if not k < 1:
    #             threshold, _ = torch.kthvalue(torch.flatten(score), k) # 해당 파라미터에서 k번째로 작은 값을 threshold로 설정
    #             zero = torch.tensor([0], dtype=torch.bool).to(mask.device)
    #             one = torch.tensor([1], dtype=torch.bool).to(mask.device)
    #             # mask를 threshold보다 작은 값은 0으로, 나머지는 1로 설정
    #             mask.copy_(torch.where(score.to(mask.device) <= threshold, zero, one))

    @torch.inference_mode()
    def _local_mask(self, sparsity):
        print("[Debug] pruners/base.py -> _local_mask()함수 호출 : pruning을 위한 mask 생성")
        print(f"[Debug][local_mask] self.scores len = {len(self.scores)}")

        for name, mask, param in self.named_masked_parameters:
            pid = id(param)

            # 1차: id 기반
            score = self.scores.get(pid, None)

            # 2차: 이름 기반 fallback
            if score is None and hasattr(self, "scores_by_name"):
                score = self.scores_by_name.get(name, None)

            has_score = score is not None
            # print(f"[Debug][local_mask] {name} pid={pid} has_score={has_score}")
            if not has_score:
                print(f"[Warn] _local_mask: score가 없는 파라미터 스킵: {name} (id={pid})")
                continue

            # sparsity dict인 경우도 같은 이슈 생길 수 있어서 비슷하게 처리하면 더 안전
            if isinstance(sparsity, dict):
                sparsity_for_this_param = sparsity.get(pid, sparsity.get(name, None))
                if sparsity_for_this_param is None:
                    print(f"[Warn] _local_mask: sparsity 정보가 없는 파라미터 스킵: {name} (id={pid})")
                    continue
            else:
                sparsity_for_this_param = sparsity

            k = int(sparsity_for_this_param * score.numel())
            if k < 1:
                continue

            threshold, _ = torch.kthvalue(score.view(-1), k)
            zero = torch.tensor([0], dtype=torch.bool, device=mask.device)
            one  = torch.tensor([1], dtype=torch.bool, device=mask.device)
            mask.copy_(torch.where(score.to(mask.device) <= threshold.to(mask.device), zero, one))


    @torch.no_grad()
    def _group_mask(self, sparsity):
        print("[Debug] pruners/base.py -> _local_mask() 함수 호출 : pruning을 위한 mask 생성")

        def _split_sizes(D: int, parts: int):
            base, rem = divmod(D, parts)
            return [(base + 1 if i < rem else base) for i in range(parts)]

        for _, mask, param in self.named_masked_parameters:
            pid   = id(param)
            score = self.scores[pid].to(mask.device)  # same device
            # sparsity 설정(스칼라 or 벡터)
            sp = sparsity[pid] if isinstance(sparsity, dict) else sparsity

            # --- [A] 구간별 sparsity (예: tensor([s1, s2, s3, s4])) ---
            if torch.is_tensor(sp) and sp.ndim == 1 and sp.numel() > 1:
                parts = int(sp.numel())

                # 점수 텐서가 2D(Linear: [D_out, D_in])라고 가정하고 입력축(열, dim=1)으로 나눔
                if score.ndim >= 2:
                    D_in  = score.shape[1]
                    sizes = _split_sizes(D_in, parts)

                    # 최종 마스크 (초기값: 전부 keep=True)
                    m = torch.ones_like(score, dtype=torch.bool)

                    c0 = 0
                    for j, (s_part, sz) in enumerate(zip(sp.tolist(), sizes)):
                        if sz <= 0:
                            continue
                        # seg: 이 구간에 해당하는 열 슬라이스
                        seg = score[:, c0:c0+sz]  # (D_out, sz)
                        # k_j = round(s_part * |seg|)
                        s_part_clamped = float(max(0.0, min(1.0, s_part)))
                        k = int(s_part_clamped * seg.numel())
                        if k >= 1:
                            thr, _ = torch.kthvalue(seg.reshape(-1), k)
                            m_seg = seg > thr               # <= thr는 0(프루닝), > thr는 1(keep)
                            m[:, c0:c0+sz] = m_seg.reshape_as(seg)
                        else:
                            # k==0이면 이 구간은 전부 keep
                            m[:, c0:c0+sz] = True
                        c0 += sz

                    mask.copy_(m.to(mask.device))

                else:
                    # 1D (드문 케이스)일 때는 마지막 축을 등분해서 동일 로직 적용
                    D     = score.numel()
                    sizes = _split_sizes(D, parts)
                    m = torch.ones_like(score, dtype=torch.bool)
                    i0 = 0
                    for j, (s_part, sz) in enumerate(zip(sp.tolist(), sizes)):
                        if sz <= 0:
                            continue
                        seg = score[i0:i0+sz]
                        s_part_clamped = float(max(0.0, min(1.0, s_part)))
                        k = int(s_part_clamped * seg.numel())
                        if k >= 1:
                            thr, _ = torch.kthvalue(seg.reshape(-1), k)
                            m[i0:i0+sz] = seg > thr
                        i0 += sz
                    mask.copy_(m.to(mask.device))

                continue  # 다음 파라미터로

            # --- [B] 스칼라 sparsity (기존 동작) ---
            if isinstance(sp, (int, float)):
                s_val = float(max(0.0, min(1.0, sp)))
            elif torch.is_tensor(sp) and sp.numel() == 1:
                s_val = float(max(0.0, min(1.0, sp.item())))
            else:
                # 예외적으로 들어온 타입은 float로 시도
                s_val = float(sp)

            k = int(s_val * score.numel())
            if k >= 1:
                thr, _ = torch.kthvalue(score.reshape(-1), k)
                # 기존 구현과 동일하게 <= thr는 0, > thr는 1
                zero = torch.tensor([0], dtype=torch.bool, device=mask.device)
                one  = torch.tensor([1], dtype=torch.bool, device=mask.device)
                mask.copy_(torch.where(score.to(mask.device) <= thr, zero, one))
            else:
                mask.fill_(True)


    @torch.inference_mode()
    def _local_mask_exact(self, sparsity, target_sparsity: float = None):
        """
        local mask지만, 전역적으로 target sparsity(=prune ratio)를 정확히 맞추도록
        param별 k를 정수 보정한 뒤, topk로 정확히 k개만 prune한다.

        - sparsity: scalar or dict
            * dict이면 {pid: sp} 우선, 없으면 {name: sp} fallback
        - target_sparsity:
            * None이면 self.original_sparsity (있으면) 사용
        """
        print("[Debug] pruners/base.py -> _local_mask_exact() called")

        # 0) target sparsity 결정
        if target_sparsity is None:
            target_sparsity = float(getattr(self, "original_sparsity", 0.0))
        target_sparsity = float(max(0.0, min(1.0, target_sparsity)))

        # 1) 마스킹 대상 파라미터/스코어 수집
        items = []  # [(name, pid, mask, param, score_tensor, sp_float, numel)]
        total_numel = 0

        for name, mask, param in self.named_masked_parameters:
            pid = id(param)

            # score 가져오기 (id 우선)
            score = self.scores.get(pid, None)
            if score is None and hasattr(self, "scores_by_name"):
                score = self.scores_by_name.get(name, None)

            if score is None:
                print(f"[Warn] _local_mask_exact: score가 없는 파라미터 스킵: {name} (id={pid})")
                continue

            # sparsity 가져오기
            if isinstance(sparsity, dict):
                sp = sparsity.get(pid, sparsity.get(name, None))
                if sp is None:
                    print(f"[Warn] _local_mask_exact: sparsity 정보가 없는 파라미터 스킵: {name} (id={pid})")
                    continue
            else:
                sp = sparsity

            # sp -> float
            if torch.is_tensor(sp):
                if sp.numel() != 1:
                    raise ValueError("[local_exact] vector sparsity는 지원 안함. (group_mask 경로를 쓰세요)")
                sp = float(sp.item())
            else:
                sp = float(sp)

            sp = float(max(0.0, min(1.0, sp)))
            n  = int(score.numel())
            if n <= 0:
                continue

            total_numel += n
            items.append((name, pid, mask, param, score, sp, n))

        if total_numel == 0 or len(items) == 0:
            print("[Warn] _local_mask_exact: 대상 파라미터가 없음")
            return

        # 2) 전역 목표 prune 개수 K_target
        K_target = int(round(target_sparsity * total_numel))
        K_target = max(0, min(total_numel, K_target))

        # 3) param별 k_i를 floor로 잡고, 잔차를 fractional part로 보정
        k_floor = []
        fracs   = []
        for (name, pid, mask, param, score, sp, n) in items:
            kf = int(math.floor(sp * n))
            kf = max(0, min(n, kf))
            k_floor.append(kf)
            fracs.append((sp * n - kf))

        K_floor_sum = int(sum(k_floor))
        residual = K_target - K_floor_sum  # +면 더 prune 필요, -면 prune 줄여야 함

        # 정렬 인덱스 준비 (fractional 큰 순서)
        # residual>0: frac 큰 것부터 +1
        # residual<0: frac 작은 것부터 -1 (이미 floor라서 작은 것부터 줄이면 오차 최소)
        order_desc = sorted(range(len(items)), key=lambda i: fracs[i], reverse=True)
        order_asc  = list(reversed(order_desc))

        k_final = k_floor[:]  # copy
        if residual > 0:
            t = residual
            for i in order_desc:
                if t <= 0:
                    break
                n = items[i][6]
                if k_final[i] < n:
                    k_final[i] += 1
                    t -= 1
            if t != 0:
                # 혹시 saturate로 못 채운 경우(드문 케이스) 다시 순회
                for i in order_desc:
                    if t <= 0:
                        break
                    n = items[i][6]
                    add = min(t, n - k_final[i])
                    if add > 0:
                        k_final[i] += add
                        t -= add

        elif residual < 0:
            t = -residual
            for i in order_asc:
                if t <= 0:
                    break
                if k_final[i] > 0:
                    k_final[i] -= 1
                    t -= 1
            if t != 0:
                for i in order_asc:
                    if t <= 0:
                        break
                    sub = min(t, k_final[i])
                    if sub > 0:
                        k_final[i] -= sub
                        t -= sub

        # sanity
        K_final_sum = int(sum(k_final))
        if K_final_sum != K_target:
            print(f"[Warn] _local_mask_exact: K_final_sum({K_final_sum}) != K_target({K_target})")
        else:
            print(f"[Debug] _local_mask_exact: exact prune matched: {K_final_sum}/{total_numel} (p={K_final_sum/total_numel:.6f})")

        # 4) 각 param에 대해 정확히 k_i개 prune (topk smallest)
        for (i, (name, pid, mask, param, score, sp, n)) in enumerate(items):
            k = int(k_final[i])
            if k <= 0:
                mask.fill_(True)
                continue
            if k >= n:
                mask.fill_(False)
                continue

            # score를 mask device로 옮겨서 topk (정확히 k개)
            sc = score.to(device=mask.device)
            sc_flat = sc.reshape(-1)

            # k smallest indices
            idx = torch.topk(sc_flat, k, largest=False, sorted=False).indices

            m_flat = torch.ones_like(sc_flat, dtype=torch.bool, device=mask.device)
            m_flat[idx] = False
            mask.copy_(m_flat.view_as(sc))

    def compute_mask(self, sparsity, scope):
        print("[Debug] pruners/base.py -> compute_mask()함수 호출 : pruning을 위한 mask 생성")
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)
        if scope == 'group':
            self._group_mask(sparsity)
        if scope == 'local_exact':
            # target은 self.original_sparsity 기반으로 정확히 맞춤
            self._local_mask_exact(sparsity, target_sparsity=getattr(self, "original_sparsity", None))

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for _, mask, param in self.named_masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for _, mask, _ in self.named_masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for _, mask, param in self.named_masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        return stats(self.named_masked_parameters)
    
    def state_dict(self):
        state_dict = OrderedDict()
        for name, _, param in self.named_masked_parameters:
            score = self.scores[id(param)]
            state_dict[name] = score.detach().cpu()
        return state_dict
    
    def load_state_dict(self, state_dict):
        print("[Debug] pruners/base.py -> load_state_dict()함수 호출 : pruning을 위한 mask")
        for name, _, param in self.named_masked_parameters:
            self.scores[id(param)] = state_dict[name].to(param.device)


    def get_params(self, model):
        params = []
        names = []

        for name, param in model.named_parameters():
            names.append(name)
            params.append(param)

        return names, params

    def model_setup_and_record_attributes(self, model):
        dtype_record = {}
        requires_grad_record = {}
        for n, p in model.state_dict().items():
            dtype_record[n] = p.data.dtype
            p.data = p.data.type(torch.bfloat16)

        # set requires_grad to be true for getting model's derivatives
        for n, p in model.named_parameters():
            requires_grad_record[n] = p.requires_grad
            p.requires_grad = True

        device = list(self.model.parameters())[0].device
        # self.model.to("cpu")

        return dtype_record, requires_grad_record, device

    def model_reset(self, model, dtype_record, requires_grad_record, device):
        # set to original requires grad
        for n, p in model.named_parameters():
            p.requires_grad = requires_grad_record[n]

        for n, p in model.state_dict().items():
            p.data = p.data.type(dtype_record[n])
            
        model.to(device)

    def convert_spec_to_list(self, spec):
        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = spec.split("-")

        num_layers = int(num_layers)
        res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = float(res_keep_ratio), float(attn_keep_ratio), float(ffn_keep_ratio)

        return num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio
    
