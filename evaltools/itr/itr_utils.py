# -*- coding: utf-8 -*-
"""Evaluation Functions + Slice(SVD/AxisVar) Utilities for Image-Text Retrieval"""

import os, re, time, datetime, hashlib
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============== utils (외부 util에 의존) ===============
import utils  # get_world_size, get_rank

import contextlib
import torch

class CUDAEventTimer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.total_ms = 0.0
        self.iters = 0
        self._start = torch.cuda.Event(enable_timing=True)
        self._end   = torch.cuda.Event(enable_timing=True)

    @contextlib.contextmanager
    def measure(self):
        if not self.enabled:
            yield
            return
        self._start.record()
        yield
        self._end.record()
        torch.cuda.synchronize()
        self.total_ms += self._start.elapsed_time(self._end)
        self.iters += 1

    @property
    def avg_ms(self):
        return self.total_ms / max(1, self.iters)


def _norm_drop(x):
    d = float(x)
    if d > 1.0:  # 5,10,15 같은 퍼센트 수치도 허용
        d = d / 100.0
    return max(0.0, min(1.0, d))

def _stable_int_from_str(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) & 0x7FFFFFFF

# =======================================================
#                      Projectors
# =======================================================

class SliceSVDProjector(nn.Module):
    r"""
    Linear(in=D,out=*) 앞단에 붙여, 입력 x의 feature 차원(D)을 축소/투영.
    공분산/PCA 또는 축-분산(axis variance) 기준으로 k개 축 유지.

    forward 등가식:
      x → (우측)부분공간 선택 V_k → W @ V_k,  x @ V_k → Linear
      == x @ (V_k V_k^T) 를 선형층 앞에서 수행한 것과 동일(부분공간 성분만 보존)

    rotation 모드:
      - 축-분산(원축)  : "axisvar_top" | "axisvar_bottom" | "axisvar_random"
      - PCA/공분산기반 : "pca_topk"    | "pca_bottomk"    | "pca_random"
      - 구버전 별칭    : "topk","bottomk","random"  → 축-분산으로 해석
    """
    def __init__(
        self,
        linear: nn.Linear,
        keep_ratio: float,
        center_input: bool = True,
        max_samples: int = 4096,
        freeze_basis: bool = True,
        rotation: str = "axisvar_top",
        base_seed: int | None = None,
        freeze_random: bool = False,
    ):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.linear        = linear
        self.D             = int(linear.in_features)
        self.keep_ratio    = float(keep_ratio)
        self.center_input  = bool(center_input)
        self.max_samples   = int(max_samples)
        self.freeze_basis  = bool(freeze_basis)
        self.rotation      = str(rotation).lower()
        self.base_seed     = None if base_seed is None else int(base_seed)
        self.freeze_random = bool(freeze_random)

        self.k    = max(1, ceil(self.D * self.keep_ratio))
        self._V_k = None  # (D, k)
        self.name = getattr(self, "name", f"{self.__class__.__name__}-{id(self)}")

        print(f"[SVD:init] {self.name} | in={self.D}, out={linear.out_features}, "
              f"keep_ratio={self.keep_ratio:.3f}, k={self.k}, rotation={self.rotation}, "
              f"center={self.center_input}, freeze_basis={self.freeze_basis}")

    # ---------------- RNG ----------------
    def _make_generator(self, device):
        if (self.base_seed is None) and (not self.freeze_random):
            return None
        base = 0 if self.base_seed is None else int(self.base_seed)
        sid  = _stable_int_from_str(self.name)
        seed = (base ^ sid) & 0x7FFFFFFF
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g

    @torch.no_grad()
    def _random_orthonormal(self, D, device, dtype):
        g = self._make_generator(device)
        R = torch.randn(D, D, device=device, dtype=dtype, generator=g) if g is not None \
            else torch.randn(D, D, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(R.float(), mode="reduced")
        return Q.to(dtype)

    # -------------- PCA builders --------------
    @torch.no_grad()
    def _build_basis_pca(self, x2d: torch.Tensor, bottomk: bool = False, random: bool = False):
        """공분산/고유벡터 이용."""
        N, D = x2d.shape
        X = x2d
        if self.max_samples and N > self.max_samples:
            idx = torch.randperm(N, device=x2d.device)[:self.max_samples]
            X = X.index_select(0, idx)
            N = X.shape[0]

        Xf = X.float()
        if self.center_input:
            Xf = Xf - Xf.mean(dim=0, keepdim=True)

        # NaN/Inf 제거
        finite_mask = torch.isfinite(Xf).all(dim=1)
        if not finite_mask.all():
            Xf = Xf[finite_mask]
            N = int(Xf.shape[0])
        if N < 1:
            # 데이터가 없으면 Random orthonormal로 폴백
            self._V_k = self._random_orthonormal(D, x2d.device, x2d.dtype)[:, :self.k].contiguous()
            print(f"[PCA:fallback] N<1 → random ortho keep k={self.k}/{D}")
            return

        # 공분산 및 고유분해
        C = Xf.T @ Xf
        C = 0.5 * (C + C.T)
        try:
            evals, V = torch.linalg.eigh(C)  # 작은→큰
        except Exception:
            # SVD 폴백
            U, S, Vh = torch.linalg.svd(Xf, full_matrices=False)
            V = Vh.transpose(0, 1)
            evals = S**2

        k = max(1, min(self.k, V.shape[1]))
        if random:
            g = self._make_generator(V.device)
            perm = torch.randperm(V.shape[1], device=V.device, generator=g) if g is not None \
                   else torch.randperm(V.shape[1], device=V.device)
            idx = torch.sort(perm[:k]).values
            V_k = V[:, idx]
            mode = "pca_random"
            kept_energy = float(evals[idx].sum().item()) if evals.ndim == 1 and idx.numel() > 0 else 0.0
        elif bottomk:
            V_k = V[:, :k]
            kept_energy = float(evals[:k].sum().item()) if evals.ndim == 1 else float(evals.sum().item())
            mode = "pca_bottomk"
        else:
            V_k = V[:, -k:]
            kept_energy = float(evals[-k:].sum().item()) if evals.ndim == 1 else float(evals.sum().item())
            mode = "pca_topk"

        self._V_k = V_k.to(x2d.dtype).contiguous()
        tot = float(evals.sum().item()) + 1e-12
        print(f"[PCA:basis/{mode}] kept k={k}/{D}, energy_keep≈{kept_energy/tot:.3f} | N={N}")

    # -------------- Axis-variance builders --------------
    @torch.no_grad()
    def _build_basis_axisvar(self, x2d: torch.Tensor, bottomk: bool = False, random: bool = False):
        """공분산/고유벡터 없이, 열별 분산(diag)만으로 k개 축 유지."""
        N, D = x2d.shape
        X = x2d
        if self.max_samples and N > self.max_samples:
            idx = torch.randperm(N, device=x2d.device)[:self.max_samples]
            X = X.index_select(0, idx)
            N = X.shape[0]

        Xf = X.float()
        if self.center_input:
            Xf = Xf - Xf.mean(dim=0, keepdim=True)

        finite_mask = torch.isfinite(Xf).all(dim=1)
        if not finite_mask.all():
            Xf = Xf[finite_mask]
            N = int(Xf.shape[0])
        if N < 1:
            I = torch.eye(D, device=x2d.device, dtype=x2d.dtype)
            self._V_k = I[:, :min(self.k, D)].contiguous()
            print(f"[AxisVar:fallback] N<1 → keep first {min(self.k,D)}/{D}")
            return

        var = Xf.pow(2).mean(dim=0)  # (D,)
        k = max(1, min(self.k, D))

        if random:
            g = self._make_generator(x2d.device)
            perm = torch.randperm(D, device=x2d.device, generator=g) if g is not None \
                   else torch.randperm(D, device=x2d.device)
            keep_idx = torch.sort(perm[:k]).values
            mode = "axisvar_random"
        elif bottomk:
            keep_idx = torch.topk(-var, k, largest=True).indices
            mode = "axisvar_bottom"
        else:
            keep_idx = torch.topk(var, k, largest=True).indices
            mode = "axisvar_top"

        I = torch.eye(D, device=x2d.device, dtype=x2d.dtype)
        V_k = I.index_select(1, keep_idx).contiguous()
        self._V_k = V_k.to(x2d.dtype)

        kept = float(var[keep_idx].sum().item())
        tot  = float(var.sum().item()) + 1e-12
        print(f"[AxisVar:basis/{mode}] kept k={k}/{D}, var_keep≈{kept/tot:.3f} | "
              f"N={N} | var(min/mean/max)={float(var.min()):.3e}/{float(var.mean()):.3e}/{float(var.max()):.3e}")

    @torch.no_grad()
    def _build_basis(self, x2d: torch.Tensor):
        rot = self.rotation
        # 축-분산 전용 모드
        if rot in ("axisvar_top", "axisvar_bottom", "axisvar_random"):
            return self._build_basis_axisvar(
                x2d,
                bottomk=(rot == "axisvar_bottom"),
                random=(rot == "axisvar_random"),
            )
        # PCA 전용 모드
        if rot in ("pca_topk", "pca_bottomk", "pca_random"):
            return self._build_basis_pca(
                x2d,
                bottomk=(rot == "pca_bottomk"),
                random=(rot == "pca_random"),
            )
        # 구버전 별칭: 축-분산으로 해석
        if rot in ("topk", "bottomk", "random"):
            return self._build_basis_axisvar(
                x2d,
                bottomk=(rot == "bottomk"),
                random=(rot == "random"),
            )
        # 기본값: 축-분산 top
        return self._build_basis_axisvar(x2d, bottomk=False, random=False)

    # ---------------- Forward ----------------
    def forward(self, x: torch.Tensor):
        if self.k >= self.D:
            return F.linear(x, self.linear.weight, self.linear.bias)

        D = self.D
        assert x.size(-1) == D, f"in_features mismatch: {x.size(-1)} != {D}"
        x2d = x.reshape(-1, D)

        if (self._V_k is None) or (not self.freeze_basis):
            self._build_basis(x2d)

        V_k = self._V_k  # (D, k)
        k = V_k.shape[1]

        # 진단: 입력 크기
        with torch.no_grad():
            try:
                print(f"[Slice] {self.name} | ||x||_mean={float(x.abs().mean()):.3e} | k={k}/{D}")
            except Exception:
                pass

        # 축소 후 선형
        x_red = x.matmul(V_k)                  # (..., k)
        Wk    = self.linear.weight.matmul(V_k) # (out, k)

        # 직교성 체크
        Ik = torch.eye(k, device=V_k.device, dtype=V_k.dtype)
        ortho_err = (V_k.transpose(0,1) @ V_k - Ik).float().norm(p='fro') / max(1, k)
        print(f"[Slice:check] {self.name} | x→{tuple(x_red.shape)} | W→{tuple(Wk.shape)} "
              f"| ortho_err={float(ortho_err):.2e}")

        return F.linear(x_red, Wk, self.linear.bias)


# ---------------- Token-axis projector ----------------
class TokenSVDProjector(nn.Module):
    """
    입력 x: (B, L, D) — 토큰 축(L)에 대해 회전/투영 적용.
    rotation: "topk"|"bottomk"|"random" (PCA 느낌)
    """
    def __init__(
        self,
        module: nn.Module,
        keep_ratio: float,
        center_input: bool = True,
        max_samples: int = 8192,
        freeze_basis: bool = True,
        rotation: str = "topk",
        base_seed: int | None = None,
        freeze_random: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        assert isinstance(module, nn.Module)
        self.module        = module
        self.keep_ratio    = float(keep_ratio)
        self.center_input  = bool(center_input)
        self.max_samples   = int(max_samples)
        self.freeze_basis  = bool(freeze_basis)
        self.rotation      = str(rotation).lower()
        self.base_seed     = None if base_seed is None else int(base_seed)
        self.freeze_random = bool(freeze_random)
        self.verbose       = bool(verbose)

        self.L   = None
        self.k   = None
        self._Uk = None
        self.name = getattr(self, "name", f"{self.__class__.__name__}-{id(self)}")

    def _make_generator(self, device='cpu'):
        if (self.base_seed is None) and (not self.freeze_random):
            return None
        base = 0 if self.base_seed is None else int(self.base_seed)
        sid  = _stable_int_from_str(self.name)
        seed = (base ^ sid) & 0x7FFFFFFF
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g

    @torch.no_grad()
    def _random_full(self, L, device, dtype):
        g = self._make_generator(device)
        R = torch.randn(L, L, device=device, dtype=dtype, generator=g) if g is not None \
            else torch.randn(L, L, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(R.float(), mode="reduced")
        return Q.to(dtype)

    @torch.no_grad()
    def _build_basis_from_batch(self, x: torch.Tensor):
        assert x.dim() >= 3, "expected (B,L,D)"
        B, L, D = x.shape[0], x.shape[1], x.shape[2]
        self.L = L
        self.k = max(1, ceil(L * self.keep_ratio))

        if self.k >= L:
            self._Uk = None
            if self.verbose:
                print(f"[TokenSVD] no-op L={L}, keep all")
            return

        X = x.permute(0, 2, 1).reshape(-1, L)  # (B*D, L)
        if self.max_samples and X.shape[0] > self.max_samples:
            idx = torch.randperm(X.shape[0], device=X.device)[:self.max_samples]
            X = X.index_select(0, idx)
        if self.center_input:
            X = X - X.mean(dim=0, keepdim=True)

        C = X.T.float() @ X.float()  # (L, L)
        evals, U = torch.linalg.eigh(C)

        if self.rotation == "random":
            g = self._make_generator(device=x.device)
            perm = torch.randperm(U.shape[1], device=x.device, generator=g) if g is not None \
                   else torch.randperm(U.shape[1], device=x.device)
            idx = torch.sort(perm[:self.k]).values
            Uk = U[:, idx]
        elif self.rotation == "bottomk":
            Uk = U[:, :self.k]
        else:
            Uk = U[:, -self.k:]

        self._Uk = Uk.to(x.dtype)

    @torch.no_grad()
    def _project_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self._Uk is None or self.k is None or self.L is None or self.k >= self.L:
            return x
        Uk  = self._Uk
        UkT = Uk.transpose(0, 1)
        x_red = torch.einsum('bld,lk->bkd', x, Uk)
        x_prj = torch.einsum('bkd,kl->bld', x_red, UkT)

        L, k = self.L, self.k
        Ik = torch.eye(k, device=Uk.device, dtype=Uk.dtype)
        ortho_err = (Uk.T @ Uk - Ik).float().norm(p='fro') / max(1, k)
        try:
            samp = x if x.shape[0] <= 4 else x[:4]
            P = (Uk.float() @ UkT.float())
            xp = torch.einsum('ij,bjd->bid', P, samp.float())
            rel_err = (x_prj[:samp.size(0)].float() - xp).norm(p='fro') / (xp.norm(p='fro').clamp_min(1e-12))
        except Exception:
            rel_err = torch.tensor(0.0, device=x.device)
        print(f"[TokenSVD:check] {self.name} | L={L}, k={k} | ortho_err={float(ortho_err):.2e} "
              f"| EquivRelErr={float(rel_err):.2e}")
        return x_prj

    def forward(self, *args, **kwargs):
        # hidden_states 인자를 우선적으로 찾음
        if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            x = kwargs["hidden_states"]
            need_kw = True
        else:
            if len(args) == 0 or (not torch.is_tensor(args[0])):
                return self.module(*args, **kwargs)
            x = args[0]
            need_kw = False

        if (self._Uk is None) or (not self.freeze_basis) or (self.L is None):
            self._build_basis_from_batch(x)

        x_proj = self._project_tokens(x)

        if need_kw:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = x_proj
            return self.module(*args, **kwargs)
        else:
            args = list(args)
            args[0] = x_proj
            return self.module(*args, **kwargs)


# --------------- replacers ---------------
def recursive_get_module(root, dotted):
    cur = root
    for p in dotted.split('.'):
        cur = getattr(cur, p)
    return cur

def replace_linear_with_svd_projector(
    model, dotted_name, keep_ratio,
    center_input=True, max_samples=4096, freeze_basis=True, rotation=None,
    base_seed=None, freeze_random=False
):
    parent = model
    *parents, leaf = dotted_name.split(".")
    for p in parents:
        parent = getattr(parent, p)
    old = getattr(parent, leaf)
    assert isinstance(old, nn.Linear), f"{dotted_name} is not Linear"

    wrapped = SliceSVDProjector(
        old,
        keep_ratio=keep_ratio,
        center_input=center_input,
        max_samples=max_samples,
        freeze_basis=freeze_basis,
        rotation=rotation,
        base_seed=base_seed,
        freeze_random=freeze_random,
    )
    setattr(parent, leaf, wrapped)
    setattr(wrapped, "name", dotted_name)
    print(f"[SVD:replace] {dotted_name} | keep_ratio={keep_ratio:.3f} | rotation={rotation}")
    return dotted_name

def replace_many_by_regex_svd(model, patterns, keep_ratio, **kw):
    replaced = []
    def walk(root, prefix=""):
        for name, m in root.named_children():
            dotted = f"{prefix}.{name}" if prefix else name
            matched_here = False
            if isinstance(m, nn.Linear):
                for pat in patterns:
                    if re.fullmatch(pat, dotted):
                        replace_linear_with_svd_projector(model, dotted, keep_ratio, **kw)
                        replaced.append(dotted)
                        matched_here = True
                        break
            # 매칭되었으면 더 깊이 들어가면 원래 m가 바뀌어서 꼬일 수 있으니 스킵
            if matched_here:
                continue
            walk(m, dotted)
    walk(model, "")
    return replaced

def replace_module_with_token_projector(model, dotted_name: str, keep_ratio: float, **kw):
    parent = model
    *parents, leaf = dotted_name.split(".")
    for p in parents:
        parent = getattr(parent, p)
    old = getattr(parent, leaf)
    assert isinstance(old, nn.Module), f"{dotted_name} is not nn.Module"

    wrapped = TokenSVDProjector(old, keep_ratio=keep_ratio, **kw)
    setattr(wrapped, "name", dotted_name)
    setattr(parent, leaf, wrapped)
    print(f"[TokenSVD:replace] {dotted_name} | keep_ratio={keep_ratio:.3f} | rotation={kw.get('rotation','topk')}")
    return dotted_name

def replace_token_many_by_regex_svd(model, patterns, keep_ratio, **kw):
    replaced = []
    def walk(root, prefix=""):
        for name, m in root.named_children():
            dotted = f"{prefix}.{name}" if prefix else name
            matched_here = False
            for pat in patterns:
                if re.fullmatch(pat, dotted):
                    replace_module_with_token_projector(model, dotted, keep_ratio, **kw)
                    replaced.append(dotted)
                    matched_here = True
                    break
            if matched_here:
                continue
            walk(m, dotted)
    walk(model, "")
    return replaced

# =======================================================
#                      Evaluations
# =======================================================

@torch.no_grad()
def blip_evaluation(model, data_loader, tokenizer, fabric, config, debug_mode=False):
    print("[Debug] evaltools/itr_utils.py -> blip_evaluation()")
    t0 = time.time()
    model.eval()
    torch.cuda.empty_cache()
    device = fabric.device

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    texts = data_loader.dataset.text
    num_text = len(texts)
    num_images = len(data_loader.dataset.image)

    text_bs = int(config.get('batch_size_test_text', 256))
    print('Computing features for evaluation...')

    score_matrix_i2t = torch.full((num_images, num_text), -100.0, device=device)
    score_matrix_t2i = torch.full((num_text, num_images), -100.0, device=device)

    # ---- text embeds ----
    text_ids_list, text_embeds_list, text_atts_list = [], [], []
    for i in range(0, num_text, text_bs):
        cur = min(num_text, i + text_bs)
        print(f"Processing text {cur}/{num_text}", end='\r' if cur < num_text else "\n")
        text = texts[i:cur]
        text_input = tokenizer(text, padding='max_length', truncation=True,
                               max_length=int(config.get("max_tokens", 35)),
                               return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds_list.append(text_embed.cpu())
        text_ids_list.append(text_input.input_ids.cpu())
        text_atts_list.append(text_input.attention_mask.cpu())
        del text_input, text_output, text_embed

    text_embeds = torch.cat(text_embeds_list, dim=0).to(device); del text_embeds_list
    text_ids    = torch.cat(text_ids_list,    dim=0).to(device); del text_ids_list
    text_atts   = torch.cat(text_atts_list,   dim=0).to(device); del text_atts_list
    text_ids[:, 0] = tokenizer.enc_token_id

    # ---- image feats ----
    image_feats_list, image_embeds_list = [], []
    for i, (image, img_id) in enumerate(data_loader):
        print(f"Processing image batch {i+1}/{len(data_loader)}", end='\r' if i != len(data_loader)-1 else "\n")
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = F.normalize(model.vision_proj(image_feat[:,0,:]), dim=-1)
        image_feats_list.append(image_feat.cpu())
        image_embeds_list.append(image_embed.cpu())
        del image, image_feat, image_embed

    torch.cuda.empty_cache()
    image_feats  = torch.cat(image_feats_list,  dim=0).to(device); del image_feats_list
    image_embeds = torch.cat(image_embeds_list, dim=0).to(device); del image_embeds_list

    sims_matrix = image_embeds @ text_embeds.t()
    step  = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end   = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(sims_matrix[start:end]):
        if debug_mode and i == 300: break
        topk_sim, topk_idx = sims.topk(k=int(config.get('k_test', 32)), dim=0)
        encoder_output = image_feats[start+i].repeat(topk_idx.numel(),1,1).to(device)
        encoder_att    = torch.ones(encoder_output.size()[:-1], dtype=torch.long, device=device)
        output = model.text_encoder(text_ids[topk_idx],
                                    attention_mask=text_atts[topk_idx],
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True)
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i, topk_idx] = score + topk_sim
        del output, encoder_att, encoder_output
        if (i+1) % 50 == 0 or (i+1) == end-start:
            print(f"[Evaluation] i2t {i+1}/{end-start} ({(i+1)/(end-start)*100:.2f}%)")

    torch.cuda.empty_cache()
    sims_matrix = sims_matrix.t()
    step  = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end   = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(sims_matrix[start:end]):
        if debug_mode and i == 300: break
        topk_sim, topk_idx = sims.topk(k=int(config.get('k_test', 32)), dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att    = torch.ones(encoder_output.size()[:-1], dtype=torch.long, device=device)
        output = model.text_encoder(text_ids[start+i].repeat(topk_idx.numel(),1),
                                    attention_mask=text_atts[start+i].repeat(topk_idx.numel(),1),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True)
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i, topk_idx] = score + topk_sim
        del output, encoder_att, encoder_output
        if (i+1) % 50 == 0 or (i+1) == end-start:
            print(f"[Evaluation] t2i {i+1}/{end-start} ({(i+1)/(end-start)*100:.2f}%)")

    fabric.barrier()
    score_matrix_i2t = fabric.all_reduce(score_matrix_i2t, reduce_op="sum")
    score_matrix_t2i = fabric.all_reduce(score_matrix_t2i, reduce_op="sum")

    print("Evaluation time {}".format(str(datetime.timedelta(seconds=int(time.time() - t0)))))
    return score_matrix_i2t.float().cpu().numpy(), score_matrix_t2i.float().cpu().numpy()


@torch.no_grad()
def xvlm_evaluation(model, data_loader, tokenizer, fabric, config, debug_mode=False):
    t0 = time.time()
    model.eval()
    torch.cuda.empty_cache()
    device = fabric.device

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    texts = data_loader.dataset.text
    num_text = len(texts)
    num_images = len(data_loader.dataset.image)
    text_bs = int(config.get('batch_size_test_text', 256))
    print('Computing features for evaluation...')

    score_matrix_i2t = torch.full((num_images, num_text), -100.0, device=device)
    score_matrix_t2i = torch.full((num_text, num_images), -100.0, device=device)

    text_feats_list, text_embeds_list, text_atts_list = [], [], []
    for i in range(0, num_text, text_bs):
        cur = min(num_text, i + text_bs)
        print(f"Processing text {cur}/{num_text}", end='\r' if cur < num_text else "\n")
        text = texts[i:cur]
        text_input = tokenizer(text, padding='max_length', truncation=True,
                               max_length=int(config.get('max_tokens', 64)),
                               return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat   = text_output.last_hidden_state
        text_embed  = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds_list.append(text_embed.cpu())
        text_feats_list.append(text_feat.cpu())
        text_atts_list.append(text_input.attention_mask.cpu())
        del text_embed, text_feat, text_input, text

    text_embeds = torch.cat(text_embeds_list, dim=0).to(device); del text_embeds_list
    text_feats  = torch.cat(text_feats_list,  dim=0).to(device); del text_feats_list
    text_atts   = torch.cat(text_atts_list,   dim=0).to(device); del text_atts_list

    image_feats_list, image_embeds_list = [], []
    for i, (image, img_id) in enumerate(data_loader):
        print(f"Processing image batch {i+1}/{len(data_loader)}", end='\r' if i != len(data_loader)-1 else "\n")
        image_feat  = model.vision_encoder(image.to(device))
        image_embed = F.normalize(model.vision_proj(image_feat[:,0,:]), dim=-1)
        image_feats_list.append(image_feat.cpu())
        image_embeds_list.append(image_embed.cpu())
        del image_embed, image_feat, image

    torch.cuda.empty_cache()
    image_feats  = torch.cat(image_feats_list,  dim=0).to(device); del image_feats_list
    image_embeds = torch.cat(image_embeds_list, dim=0).to(device); del image_embeds_list

    sims_matrix = image_embeds @ text_embeds.t()
    step  = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end   = min(sims_matrix.size(0), start+step)

    for i, sims in enumerate(sims_matrix[start:end]):
        if debug_mode and i == 300: break
        topk_sim, topk_idx = sims.topk(k=int(config.get('k_test', 32)), dim=0)
        encoder_output = image_feats[start+i].repeat(topk_idx.numel(),1,1)
        encoder_att    = torch.ones(encoder_output.size()[:-1], dtype=torch.long, device=device)
        output = model.text_encoder(
            encoder_embeds=text_feats[topk_idx],
            attention_mask=text_atts[topk_idx],
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode='fusion'
        )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i, topk_idx] = score
        del output, encoder_att, encoder_output
        if (i+1) % 50 == 0 or (i+1) == end-start:
            print(f"[Evaluation] i2t {i+1}/{end-start} ({(i+1)/(end-start)*100:.2f}%)")

    torch.cuda.empty_cache()
    sims_matrix = sims_matrix.t()
    step  = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end   = min(sims_matrix.size(0), start+step)

    for i, sims in enumerate(sims_matrix[start:end]):
        if debug_mode and i == 300: break
        topk_sim, topk_idx = sims.topk(k=int(config.get('k_test', 32)), dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att    = torch.ones(encoder_output.size()[:-1], dtype=torch.long, device=device)
        output = model.text_encoder(
            encoder_embeds=text_feats[start+i].repeat(topk_idx.numel(),1,1),
            attention_mask=text_atts[start+i].repeat(topk_idx.numel(),1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode='fusion'
        )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i, topk_idx] = score
        del output, encoder_att, encoder_output
        if (i+1) % 50 == 0 or (i+1) == end-start:
            print(f"[Evaluation] t2i {i+1}/{end-start} ({(i+1)/(end-start)*100:.2f}%)")

    fabric.barrier()
    score_matrix_i2t = fabric.all_reduce(score_matrix_i2t, reduce_op="sum")
    score_matrix_t2i = fabric.all_reduce(score_matrix_t2i, reduce_op="sum")

    print("Evaluation time {}".format(str(datetime.timedelta(seconds=int(time.time() - t0)))))
    return score_matrix_i2t.float().cpu().numpy(), score_matrix_t2i.float().cpu().numpy()


@torch.no_grad()
def clip_evaluation(model, data_loader, tokenizer, fabric, config, debug_mode=False):
    print("[Debug] evaltools/itr_utils.py -> clip_evaluation()")
    t0 = time.time()
    model.eval()
    torch.cuda.empty_cache()
    device = fabric.device

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    texts = data_loader.dataset.text
    num_text = len(texts)
    num_images = len(data_loader.dataset.image)

    text_bs = int(config.get('batch_size_test_text', 64))
    max_len = int(config.get('max_tokens', 77))

    # ---- runtime slice 주입 (dim/token) ----
    inner = getattr(model, "model", model)
    rts = config.get("runtime_slice")
    if rts:
        enabled = bool(rts.get("enabled", True))
        if enabled:
            axis = str(rts.get("axis", "dim")).lower()
            drop = _norm_drop(rts.get("drop_percent", 0.0))
            keep_ratio = 1.0 - drop
            center_input = bool(rts.get("center_input", True))
            freeze_basis = bool(rts.get("freeze_basis", True))
            rotation     = str(rts.get("rotation", "axisvar_top")).lower()  # 기본은 원축 Top
            max_samples  = int(rts.get("max_samples", 4096))
            base_seed    = rts.get("base_seed")
            print(f"[SliceConfig] axis={axis} keep_ratio={keep_ratio:.3f} rotation={rotation} "
                  f"center={center_input} freeze_basis={freeze_basis}")

            if axis == "token":
                pats_tok = [
                    r"text_model\.encoder\.layers\.\d+\.self_attn$",
                    r"vision_model\.encoder\.layers\.\d+\.self_attn$",
                    r"(?:visual\.)?transformer\.resblocks\.\d+\.attn$",
                ]
                replaced_tok = replace_token_many_by_regex_svd(
                    inner, pats_tok, keep_ratio,
                    center_input=center_input, max_samples=int(rts.get("max_samples", 8192)),
                    freeze_basis=freeze_basis, rotation=rotation,
                    base_seed=base_seed, freeze_random=bool(rts.get("freeze_random", False)),
                    verbose=bool(rts.get("verbose", False)),
                )
                print("[TokenSVD] replaced:", len(replaced_tok))
                if len(replaced_tok) == 0:
                    print("[ERROR] TokenSVD: No module matched — check dotted names.")
            else:
                pats = [
                    r"text_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
                    r"vision_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
                    r"(?:visual\.)?transformer\.resblocks\.\d+\.attn\.(?:q_proj|k_proj|v_proj|out_proj)",
                ]
                replaced = replace_many_by_regex_svd(
                    inner, pats, keep_ratio,
                    center_input=center_input, max_samples=max_samples,
                    freeze_basis=freeze_basis, rotation=rotation, base_seed=base_seed
                )
                print("[SlicePCA] replaced:", len(replaced))
                if len(replaced) == 0:
                    print("[ERROR] SlicePCA: No Linear matched — check regex/dotted names.")

    # ---- text embeds ----
    print("[Debug] 텍스트 임베딩 추출...")
    text_embeds_list = []
    for i in range(0, num_text, text_bs):
        j = min(num_text, i + text_bs)
        batch_texts = texts[i:j]
        text_inputs = tokenizer(batch_texts, padding="max_length", truncation=True,
                                max_length=max_len, return_tensors="pt").to(device)
        feats = model.encode_text(text_inputs.input_ids, text_inputs.attention_mask)
        feats = F.normalize(feats, dim=-1)
        text_embeds_list.append(feats.cpu())
        del text_inputs, feats

    text_embeds = torch.cat(text_embeds_list, dim=0).to(device); del text_embeds_list
    torch.cuda.empty_cache()

    # ---- image embeds ----
    print("[Debug] 이미지 임베딩 추출...")
    image_embeds_list = []
    for i, (images, img_ids) in enumerate(data_loader):
        if debug_mode and i % 10 == 0:
            print(f"  - image batch {i+1}/{len(data_loader)}")
        images = images.to(device, non_blocking=True)
        feats  = model.encode_image(images)
        feats  = F.normalize(feats, dim=-1)
        image_embeds_list.append(feats.cpu())
        del images, feats

    image_embeds = torch.cat(image_embeds_list, dim=0).to(device); del image_embeds_list
    torch.cuda.empty_cache()

    # ---- similarity ----
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "logit_scale"):
        logit_scale = float(inner.logit_scale.exp().item())
    else:
        logit_scale = 1.0
    sims = (image_embeds @ text_embeds.t()) * logit_scale

    # ---- gather to matrices ----
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    score_matrix_i2t = torch.zeros((len(image_embeds), len(text_embeds)), device=device)
    score_matrix_t2i = torch.zeros((len(text_embeds), len(image_embeds)), device=device)

    step  = sims.size(0) // num_tasks + 1
    start = rank * step
    end   = min(sims.size(0), start + step)
    score_matrix_i2t[start:end, :] = sims[start:end, :]
    score_matrix_t2i[:, start:end] = sims.t()[:, start:end]

    fabric.barrier()
    score_matrix_i2t = fabric.all_reduce(score_matrix_i2t, reduce_op="sum")
    score_matrix_t2i = fabric.all_reduce(score_matrix_t2i, reduce_op="sum")

    print("Evaluation time {}".format(str(datetime.timedelta(seconds=int(time.time() - t0)))))
    return score_matrix_i2t.float().cpu().numpy(), score_matrix_t2i.float().cpu().numpy()


@torch.no_grad()
def clipG_evaluation(model, data_loader, tokenizer, fabric, config, debug_mode=False):
    """
    CLIP-G: encode_image / encode_text 제공 + inner.logit_scale 사용
    runtime_slice(axis=dim|token) 적용 가능

    Timing:
      - 측정 포함: encode_text/encode_image (+ optional normalize), similarity matmul
      - 측정 제외: tokenizer, H2D(.to), D2H(.cpu), empty_cache, all_reduce/barrier(통신)
    """
    print("[Debug] evaltools/itr_utils.py -> clipG_evaluation()")
    t0 = time.time()
    model.eval()
    torch.cuda.empty_cache()
    device = fabric.device

    # -------------------------
    # inference-only timers
    # -------------------------
    t_text = CUDAEventTimer(enabled=True)
    t_img  = CUDAEventTimer(enabled=True)
    t_sim  = CUDAEventTimer(enabled=True)

    # "encode만" vs "encode+normalize" 포함 여부
    MEASURE_WITH_NORMALIZE = True

    # 워밍업 (타이머 없이 몇 번 실행)
    WARMUP_TEXT_ITERS = 2
    WARMUP_IMG_ITERS  = 2

    # DDP info
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    texts = data_loader.dataset.text
    num_text = len(texts)
    num_images = len(data_loader.dataset.image)

    text_bs = int(config.get('batch_size_train', 16))
    max_len = int(config.get('max_tokens', getattr(model, "max_tokens", 77)))

    inner = getattr(model, "model", model)

    # ---- runtime slice 주입 ---- (모델 준비 단계: inference-only 측정 대상 아님)
    rts = config.get("runtime_slice")
    if rts:
        enabled = bool(rts.get("enabled", True))
        if enabled:
            axis = str(rts.get("axis", "dim")).lower()
            drop = _norm_drop(rts.get("drop_percent", 0.0))
            keep_ratio  = 1.0 - drop
            rotation    = str(rts.get("rotation", "axisvar_top")).lower()
            center_input= bool(rts.get("center_input", True))
            freeze_basis= bool(rts.get("freeze_basis", True))
            max_samples = int(rts.get("max_samples", 4096))
            base_seed   = rts.get("base_seed")

            print(f"[SliceConfig] axis={axis} keep_ratio={keep_ratio:.3f} rotation={rotation} "
                  f"center={center_input} freeze_basis={freeze_basis}")

            if axis == "token":
                pats_tok = [
                    r"text_model\.encoder\.layers\.\d+\.self_attn$",
                    r"vision_model\.encoder\.layers\.\d+\.self_attn$",
                    r"(?:visual\.)?transformer\.resblocks\.\d+\.attn$",
                ]
                replaced_tok = replace_token_many_by_regex_svd(
                    inner, pats_tok, keep_ratio,
                    center_input=bool(rts.get("center_input", False)),
                    max_samples=int(rts.get("max_samples", 8192)),
                    freeze_basis=freeze_basis, rotation=rotation, base_seed=base_seed,
                    freeze_random=bool(rts.get("freeze_random", False)),
                    verbose=bool(rts.get("verbose", False)),
                )
                print("[TokenSVD] replaced:", len(replaced_tok))
                if len(replaced_tok) == 0:
                    print("[ERROR] TokenSVD: No module matched — check dotted names.")
            else:
                pats = [
                    r"text_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
                    r"vision_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
                    r"(?:visual\.)?transformer\.resblocks\.\d+\.attn\.(?:q_proj|k_proj|v_proj|out_proj)",
                ]
                replaced = replace_many_by_regex_svd(
                    inner, pats, keep_ratio,
                    center_input=center_input, max_samples=max_samples,
                    freeze_basis=freeze_basis, rotation=rotation, base_seed=base_seed
                )
                print("[SlicePCA] replaced:", len(replaced))
                if len(replaced) == 0:
                    print("[ERROR] SlicePCA: No Linear matched — check regex/dotted names.")

    # ---- text embeds ----
    print("[Debug] 텍스트 임베딩 추출...")
    text_embeds_list = []

    warm = 0
    for i in range(0, num_text, text_bs):
        j = min(num_text, i + text_bs)
        batch_texts = texts[i:j]

        # tokenizer는 측정 제외
        text_inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        # H2D 복사도 측정 제외
        input_ids = text_inputs.input_ids.to(device, non_blocking=True)
        attn_mask = text_inputs.attention_mask.to(device, non_blocking=True)

        # 워밍업 (타이머 X)
        if warm < WARMUP_TEXT_ITERS:
            feats = model.encode_text(input_ids, attn_mask)
            if MEASURE_WITH_NORMALIZE:
                feats = F.normalize(feats, dim=-1)
            warm += 1
        else:
            # ✅ inference-only: encode_text(+optional normalize)만
            with t_text.measure():
                feats = model.encode_text(input_ids, attn_mask)
                if MEASURE_WITH_NORMALIZE:
                    feats = F.normalize(feats, dim=-1)

        # D2H(.cpu)는 측정 제외
        text_embeds_list.append(feats.cpu())

        del text_inputs, input_ids, attn_mask, feats

    text_embeds = torch.cat(text_embeds_list, dim=0).to(device)
    del text_embeds_list
    torch.cuda.empty_cache()

    # ---- image embeds ----
    print("[Debug] 이미지 임베딩 추출...")
    image_embeds_list = []

    warm = 0
    for i, (images, img_ids) in enumerate(data_loader):
        if debug_mode and i % 10 == 0:
            print(f"  - image batch {i+1}/{len(data_loader)}")

        # H2D 복사 측정 제외
        images = images.to(device, non_blocking=True)

        # 워밍업 (타이머 X)
        if warm < WARMUP_IMG_ITERS:
            feats = model.encode_image(images)
            if MEASURE_WITH_NORMALIZE:
                feats = F.normalize(feats, dim=-1)
            warm += 1
        else:
            # ✅ inference-only: encode_image(+optional normalize)만
            with t_img.measure():
                feats = model.encode_image(images)
                if MEASURE_WITH_NORMALIZE:
                    feats = F.normalize(feats, dim=-1)

        # D2H(.cpu) 측정 제외
        image_embeds_list.append(feats.cpu())
        del images, feats

    image_embeds = torch.cat(image_embeds_list, dim=0).to(device)
    del image_embeds_list
    torch.cuda.empty_cache()

    # ---- similarity ----
    inner2 = getattr(model, "model", None)
    logit_scale = float(inner2.logit_scale.exp().item()) if (inner2 is not None and hasattr(inner2, "logit_scale")) else 1.0

    # ✅ inference-only: GPU matmul (원하면 제외 가능)
    with t_sim.measure():
        sims = (image_embeds @ text_embeds.t()) * logit_scale

    # ---- gather (통신: inference-only에서 제외) ----
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    I, T = sims.size(0), sims.size(1)

    score_matrix_i2t = torch.zeros((I, T), device=device)
    score_matrix_t2i = torch.zeros((T, I), device=device)

    step  = I // num_tasks + 1
    start = rank * step
    end   = min(I, start + step)

    score_matrix_i2t[start:end, :] = sims[start:end, :]
    score_matrix_t2i[:, start:end] = sims.t()[:, start:end]

    fabric.barrier()
    score_matrix_i2t = fabric.all_reduce(score_matrix_i2t, reduce_op="sum")
    score_matrix_t2i = fabric.all_reduce(score_matrix_t2i, reduce_op="sum")

    # ---- report ----
    if fabric.is_global_zero:
        print(
            f"[Inference-only timing]\n"
            f"  text encode: total={t_text.total_ms/1000:.6f}s  "
            f"avg={t_text.total_ms/max(1,num_text):.4f}ms/text  (N_text={num_text})\n"
            f"  img  encode: total={t_img.total_ms/1000:.6f}s  "
            f"avg={t_img.total_ms/max(1,num_images):.4f}ms/image (N_img={num_images})\n"
            f"  sim  matmul: total={t_sim.total_ms/1000:.6f}s  avg={t_sim.avg_ms:.3f}ms/call\n"
            f"  total inference-only: {(t_text.total_ms+t_img.total_ms+t_sim.total_ms)/1000:.6f}s\n"
            f"  excluded: tokenizer / H2D / D2H / all_reduce+barrier"
        )

    print("Evaluation time {}".format(str(datetime.timedelta(seconds=int(time.time() - t0)))))

    return (
        score_matrix_i2t.float().cpu().numpy(),
        score_matrix_t2i.float().cpu().numpy(),
    )

@torch.no_grad()
def blip2_evaluation(model, data_loader, tokenizer, fabric, config, debug_mode=False):
    print("[Debug] evaltools/itr_utils.py -> blip2_evaluation()")
    t0 = time.time()

    base   = getattr(model, "model", model)
    device = getattr(fabric, "device", next(base.parameters()).device)
    param_dtype = next(base.parameters()).dtype
    processor = getattr(model, "processor", None)

    base.eval()

    texts   = data_loader.dataset.text
    T       = len(texts)
    max_len = int(config.get("max_tokens", 64))
    text_bs = int(config.get("batch_size_test_text", 256))

    # text token blocks
    text_blocks = []
    for s in range(0, T, text_bs):
        e = min(T, s + text_bs)
        enc = tokenizer(texts[s:e], padding="max_length", truncation=True,
                        max_length=max_len, return_tensors="pt")
        text_blocks.append({k: v.to(device) for k, v in enc.items()})

    # image-text contrast
    I = len(getattr(data_loader.dataset, "image", [])) or sum(1 for _ in data_loader)
    sims_i2t   = torch.empty(I, T, device=device)
    image_bank = [None] * I
    cur = 0

    for batch in data_loader:
        if isinstance(batch, dict):
            images = batch["image"]
            idxs   = batch.get("index") or batch.get("idx")
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()
        else:
            images, idxs = batch
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()

        if isinstance(images, list):
            assert processor is not None, "processor required when images is a list"
            enc = processor(images=images, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(device=device, dtype=param_dtype, non_blocking=True)
        else:
            pixel_values = images.to(device, non_blocking=True)
            if pixel_values.is_floating_point() and pixel_values.dtype != param_dtype:
                pixel_values = pixel_values.to(dtype=param_dtype)

        B = pixel_values.size(0)
        row_slice = slice(cur, cur + B)

        col = 0
        for blk in text_blocks:
            out = base(pixel_values=pixel_values,
                       input_ids=blk["input_ids"],
                       attention_mask=blk["attention_mask"],
                       return_dict=True,
                       use_image_text_matching_head=False)
            logits = out.logits_per_image  # (B, Tk)
            Tk = logits.size(1)
            sims_i2t[row_slice, col:col+Tk] = logits
            col += Tk

        if isinstance(idxs, list):
            for k, gi in enumerate(idxs):
                image_bank[gi] = pixel_values[k].detach().cpu()
        else:
            for k in range(B):
                image_bank[cur + k] = pixel_values[k].detach().cpu()

        cur += B

    # ITM re-rank (optional)
    k_test = int(config.get("k_test", 0))
    if k_test > 0:
        text_ids  = torch.cat([blk["input_ids"] for blk in text_blocks], dim=0)
        text_atts = torch.cat([blk["attention_mask"] for blk in text_blocks], dim=0)

        # i→t
        score_i2t = torch.full_like(sims_i2t, -100.0)
        for i in range(I):
            sims = sims_i2t[i]
            topk_sim, topk_idx = sims.topk(k=min(k_test, T), dim=0)
            pv = image_bank[i].unsqueeze(0).repeat(topk_idx.numel(), 1, 1, 1).to(device)
            itm = base(pixel_values=pv,
                       input_ids=text_ids[topk_idx],
                       attention_mask=text_atts[topk_idx],
                       return_dict=True,
                       use_image_text_matching_head=True).logits_per_image
            if itm.ndim == 2 and itm.size(-1) == 2:
                itm = itm[:, 1]
            score_i2t[i, topk_idx] = itm.float() + topk_sim

        # t→i
        sims_t2i = sims_i2t.t().contiguous()
        score_t2i = torch.full((T, I), -100.0, device=device)
        for t in range(T):
            sims = sims_t2i[t]
            topk_sim, topk_idx = sims.topk(k=min(k_test, I), dim=0)
            pv = torch.stack([image_bank[j] for j in topk_idx.tolist()], dim=0).to(device)
            ids = text_ids[t].unsqueeze(0).repeat(topk_idx.numel(), 1)
            ats = text_atts[t].unsqueeze(0).repeat(topk_idx.numel(), 1)
            itm = base(pixel_values=pv,
                       input_ids=ids,
                       attention_mask=ats,
                       return_dict=True,
                       use_image_text_matching_head=True).logits_per_image
            if itm.ndim == 2 and itm.size(-1) == 2:
                itm = itm[:, 1]
            score_t2i[t, topk_idx] = itm.float() + topk_sim

        sims_i2t = score_i2t
        sims_t2i = score_t2i
    else:
        sims_t2i = sims_i2t.t()

    print("Evaluation time {}".format(str(datetime.timedelta(seconds=int(time.time() - t0)))))
    return sims_i2t.float().cpu().numpy(), sims_t2i.float().cpu().numpy()
