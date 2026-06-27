import time
import datetime
import math
import os
import re
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from pruners.base import Pruner
from pruners.accumulators import forward_output, region_forward_output
from utils.prune_utils import make_prunable, recursive_getattr
from utils.functions import detect_modality_fn


_LAYER_PATTS = [
    re.compile(r"text_model\.encoder\.layers\.(\d+)\."),
    re.compile(r"transformer\.resblocks\.(\d+)\."),
]


def _get_layer_idx_from_name(name: str):
    for p in _LAYER_PATTS:
        m = p.search(name)
        if m:
            return int(m.group(1))
    return None


def _make_mask(att, B, L, device, dtype):
    """
    Normalize attention mask to (B, L, 1).
    If shape does not match, fallback to all-ones.
    """
    if att is None:
        return torch.ones(B, L, 1, device=device, dtype=dtype)

    if att.dim() == 2 and att.shape == (B, L):
        return att.unsqueeze(-1).to(device=device, dtype=dtype, non_blocking=True)

    if att.dim() == 3 and att.shape[0] == B and att.shape[1] == L:
        return att.to(device=device, dtype=dtype, non_blocking=True)

    return torch.ones(B, L, 1, device=device, dtype=dtype)


class Porta(Pruner):
    """
    Optimized col-stat version.

    Shared statistic:
        col_energy_j = E[x_j^2] ~= Var[x_j] under mean-zero approximation.

    Used for:
        1) intra-layer mask score:
            score_ij = |W_ij| * sqrt(col_energy_j)

        2) layer-wise distribution scalar:
            s_layer = dot(col_energy, ||W_:,j||_2 / sqrt(D_out)) / sum(col_energy)
    """

    def __init__(self, model, *args, **kwargs):
        make_prunable(
            model,
            mask_dtype=torch.bool,
            pattern_lock=True,
            mask_on_the_fly=False,
            store_input=True,
        )
        super().__init__(model, *args, **kwargs)

        self.detect_modality_fn = partial(detect_modality_fn, self.model.name)
        self.forward_output = partial(forward_output, self.model.name)
        self.region_forward_output = partial(region_forward_output, self.model.name)

        self.scores_computed = False
        self.is_one_shot = True
        self.modifies_weights = False
        self.name = "porta"
        self.output_dir = str(Path(kwargs.get("output_dir", getattr(self, "output_dir", "."))).resolve())

        self.prune_modalities = set(kwargs.get("prune_modalities", ("vision", "text", "fusion")))

        # Only statistic we actually use.
        self.col_ss = {}       # pid -> (D_in,) sum of x^2
        self.col_cnt = {}      # pid -> scalar effective token/sample count

        # Cache maps.
        self.pid2fullname = {}
        self.pid2module = {}
        self.pid2numel = {}

        self._build_pid_maps()

    def _build_pid_maps(self):
        self.pid2fullname.clear()
        self.pid2module.clear()
        self.pid2numel.clear()

        for name, _, param in self.named_masked_parameters:
            pid = id(param)
            self.pid2fullname[pid] = name
            self.pid2numel[pid] = int(param.numel())

            module_name = ".".join(name.split(".")[:-1])
            try:
                self.pid2module[pid] = recursive_getattr(self.model, module_name)
            except Exception:
                self.pid2module[pid] = None

    def _extract_block_smart(self, fullname: str) -> int:
        for key in [
            "vision_model.encoder.layers",
            "qformer.encoder.layer",
            "language_model.model.layers",
        ]:
            if key in fullname:
                try:
                    return int(fullname.split(key + ".")[1].split(".", 1)[0])
                except Exception:
                    pass

        parts = fullname.split(".")
        if parts and parts[0] == "module":
            parts = parts[1:]
        if parts:
            parts = parts[:-1]

        for i in range(len(parts) - 1, 0, -1):
            parent_path = ".".join(parts[:i])
            child_token = parts[i]
            try:
                parent = recursive_getattr(self.model, parent_path)
            except Exception:
                continue
            if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_token.isdigit():
                return int(child_token)

        return -1

    def _parse_role_smart(self, name: str, module, param) -> str:
        nm = name.lower()

        if isinstance(module, nn.MultiheadAttention):
            if getattr(module, "in_proj_weight", None) is param:
                return "attn_qkv"
            if hasattr(module, "out_proj") and getattr(module.out_proj, "weight", None) is param:
                return "attn_out"

        for attr, role in [
            ("q_proj", "attn_q"),
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("out_proj", "attn_out"),
            ("qkv", "attn_qkv"),
            ("proj", "attn_out"),
        ]:
            if hasattr(module, attr):
                sub = getattr(module, attr)
                if isinstance(sub, nn.Linear) and getattr(sub, "weight", None) is param:
                    return role

        if "qformer.encoder.layer" in nm:
            if any(s in nm for s in ["query", "q_proj", ".q."]):
                return "attn_q"
            if any(s in nm for s in ["key", "k_proj", ".k."]):
                return "attn_k"
            if any(s in nm for s in ["value", "v_proj", ".v."]):
                return "attn_v"
            if "out_proj" in nm:
                return "attn_out"
            if "fc1" in nm:
                return "mlp_fc1"
            if "fc2" in nm:
                return "mlp_fc2"

        if "language_model.model.layers" in nm:
            if "self_attn.q_proj" in nm:
                return "attn_q"
            if "self_attn.k_proj" in nm:
                return "attn_k"
            if "self_attn.v_proj" in nm:
                return "attn_v"
            if "self_attn.out_proj" in nm:
                return "attn_out"
            if ".fc1" in nm:
                return "mlp_fc1"
            if ".fc2" in nm:
                return "mlp_fc2"

        if any(s in nm for s in ["query", "q_proj", ".q."]):
            return "attn_q"
        if any(s in nm for s in ["key", "k_proj", ".k."]):
            return "attn_k"
        if any(s in nm for s in ["value", "v_proj", ".v."]):
            return "attn_v"
        if "out_proj" in nm or "attn.out" in nm:
            return "attn_out"
        if "in_proj_weight" in nm or ".qkv." in nm:
            return "attn_qkv"
        if "intermediate.dense" in nm or "fc1" in nm or "mlp.fc1" in nm:
            return "mlp_fc1"
        if nm.endswith("output.dense.weight") or "fc2" in nm or "mlp.fc2" in nm:
            return "mlp_fc2"

        return "other"

    def _flag_inputs_for_prunable_modules(self):
        """
        Only prunable modules store input_history.
        This keeps activation memory lower during calibration.
        """
        for module in self.model.modules():
            if hasattr(module, "store_input_flag"):
                module.store_input_flag = False

        for name, _, _ in self.named_masked_parameters:
            module_name = ".".join(name.split(".")[:-1])
            module = recursive_getattr(self.model, module_name)
            if hasattr(module, "store_input_flag"):
                module.store_input_flag = True

    def _ensure_col_buffer(self, pid: int, D: int, device):
        if pid not in self.col_ss:
            self.col_ss[pid] = torch.zeros(D, dtype=torch.float32, device=device)
            self.col_cnt[pid] = torch.zeros((), dtype=torch.float32, device=device)
            return

        if self.col_ss[pid].device != device:
            self.col_ss[pid] = self.col_ss[pid].to(device, non_blocking=True)
            self.col_cnt[pid] = self.col_cnt[pid].to(device, non_blocking=True)

        if self.col_ss[pid].numel() != D:
            # Shape mismatch should rarely happen, but keep safe behavior.
            old = self.col_ss[pid]
            new = torch.zeros(D, dtype=torch.float32, device=device)
            keep = min(old.numel(), D)
            if keep > 0:
                new[:keep].copy_(old[:keep])
            self.col_ss[pid] = new

    @torch.inference_mode()
    def _accumulate_col_stats_from_inputs(self, text_att=None):
        """
        Accumulate only the shared statistic:
            col_ss  += sum x^2 over valid tokens
            col_cnt += number of valid tokens

        No row stats, no covariance, no Gram.
        """
        for name, _, param in self.named_masked_parameters:
            pid = id(param)
            module = self.pid2module.get(pid, None)
            if module is None:
                continue

            inputs = getattr(module, "input_history", None)
            if not inputs:
                continue

            modality = self.detect_modality_fn(name)

            for idx, x in enumerate(inputs):
                if x is None:
                    continue

                x = x.detach()
                if x.ndim == 2:
                    # (N, D) -> (N, 1, D)
                    x = x.unsqueeze(1)
                elif x.ndim != 3:
                    inputs[idx] = None
                    continue

                x = x.to(torch.float32)
                B, L, D = x.shape
                device = x.device

                self._ensure_col_buffer(pid, D, device)

                x2 = x.square()

                if modality in ("text", "fusion"):
                    mask = _make_mask(text_att, B, L, device, torch.float32)
                    self.col_ss[pid].add_((x2 * mask).sum(dim=(0, 1)))
                    self.col_cnt[pid].add_(mask.sum())
                elif modality == "vision":
                    self.col_ss[pid].add_(x2.sum(dim=(0, 1)))
                    self.col_cnt[pid].add_(torch.tensor(B * L, device=device, dtype=torch.float32))
                else:
                    # Unknown modality: safest fallback is all tokens valid.
                    self.col_ss[pid].add_(x2.sum(dim=(0, 1)))
                    self.col_cnt[pid].add_(torch.tensor(B * L, device=device, dtype=torch.float32))

                # Release activation immediately.
                inputs[idx] = None

            module.input_history = []

    def _col_energy(self, pid: int, param: torch.Tensor, eps: float = 1e-5):
        """
        Return E[x^2] for input columns.
        """
        if pid not in self.col_ss or pid not in self.col_cnt:
            return None

        device = param.device
        ss = self.col_ss[pid].to(device=device, dtype=torch.float32, non_blocking=True)
        cnt = self.col_cnt[pid].to(device=device, dtype=torch.float32, non_blocking=True).clamp_min(1.0)

        return (ss / cnt).clamp_min(0.0).add_(eps)

    @torch.no_grad()
    def _build_scores_and_layer_scalars(self, eps: float = 1e-6):
        """
        One pass over parameters:
          - fills self.scores for compute_mask()
          - returns layer-wise scalar dict for distribution

        This removes the duplicated work of:
          finalize_varcolnorm_to_scalars_mean0() + score()
        """
        scalars = {}

        for name, _, param in self.named_masked_parameters:
            pid = id(param)
            W = param.detach().to(torch.float32)
            Wabs = W.abs()

            energy = self._col_energy(pid, param, eps=eps)

            if energy is None or energy.numel() != W.shape[1]:
                # Fallback: magnitude-only, and skip layer scalar.
                self.scores[pid] = Wabs.detach().cpu()
                continue

            # 1) Intra-layer pruning score: |W| * sqrt(E[x^2])
            col_scale = torch.sqrt(energy).to(dtype=Wabs.dtype)
            score = Wabs * col_scale.unsqueeze(0)
            self.scores[pid] = score.detach().cpu()

            # 2) Layer scalar for sparsity distribution.
            #    s = dot(E[x^2], column RMS of W) / sum(E[x^2])
            D_out = max(W.shape[0], 1)
            col_w_rms = torch.linalg.vector_norm(W, ord=2, dim=0) / math.sqrt(D_out)
            s = torch.dot(energy, col_w_rms) / energy.sum().clamp_min(1e-12)

            module = self.pid2module.get(pid, None)
            modality = self.detect_modality_fn(name)

            if modality not in self.prune_modalities:
                continue

            key = (
                modality,
                self._extract_block_smart(self.pid2fullname.get(pid, name)),
                self._parse_role_smart(name, module, param),
                pid,
            )
            scalars[key] = s.detach()

        return scalars

    @torch.no_grad()
    def _linear_unimportance_distribute(self, scalars_per_layer: dict, target_sparsity: float, strength: float = 1.0):
        """
        importance s_i -> p_i = s_i / sum(s)
        unimportance u_i = (1 - p_i)^strength
        prune_i = target_sparsity * u_i / weighted_mean(u)
        """
        if not scalars_per_layer:
            return {}

        vals = {}
        total = 0.0
        for k, s in scalars_per_layer.items():
            v = float(s.item()) if torch.is_tensor(s) else float(s)
            vals[k] = v
            total += v

        if total <= 1e-12:
            p = {k: 1.0 / len(vals) for k in vals}
        else:
            p = {k: v / total for k, v in vals.items()}

        u = {k: (max(0.0, 1.0 - pi) + 1e-8) ** float(strength) for k, pi in p.items()}

        weights = {}
        sum_w = 0.0
        for k in scalars_per_layer:
            pid = k[-1]
            w = float(self.pid2numel.get(pid, 1))
            weights[k] = w
            sum_w += w

        mean_u = sum(weights[k] * u[k] for k in u) / max(sum_w, 1e-12)
        coef = float(target_sparsity) / max(mean_u, 1e-12)

        distribution = {}
        for k in scalars_per_layer:
            pid = k[-1]
            distribution[pid] = max(0.0, min(1.0, coef * u[k]))


        return distribution

    @torch.no_grad()
    def prune(self, target_sparsity, model, dataloader, device, fabric, num_batches_per_step, **kwargs):
        time_in = time.time()

        is_combined = "region_loader" in kwargs and kwargs["region_loader"] is not None
        if is_combined:
            dataloader = CombinedLoader((dataloader, kwargs["region_loader"]), mode="min_size")

        if len(dataloader) == 0:
            raise RuntimeError("DataLoader is empty.")

        alpha = float(kwargs.get("alpha", 1.0))

        self._flag_inputs_for_prunable_modules()

        for batch_idx, batch in enumerate(dataloader):
            if is_combined:
                general_batch, _region_batch = batch
            else:
                general_batch = batch

            if hasattr(model, "is_vlm") and model.is_vlm:
                if isinstance(general_batch, dict):
                    text_att = general_batch.get("attention_mask", None)
                else:
                    text_att = general_batch[2]
            else:
                text_att = None

            _ = self.forward_output(model, general_batch, device, modality="fusion")

            # Accumulate and immediately clear input_history.
            self._accumulate_col_stats_from_inputs(text_att=text_att)

            is_end_step = (
                (batch_idx + 1) % num_batches_per_step == 0
                or batch_idx == len(dataloader) - 1
            )

            if is_end_step:
                # One pass: build both mask scores and layer scalars.
                scalars_per_layer = self._build_scores_and_layer_scalars(eps=1e-5)

                distribution = self._linear_unimportance_distribute(
                    scalars_per_layer,
                    target_sparsity,
                    strength=alpha,
                )

                self.compute_mask(distribution, scope="local")
                break

        self.scoring_time = int(time.time() - time_in)
        print(f"[{self.name}] Total pruning time (hh:mm:ss) = {datetime.timedelta(seconds=self.scoring_time)}")

    def reset(self):
        """
        Minimal reset for repeated pruning runs.
        """
        self.scores_computed = False
        self.col_ss.clear()
        self.col_cnt.clear()

        for _, mask, param in self.named_masked_parameters:
            mask.fill_(1)
            if mask.grad is not None:
                mask.grad.zero_()
            if param.grad is not None:
                param.grad.zero_()
            self.scores[id(param)] = torch.zeros_like(param, dtype=torch.float32, device="cpu")