# models/clip_vqa.py
# CLIPVQA: VQA-style training & inference built on top of CLIP bi-encoder
# - Training: question+candidate answers -> concatenate tokens per (Q,A_i),
#             compute image-text similarities, apply soft-label NLL with per-question
#             candidate weights (k, weights) like BLIP's scheme.
# - Inference (rank): pick best answer index per question among the provided candidates.
# - Pruned checkpoints: load masks with key normalization compatible with CLIPRetrieval.

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# -------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------

def _as_namespace(batch_enc) -> Any:
    """Accepts a HF BatchEncoding, dict, or a simple namespace-like object; returns obj with
    .input_ids and .attention_mask tensors."""
    if batch_enc is None:
        return None
    if hasattr(batch_enc, "input_ids") and hasattr(batch_enc, "attention_mask"):
        return batch_enc
    # dict-like
    if isinstance(batch_enc, dict):
        class _T: ...
        t = _T()
        t.input_ids = batch_enc["input_ids"]
        t.attention_mask = batch_enc["attention_mask"]
        return t
    raise TypeError("Unsupported batch encoding type for question/answer.")


def normalize_clip_mask_keys(mask_sd: dict, target_keys: set):
    """Normalize pruning mask keys to this module's state_dict schema.
    - Drop bias masks
    - Map 'vision_proj.'/'text_proj.' -> 'visual_projection.'/'text_projection.'
    - Prefix 'model.' when keys refer to inner CLIP model
    - Keep only keys that exist in target state_dict
    """
    fixed = {}
    for k, v in mask_sd.items():
        if k.endswith(".bias_pruning_mask"):
            continue
        kk = k.replace("vision_proj.", "visual_projection.")
        kk = kk.replace("text_proj.",   "text_projection.")
        if kk.startswith(("text_model.", "vision_model.", "text_projection.", "visual_projection.")):
            kk = "model." + kk
        if kk in target_keys:
            fixed[kk] = v
    return fixed

# -------------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------------
class CLIPVQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("[Debug] clip_vqa.py : CLIPVQA 클래스의 init()함수 호출")
        self.model = CLIPModel.from_pretrained(config["model_name"])  # HF repo id or local dir
        self.max_tokens = config.get("max_tokens", 77)
        self.pad_token_id = config.get("pad_token_id", 0)
        self.tokenizer = CLIPTokenizer.from_pretrained(config['model_name'])
        # Metadata for pruning utils / outside code
        setattr(self, "name", "clip")
        setattr(self, "is_vlm", True)
        setattr(self, "needs_tie", False)

    # -------------------- Encoders --------------------
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        print("[Debug] clip_vqa.py : CLIPVQA 클래스의 encode_image()함수 호출")
        feats = self.model.get_image_features(pixel_values=images)
        return F.normalize(feats, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        print("[Debug] clip_vqa.py : CLIPVQA 클래스의 encode_text()함수 호출")
        feats = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(feats, dim=-1)

    # -------------------- Token ops -------------------
    def _repeat_question_for_candidates(self, q_ids: torch.Tensor, q_att: torch.Tensor, k: List[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeat each question row n=k[b] times so it aligns with flattened candidate answers.
        Returns tensors of shape (sum(k), Lq).
        """
        device = q_ids.device
        reps = torch.tensor(k, device=device, dtype=torch.long)
        # Repeat via index_select on expanded index list
        idx = torch.repeat_interleave(torch.arange(q_ids.size(0), device=device, dtype=torch.long), reps)
        q_ids_rep = q_ids.index_select(0, idx)
        q_att_rep = q_att.index_select(0, idx)
        return q_ids_rep, q_att_rep

    def _concat_q_a(self, q_ids: torch.Tensor, q_att: torch.Tensor,
                     a_ids: torch.Tensor, a_att: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate (Q, A) token sequences row-wise, then trim/pad to max_tokens.
        All inputs have shape (N, Lq or La). Outputs: (N, Lmax)
        """
        # Concatenate then slice to max length
        ids_cat = torch.cat([q_ids, a_ids], dim=1)
        att_cat = torch.cat([q_att, a_att], dim=1)
        if ids_cat.size(1) > self.max_tokens:
            ids_cat = ids_cat[:, : self.max_tokens]
            att_cat = att_cat[:, : self.max_tokens]
        else:
            # Pad to max_tokens for shape stability (optional)
            pad_len = self.max_tokens - ids_cat.size(1)
            if pad_len > 0:
                pad_ids = torch.full((ids_cat.size(0), pad_len), self.pad_token_id, device=ids_cat.device, dtype=ids_cat.dtype)
                pad_att = torch.zeros((att_cat.size(0), pad_len), device=att_cat.device, dtype=att_cat.dtype)
                ids_cat = torch.cat([ids_cat, pad_ids], dim=1)
                att_cat = torch.cat([att_cat, pad_att], dim=1)
        return ids_cat, att_cat

    # -------------------- Forward ---------------------
    def forward(
        self,
        image: torch.Tensor,
        question,              # BatchEncoding-like (with .input_ids, .attention_mask)
        answer=None,           # BatchEncoding-like of flattened candidates across batch
        *,
        train: bool = True,
        k: List[int] | None = None,     # number of candidates per question (len=k == B)
        weights: torch.Tensor | None = None,  # shape (sum(k),)
        inference: str = "rank",
        k_test: int | None = None,
        **kwargs,
    ):
        """If train=True: returns scalar loss.
           If train=False and inference=='rank': returns LongTensor of absolute indices in the
           flattened candidate list (length = batch size), i.e., like BLIP's rank_answer.
        """
        print("[Debug] clip_vqa.py : CLIPVQA 클래스의 forward()함수 호출")
        question = _as_namespace(question)
        answer   = _as_namespace(answer) if answer is not None else None

        if train:
            assert k is not None and weights is not None and answer is not None, "Training requires k, weights, and answer tokens."

            # 1) Build (Q,A_i) concatenated tokens aligned with flattened answers
            q_ids_rep, q_att_rep = self._repeat_question_for_candidates(question.input_ids, question.attention_mask, k)
            qa_ids, qa_att = self._concat_q_a(q_ids_rep, q_att_rep, answer.input_ids, answer.attention_mask)

            # 2) Encode
            img_embeds = self.encode_image(image)                    # (B, D)
            txt_embeds = self.encode_text(qa_ids, qa_att)            # (sum(k), D)
            logit_scale = self.model.logit_scale.exp()

            # 3) Compute per-question soft-label NLL
            B = image.size(0)
            weights = weights.to(img_embeds.device)
            # offsets for slicing
            with torch.no_grad():
                k_tensor = torch.as_tensor(k, device=img_embeds.device, dtype=torch.long)
                starts = F.pad(k_tensor.cumsum(dim=0), (1, 0))[:-1]  # shift-right with 0 at start

            total = img_embeds.new_zeros(())
            for b in range(B):
                s = starts[b].item()
                e = (starts[b] + k_tensor[b]).item()
                # Similarity of this image vs its candidate answers
                sims = (txt_embeds[s:e] @ img_embeds[b].unsqueeze(-1)).squeeze(-1) * logit_scale  # (k_b,)
                logp = sims.log_softmax(dim=0)
                w = weights[s:e]
                total = total - (w * logp).sum()
            loss = total / B
            return loss

        # ---------------- Inference ----------------
        assert inference in {"rank"}, "Only 'rank' inference is supported for CLIPVQA."
        assert answer is not None and k is not None, "Inference requires answer tokens and k."

        # Build (Q,A_i) concatenated tokens per candidate
        q_ids_rep, q_att_rep = self._repeat_question_for_candidates(question.input_ids, question.attention_mask, k)
        qa_ids, qa_att = self._concat_q_a(q_ids_rep, q_att_rep, answer.input_ids, answer.attention_mask)

        # Encode
        img_embeds = self.encode_image(image)                # (B, D)
        txt_embeds = self.encode_text(qa_ids, qa_att)        # (sum(k), D)
        logit_scale = self.model.logit_scale.exp()

        # For each question, pick best candidate's absolute index in the flattened pool
        B = image.size(0)
        with torch.no_grad():
            k_tensor = torch.as_tensor(k, device=img_embeds.device, dtype=torch.long)
            starts = F.pad(k_tensor.cumsum(dim=0), (1, 0))[:-1]
        out = []
        for b in range(B):
            s = starts[b].item()
            e = (starts[b] + k_tensor[b]).item()
            sims = (txt_embeds[s:e] @ img_embeds[b].unsqueeze(-1)).squeeze(-1) * logit_scale
            if k_test is not None and k_test > 0 and k_test < (e - s):
                # restrict to top-k_test before argmax (optional)
                topk_vals, topk_idx = torch.topk(sims, k=k_test, dim=0)
                # choose best among those
                rel = int(topk_idx[topk_vals.argmax()].item())
            else:
                rel = int(sims.argmax().item())
            out.append(s + rel)
        return torch.as_tensor(out, device=image.device, dtype=torch.long)

    # -------------------- Weights / Masks --------------------
    def load_pretrained(self, weights_ckpt, config=None, is_eval: bool = False):
        """Flexible loader: keep HF weights if empty string; re-load from HF repo name;
        or load a local file (supports raw state_dict or wrappers)."""
        if not weights_ckpt:
            print("[CLIPVQA.load_pretrained] Empty path -> keep HF weights as-is.")
            return
        if isinstance(weights_ckpt, str) and not os.path.exists(weights_ckpt) and "/" in weights_ckpt:
            print(f"[CLIPVQA.load_pretrained] Re-loading from HF repo: {weights_ckpt}")
            self.model = CLIPModel.from_pretrained(weights_ckpt)
            return
        print(f"[CLIPVQA.load_pretrained] Loading weights from: {weights_ckpt}")
        ckpt = torch.load(weights_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        msg = self.load_state_dict(state_dict, strict=False)
        print("[CLIPVQA.load_pretrained] missing keys:", [k for k in msg.missing_keys if "pruning_mask" not in k])
        print("[CLIPVQA.load_pretrained] unexpected keys:", msg.unexpected_keys)

    def load_from_pruned_pretrained(self, pretraining_weights, mask_path, config=None, is_eval: bool = False):
        print("-" * 80)
        print("[CLIPVQA] load_from_pruned_pretrained(): applying weights (if any) and pruning masks")
        # Then pruning masks
        print(f"[CLIPVQA] Loading pruning mask from: {mask_path}")
        mask_sd_raw = torch.load(mask_path, map_location="cpu")
        if isinstance(mask_sd_raw, dict) and ("state_dict" in mask_sd_raw or "model_state" in mask_sd_raw):
            mask_sd_raw = mask_sd_raw.get("state_dict", mask_sd_raw.get("model_state"))
        target_keys = set(self.state_dict().keys())
        mask_sd = normalize_clip_mask_keys(mask_sd_raw, target_keys)
        msg = self.load_state_dict(mask_sd, strict=False)
        miss = [k for k in msg.missing_keys if k.endswith("_pruning_mask")]
        print("[CLIPVQA] mask missing keys:", miss[:10], "..." if len(miss) > 10 else "")
        print("[CLIPVQA] mask unexpected keys:", [k for k in msg.unexpected_keys][:10], "...")
        print("-" * 80)
