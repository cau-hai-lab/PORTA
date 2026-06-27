# --- 상단 import 정리 ---
import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from huggingface_hub import snapshot_download
import os
from typing import List, Any
# ... _CLIP_ALIASES, resolve_clip_id, normalize_clip_mask_keys 동일 ...
_CLIP_ALIASES = {
    "clipL": "openai/clip-vit-large-patch14",
    "clipL-336": "openai/clip-vit-large-patch14-336",
    "clipG": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",  # ViT-bigG/14 (LAION) 
}

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


def resolve_clip_id(name_or_id: str) -> str:
    """'clipG' 같은 별칭을 HF repo id로 변환. 슬래시('/')가 있으면 그대로 사용."""
    if "/" in name_or_id:
        return name_or_id
    return _CLIP_ALIASES.get(name_or_id, name_or_id)
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
def normalize_clip_mask_keys(mask_sd: dict, target_keys: set):
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

def _maybe_build_safetensors_index(local_dir: str):
    """
    pytorch_model.bin.index.json -> model.safetensors.index.json 생성.
    파일명 매핑 규칙:
      'pytorch_model-00001-of-00008.bin' -> 'model-00001-of-00008.safetensors'
    """
    safe_index = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.exists(safe_index):
        return

    bin_index = os.path.join(local_dir, "pytorch_model.bin.index.json")
    if not os.path.exists(bin_index):
        return  # bin index 자체가 없으면 건드리지 않음

    with open(bin_index, "r") as f:
        data = json.load(f)

    wmap = data.get("weight_map", {})
    def _map_name(v: str) -> str:
        # 접두사 & 확장자 모두 치환
        v2 = v.replace("pytorch_model-", "model-").replace(".bin", ".safetensors")
        return v2

    new_map = {k: _map_name(v) for k, v in wmap.items()}

    # 존재하는 샤드 용량 합산(없어도 동작엔 문제 없음)
    shard_files = set(new_map.values())
    total_size = 0
    for s in shard_files:
        p = os.path.join(local_dir, s)
        if os.path.exists(p):
            total_size += os.path.getsize(p)

    with open(safe_index, "w") as f:
        json.dump({"metadata": {"total_size": int(total_size)}, "weight_map": new_map}, f)
    print(f"[Debug] 생성: {safe_index}")

class CLIPGVQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        resolved_id = resolve_clip_id(config['model_name'])
        self.model_id = resolved_id
        print(f"[Debug] CLIP-ViT-bigG-14 로딩 준비: {resolved_id}")

        # --- DDP 중복 다운로드 회피 ---
        rank = int(os.environ.get("RANK", "0"))
        patterns = [
            "*.safetensors", "pytorch_model.bin.index.json",
            "config.json", "open_clip_config.json",
            "tokenizer.json","tokenizer_config.json","vocab.json","merges.txt",
            "preprocessor_config.json","special_tokens_map.json",
            "*.md","*.txt",
        ]
        if rank == 0:
            snapshot_download(resolved_id, allow_patterns=patterns)
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass

        # 캐시 재확인 (오프라인 로드)
        local_dir = snapshot_download(resolved_id, allow_patterns=patterns, local_files_only=True)

        # --- safetensors 인덱스 없으면 만들어주기(샤드형) ---
        _maybe_build_safetensors_index(local_dir)

        # --- 모델 로드: safetensors -> 단일 safetensors -> bin fallback ---
        try:
            self.model = CLIPModel.from_pretrained(
                local_dir, use_safetensors=True, low_cpu_mem_usage=True, device_map=None
            )
        except Exception as e:
            print(f"[Warn] safetensors 샤드 로드 실패: {e}")
            single_safe = os.path.join(local_dir, "open_clip_model.safetensors")
            if os.path.exists(single_safe):
                print("[Info] 단일 safetensors 파일로 시도합니다.")
                self.model = CLIPModel.from_pretrained(
                    local_dir, use_safetensors=True, low_cpu_mem_usage=True,
                    device_map=None, weights_name="open_clip_model.safetensors"
                )
            else:
                print("[Info] bin 가중치 fallback 로 시도합니다.")
                # bin 필요 시 다시 받아오기(최초 allow_patterns가 *.bin을 포함 안 했으면)
                try:
                    snapshot_download(resolved_id, allow_patterns=["*.bin", "pytorch_model.bin.index.json"])
                except Exception:
                    pass
                self.model = CLIPModel.from_pretrained(
                    resolved_id, use_safetensors=False, low_cpu_mem_usage=True
                )

        self.model = self.model.float()  # dtype은 이후 Fabric/AMP에서 조정
        self.tokenizer = CLIPTokenizer.from_pretrained(local_dir)

        self.max_tokens = 77
        self.pad_token_id = 0
        setattr(self, "name", "clipG")
        setattr(self, "is_vlm", True)
        setattr(self, "needs_tie", False)

    # === L14와 동일 인터페이스 보장 ===
    def encode_image(self, images):
        print("[Debug] clip_vqa.py : CLIPGVQA 클래스의 encode_image()함수 호출")
        feats = self.model.get_image_features(pixel_values=images)
        return F.normalize(feats, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        print("[Debug] clip_vqa.py : CLIPGVQA 클래스의 encode_text()함수 호출")
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


    # 시그니처 호환(선택적 pretraining_weights 지원)
    def load_pretrained(self, weights_ckpt: str = "",config=None, is_eval: bool = False):
        if not weights_ckpt:
            print("[CLIP.load_pretrained] Empty -> keep HF weights.")
            return
        if isinstance(weights_ckpt, str) and not os.path.exists(weights_ckpt) and "/" in weights_ckpt:
            repo = resolve_clip_id(weights_ckpt)
            print(f"[CLIP.load_pretrained] Re-loading from HF: {repo}")
            self.model = CLIPModel.from_pretrained(repo)
            self.tokenizer = CLIPTokenizer.from_pretrained(repo)
            self.model_id = repo
            return
        print(f"[CLIP.load_pretrained] Loading from file: {weights_ckpt}")
        ckpt = torch.load(weights_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        msg = self.load_state_dict(state_dict, strict=False)
        print("[CLIP.load_pretrained] missing (w/o pruning_mask):",
              [k for k in msg.missing_keys if "pruning_mask" not in k])
        print("[CLIP.load_pretrained] unexpected:", msg.unexpected_keys)

    def load_from_pruned_pretrained(self, pretraining_weights: str | None = None,
                                    mask_path: str | None = None,
                                    config=None, is_eval: bool = False):
        print("-" * 80)
        print("[CLIP] load_from_pruned_pretrained(): weights + pruning masks")

        if pretraining_weights:
            self.load_pretrained(pretraining_weights, is_eval=is_eval)

        if mask_path:
            print(f"[CLIP] mask: {mask_path}")
            mask_sd_raw = torch.load(mask_path, map_location="cpu")
            if isinstance(mask_sd_raw, dict) and ("state_dict" in mask_sd_raw or "model_state" in mask_sd_raw):
                mask_sd_raw = mask_sd_raw.get("state_dict", mask_sd_raw.get("model_state"))
            target_keys = set(self.state_dict().keys())
            mask_sd = normalize_clip_mask_keys(mask_sd_raw, target_keys)
            msg = self.load_state_dict(mask_sd, strict=False)
            miss = [k for k in msg.missing_keys if k.endswith("_pruning_mask")]
            print("[CLIP] mask missing keys:", miss[:10], "..." if len(miss) > 10 else "")
            print("[CLIP] mask unexpected keys:", [k for k in msg.unexpected_keys][:10], "...")
        print("-" * 80)
