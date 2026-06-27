import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


_QWEN_ALIASES = {
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-2b-instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-3b-instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
}


def resolve_qwen_vl_id(name_or_id: str) -> str:
    if "/" in name_or_id:
        return name_or_id
    return _QWEN_ALIASES.get(name_or_id.lower(), name_or_id)


def normalize_qwen_mask_keys(mask_sd: dict, target_keys: set) -> dict:
    fixed = {}
    for k, v in mask_sd.items():
        if not k.endswith("_pruning_mask"):
            continue
        if k.endswith(".bias_pruning_mask"):
            continue

        kk = k[len("module."):] if k.startswith("module.") else k
        candidates = [kk]
        if kk.startswith(("visual.", "model.", "lm_head.")):
            candidates.append("model." + kk)
        if kk.startswith("model.model."):
            candidates.append(kk[len("model."):])

        for cand in candidates:
            if cand in target_keys:
                fixed[cand] = v
                break

    print(f"[normalize_qwen_mask_keys] mapped={len(fixed)}, raw={len(mask_sd)}")
    return fixed


def params_path_from_mask(mask_path: str | os.PathLike) -> Path:
    path = Path(mask_path)
    return path.with_name(path.name.replace("_pruning_masks.pth", "_params.pth"))


class QwenVLVQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_id = resolve_qwen_vl_id(config["model_name"])
        if "Qwen2.5-VL" in self.model_id:
            raise RuntimeError(
                "This environment has transformers 4.46.3 without "
                "Qwen2_5_VLForConditionalGeneration. Use Qwen2-VL-2B here, "
                "or update transformers/qwen-vl-utils before running Qwen2.5-VL."
            )

        dtype_str = str(config.get("qwen_dtype", config.get("llava_dtype", "bf16"))).lower()
        if "bf16" in dtype_str:
            torch_dtype = torch.bfloat16
        elif "16" in dtype_str:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        print(f"[QwenVLVQA] loading {self.model_id} dtype={torch_dtype}")
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=None,
            trust_remote_code=True,
        )

        self.gen_cfg = config.get("qwen_gen", config.get("llava_gen", {}))
        self.gen_cfg.setdefault("max_new_tokens", 6)
        self.gen_cfg.setdefault("num_beams", 1)
        self.gen_cfg.setdefault("do_sample", False)

        setattr(self, "name", "qwen_vl")
        setattr(self, "is_vlm", True)
        setattr(self, "needs_tie", False)

    def forward(
        self,
        image: torch.Tensor,
        question,
        answer=None,
        *,
        train: bool = True,
        k=None,
        weights=None,
        inference: str = "generate",
        k_test: int | None = None,
        **kwargs,
    ):
        if train:
            raise NotImplementedError("QwenVLVQA is wired for zero-shot VQA evaluation.")
        if inference != "generate":
            raise ValueError(f"Unsupported inference mode for QwenVLVQA: {inference}")

        device = image.device
        pil_imgs = []
        for img in image:
            if img.dtype != torch.float32:
                img = img.to(torch.float32)
            pil_imgs.append(to_pil_image(img.cpu()))

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q},
                    ],
                }
            ]
            for q in question
        ]
        prompts = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        proc_inputs = self.processor(
            text=prompts,
            images=pil_imgs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        generate_kwargs = dict(
            **proc_inputs,
            max_new_tokens=self.gen_cfg.get("max_new_tokens", 6),
            num_beams=self.gen_cfg.get("num_beams", 1),
            do_sample=self.gen_cfg.get("do_sample", False),
        )
        if os.environ.get("QWEN_VQA_DISABLE_CUDNN", "1") != "0":
            with torch.backends.cudnn.flags(enabled=False):
                gen_out = self.model.generate(**generate_kwargs)
        else:
            gen_out = self.model.generate(**generate_kwargs)
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(proc_inputs.input_ids, gen_out)
        ]
        answers = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [a.strip() for a in answers]

    def load_pretrained(self, weights_ckpt: str = "", config=None, is_eval: bool = False):
        if not weights_ckpt:
            print("[QwenVLVQA.load_pretrained] Empty -> keep HF weights initialized in __init__.")
            return
        if not os.path.exists(weights_ckpt):
            raise FileNotFoundError(weights_ckpt)

        ckpt = torch.load(weights_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        msg = self.load_state_dict(state_dict, strict=False)
        print("[QwenVLVQA.load_pretrained] missing keys:", len(msg.missing_keys))
        print("[QwenVLVQA.load_pretrained] unexpected keys:", len(msg.unexpected_keys))

    def load_from_pruned_pretrained(self, mask_path: str | None = None, config=None, is_eval: bool = False):
        print("-" * 80)
        print(f"[QwenVLVQA] load_from_pruned_pretrained(): mask = {mask_path}")
        if mask_path is None:
            raise ValueError("Qwen pruned VQA requires --mask. Use --dense for dense evaluation.")

        params_path = params_path_from_mask(mask_path)
        if params_path.exists():
            params_sd = torch.load(params_path, map_location="cpu")
            if isinstance(params_sd, dict) and ("state_dict" in params_sd or "model_state" in params_sd):
                params_sd = params_sd.get("state_dict", params_sd.get("model_state"))
            msg = self.model.load_state_dict(params_sd, strict=False)
            print(f"[QwenVLVQA] loaded pruned params: {params_path}")
            print("[QwenVLVQA] params missing keys:", len(msg.missing_keys))
            print("[QwenVLVQA] params unexpected keys:", len(msg.unexpected_keys))

        mask_sd_raw = torch.load(mask_path, map_location="cpu")
        if isinstance(mask_sd_raw, dict) and ("state_dict" in mask_sd_raw or "model_state" in mask_sd_raw):
            mask_sd_raw = mask_sd_raw.get("state_dict", mask_sd_raw.get("model_state"))

        target_keys = set(self.state_dict().keys())
        mask_sd = normalize_qwen_mask_keys(mask_sd_raw, target_keys)
        if not mask_sd:
            raise RuntimeError(
                f"No pruning masks from {mask_path} matched Qwen-VL. "
                "Do not reuse LLaVA masks for Qwen; generate Qwen masks first."
            )

        msg = self.load_state_dict(mask_sd, strict=False)
        print("[QwenVLVQA] mapped mask keys:", len(mask_sd))
        print("[QwenVLVQA] unexpected keys:", len(msg.unexpected_keys))
        print("-" * 80)
