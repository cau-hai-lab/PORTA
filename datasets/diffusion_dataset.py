# datasets/diffusion_dataset.py
import os, json, random
from typing import List, Tuple
from torch.utils.data import Dataset

# class DiffusionEvalDataset(Dataset):
#     def __init__(
#         self,
#         ann_file: str,
#         image_root: str,
#         text_key: str = "caption",
#         image_key: str = "image",
#         mode: str = "per_image",
#         caption_policy: str = "first",
#         seed: int = 42,
#         limit: int | None = None,
#     ):
#         assert mode in ("per_image", "per_caption")
#         assert caption_policy in ("first", "random")

#         self.ann_file = ann_file
#         self.image_root = image_root
#         self.text_key = text_key
#         self.image_key = image_key
#         self.mode = mode
#         self.caption_policy = caption_policy
#         self.seed = seed

#         # (caption_text, ref_image_abs_path)
#         self.limit = limit
#         self._items: List[Tuple[str, str]] = []
#         self._load()

#     def _load(self):
#         rng = random.Random(self.seed)

#         def _abs_img(p: str) -> str:
#             # ann["image"] 예: "val2014/COCO_val2014_000000xxxxxx.jpg"
#             ap = os.path.join(self.image_root, p)
#             return os.path.abspath(ap)

#         def _append_per_image(obj):
#             caps = obj.get(self.text_key, []) or []
#             if not isinstance(caps, list) or len(caps) == 0:
#                 return
#             cap = caps[0] if self.caption_policy == "first" else rng.choice(caps)
#             if isinstance(cap, str) and cap.strip():
#                 self._items.append((cap.strip(), _abs_img(obj[self.image_key])))

#         def _append_per_caption(obj):
#             caps = obj.get(self.text_key, []) or []
#             for cap in caps:
#                 if isinstance(cap, str) and cap.strip():
#                     self._items.append((cap.strip(), _abs_img(obj[self.image_key])))

#         path = self.ann_file.lower()
#         if path.endswith(".json"):
#             data = json.load(open(self.ann_file, "r", encoding="utf-8"))
#             assert isinstance(data, list), "JSON must be a list of objects"
#             for obj in data:
#                 (_append_per_image if self.mode == "per_image" else _append_per_caption)(obj)
#         elif path.endswith(".jsonl"):
#             with open(self.ann_file, "r", encoding="utf-8") as f:
#                 for line in f:
#                     line = line.strip()
#                     if not line:
#                         continue
#                     obj = json.loads(line)
#                     (_append_per_image if self.mode == "per_image" else _append_per_caption)(obj)
#         else:
#             raise ValueError(f"Unsupported annotation file: {self.ann_file}")

#         if len(self._items) == 0:
#             raise ValueError(f"No (caption, image) pairs found in {self.ann_file}")

#         # 빠른 존재 확인(처음 100개만)
#         missing = []
#         for _, p in self._items[:100]:
#             if not os.path.exists(p):
#                 missing.append(p)
#         if missing:
#             print(f"[Warn] Some image paths don't exist (first 5): {missing[:5]}")
        
#         if self.limit is not None:
#             self._items = self._items[: int(self.limit)]

#     def __len__(self):
#         return len(self._items)

#     def __getitem__(self, idx: int):
#         cap, ref_path = self._items[idx]
#         return cap, idx, ref_path
# datasets/diffusion_dataset.py
import os, json, random
from typing import List, Tuple, Optional
from torch.utils.data import Dataset

class DiffusionEvalDataset(Dataset):
    def __init__(
        self,
        ann_file: str,
        image_root: str,
        text_key: str = "caption",
        image_key: str = "image",
        mode: str = "per_image",
        caption_policy: str = "first",
        seed: int = 42,
        limit: Optional[int] = None,
        require_images: bool = True,   # ★ FID 돌릴 때만 True, 나머지 지표는 False 권장
    ):
        assert mode in ("per_image", "per_caption")
        assert caption_policy in ("first", "random")

        self.ann_file = ann_file
        self.image_root = image_root
        self.text_key = text_key
        self.image_key = image_key
        self.mode = mode
        self.caption_policy = caption_policy
        self.seed = seed
        self.limit = limit
        self.require_images = require_images  # ★ 추가

        # (caption_text, ref_image_abs_path or "")
        self._items: List[Tuple[str, str]] = []
        self._load()

    def _load(self):
        rng = random.Random(self.seed)

        def _abs_img(p: str) -> str:
            if not p:  # 빈 문자열/None 허용
                return ""
            ap = os.path.join(self.image_root, p)
            return os.path.abspath(ap)

        def _ensure_caps(obj):
            caps = obj.get(self.text_key, []) or []
            # ★ 문자열 캡션도 허용
            if isinstance(caps, str):
                caps = [caps]
            return [c.strip() for c in caps if isinstance(c, str) and c.strip()]

        def _append_per_image(obj):
            caps = _ensure_caps(obj)
            if not caps:
                return
            cap = caps[0] if self.caption_policy == "first" else rng.choice(caps)
            img_rel = obj.get(self.image_key, "") or ""
            ref = _abs_img(img_rel) if self.require_images else ""  # ★ 이미지 요구 안 하면 빈 문자열
            self._items.append((cap, ref))

        def _append_per_caption(obj):
            caps = _ensure_caps(obj)
            img_rel = obj.get(self.image_key, "") or ""
            ref = _abs_img(img_rel) if self.require_images else ""
            for cap in caps:
                self._items.append((cap, ref))

        path = self.ann_file.lower()
        if path.endswith(".json"):
            data = json.load(open(self.ann_file, "r", encoding="utf-8"))
            assert isinstance(data, list), "JSON must be a list of objects"
            for obj in data:
                (_append_per_image if self.mode == "per_image" else _append_per_caption)(obj)
        elif path.endswith(".jsonl"):
            with open(self.ann_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    (_append_per_image if self.mode == "per_image" else _append_per_caption)(obj)
        else:
            raise ValueError(f"Unsupported annotation file: {self.ann_file}")

        if len(self._items) == 0:
            raise ValueError(f"No captions found in {self.ann_file}")

        # ★ 존재 확인은 이미지가 필요할 때만
        if self.require_images:
            missing = []
            for _, p in self._items[:100]:
                if not (p and os.path.isfile(p)):
                    missing.append(p)
            if missing:
                print(f"[Warn] Some image files don't exist (first 5): {missing[:5]}")

        if self.limit is not None:
            self._items = self._items[: int(self.limit)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx: int):
        cap, ref_path = self._items[idx]
        return cap, idx, ref_path
