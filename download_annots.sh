#!/bin/bash
pip install gdown

# ──────────────────────────────────────────────────────────────────────────────
# finetune.tar.gz
#   data/finetune/itr/   : MSCOCO & Flickr30k retrieval annotations
#   data/finetune/ic/    : COCO Karpathy image captioning annotations
#   data/finetune/vqa/   : ScienceQA (train/val/test), answer lists
# ──────────────────────────────────────────────────────────────────────────────
gdown https://drive.google.com/uc?id=1-CHaAm5yxmEQwGN58y7g77Qjc7kmniOu
tar -xvf finetune.tar.gz
rm finetune.tar.gz

# ──────────────────────────────────────────────────────────────────────────────
# imagenet-annots.tar.gz
#   data/imagenet/annots/ : ImageNet-1K class index mapping
#   (images must be downloaded separately — see README)
# ──────────────────────────────────────────────────────────────────────────────
gdown https://drive.google.com/uc?id=1-Um5jhtLsI8hQjgaRYU9XO5C0Pc3AOwP
tar -xvf imagenet-annots.tar.gz
rm imagenet-annots.tar.gz

# ──────────────────────────────────────────────────────────────────────────────
# flowers102-annots.tar.gz
#   data/flowers102/annots/ : Flowers-102 split annotations
#   (images must be downloaded separately — see README)
# ──────────────────────────────────────────────────────────────────────────────
gdown https://drive.google.com/uc?id=1-XTwKu28YY-vlB9zbgyi12fkqPV7nl9v
tar -xvf flowers102-annots.tar.gz
rm flowers102-annots.tar.gz

# ──────────────────────────────────────────────────────────────────────────────
# weights.tar.gz
#   weights/ : BLIP-base pretrained weights
# ──────────────────────────────────────────────────────────────────────────────
gdown https://drive.google.com/uc?id=1-XwsiOiIbx2dI7yYDcUP52OguF73aJ3Z
tar -xvf weights.tar.gz
rm weights.tar.gz

# ──────────────────────────────────────────────────────────────────────────────
# clip-vit-weights.tar.gz
#   data/ : CLIP ViT weights used by the codebase
# ──────────────────────────────────────────────────────────────────────────────
gdown https://drive.google.com/uc?id=1-Z-l1xggltfZJgFQZowxJRAz3y0jy01X
tar -xvf clip-vit-weights.tar.gz
rm clip-vit-weights.tar.gz

echo ""
echo "Download complete."
echo ""
echo "Annotation files now available:"
echo "  data/finetune/itr/   — MSCOCO & Flickr30k retrieval"
echo "  data/finetune/ic/    — COCO captioning (OpenFlamingo)"
echo "  data/finetune/vqa/   — ScienceQA (Qwen2-VL)"
echo "  data/imagenet/       — ImageNet-1K class index"
echo "  data/flowers102/     — Flowers-102 splits"
echo ""
echo "Images must be downloaded separately (see README):"
echo "  COCO       : https://cocodataset.org/#download"
echo "  Flickr30k  : https://shannon.cs.illinois.edu/DenotationGraph/"
echo "  ImageNet-1K: https://www.kaggle.com/c/imagenet-object-localization-challenge/data"
echo "  Flowers-102: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
echo "  CIFAR-10/100: downloaded automatically via torchvision"
echo "  ScienceQA  : annotations only, no image download required"
