# [ECCV 2026] Prune Once: Retraining-Free Task-Agnostic Pruning for Vision-Language Models

Official implementation of **"Prune Once: Retraining-Free Task-Agnostic Pruning for Vision-Language Models"**, accepted at ECCV 2026.

**Authors**: Minseok Kang\*, Hyunwoo Kim\*, Chanyoung Kim, Minwoo Kim, Jaekoo Lee†, Dahuin Jung†

\* Equal contribution. &nbsp; † Corresponding authors.

**Affiliations**: Chung-Ang University &nbsp;|&nbsp; Soongsil University &nbsp;|&nbsp; Kookmin University

---

We propose **PORTA** (**P**rune-**O**nce: **R**etraining-free **T**ask-**A**gnostic pruning), a one-shot VLM pruning framework that requires no retraining and transfers directly across heterogeneous downstream tasks.

Existing methods such as Wanda and SparseGPT rely on activation *magnitude*, which exhibits strong modality-dependent bias across vision and language encoders and degrades reliability in task-agnostic settings.
PORTA instead uses activation *variance* as a modality-agnostic importance signal — high-variance features respond broadly across inputs and capture generalizable structure, while low-variance features encode localized, input-specific patterns.
Concretely, PORTA scores each weight by:

$$S_{i,j} = v_j \cdot |W_{i,j}|, \quad v_j = \mathrm{Var}_{b,t}(X_{b,t,j})$$

and allocates layer-wise pruning ratios based on the variance-weighted output feature utilization of each layer, avoiding the limitations of uniform sparsity at high compression.

## Updates
- **[2026]** Code released alongside ECCV 2026 camera-ready.

## Installation

This project was developed with Python 3.10.8 and tested on Ubuntu 22.04. All experiments were conducted on a single NVIDIA L40S GPU.

**Conda (recommended):**
```bash
conda env create -f deps/environment.yaml
conda activate porta
```

**pip:**
```bash
pip install -r deps/requirements.txt
```

For Image Captioning evaluation, a Java runtime is also required:
```bash
sudo apt install default-jre default-jdk
```

## Data Preparation

### Annotations & Weights

Download annotation archives and model weights using `download_annots.sh` (requires `gdown`). After extraction, the following structure will be created:

```
data/
├── finetune/
│   ├── itr/          # MSCOCO & Flickr30k retrieval annotations
│   ├── ic/           # COCO image captioning annotations
│   └── vqa/          # ScienceQA annotations
├── imagenet/         # ImageNet-1K class index
└── flowers102/       # Flowers-102 split annotations
weights/              # BLIP-base pretrained weights
```

### Images

Place image files under `images/` with the following structure:

```
images/
├── coco/
│   ├── train2014/
│   ├── val2014/
│   └── test2015/
├── flickr30k/
├── imagenet/
│   └── val/
├── flowers102/
├── cifar/            # auto-downloaded via torchvision
└── vg/               # optional
```

| Dataset | Used for | Download |
|---|---|---|
| MSCOCO | Calibration, retrieval (Table 3) | [cocodataset.org](https://cocodataset.org/#download) — `train2014`, `val2014`, `test2015` |
| Flickr30k | Retrieval (Table 4) | [DenotationGraph](https://shannon.cs.illinois.edu/DenotationGraph/) |
| ImageNet-1K | Zero-shot classification (Table 3) | [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data) — validation split |
| CIFAR-10 / CIFAR-100 | Zero-shot classification (Table 3) | Downloaded automatically via torchvision |
| Flowers-102 | Zero-shot classification (Table 3) | [VGG](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) |
| ScienceQA | VQA (Table 5) | Annotations included in `data/finetune/vqa/`; no image download required |
| Visual Genome | Calibration robustness (Fig. 4, optional) | [visualgenome.org](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) |

> **Calibration.** All main experiments calibrate on MSCOCO (82,783 image–text pairs). PORTA is robust to the calibration source — performance varies by less than ~1% across MSCOCO, Flickr30k, and Visual Genome (Fig. 4).

## Pruning

The main entry point is `prune.py`. Pruning runs on a single GPU with FP32 precision by default.

### Supported Models

| Model | Used for |
|---|---|
| OpenAI CLIP-ViT-bigG | Retrieval (MSCOCO, Flickr30k), Classification (Table 3, 4) |
| BLIP-base | Retrieval (Flickr30k) (Table 4) |
| Qwen2-VL | VQA (ScienceQA) (Table 5) |
| OpenFlamingo | Image Captioning (Appendix C) |
| SDXL-1.0 (CLIP text encoder) | Text-to-Image Generation (Appendix B) |

### Baseline Pruners

| Name | Reference |
|---|---|
| `wanda` | [Sun et al., ICLR 2024](https://arxiv.org/abs/2306.11695) |
| `sparsegpt` | [Frantar & Alistarh, ICML 2023](https://arxiv.org/abs/2301.00774) |
| `ecoflap` | [Sung et al., ICLR 2024](https://arxiv.org/abs/2310.02998) |
| `multiflow` | [Farina et al., CVPR 2024](https://arxiv.org/abs/2404.05621) |

## Main Results

**Zero-shot retrieval and classification on CLIP-ViT-bigG** (Table 3):

| Sparsity | Method | TR@1 | TR@5 | IR@1 | IR@5 | C10 | C100 | IN | F102 |
|---|---|---|---|---|---|---|---|---|---|
| 0% | Dense | 68.56 | 87.70 | 52.28 | 76.09 | 97.04 | 87.50 | 78.45 | 80.19 |
| 55% | Wanda | 66.30 | 86.44 | 48.62 | 73.05 | 96.12 | 79.92 | 69.50 | 67.72 |
| 55% | SparseGPT | 65.78 | 86.44 | 48.57 | **73.29** | 95.07 | 79.95 | 68.61 | 65.16 |
| 55% | ECoFLaP | 51.14 | 76.04 | 33.14 | 58.44 | 80.69 | 51.79 | 43.11 | 30.00 |
| 55% | **PORTA** | **66.28** | **87.00** | **48.70** | 73.02 | **95.58** | **80.27** | **70.29** | **68.73** |
| 60% | Wanda | 59.78 | 82.82 | 42.55 | 68.01 | 93.20 | 70.28 | 57.35 | 51.48 |
| 60% | SparseGPT | 57.52 | 81.88 | 41.77 | 66.97 | **95.16** | 74.23 | 55.26 | 41.37 |
| 60% | ECoFLaP | 35.44 | 60.56 | 19.12 | 40.29 | 57.98 | 32.60 | 25.73 | 14.21 |
| 60% | **PORTA** | **60.14** | **83.04** | **43.11** | **68.18** | 95.06 | **75.59** | **59.66** | **55.50** |
| 65% | Wanda | 27.30 | 51.04 | 16.96 | 35.61 | 69.81 | 28.86 | 30.06 | 17.30 |
| 65% | SparseGPT | 25.10 | 49.10 | 15.93 | 35.00 | 86.35 | 49.23 | 25.96 | 12.09 |
| 65% | ECoFLaP | 13.60 | 29.62 | 7.22 | 18.32 | 33.18 | 16.70 | 10.20 | 6.66 |
| 65% | **PORTA** | **28.14** | **51.26** | **17.31** | **37.32** | **90.49** | **62.29** | **31.38** | **18.27** |

C10: CIFAR-10, C100: CIFAR-100, IN: ImageNet-1K, F102: Flowers-102.

**Zero-shot retrieval on BLIP-base at 45% sparsity** (Table 4, Flickr30k): PORTA achieves mean **90.26**, outperforming SparseGPT (89.95) and Wanda (89.81).

**Zero-shot VQA on Qwen2-VL at 50% sparsity** (Table 5, ScienceQA): PORTA achieves average accuracy **54.30**, outperforming Wanda (53.43) and SparseGPT (50.78).

**Pruning time on CLIP** (Table 8): PORTA completes in ~195s — **3.96× faster than SparseGPT** and **2.06× faster than ECoFLaP**.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{kang2026porta,
  title     = {Prune Once: Retraining-Free Task-Agnostic Pruning for Vision-Language Models},
  author    = {Kang, Minseok and Kim, Hyunwoo and Kim, Chanyoung and Kim, Minwoo and Lee, Jaekoo and Jung, Dahuin},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant (RS-2025-00555943); by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants (No. RS-2021-II211341; Artificial Intelligence Graduate School Program (Chung-Ang University), RS-2026-25513331; Digital Columbus Project, and IITP-2026-RS-2026-25546026; Leading Generative AI Human Resources Development) funded by the Korea government (MSIT).

Parts of this codebase build on [MULTIFLOW](https://github.com/FarinaMatteo/multiflow) (Farina et al., CVPR 2024), [BLIP](https://github.com/salesforce/BLIP), [Wanda](https://github.com/locuslab/wanda), and [SparseGPT](https://github.com/IST-DASLab/sparsegpt). We sincerely thank all authors for releasing their code.

## Contact

For questions or issues, please open a GitHub issue or contact us at **dahuinjung@cau.ac.kr**.
