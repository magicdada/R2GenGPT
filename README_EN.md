# R2GenGPT - Chest X-Ray Report Generation with Swin Transformer + LLaMA-2

## Overview

This project is based on the R2GenGPT framework, using Swin Transformer as the visual encoder and LLaMA-2-7B-Chat as the language generator for automatic chest X-ray diagnostic report generation. It supports dual-view input (Frontal + Lateral) and provides interpretability features including occlusion-based attention heatmaps, tri-state gating classification, and confidence scoring.

## Model Architecture

```
Chest X-Ray Images (Frontal + Lateral)
    ↓
Swin Transformer (Visual Encoder, microsoft/swin-base-patch4-window7-224)
    ↓
MultiGranularityFusion (Multi-granularity Visual Feature Fusion, optional)
    ├── Stage 2 (28×28, 256d) → AdaptivePool(7×7) → Linear(256→1024)
    ├── Stage 3 (14×14, 512d) → AdaptivePool(7×7) → Linear(512→1024)
    └── Stage 4 (7×7, 1024d) ─────────────────────────────────────
    → Learnable Weighted Fusion
    ↓
EnhancedProjection / Linear (Visual → Language Space Mapping)
    ↓
LayerNorm + Prompt Wrapping
    ↓
LLaMA-2-7B-Chat (4-bit NF4 Quantization, Report Generation)
    ↓
Diagnostic Report Text
```

## Project Structure

```
R2GenGPT/
├── configs/
│   └── config.py                -- Training hyperparameters (dataset/model/decoding/training strategy)
├── data/
│   ├── iu_xray/                 -- IU X-Ray dataset (annotation.json + images/)
│   └── mimic_cxr/               -- MIMIC-CXR dataset (annotation.json + images/)
├── dataset/
│   ├── data_helper.py           -- Data preprocessing (image transforms, annotation parsing)
│   └── data_module.py           -- PyTorch Lightning DataModule
├── evalcap/                     -- Evaluation metrics
│   ├── bleu/                    -- BLEU score
│   ├── cider/                   -- CIDEr score
│   ├── meteor/                  -- METEOR score
│   └── rouge/                   -- ROUGE-L score
├── lightning_tools/
│   ├── callbacks.py             -- Training callbacks (TensorBoard logging, checkpoint saving)
│   └── optim.py                 -- Optimizer configuration
├── models/
│   └── R2GenGPT.py              -- Core model (with MultiGranularityFusion + EnhancedProjection)
├── save/                        -- Training outputs (checkpoints, evaluation results)
│   └── iu_xray/
│       ├── v1_shallow/          -- Shallow baseline experiment
│       ├── v1_delta/            -- Delta baseline experiment
│       ├── v1_deep/             -- Deep baseline experiment
│       └── v2_full/             -- Full improvement experiment
├── scripts/                     -- Training scripts
├── api.py                       -- FastAPI inference service (heatmap/tri-state gating/confidence)
├── train.py                     -- Training entry point
├── requirements.txt             -- Python dependencies
└── LICENSE
```

## Baseline Experiment Results (IU X-Ray)

Three training strategies compared, all using Swin-Base + LLaMA-2-7B-Chat (4-bit quantization):

| Strategy | Freeze Visual Encoder | Visual LoRA | LLM LoRA | Trainable Params |
|----------|----------------------|-------------|----------|-----------------|
| Shallow | Yes | No | No | Linear projection only |
| Delta | Yes | No | No | Linear projection only (with delta weights) |
| Deep | No | No | No | Linear + full visual encoder |

### Best Validation Metrics

| Model | BLEU_4 | CIDEr | ROUGE_L | METEOR |
|-------|--------|-------|---------|--------|
| Shallow | 0.1283 | 0.3154 | 0.3406 | - |
| Delta | 0.1480 | 0.4139 | 0.3482 | - |
| Deep | 0.1517 | 0.4170 | 0.3522 | - |

### Delta Test Set Results

| BLEU_4 | CIDEr | ROUGE_L | METEOR |
|--------|-------|---------|--------|
| 0.1630 | 0.5400 | 0.3721 | 0.4106 |

Delta was selected as the deployment model for its training stability, absence of overfitting, and strong CIDEr performance.

## Model Improvements

Two optional modules added on top of the Delta baseline:

### Multi-Granularity Visual Feature Fusion (MultiGranularityFusion)

The original model only uses the final 7×7 feature map from Swin Transformer, losing fine-grained details from earlier stages (e.g., edges and textures of small lesions). The improvement extracts features from Stages 2/3/4 at three scales, aligns spatial dimensions via adaptive pooling and channel dimensions via linear projection, then fuses them with learnable weights.

### Enhanced Projection Module (EnhancedProjection)

The original model uses a single Linear layer to map visual features to LLaMA's input space. The improvement replaces it with a 2-layer MLP (Linear → GELU → Dropout → Linear) for stronger visual-to-language mapping capability.

### Ablation Study Plan

| Experiment | use_multi_granularity | use_enhanced_proj | Save Path |
|-----------|---|---|---|
| Baseline (completed) | False | False | save/iu_xray/v1_delta |
| Exp A: MGVF only | True | False | save/iu_xray/v2_mgvf |
| Exp B: EP only | False | True | save/iu_xray/v2_ep |
| Exp C: Full improvement | True | True | save/iu_xray/v2_full |

## Inference API (api.py)

FastAPI inference service providing:

- Accepts Frontal + Lateral dual-view chest X-ray images
- Generates diagnostic reports (Findings + Impression)
- Occlusion-based (Preserve mode) per-word attention heatmaps
- Tri-state gating: normal / findings / uncertain
- Image contribution assessment (full-occlusion drop)
- Per-finding confidence scores
- Frontal/Lateral view labeling

### Heatmap Methodology

Occlusion-based Causal Perturbation with preserve mode:

1. Divide the image into a 7×7 grid
2. For each region, preserve only that region and occlude the rest, then observe model output changes
3. Generate per-word heatmaps showing which image regions correspond to each diagnostic finding
4. Tri-state gating: no heatmap for all-negative reports; keyword-level maps only for positive findings

### Start the Service

```bash
python api.py
```

The service listens on `0.0.0.0:8000` by default.

## Training

### Requirements

- Python 3.10+
- CUDA 11.8+
- GPU: RTX 4060 8GB (with 4-bit quantization)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Training Configuration

Modify default parameters in `configs/config.py` or pass via command line:

```bash
# Delta baseline training
python train.py --savedmodel_path save/iu_xray/v1_delta

# Full improvement experiment
python train.py --use_multi_granularity True --use_enhanced_proj True --savedmodel_path save/iu_xray/v2_full
```

Alternatively, modify the defaults in `config.py` and run `train.py` directly in your IDE.

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| dataset | iu_xray | Dataset (iu_xray / mimic_cxr) |
| vision_model | microsoft/swin-base-patch4-window7-224 | Visual encoder |
| llama_model | meta-llama/Llama-2-7b-chat-hf | Language model |
| low_resource | True | Enable 4-bit quantization |
| freeze_vm | False | Freeze visual encoder |
| use_multi_granularity | False | Enable multi-granularity feature fusion |
| use_enhanced_proj | False | Enable enhanced projection module |
| batch_size | 1 | Training batch size |
| accumulate_grad_batches | 4 | Gradient accumulation steps |
| max_epochs | 15 | Maximum training epochs |
| learning_rate | 1e-4 | Learning rate |
| beam_size | 3 | Beam search width |

### Testing

```bash
python train.py --test --ckpt_file <checkpoint_path>
```

## Dataset Preparation

IU X-Ray: download from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view)

MIMIC-CXR: download the preprocessed annotation file from [here](https://drive.google.com/file/d/14689ztodTtrQJYs--ihB_hgsPMMNHX-H/view?usp=sharing) and images from the [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

Place downloaded data in the `./data` directory.

## Notes

- Swin model must be `microsoft/swin-base-patch4-window7-224` (ImageNet-1K), NOT the in22k variant
- LLaMA-2 requires access approval on HuggingFace
- Dual-view input (Frontal + Lateral) yields better results than single-view; features from both views are averaged
- With 4-bit quantization, batch_size=1 + accumulate_grad_batches=4 fits within 8GB VRAM

## Acknowledgement

- [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) This project is built upon the R2GenGPT framework
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) Some codes are based on MiniGPT-4
- [Llama-2](https://github.com/facebookresearch/llama) Meta's LLaMA-2 large language model

## License

This repository is under [BSD 3-Clause License](LICENSE).