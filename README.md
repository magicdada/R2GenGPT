# R2GenGPT - 基于Swin Transformer + LLaMA-2的胸部X光报告生成模型

## 项目简介

本项目基于 R2GenGPT 框架，使用 Swin Transformer 作为视觉编码器、LLaMA-2-7B-Chat 作为语言生成器，实现胸部X光影像的自动诊断报告生成。支持前后位（Frontal）和侧位（Lateral）双视图输入，并提供基于遮挡扰动的注意力热力图、三态门控分类和置信度评估等可解释性功能。

## 模型架构

```
胸部X光影像（Frontal + Lateral）
    ↓
Swin Transformer（视觉编码器，microsoft/swin-base-patch4-window7-224）
    ↓
MultiGranularityFusion（多粒度视觉特征融合，可选）
    ├── Stage 2 (28×28, 256d) → AdaptivePool(7×7) → Linear(256→1024)
    ├── Stage 3 (14×14, 512d) → AdaptivePool(7×7) → Linear(512→1024)
    └── Stage 4 (7×7, 1024d) ─────────────────────────────────────
    → 可学习加权融合
    ↓
EnhancedProjection / Linear（视觉→语言空间映射）
    ↓
LayerNorm + Prompt Wrapping
    ↓
LLaMA-2-7B-Chat（4-bit NF4量化，报告生成）
    ↓
诊断报告文本
```

## 项目结构

```
R2GenGPT/
├── configs/
│   └── config.py                -- 训练超参数配置（数据集/模型/解码/训练策略）
├── data/
│   ├── iu_xray/                 -- IU X-Ray数据集（annotation.json + images/）
│   └── mimic_cxr/               -- MIMIC-CXR数据集（annotation.json + images/）
├── dataset/
│   ├── data_helper.py           -- 数据预处理（图像变换、标注解析）
│   └── data_module.py           -- PyTorch Lightning DataModule
├── evalcap/                     -- 评估指标
│   ├── bleu/                    -- BLEU评分
│   ├── cider/                   -- CIDEr评分
│   ├── meteor/                  -- METEOR评分
│   └── rouge/                   -- ROUGE-L评分
├── lightning_tools/
│   ├── callbacks.py             -- 训练回调（TensorBoard日志、检查点保存）
│   └── optim.py                 -- 优化器配置
├── models/
│   └── R2GenGPT.py              -- 模型核心代码（含MultiGranularityFusion + EnhancedProjection）
├── save/                        -- 训练产出（检查点、评估结果）
│   └── iu_xray/
│       ├── v1_shallow/          -- Shallow基线实验
│       ├── v1_delta/            -- Delta基线实验
│       ├── v1_deep/             -- Deep基线实验
│       └── v2_full/             -- 完整改进实验
├── scripts/                     -- 训练脚本
├── api.py                       -- FastAPI推理服务（含热力图生成、三态门控、置信度评估）
├── train.py                     -- 训练入口
├── requirements.txt             -- Python依赖
└── LICENSE
```

## 基线实验结果（IU X-Ray）

三种训练策略对比，均使用 Swin-Base + LLaMA-2-7B-Chat（4-bit量化）：

| 策略 | 冻结视觉编码器 | 视觉LoRA | LLM LoRA | 训练参数 |
|------|--------------|---------|---------|---------|
| Shallow | 是 | 否 | 否 | 仅Linear映射层 |
| Delta | 是 | 否 | 否 | 仅Linear映射层（加载delta权重） |
| Deep | 否 | 否 | 否 | Linear + 完整视觉编码器 |

### 验证集最佳指标

| 模型 | BLEU_4 | CIDEr | ROUGE_L | METEOR |
|------|--------|-------|---------|--------|
| Shallow | 0.1283 | 0.3154 | 0.3406 | - |
| Delta | 0.1480 | 0.4139 | 0.3482 | - |
| Deep | 0.1517 | 0.4170 | 0.3522 | - |

### Delta 测试集结果

| BLEU_4 | CIDEr | ROUGE_L | METEOR |
|--------|-------|---------|--------|
| 0.1630 | 0.5400 | 0.3721 | 0.4106 |

Delta 方案被选为部署模型：训练稳定、无过拟合、CIDEr 表现优秀。

## 模型改进

在 Delta 基线基础上，新增两个可选模块：

### 多粒度视觉特征融合（MultiGranularityFusion）

原模型仅使用 Swin Transformer 最后一层的 7×7 特征图，丢失了浅层的细节信息（如小病灶的边缘、纹理）。改进方案提取 Stage 2/3/4 三个尺度的特征，通过自适应池化对齐空间维度、线性层对齐通道维度，最终用可学习权重加权融合。

### 增强映射模块（EnhancedProjection）

原模型使用单层 Linear 将视觉特征映射到 LLaMA 的输入空间。改进方案使用两层 MLP（Linear → GELU → Dropout → Linear），增强视觉到语言空间的映射能力。

### 消融实验计划

| 实验 | use_multi_granularity | use_enhanced_proj | 保存路径 |
|-----|---|---|---|
| Baseline（已完成） | False | False | save/iu_xray/v1_delta |
| 实验A：仅多粒度融合 | True | False | save/iu_xray/v2_mgvf |
| 实验B：仅增强映射 | False | True | save/iu_xray/v2_ep |
| 实验C：完整改进 | True | True | save/iu_xray/v2_full |

## 推理API（api.py）

FastAPI 推理服务，提供以下功能：

- 接收前后位 + 侧位双视图胸部X光影像
- 调用模型生成诊断报告（Findings + Impression）
- 基于遮挡扰动（Occlusion Preserve-mode）的逐词注意力热力图
- 三态门控分类：normal（正常）/ findings（有发现）/ uncertain（不确定）
- 图像贡献度评估（full-occlusion drop）
- 逐发现置信度评分
- Frontal/Lateral 视图标注

### 热力图技术方案

采用遮挡扰动法（Occlusion-based Causal Perturbation），preserve 模式：

1. 将影像划分为 7×7 网格区域
2. 对每个区域，保留该区域、遮挡其余区域，观察模型输出变化
3. 生成逐词热力图，展示每个诊断发现对应的影像区域
4. 三态门控：全阴性报告不显示热力图，仅阳性发现生成关键词级别热力图

### 启动方式

```bash
python api.py
```

服务默认监听 `0.0.0.0:8000`。

## 训练

### 环境要求

- Python 3.10+
- CUDA 11.8+
- GPU: RTX 4060 8GB（使用4-bit量化）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练配置

修改 `configs/config.py` 中的默认参数，或通过命令行传递：

```bash
# Delta基线训练
python train.py --savedmodel_path save/iu_xray/v1_delta

# 完整改进实验
python train.py --use_multi_granularity True --use_enhanced_proj True --savedmodel_path save/iu_xray/v2_full
```

也可以直接修改 `config.py` 中的默认值后在 IDE 中运行 `train.py`。

### 关键训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| dataset | iu_xray | 数据集（iu_xray / mimic_cxr） |
| vision_model | microsoft/swin-base-patch4-window7-224 | 视觉编码器 |
| llama_model | meta-llama/Llama-2-7b-chat-hf | 语言模型 |
| low_resource | True | 启用4-bit量化 |
| freeze_vm | False | 是否冻结视觉编码器 |
| use_multi_granularity | False | 启用多粒度特征融合 |
| use_enhanced_proj | False | 启用增强映射模块 |
| batch_size | 1 | 训练批次大小 |
| accumulate_grad_batches | 4 | 梯度累积步数 |
| max_epochs | 15 | 最大训练轮数 |
| learning_rate | 1e-4 | 学习率 |
| beam_size | 3 | Beam Search宽度 |

### 测试

```bash
python train.py --test --ckpt_file <checkpoint_path>
```

## 数据集准备

IU X-Ray：从[这里](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view)下载

MIMIC-CXR：预处理标注文件从[这里](https://drive.google.com/file/d/14689ztodTtrQJYs--ihB_hgsPMMNHX-H/view?usp=sharing)下载，影像数据从[官网](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)下载

下载后放入 `./data` 目录。

## 注意事项

- Swin 模型必须使用 `microsoft/swin-base-patch4-window7-224`（ImageNet-1K），不要使用 in22k 版本
- LLaMA-2 需要在 HuggingFace 申请访问权限
- 双视图输入（Frontal + Lateral）比单视图效果更好，两个视图的特征取平均融合
- 4-bit 量化下 batch_size=1 + accumulate_grad_batches=4 可在 8GB 显存下训练

## 致谢

- [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) 本项目基于 R2GenGPT 框架开发
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) 部分代码参考自 MiniGPT-4
- [Llama-2](https://github.com/facebookresearch/llama) Meta 的 LLaMA-2 大语言模型

## 许可证

本项目基于 [BSD 3-Clause License](LICENSE) 发布。