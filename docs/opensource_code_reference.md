# 可用开源代码与工具库详细参考

---

## 一、有开源代码的论文方案

### 1. Wukong（非官方实现）

| 项目 | 详情 |
|------|------|
| **论文** | Wukong: Towards a Scaling Law for Large-Scale Recommendation |
| **机构/年份** | Meta, ICML 2024 |
| **GitHub** | [clabrugere/wukong-recommendation](https://github.com/clabrugere/wukong-recommendation) |
| **框架** | PyTorch + TensorFlow |
| **官方/非官方** | ⚠️ 非官方实现 |
| **核心思想** | 基于堆叠分解机（Stacked Factorization Machines）的可 scale 架构，展示推荐模型的 scaling law，跨越两个数量级的模型复杂度（>100 GFLOP/example） |
| **与竞赛关联** | 提供了可堆叠架构的 scaling law 参考；但不包含序列建模部分 |

**安装使用**：
```bash
git clone https://github.com/clabrugere/wukong-recommendation.git
cd wukong-recommendation
pip install -e .
```

---

### 2. InterFormer（底层 Kernel 部分开源）

| 项目 | 详情 |
|------|------|
| **论文** | InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction |
| **机构/年份** | Meta, CIKM 2025 |
| **完整模型代码** | ❌ **未开源** — 完整的 InterFormer 模型架构和训练代码未公开 |
| **底层 Kernel** | ✅ **GDPA (Generalized Dot Product Attention)** — InterFormer 使用的核心注意力算子 |
| **GDPA 仓库** | [facebookresearch/ads_model_kernel_library](https://github.com/facebookresearch/ads_model_kernel_library) → `gdpa/` 目录 |
| **框架** | PyTorch，基于 Flash Attention 4 + NVIDIA CUTE-DSL |
| **硬件要求** | ⚠️ **Hopper (SM90) 或 Blackwell (SM100) GPU**（如 H100/B200），CUDA ≥ 12.0 |
| **许可** | Apache 2.0 |
| **核心思想** | 交错式异构特征交互学习，实现双向信息流，避免信息聚合过激；已部署于 Meta Ads (GEM) |
| **与竞赛关联** | 直接处理异构特征交互（稀疏+稠密），非常匹配竞赛场景；模型架构需根据论文自行实现 |

**GDPA 安装与使用**：
```bash
git clone https://github.com/facebookresearch/ads_model_kernel_library.git
cd ads_model_kernel_library/gdpa
pip install nvidia-cutlass-dsl>=4.1.0 torch einops
python tests/benchmark_attn.py  # 运行 BF16 benchmark
```

> ⚠️ GDPA 是将 softmax 替换为其他激活函数的注意力变体 kernel，也被用于 Meta 的 Kunlun 和 GEM 模型。它只是底层算子，**不包含 InterFormer 的完整模型定义、数据处理和训练流程**。

---

### 3. AutoInt（官方代码 + 多个实现）

| 项目 | 详情 |
|------|------|
| **论文** | AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks |
| **年份** | CIKM 2019 |
| **官方 GitHub** | [shichence/AutoInt](https://github.com/shichence/AutoInt) |
| **迁移仓库** | [DeepGraphLearning/RecommenderSystems/featureRec](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/featureRec) |
| **框架** | TensorFlow (官方), PyTorch (DeepCTR-Torch, RecBole) |
| **官方/非官方** | ✅ 官方 + 多个社区实现 |
| **核心思想** | 用多头自注意力机制自动学习高阶特征交叉，是最早将 Transformer 引入特征交叉的工作之一 |
| **与竞赛关联** | 可作为特征交叉 Transformer 的基础组件，OneTrans 等工作在此基础上演进 |

**通过 DeepCTR-Torch 使用**：
```python
from deepctr_torch.models import AutoInt
model = AutoInt(linear_feature_columns, dnn_feature_columns, att_layer_num=3)
model.fit(x_train, y_train, batch_size=256, epochs=10)
```

---

### 4. DCN V2（多个社区实现）

| 项目 | 详情 |
|------|------|
| **论文** | DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems |
| **机构/年份** | Google, WWW 2021 |
| **可用实现** | DeepCTR-Torch / TensorFlow Recommenders / RecBole |
| **框架** | PyTorch / TensorFlow |
| **核心思想** | 深度交叉网络的改进版，用交叉层显式建模有界度数的特征交叉，Mix 版本提升表达能力 |
| **与竞赛关联** | 经典的特征交叉 baseline，可作为统一架构中的特征交叉组件 |

**通过 DeepCTR-Torch 使用**：
```python
from deepctr_torch.models import DCNMix  # DCN V2 Mix 版本
model = DCNMix(linear_feature_columns, dnn_feature_columns, cross_num=3)
```

**通过 TensorFlow Recommenders 使用**：
```python
import tensorflow_recommenders as tfrs
cross_layer = tfrs.layers.dcn.Cross(projection_dim=64, kernel_initializer="glorot_uniform")
```

---

### 5. GDCN（官方代码）

| 项目 | 详情 |
|------|------|
| **论文** | GDCN: Towards Deeper, Lighter and Interpretable Cross Network for CTR Prediction |
| **年份** | CIKM 2023 |
| **GitHub** | [anonctr/GDCN](https://github.com/anonctr/GDCN) |
| **框架** | PyTorch |
| **官方/非官方** | ✅ 官方 |
| **核心思想** | 门控深度交叉网络 + Field-level 维度优化（FDO），捕获显式高阶特征交叉并动态过滤重要交互 |
| **与竞赛关联** | DCN V2 的改进，适合作为更强的特征交叉 baseline |

---

### 6. DeepFM / xDeepFM（多个社区实现）

| 项目 | 详情 |
|------|------|
| **DeepFM 论文** | DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (Huawei, IJCAI 2017) |
| **xDeepFM 论文** | xDeepFM: Combining Explicit and Implicit Feature Interactions (KDD 2018) |
| **可用实现** | DeepCTR-Torch / RecBole / FuxiCTR |
| **框架** | PyTorch / TensorFlow |
| **核心思想** | DeepFM: FM + DNN 并行融合；xDeepFM: CIN (Compressed Interaction Network) 构建显式交叉 |
| **与竞赛关联** | 经典 baseline，但不含序列建模 |

---

## 二、推荐工具库

### 1. DeepCTR-Torch ⭐⭐⭐⭐⭐（最推荐）

| 项目 | 详情 |
|------|------|
| **GitHub** | [shenweichen/DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) |
| **Stars** | 3.4k+ |
| **框架** | PyTorch |
| **文档** | [deepctr-torch.readthedocs.io](https://deepctr-torch.readthedocs.io/) |
| **安装** | `pip install deepctr-torch` |

**包含的模型（与竞赛相关的）**：

| 类别 | 模型 |
|------|------|
| **特征交叉** | DeepFM, DCN, DCN V2 (DCNMix), xDeepFM, AutoInt, PNN, AFM, NFM, FiBiNET, IFM, DIFM, AFN |
| **序列建模** | DIN (Deep Interest Network), DIEN (Deep Interest Evolution Network) |
| **经典** | Wide & Deep, CCPM, ONN |
| **多任务** | SharedBottom, ESMM, MMOE, PLE |

**优势**：
- 模块化设计，核心组件（DNN, CrossNet, InteractionLayer）可单独复用
- `model.fit()` / `model.predict()` 接口简洁
- 适合快速搭建 baseline 并在此基础上改造

---

### 2. FuxiCTR ⭐⭐⭐⭐

| 项目 | 详情 |
|------|------|
| **GitHub** | [reczoo/FuxiCTR](https://github.com/reczoo/FuxiCTR) |
| **框架** | PyTorch + TensorFlow |
| **文档** | [fuxictr.github.io](https://fuxictr.github.io/) |
| **安装** | `pip install fuxictr` |

**特色功能**：
- **可配置性**：通过 YAML 配置文件定义数据预处理和模型参数
- **可调参**：内置自动超参搜索（支持网格搜索）
- **可复现**：提供标准 Benchmark 设置和排行榜
- 支持 30+ CTR 预测模型
- 曾用于 WWW 2025 多模态 CTR 预测挑战赛的 baseline

**优势**：
- 适合做公平的模型对比实验
- 配置驱动，无需改代码即可切换模型
- 有标准化的 benchmark 流程

---

### 3. RecBole ⭐⭐⭐⭐

| 项目 | 详情 |
|------|------|
| **GitHub** | [RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole) |
| **Stars** | 3.5k+ |
| **框架** | PyTorch |
| **文档** | [recbole.io](https://recbole.io/) |
| **安装** | `pip install recbole` |

**模型覆盖（100+ 算法，四大类）**：

| 类别 | 代表模型 |
|------|----------|
| **通用推荐** | BPR, NeuMF, LightGCN, MultiVAE, SimpleX |
| **序列推荐** | GRU4Rec, SASRec, BERT4Rec, LightSANs, NARM |
| **上下文感知** | FM, DeepFM, Wide&Deep, DCN, AutoInt |
| **知识图谱** | CKE, KGAT, RippleNet |

**扩展子包**：
- **RecBole-TRM**：Transformer 系列推荐模型
- **RecBole-GNN**：图神经网络系列
- **RecBole-DA**：数据增强（CL4SRec, DuoRec 等）

**优势**：
- 模型种类最全，覆盖推荐全流程
- 统一的数据加载、训练、评估框架
- 适合做学术研究和全面的模型对比

---

## 三、竞赛中的推荐使用策略

```
                     ┌─────────────────────────────────┐
                     │  Phase 1: 快速搭建 Baseline      │
                     │  使用 DeepCTR-Torch              │
                     │  模型: DCN V2 + DIN              │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │  Phase 2: 改造为统一架构          │
                     │  参考 OneTrans 论文               │
                     │  复用 AutoInt 的自注意力层         │
                     │  复用 DCN V2 的交叉层             │
                     │  搭建统一 Tokenizer               │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │  Phase 3: 优化 Scaling + 延迟     │
                     │  参考 RankMixer / Wukong          │
                     │  MoE 扩展、延迟优化、量化          │
                     └─────────────────────────────────┘
```

---

## 四、快速安装命令

```bash
# DeepCTR-Torch（最推荐，快速上手）
pip install deepctr-torch

# FuxiCTR（配置化实验）
pip install fuxictr

# RecBole（全面的模型库）
pip install recbole

# Wukong 非官方实现
git clone https://github.com/clabrugere/wukong-recommendation.git

# GDCN 官方实现
git clone https://github.com/anonctr/GDCN.git

# AutoInt 官方实现
git clone https://github.com/shichence/AutoInt.git
```
