# KDD Cup 2026 可用 Baseline 开源代码评估

> 基于竞赛要求（统一 tokenization + 可堆叠 backbone，CVR 预测，推理延迟限制），对所有可获取的开源代码进行系统性评估。

---

## 📋 竞赛核心要求回顾

| 要求 | 详情 |
|------|------|
| **任务** | CVR 预测（二分类，AUC 评估） |
| **输入** | 异构多域特征：user_feature (57个) + item_feature (15个) + 3 种行为序列 (最长~4000步) |
| **特征类型** | int_value / int_array / float_array / 混合类型 |
| **序列建模** | 3 种并行对齐序列 (action_seq×10属性, content_seq×9属性, item_seq×12属性) |
| **架构主题** | 统一 tokenization + 同质可堆叠 backbone |
| **关键约束** | ⚠️ **严格推理延迟限制**（超时淘汰） |

---

## 🏆 候选 Baseline 总览与排名

| 排名 | 方案 | 仓库 | 匹配度 | 推荐理由 |
|:----:|------|------|:------:|---------|
| **🥇** | **DLRM-v3 (HSTU)** | [meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders) | ⭐⭐⭐⭐⭐ | **唯一同时具备统一 backbone + 序列建模 + 特征交叉 + 工业级优化的开源方案** |
| **🥈** | **DeepCTR-Torch (DIN/DIEN)** | [shenweichen/DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) | ⭐⭐⭐⭐ | 成熟框架，含序列建模 + 特征交叉，快速出分 |
| **🥉** | **FuxiCTR** | [reczoo/FuxiCTR](https://github.com/reczoo/FuxiCTR) | ⭐⭐⭐⭐ | 配置化实验框架，30+ 模型，自动调参 |
| 4 | **Wukong** (非官方) | [clabrugere/wukong-recommendation](https://github.com/clabrugere/wukong-recommendation) | ⭐⭐⭐ | 可堆叠架构 + Scaling Law，但无序列建模 |
| 5 | **GDCN** (官方) | [anonctr/GDCN](https://github.com/anonctr/GDCN) | ⭐⭐⭐ | 门控深度交叉网络，好的特征交叉组件 |
| 6 | **OpenOneRec** | [Kuaishou-OneRec/OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) | ⭐ | LLM 范式，不适合本竞赛 |

---

## 🥇 第一优先：DLRM-v3 (基于 HSTU)

> [!IMPORTANT]
> **这是目前与比赛最匹配的开源代码。** 它是 Meta 将 HSTU (Hierarchical Sequential Transduction Unit) 用于工业级排序的完整实现，包含训练和推理代码，且专注于高效推理。

### 基本信息

| 项目 | 详情 |
|------|------|
| **仓库** | [meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders) |
| **Stars** | 1.8k ⭐ |
| **许可** | Apache 2.0 |
| **框架** | PyTorch + fbgemm_gpu + torchrec |
| **论文** | *Actions Speak Louder than Words* (ICML 2024) |
| **核心组件** | HSTU + M-FALCON + DLRM-v3 ranking model |

### 与竞赛的匹配度分析

| 竞赛要求 | DLRM-v3 支持 | 详情 |
|---------|:------:|------|
| **统一 backbone** | ✅ | HSTU 用统一的 Transformer 变体处理序列和特征交叉 |
| **可堆叠** | ✅ | HSTU blocks 可堆叠，支持 scaling law |
| **序列建模** | ✅ | 核心能力，序列Transducer 架构 |
| **特征交叉** | ✅ | DLRM-v3 模式支持稀疏+稠密特征交互 |
| **排序/CTR/CVR 预测** | ✅ | `train_ranker.py` 直接用于排序任务 |
| **推理优化** | ✅ | 包含高效 CUDA kernel (Flash Attention V3)、MLPerf 推理基准 |
| **异构特征处理** | ⚠️ | 需要适配比赛的具体特征 schema |
| **多 GPU** | ✅ | 原生支持分布式训练 |

### 关键代码结构

```
generative-recommenders/
├── generative_recommenders/
│   └── dlrm_v3/
│       ├── train/
│       │   └── train_ranker.py    ← 排序模型训练
│       └── inference/
│           └── main.py            ← 推理服务
├── configs/                       ← gin 配置文件
├── ops/
│   ├── triton/                    ← Triton kernels
│   └── cpp/
│       └── hstu_attention/        ← Flash Attention V3 based HSTU
├── main.py                        ← 序列推荐训练
└── preprocess_public_data.py      ← 数据预处理
```

### 快速上手

```bash
# 安装
git clone https://github.com/meta-recsys/generative-recommenders.git
cd generative-recommenders
pip install -r requirements.txt

# 训练排序模型（4 GPU）
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 \
  generative_recommenders/dlrm_v3/train/train_ranker.py \
  --dataset debug --mode train

# 推理基准
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 \
  generative_recommenders/dlrm_v3/inference/main.py \
  --dataset debug
```

### 需要适配的工作

| 适配项 | 工作量 | 说明 |
|-------|:------:|------|
| **数据格式转换** | 中 | 将比赛 Parquet 的嵌套 struct 转为 DLRM-v3 的输入格式 |
| **特征编码** | 中 | 适配比赛的 int_value/int_array/float_array/混合类型 |
| **多序列处理** | 中 | 比赛有 3 种行为序列，需要融合策略 |
| **输出层** | 低 | 改为 CVR 二分类 |
| **硬件依赖** | ⚠️ | 需要 GPU (推荐 A100/H100)；部分高效 kernel 需 Hopper 架构 |

> [!WARNING]
> DLRM-v3 依赖 `fbgemm_gpu` 和 `torchrec`（Meta 自有的 PyTorch 扩展），安装配置可能比较复杂。建议在 CUDA 12.4 + Python 3.10 环境下使用。

---

## 🥈 第二优先：DeepCTR-Torch

### 基本信息

| 项目 | 详情 |
|------|------|
| **仓库** | [shenweichen/DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) |
| **Stars** | 3.4k+ ⭐ |
| **许可** | Apache 2.0 |
| **框架** | PyTorch |
| **安装** | `pip install deepctr-torch` |

### 与竞赛的匹配度

| 竞赛要求 | 支持 | 详情 |
|---------|:----:|------|
| **序列建模** | ✅ | DIN + DIEN (用户行为序列建模) |
| **特征交叉** | ✅ | DCN V2, AutoInt, DeepFM, xDeepFM 等 |
| **统一 backbone** | ❌ | 传统的 "编码 → 交叉" 两阶段範式，非统一架构 |
| **可堆叠** | ⚠️ | 交叉层可堆叠，但不是统一 block |
| **CVR 预测** | ✅ | 直接支持二分类 + 多任务 (ESMM, MMOE) |
| **推理优化** | ❌ | 无工业级推理优化 |

### 推荐使用模式

```python
from deepctr_torch.models import DIN  # 或 DIEN, DCNMix

# 1. 定义特征列
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat

user_features = [SparseFeat('gender', 3), SparseFeat('age', 10), ...]
item_features = [SparseFeat('item_cat1', 200), ...]
behavior_features = [VarLenSparseFeat(
    SparseFeat('hist_item_id', 100000), maxlen=50
)]

# 2. 构建 DIN 模型
model = DIN(user_features + item_features, behavior_features,
            dnn_hidden_units=(256, 128), att_hidden_size=(64, 16))

# 3. 训练
model.compile('adam', 'binary_crossentropy', metrics=['auc'])
model.fit(x_train, y_train, batch_size=256, epochs=10)
```

### 优势与限制

| ✅ 优势 | ❌ 限制 |
|---------|---------|
| 安装简单，API 成熟 | 非统一架构，不匹配竞赛主题 |
| 快速出 baseline 分数 | 无推理延迟优化 |
| DIN/DIEN 处理行为序列 | 不支持超长序列 (>1000) |
| 模块可复用 | 无法处理竞赛的 3 种并行序列 |

---

## 🥉 第三优先：FuxiCTR

### 基本信息

| 项目 | 详情 |
|------|------|
| **仓库** | [reczoo/FuxiCTR](https://github.com/reczoo/FuxiCTR) |
| **Stars** | 800+ |
| **安装** | `pip install fuxictr` |
| **特色** | YAML 配置驱动 + 自动超参搜索 |

### 与竞赛的匹配度

| 竞赛要求 | 支持 | 详情 |
|---------|:----:|------|
| **序列建模** | ✅ | BST, DIN, DIEN, SIM 等 |
| **特征交叉** | ✅ | 30+ CTR 模型 |
| **统一 backbone** | ❌ | 传统 DLRM 范式 |
| **实验管理** | ✅ | YAML 配置 + 自动调参 + 标准 benchmark |

### 适合场景

- 快速对比多个 baseline 模型效果
- 系统性调参找最优配置
- 作为最终方案的效果下限参考

---

## 4. Wukong（非官方实现）

| 项目 | 详情 |
|------|------|
| **仓库** | [clabrugere/wukong-recommendation](https://github.com/clabrugere/wukong-recommendation) |
| **Stars** | 83 |
| **框架** | PyTorch + TensorFlow |

### 与竞赛的匹配度

| 竞赛要求 | 支持 | 详情 |
|---------|:----:|------|
| **可堆叠 backbone** | ✅ | 堆叠分解机 (LCB + FMB) 设计 |
| **Scaling Law** | ✅ | 论文核心贡献 |
| **特征交叉** | ✅ | 显式高阶特征交叉 |
| **序列建模** | ❌ | **完全不包含序列建模** |
| **推理优化** | ❌ | 无推理优化 |

### 可利用的组件

- **LCB (Linear Compression Block)**: 可作为特征压缩/对齐组件
- **FMB (Factorization Machine Block)**: 可作为特征交叉组件
- **分层堆叠架构**: 可参考堆叠 block 的残差连接设计

```python
from model.pytorch import WukongTorch

model = WukongTorch(
    num_layers=3,           # 可堆叠层数
    num_emb=NUM_EMBEDDING,
    dim_emb=128,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    num_emb_lcb=16,
    num_emb_fmb=16,
    rank_fmb=24,
    dim_output=1,
)
```

---

## 5. GDCN（官方代码）

| 项目 | 详情 |
|------|------|
| **仓库** | [anonctr/GDCN](https://github.com/anonctr/GDCN) |
| **论文** | CIKM 2023 |
| **核心** | 门控深度交叉网络 + Field-level 维度优化 |

- ✅ DCN V2 的改进版，更强的特征交叉能力
- ❌ 不包含序列建模
- 🔧 可作为统一架构中特征交叉模块的参考

---

## ❌ 不推荐：OpenOneRec

| 项目 | 详情 |
|------|------|
| **仓库** | [Kuaishou-OneRec/OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) |
| **结论** | **不适合本竞赛** |

| 对比维度 | 竞赛要求 | OpenOneRec |
|---------|---------|------------|
| 范式 | 判别式 (CVR 概率) | 生成式 (LLM 指令跟随) |
| 模型 | 轻量级排序模型 | Qwen3 1.7B/8B |
| 延迟 | 严格延迟限制 | LLM 推理极慢 |
| 特征 | 多域异构特征交叉 | 物品 ID 序列 + 文本 |

> [!CAUTION]
> OpenOneRec 的 LLM 方案在本竞赛的延迟约束下 **完全不可行**。

---

## ❌ 暂无：官方 Baseline

截至目前（2026-03-28），**腾讯 TAAC2026 / KDD Cup 2026 尚未发布官方 baseline 代码**。根据比赛日程：
- Phase 3（4月24日）发布训练数据时，可能会同步发布官方 baseline
- 建议持续关注 [algo.qq.com](https://algo.qq.com) 和比赛 QQ 群

---

## 📊 综合对比矩阵

| 维度 | DLRM-v3 | DeepCTR | FuxiCTR | Wukong | GDCN |
|------|:-------:|:-------:|:-------:|:------:|:----:|
| **统一 backbone** | ✅ | ❌ | ❌ | ⚠️ | ❌ |
| **可堆叠** | ✅ | ⚠️ | ⚠️ | ✅ | ⚠️ |
| **序列建模** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **特征交叉** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **异构特征** | ⚠️ | ✅ | ✅ | ⚠️ | ✅ |
| **推理优化** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **工业级就绪** | ✅ | ⚠️ | ⚠️ | ❌ | ❌ |
| **上手难度** | 😰 高 | 😊 低 | 😊 低 | 😐 中 | 😐 中 |
| **代码活跃度** | 🟢 高 (330 commits, 44 contributors) | 🟡 中 | 🟡 中 | 🔴 低 (13 commits) | 🔴 低 |

---

## 🎯 推荐 Baseline 策略

```
                  ┌─────────────────────────────────────────┐
                  │          推荐的分阶段策略                  │
                  └─────────────────────────────────────────┘

Phase 1: 快速出分 (1-2天)
┌─────────────────────────────────────────────────────────┐
│  DeepCTR-Torch: DIN + DCN V2                            │
│  - 快速验证数据 pipeline                                  │
│  - 建立 AUC 效果下限                                      │
│  - 验证特征工程                                           │
└───────────────────────┬─────────────────────────────────┘
                        │
Phase 2: 核心架构 (1-2周)
┌───────────────────────▼─────────────────────────────────┐
│  DLRM-v3 (HSTU) 适配                                    │
│  - 将比赛数据转为 DLRM-v3 输入格式                         │
│  - 适配 3 种并行序列的融合策略                              │
│  - 利用 HSTU 的统一 backbone 处理序列+特征交叉              │
│  - 复用其推理优化 kernel                                   │
└───────────────────────┬─────────────────────────────────┘
                        │
Phase 3: 自研统一架构 (持续优化)
┌───────────────────────▼─────────────────────────────────┐
│  基于 OneTrans/TokenMixer-Large 论文自行实现               │
│  - 参考 DLRM-v3 的工业级设计                               │
│  - 参考 Wukong 的可堆叠设计                                │
│  - 实现 Pyramid Stack + KV Caching 满足延迟约束            │
│  - 加入 MoE 进一步提升效果                                 │
└─────────────────────────────────────────────────────────┘
```

> [!TIP]
> **核心发现**: Meta 的 **DLRM-v3** (基于 HSTU) 是目前唯一同时满足「统一 backbone + 序列建模 + 特征交叉 + 推理优化」的开源工业级方案。虽然上手难度较高（依赖 fbgemm/torchrec），但它的架构理念与竞赛主题高度一致，且包含完整的训练和推理 pipeline，是最值得投入时间适配的 baseline。
