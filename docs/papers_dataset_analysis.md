# Refs 目录参考论文 — 数据集来源分析

> 本文档详细分析了 `Refs/` 目录中五篇参考论文所使用的数据集，包括数据来源、规模、特征构成以及是否可公开获取。

---

## 总览

| 论文 | 机构 | 数据集来源 | 数据 规模 | 特征数量 | 用户行为序列 | 公开可用 |
|------|------|-----------|----------|---------|-------------|---------|
| **OneTrans** | ByteDance | 字节内部生产日志 | 未公开（工业级） | 未公开 | ✅ 多行为序列 | ❌ |
| **MixFormer** | ByteDance | 抖音推荐系统日志 | 万亿级交互记录 | 300+ 特征 | ✅ 多行为序列 | ❌ |
| **TokenMixer-Large** | ByteDance | 抖音电商/广告/直播 | 4亿+/天 (电商), 170亿/天 (直播) | 500+ 特征 | ✅ 多行为序列 | ❌ |
| **InterFormer** | Meta | Meta 广告系统内部 | 100亿级+样本 | 数百特征 | ✅ 用户行为序列 | ❌ |
| **HyFormer** | 未署名(疑似腾讯/字节) | 内部工业数据 + Criteo/Avazu | 工业级 + 公开基准 | 未公开 | ✅ | 部分 ✅ (Criteo/Avazu) |

> [!IMPORTANT]
> **所有五篇论文的核心实验都基于大规模内部工业数据集**，均不对外公开。这与工业界推荐系统论文的通用惯例一致——模型在真实生产环境中验证，数据含有用户隐私和商业敏感信息。

---

## 1. OneTrans (ByteDance, 2025)

**论文全称**: *OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender*

### 数据集来源
- **来源**: 字节跳动内部**大规模工业排序场景**的生产日志 (production logs)
- **隐私合规**: 论文明确声明所有个人可识别信息已被匿名化和哈希处理

### 数据集特征
| 属性 | 详情 |
|------|------|
| **数据划分** | 按时间顺序 (chronologically)，所有特征在曝光时快照，防止时间泄漏 |
| **标签** | 点击 (clicks) 和订单 (orders)，在固定时间窗口内聚合 |
| **序列特征** | 多行为序列 (multi-behavior sequences)，包括不同行为类型 |
| **非序列特征** | 用户画像、物品属性、上下文特征 |
| **序列长度** | 初始 token 序列长度 1190，通过 Pyramid 调度线性递减至 12 |

### 评估任务
- **CTR** (Click-Through Rate) 和 **CVR** (Conversion Rate) 两个二分类任务
- 使用 AUC 和 UAUC (impression-weighted user-level AUC) 评估
- 采用 **Next-batch evaluation** 方法（按时间顺序处理数据）

### 在线 A/B 测试
- 在字节两个大规模工业场景进行：**Feeds**（信息流推荐）和 **Mall**（整体商城）
- 使用 1.5 年生产数据训练
- OneTransL vs RankMixer+Transformer：
  - Feeds: **+4.35% order/u, +5.68% gmv/u, -3.91% 延迟**
  - Mall: **+2.58% order/u, +3.67% gmv/u, -3.26% 延迟**

### 🔍 数据可获得性
> **❌ 不可公开获取。** 数据来自字节跳动内部生产系统。

---

## 2. MixFormer (ByteDance, 2026)

**论文全称**: *MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders*

### 数据集来源
- **来源**: **抖音推荐系统 (Douyin recommendation system)** 的大规模离线数据集
- **时间跨度**: 连续两周

### 数据集特征
| 属性 | 详情 |
|------|------|
| **规模** | **万亿级 (trillions)** 用户-物品交互记录 |
| **特征数量** | **300+** 特征 |
| **非序列特征** | 分类特征 (categorical)、数值特征 (numerical)、交叉特征 (cross features)，来自用户画像、物品属性、上下文 |
| **序列特征** | 用户历史行为序列，每个 action 包含：物品标识 (item identifier)、行为类型 (action type)、时间戳 (timestamp)、侧信息属性 (side attributes) |

### 评估任务
- **CTR 预测** (二分类问题)
- 评估指标：AUC 和 UAUC
- 效率指标：参数量 (Parameters) 和 FLOPs

### 在线 A/B 测试
- 场景: **抖音 (Douyin)** 和 **抖音极速版 (Douyin Lite)** 的信息流推荐
- 基线: STCA→RankMixer (10亿+参数)
- 关键指标: Active Days (日活)、Duration (使用时长)、Finish/Like/Comment

### 训练配置
- 数百 GPU 的混布式分布训练
- Sparse 部分异步更新，Dense 部分同步更新
- Dense: RMSProp (lr=0.01)，Sparse: Adagrad
- BatchSize = 1500

### 🔍 数据可获得性
> **❌ 不可公开获取。** 数据来自抖音内部推荐系统日志。

---

## 3. TokenMixer-Large (ByteDance, 2026)

**论文全称**: *TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders*

### 数据集来源
TokenMixer-Large 在 **三个不同的字节跳动业务场景** 中进行了实验：

| 场景 | 数据来源 | 每日样本量(采样后) | 特征数 | 训练时长 |
|------|---------|-----------------|--------|---------|
| **抖音电商主Feed** (E-Commerce) | 平台在线日志和用户反馈 | ~**4亿**条/天 | **500+** | 2年数据 |
| **抖音广告** (Feed Ads) | 广告投放日志 | ~**3亿**条/天 | 未公开 | 未公开 |
| **抖音直播** (Live Streaming) | 直播互动日志 | ~**170亿**条/天 | 未公开 | 未公开 |

### 数据集特征
| 属性 | 详情 (电商场景) |
|------|------|
| **特征类型** | 数值特征 (numerical)、ID特征 (ID-based)、交叉特征 (cross)、序列特征 (sequential) |
| **用户规模** | 数亿独立用户 (hundreds of millions of unique users) |
| **标签** | 商品点击 (product clicks)、转化 (conversions)、GMV |

### 评估指标
- **效果**: AUC, UAUC (主要使用 CTR/CVR 任务)
- **效率**: Dense 参数量、Training FLOPs/Batch (batch=2048)、**MFU** (Model FLOPs Utilization)

### 模型规模与在线表现
| 场景 | 基线 | TokenMixer-Large 规模 | 在线增益 |
|------|------|---------------------|---------|
| 广告 (Feed Ads) | RankMixer-1B | **7B** (离线15B) | ADSS **+2.0%** |
| 电商 (E-Commerce) | RankMixer-150M | **4B** (离线7B) | GMV **+2.98%**, 订单 **+1.66%** |
| 直播 (Live Streaming) | RankMixer-500M | **2B** (离线4B) | 收入 **+1.4%** |

### 训练配置
- 64 GPU (电商) / 256 GPU (广告/直播)
- Sparse: Adagrad (lr=0.05)，Dense: Adagrad (lr=0.01)
- FP8 E4M3 推理量化 (1.7x 加速，无精度损失)

### 🔍 数据可获得性
> **❌ 不可公开获取。** 数据来自抖音三个不同业务场景的内部生产数据。

---

## 4. InterFormer (Meta, 2024/2025)

**论文全称**: *InterFormer: Towards Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction*

### 数据集来源
- **来源**: **Meta 广告系统** 内部大规模工业数据集
- 论文最初发表于 2024 年 (arXiv: 2411.09852)，后被 CIKM 2025 接收

### 数据集特征
| 属性 | 详情 |
|------|------|
| **规模** | 100亿级+ 样本 (数十亿日志) |
| **特征类型** | 稀疏特征 (user/item/context IDs) 和 稠密特征 (numerical) |
| **序列特征** | 用户行为序列 (user behavior sequences) |
| **架构特点** | 双分支：Interaction Arch (特征交叉) + Sequence Arch (序列建模)，通过 bidirectional cross-bridge 连接 |

### 评估任务
- CTR 预测 (Click-Through Rate)
- 部署于 Meta Ads 生产系统

### 🔍 数据可获得性
> **❌ 不可公开获取。** 数据来自 Meta 广告系统内部日志。
> 
> **⚠️ 代码部分可获取**: 作者 Zhichen Zeng 的个人主页 (zhichenzeng.github.io) 提供了部分代码实现。

---

## 5. HyFormer (2025)

**论文全称**: *HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction*

### 数据集来源
HyFormer 是此次分析中**唯一使用了公开数据集**的论文：

| 数据集 | 类型 | 规模 | 公开 |
|--------|------|------|------|
| **内部工业数据集** | 大规模生产日志 | 工业级 | ❌ |
| **Criteo Kaggle** | CTR 预测基准 | 4500万行, 13 数值 + 26 分类特征 | ✅ |
| **Avazu** | CTR 预测基准 | ~4000万条广告点击日志 | ✅ |

### 架构特点
- **双组件统一**: Query Decoding (序列建模 → 特征交叉) + Query Boosting (特征交叉 → 序列建模)
- 提出 Hybrid Transformer 统一框架

### 公开数据集详情

#### Criteo Kaggle Dataset
- **下载**: [Kaggle Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)
- **规模**: ~4500万行
- **特征**: 13 个数值特征 (I1-I13) + 26 个分类特征 (C1-C26)
- **标签**: 点击/未点击 (二分类)
- **注意**: 无用户行为序列特征

#### Avazu Dataset
- **下载**: [Kaggle Avazu CTR Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)
- **规模**: ~4000万条
- **特征**: 23 个特征字段 (含 click, hour, banner_pos, site/app 属性, device 属性等)
- **标签**: 点击/未点击 (二分类)
- **注意**: 无用户行为序列特征

### 🔍 数据可获得性
> **部分可获取。** 公开基准 Criteo 和 Avazu 可从 Kaggle 下载，但核心工业数据集不可获取。

---

## 📊 数据集对比分析

### 各论文数据集规模对比

```
TokenMixer-Large (直播)  ████████████████████████████████████████ 170亿/天
MixFormer                █████████████  万亿级(两周)
OneTrans                 ████████  工业级(1.5年)
TokenMixer-Large (电商)  ████  4亿/天
TokenMixer-Large (广告)  ███  3亿/天
InterFormer              ██████████  100亿级+
HyFormer (工业)          ████  工业级
HyFormer (Criteo)        ▌  4500万
HyFormer (Avazu)         ▌  4000万
```

### 数据特征维度对比

| 论文 | 特征总数 | 序列特征 | 序列长度 | 多行为序列 |
|------|---------|---------|---------|-----------|
| OneTrans | 未公开 | ✅ | 1190 tokens | ✅ (timestamp-aware/agnostic) |
| MixFormer | 300+ | ✅ | 可达10,000+ | ✅ |
| TokenMixer-Large | 500+ | ✅ | 未公开 | ✅ (短期DIN/长期SIM/超长LONGER) |
| InterFormer | 数百 | ✅ | 未公开 | ✅ |
| HyFormer | 未公开/39(Criteo)/23(Avazu) | ✅(工业)/❌(公开) | 未公开 | ✅(工业)/❌(公开) |

---

## 🎯 对比赛 (KDD Cup 2026) 的启示

### 1. 数据差距分析

比赛提供的 `TAAC2026/data_sample_1000` 样本数据有 **1000 条记录**，包含：
- 149 个特征 (78 int_value + 52 int_array + 19 float_array)
- 3 个时间对齐的序列特征 (action_seq, content_seq, item_seq)
- 4 个标签 (二分类)

**与论文数据的关键差距：**

| 对比维度 | 论文数据 | 比赛数据 |
|---------|---------|---------|
| 样本量 | 亿~万亿级 | 待发布 (4月24日Phase 3) |
| 特征数 | 300-500+ | 149 |
| 序列类型 | 多类型行为序列 | 3类对齐序列 |
| 数据跨度 | 数周~数年 | 待确认 |

### 2. 可利用的公开数据集

根据分析结果，推荐以下公开数据集用于预赛开发：

| 数据集 | 最佳用途 | 与比赛相似度 |
|--------|---------|------------|
| **Criteo Kaggle** | 特征交叉 baseline 快速验证 | ⭐⭐ (无序列) |
| **Avazu** | CTR 预测基准测试 | ⭐⭐ (无序列) |
| **Taobao Ad Display/Click** | DIN/DIEN 序列建模验证 | ⭐⭐⭐ (有行为序列) |
| **Ali-CCP** | 多任务 CTR/CVR 学习 | ⭐⭐⭐ (多目标) |

### 3. 核心结论

> [!CAUTION]
> **所有第一梯队论文 (OneTrans, MixFormer, TokenMixer-Large) 均只使用了内部数据进行训练和验证。** 
> 
> 这意味着：
> 1. **无法直接复现论文实验** — 需要在比赛数据或公开数据上独立验证
> 2. **论文中的超参数可能不适用** — 需要根据比赛数据规模重新调优
> 3. **公开基准 (Criteo/Avazu) 缺少序列特征** — 不适合验证统一序列+特征交叉架构的核心创新
> 4. **模型规模需大幅缩小** — 论文模型 1B-15B 参数，比赛需在严格延迟限制下运行

> [!TIP]
> **建议策略**: 
> - 使用 **Taobao Ad** 数据集预训练和验证序列建模组件
> - 使用 **Criteo** 数据集验证特征交叉组件  
> - 等待 4 月 24 日官方训练数据发布后，在真实数据上调优统一架构
> - 参考 OneTrans 的 **Pyramid Stack** 和 **KV Caching** 设计满足延迟要求
