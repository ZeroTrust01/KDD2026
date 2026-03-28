# KDD Cup 2026 相关论文分析

> 来源: [Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising](https://github.com/guyulongcs/Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising)
>
> 筛选标准: 符合比赛主题「**统一 tokenization 方案 + 同质可堆叠 backbone，在单一架构中同时建模序列行为与多域特征交叉**」的论文

---

## 🔴 第一梯队：直接匹配比赛主题（统一序列+特征交叉的 Transformer 架构）

这些论文的核心思想与比赛要求高度一致：将序列建模与特征交叉统一到同一个 Transformer/Mixer 架构中。

| # | 论文 | 机构 | 年份 | 核心思想 | 源码 |
|---|------|------|------|----------|------|
| 1 | **OneTrans** — Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender | ByteDance | 2025 | 用统一 tokenizer 将序列行为和非序列特征转为统一 token 序列，用可堆叠的 OneTrans Block 同时建模 | ❌ 无公开代码 |
| 2 | **MixFormer** — Co-Scaling Up Dense and Sequence in Industrial Recommenders | ByteDance | 2026 | 统一 Transformer 风格架构，解决 dense 特征交叉与序列建模的协同 scaling 问题 | ❌ 无公开代码 |
| 3 | **TokenMixer-Large** — Scaling Up Large Ranking Models in Industrial Recommenders | ByteDance | 2026 | OneTrans/RankMixer 的大规模演进版，统一 token mixing 实现特征交叉+序列建模 | ❌ 无公开代码 |
| 4 | **Hiformer** — Heterogeneous Feature Interactions Learning with Transformers for Recommender Systems | 2023/2025 | 异构感知 Transformer，区分稀疏/稠密特征的异构交互，提出低秩注意力加速 | ❌ 无公开代码 |
| 5 | **InterFormer** — Effective Heterogeneous Interaction Learning for CTR Prediction | Meta | 2025 | 学习异构特征交互的高效 Transformer，已部署于 Meta Ads | ⚠️ 作者主页有代码 |
| 6 | **HHFT** — Hierarchical Heterogeneous Feature Transformer for Recommendation Systems | Alibaba | 2025 | 分层处理异构特征（稀疏/稠密），统一 Transformer 建模 | ❌ 无公开代码 |

> [!IMPORTANT]
> **OneTrans** 与比赛主题完全匹配（您已经在项目 Refs 中保存了该论文），是最直接的参考方案。**MixFormer** 和 **TokenMixer-Large** 是其进化版本。

---

## 🟠 第二梯队：可堆叠 Backbone / Scaling Law 相关

这些论文提出了可 scale 的统一 backbone 架构，关注模型规模化和推理效率。

| # | 论文 | 机构 | 年份 | 核心思想 | 源码 |
|---|------|------|------|----------|------|
| 7 | **RankMixer** — Scaling Up Ranking Models in Industrial Recommenders | ByteDance | 2025 | 用多头 token mixing 替代注意力，统一特征交叉架构，支持 MoE 扩展到十亿参数 | ❌ 无公开代码 |
| 8 | **Wukong** — Towards a Scaling Law for Large-Scale Recommendation | Meta | 2024 | 基于堆叠分解机的可 scale 架构，展示推荐模型的 scaling law | ✅ 非官方代码: [clabrugere/wukong-recommendation](https://github.com/clabrugere/wukong-recommendation) |
| 9 | **Zenith** — Scaling up Ranking Models for Billion-scale Livestreaming Recommendation | ByteDance | 2026 | 大规模直播推荐的 Ranking 模型 scaling | ❌ 无公开代码 |
| 10 | **Pyramid Mixer** — Multi-dimensional Multi-period Interest Modeling for Sequential Recommendation | ByteDance | 2025 | 多维多周期的序列兴趣建模 Mixer | ❌ 无公开代码 |
| 11 | **Meta Lattice** — Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations | Meta | 2026 | 工业级广告推荐模型空间重设计 | ❌ 无公开代码 |

---

## 🟡 第三梯队：长序列用户行为建模（序列建模端）

这些论文专注于序列建模的效率和效果，可作为序列建模组件的参考。

| # | 论文 | 机构 | 年份 | 核心思想 | 源码 |
|---|------|------|------|----------|------|
| 12 | **LONGER** — Scaling Up Long Sequence Modeling in Industrial Recommenders | ByteDance | 2025 | 工业级长序列建模 scaling | ❌ 无公开代码 |
| 13 | **STCA** — Make It Long, Keep It Fast: End-to-End 10k-Sequence Modeling at Billion Scale on Douyin | ByteDance | 2025 | 端到端万级序列建模，保证推理速度 | ❌ 无公开代码 |
| 14 | **GR** — Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations | Meta | 2024 | 万亿参数级序列 Transducer 用于生成式推荐 | ❌ 无公开代码 |
| 15 | **SIM** — Search-based User Interest Modeling with Lifelong Sequential Behavior Data | Alibaba | 2020 | 基于搜索的长序列用户兴趣建模 | ✅ [可通过 DeepCTR 使用](https://github.com/shenweichen/DeepCTR) |
| 16 | **ETA** — Efficient Long Sequential User Data Modeling for CTR Prediction | Alibaba | 2022 | 高效长序列建模 | ❌ 无公开代码 |

---

## 🟢 第四梯队：特征交叉基础模型（特征交叉端，有成熟代码）

这些是特征交叉方向的经典/基线模型，代码成熟，可作为 baseline 或组件复用。

| # | 论文 | 机构 | 年份 | 核心思想 | 源码 |
|---|------|------|------|----------|------|
| 17 | **AutoInt** — Automatic Feature Interaction Learning via Self-Attentive Neural Networks | - | 2019 | 用自注意力自动学习特征交叉 | ✅ [shichence/AutoInt](https://github.com/shichence/AutoInt) / DeepCTR / RecBole |
| 18 | **DCN V2** — Improved Deep & Cross Network | 2021 | 深度交叉网络的改进版 | ✅ [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) / TF Recommenders |
| 19 | **GDCN** — Towards Deeper, Lighter and Interpretable Cross Network | - | 2023 | 门控深度交叉网络 | ✅ [anonctr/GDCN](https://github.com/anonctr/GDCN) |
| 20 | **DeepFM** — A Factorization-Machine based Neural Network for CTR Prediction | Huawei | 2017 | FM + DNN 并行结构 | ✅ DeepCTR / RecBole |
| 21 | **xDeepFM** — Combining Explicit and Implicit Feature Interactions | - | 2018 | 显式+隐式特征交叉 | ✅ DeepCTR / RecBole |
| 22 | **FAT** — From Scaling to Structured Expressivity: Rethinking Transformers for CTR Prediction | Alibaba | 2025 | 重新思考 Transformer 用于 CTR 的结构化表达能力 | ❌ 无公开代码 |
| 23 | **DHEN** — A Deep and Hierarchical Ensemble Network for Large-Scale CTR Prediction | Meta | 2022 | 深层分层集成网络 | ❌ 无公开代码 |

---

## 📑 总结与建议

### 与比赛最相关的核心论文
1. **OneTrans** ⭐⭐⭐⭐⭐ — 与比赛主题完全一致：统一 tokenizer + 可堆叠 block，同时建模序列和特征交叉
2. **MixFormer** ⭐⭐⭐⭐⭐ — OneTrans 的升级，协同 scaling dense + sequence
3. **TokenMixer-Large** ⭐⭐⭐⭐⭐ — 大规模版本
4. **RankMixer** ⭐⭐⭐⭐ — 通用的统一特征交叉 Mixer 架构
5. **Hiformer / InterFormer / HHFT** ⭐⭐⭐⭐ — 异构特征的 Transformer 处理

### 可用开源代码汇总

| 论文 | 代码链接 | 说明 |
|------|----------|------|
| Wukong | [clabrugere/wukong-recommendation](https://github.com/clabrugere/wukong-recommendation) | 非官方实现 (PyTorch + TF) |
| InterFormer | 作者主页 zhichenzeng.github.io | 官方代码 |
| AutoInt | [shichence/AutoInt](https://github.com/shichence/AutoInt) | 官方代码 |
| DCN V2 | [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) | 社区实现 |
| GDCN | [anonctr/GDCN](https://github.com/anonctr/GDCN) | 官方代码 |
| DeepFM / xDeepFM | [DeepCTR](https://github.com/shenweichen/DeepCTR) / RecBole | 社区实现 |

### 推荐工具库
- **[DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)** — 包含 DCN V2、AutoInt、DeepFM、xDeepFM 等大量 baseline 实现
- **[RecBole](https://github.com/RUCAIBox/RecBole)** — 全面的推荐系统框架
- **[FuxiCTR](https://github.com/xue-pai/FuxiCTR)** — CTR 预测基准框架

> [!WARNING]
> 第一梯队的核心论文（OneTrans、MixFormer、TokenMixer-Large）均未开源。参赛方案需要根据论文描述自行实现。建议以 OneTrans 论文为蓝本，参考已有开源代码（如 AutoInt/DCN V2 的 Transformer 实现）搭建基础框架。
