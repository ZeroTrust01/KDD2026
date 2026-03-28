# 基于 5 篇参考论文的模块替代实现落地方案

> 目标：在 `OneTrans / MixFormer / HyFormer / InterFormer / TokenMixer-Large` 都缺少完整公开实现的前提下，拆解其关键模块，判断哪些可以直接用开源库替代，哪些只能部分替代，哪些必须自研。

---

## 1. 先给结论

这 5 篇论文里，**最容易落地的不是“整篇复现”，而是“按模块拼装”**。

推荐采用下面的策略：

1. **数据与特征侧**：用 `DeepCTR-Torch` / `FuxiCTR` / `TorchRec` 解决稀疏特征、稠密特征、变长序列特征的输入组织问题。
2. **特征交叉侧**：用 `AutoInt` / `DCNMix` / `GDCN` 替代论文中的显式或隐式 feature interaction 模块。
3. **序列建模侧**：用 `DIN` / `DIEN` / `SIM` / `TransAct` / `SASRec` 替代论文中的行为序列编码器。
4. **统一 Backbone 侧**：用 `PyTorch TransformerEncoderLayer` 或 `xFormers` / `FlashAttention` 自己搭一个统一 block，把“序列 token”和“非序列 token”拼在一起训练。
5. **训练与大规模 embedding 侧**：如果特征规模大，再引入 `TorchRec`；如果只是单机或小规模竞赛，优先用 `DeepCTR-Torch` / `FuxiCTR`，工程成本最低。

一句话总结：

- **可直接替代**：embedding、特征交叉、序列编码、基础 attention、长序列基线
- **可部分替代**：异构交互、统一 tokenizer、token mixing、MoE 扩展
- **必须自研**：OneTrans/MixFormer/HyFormer/InterFormer/TokenMixer-Large 的“统一 block 设计本身”

---

## 2. 逐模块替代映射

| 论文中的模块 | 是否可开源替代 | 推荐替代方案 | 说明 |
|---|---|---|---|
| 稀疏特征 embedding | ✅ 直接替代 | TorchRec / DeepCTR-Torch / FuxiCTR | 标准做法，开源支持成熟 |
| 稠密特征投影 | ✅ 直接替代 | `nn.Linear` / MLP / DeepCTR 输入层 | 几乎没有实现门槛 |
| 变长行为序列输入 | ✅ 直接替代 | TorchRec `KeyedJaggedTensor` / FuxiCTR / DeepCTR sequence features | 主要是工程表示问题 |
| 特征交叉层 | ✅ 直接替代 | AutoInt / DCNMix / GDCN | 这部分开源最成熟 |
| 序列编码器 | ✅ 直接替代 | DIN / DIEN / SIM / TransAct / SASRec | 按时延和序列长度选 |
| 基础 Transformer block | ✅ 直接替代 | PyTorch TransformerEncoderLayer / xFormers / FlashAttention | 可做统一 backbone 底座 |
| 异构 attention | ⚠️ 部分替代 | AutoInt + type embedding + 自定义 attention mask / type-specific QKV | 要自己加 token type 逻辑 |
| OneTrans 统一 tokenizer | ⚠️ 部分替代 | DeepCTR/FuxiCTR 特征编码 + 自定义 token packer | 框架能做输入编码，但“统一成 token 序列”要自己写 |
| MixFormer 联合 co-scaling 结构 | ⚠️ 部分替代 | 统一 block + 可配深度/宽度/序列长度实验脚本 | 论文里的 scaling 规律不能直接复用，只能做近似实验 |
| TokenMixer / token mixing 算子 | ⚠️ 部分替代 | MLP-Mixer 风格 token mixing / 1D conv / grouped linear | 需要自己实现推荐场景版本 |
| GDPA / 特殊 kernel | ⚠️ 部分替代 | FlashAttention / xFormers；H100/B200 可选 GDPA | InterFormer 只有底层 kernel 开源 |
| Sparse MoE 扩展 | ⚠️ 部分替代 | Tutel / DeepSpeed-MoE | 可用于 TokenMixer-Large 风格扩容 |
| 论文级统一 block 设计 | ❌ 不能直接替代 | 必须自研 | 这是论文真正的创新点 |

---

## 3. 按 5 篇论文分别看怎么“拼”

### 3.1 OneTrans

OneTrans 的核心不是某个单独算子，而是：

1. 把序列特征和非序列特征统一成 token
2. 用统一 block 同时建模 feature interaction 和 sequence modeling
3. 支持缓存和高效推理

**可替代模块**

- 稀疏/稠密特征编码：`DeepCTR-Torch` / `FuxiCTR`
- 序列建模基线：`DIN` / `DIEN` / `SASRec` / `TransAct`
- attention block：`PyTorch TransformerEncoderLayer` / `xFormers` / `FlashAttention`
- feature interaction 先验：`AutoInt` / `DCNMix`

**必须自己实现**

- unified tokenizer
- non-seq token 与 seq token 的统一拼接方式
- OneTrans block 中的共享参数/分类型参数逻辑
- KV cache / request-level cache 逻辑

**最现实的替代落地**

- 先用 `DeepCTR-Torch` 管特征定义
- 自己写一个 `TokenPacker`：
  - sparse field -> embedding token
  - dense field -> linear projection token
  - history sequence -> sequence tokens
- 用 `TransformerEncoderLayer(batch_first=True)` 先堆 2 到 4 层
- 输出侧接 `MLP + sigmoid`

这能得到一个“简化版 OneTrans baseline”。

### 3.2 MixFormer

MixFormer 的重点是 **dense 侧与 sequence 侧的联合 co-scaling**，本质上是在统一架构内分配容量。

**可替代模块**

- tokenizer：沿用 OneTrans 的自定义实现
- backbone：`TransformerEncoderLayer` / `xFormers`
- dense interaction 侧：`DCNMix` / `GDCN`
- sequence 侧：`SIM` / `TransAct` / `SASRec`

**必须自己实现**

- dense capacity 和 sequence length 的联合 scaling 方案
- block 内部容量分配策略
- 与论文一致的训练/推理优化策略

**最现实的替代落地**

- 统一 backbone 不变
- 用配置控制三组变量：
  - `num_dense_tokens`
  - `max_seq_len`
  - `num_layers / hidden_dim`
- 做一个小型 scaling sweep，观察 AUC 和 latency

也就是说，**MixFormer 更像实验设计问题，不是代码库问题**。

### 3.3 HyFormer

HyFormer 强调重新审视“序列建模”和“特征交叉”在统一模型中的角色，属于结构融合类论文。

**可替代模块**

- sequence encoder：`DIN` / `DIEN` / `SIM` / `TransAct`
- feature interaction：`AutoInt` / `DCNMix` / `GDCN`
- unified block 底座：`TransformerEncoderLayer`

**必须自己实现**

- 两类信息在 block 内部的协同路径
- 论文中的 hybrid 设计与信息流
- 可能的分层或分支聚合机制

**最现实的替代落地**

- 先做双路输入：
  - 路 1：sequence tokens
  - 路 2：context/feature tokens
- 在统一 encoder 内加 `token_type_embedding`
- 输出端加两种 pooling：
  - sequence-aware pooling
  - field-aware pooling
- 再 concat 做预测

这会比硬复现 HyFormer 稳定得多。

### 3.4 InterFormer

InterFormer 的特点是 **异构特征交互**。它是这 5 篇里唯一有“相关官方代码”的，但公开的是 `GDPA` kernel。

**可替代模块**

- 异构 token 表示：`DeepCTR-Torch` / `TorchRec`
- attention backbone：`xFormers` / `FlashAttention`
- 如果硬件是 H100/B200：可尝试 `facebookresearch/ads_model_kernel_library` 里的 `GDPA`

**必须自己实现**

- InterFormer 的异构交互图式
- 双向信息流和 block 排布
- 不同 token type 的参数化策略

**最现实的替代落地**

- 先不追求 GDPA
- 用标准 attention 做异构交互 baseline：
  - 加 `token_type_embedding`
  - 对 dense/sparse/seq token 使用 type-specific projection
  - 用 attention mask 控制部分交互
- 如果后续 GPU 条件允许，再替换成 GDPA kernel

所以对竞赛来说，**InterFormer 最值得借的是“异构交互思路”，不是它的底层 kernel**。

### 3.5 TokenMixer-Large

TokenMixer-Large 的关键是：

1. token mixing 替代或弱化标准 attention
2. 更深网络的稳定训练
3. Sparse MoE 扩容

**可替代模块**

- token mixer 原型：自定义 MLP-Mixer 风格 mixing / 1D conv / grouped linear
- scaling 扩容：`Tutel`
- 训练底座：PyTorch + `torch.compile`

**必须自己实现**

- 推荐场景下的 token mixing block
- mixing-and-reverting 机制
- 与论文一致的 residual 和 auxiliary loss 设计

**最现实的替代落地**

- 先做一个轻量版本：
  - channel mixing：FFN
  - token mixing：`Linear(L, L)` 或 depthwise 1D conv
- 不要一开始就上 MoE
- 先验证 token mixer 对 latency 是否优于 attention
- 如果收益明显，再把 FFN 替换成 Tutel MoE

---

## 4. 推荐的开源库选型

### 4.1 第一优先级：快速出 baseline

#### DeepCTR-Torch

适合用途：

- 快速定义 sparse/dense/sequence 特征
- 直接拿 `AutoInt` / `DCNMix` / `DIN` / `DIEN` 做基线
- 从源码里拆交叉层和输入层，拼进自定义 unified model

最适合替代的模块：

- feature columns
- embedding lookup
- AutoInt interaction layer
- DIN / DIEN 用户行为建模
- DCNMix cross network

不适合的点：

- 不适合作为最终统一 backbone 框架
- 对“统一 token 序列”范式支持不原生

### 4.2 第二优先级：做规范实验

#### FuxiCTR

适合用途：

- 配置化管理实验
- 快速跑 CTR 各类基线
- 作为数据处理和训练脚手架

最适合替代的模块：

- 数据预处理
- baseline 对照实验
- 长序列建模对照

不适合的点：

- 论文里的新型统一 block 还是得自己写

### 4.3 第三优先级：长序列模块补齐

#### LongCTR

适合用途：

- 直接参考长序列 CTR 模型实现
- 借 `SIM` / `TransAct` / `DIEN` / `DIN`

最适合替代的模块：

- 长行为序列 encoder
- retrieval + compression 风格的序列表示

### 4.4 第四优先级：大 embedding 和工业输入表示

#### TorchRec

适合用途：

- 特征规模大
- embedding table 很多
- 需要高效变长 sparse 表示

最适合替代的模块：

- `KeyedJaggedTensor`
- `EmbeddingBagCollection`
- embedding sharding

不适合的点：

- 小规模竞赛原型会增加工程复杂度

### 4.5 第五优先级：统一 block 加速

#### xFormers / FlashAttention

适合用途：

- 自己写统一 transformer block
- 降低 attention 显存和计算成本

最适合替代的模块：

- self-attention
- memory-efficient attention
- 更深网络训练

### 4.6 第六优先级：MoE 扩容

#### Tutel

适合用途：

- 后期做 TokenMixer-Large 风格扩容
- 想做 sparse expert 层

不建议一开始就用：

- 竞赛 baseline 阶段
- 单卡小样本验证阶段

---

## 5. 推荐的“最省时间”落地组合

### 方案 A：最稳妥，适合先拿可用成绩

目标：先做一个可训练、可提交、延迟可控的 unified baseline。

组合：

- 输入与特征：`DeepCTR-Torch`
- 特征交叉：`DCNMix`
- 序列建模：`DIN` 或 `DIEN`
- 统一 backbone：自写 2 到 4 层 `TransformerEncoderLayer`

做法：

1. 用 DeepCTR 的特征定义方式准备 sparse/dense/seq 输入
2. 自写 `TokenPacker`
3. 把所有输入变成统一 token 序列
4. 堆统一 encoder
5. 输出端接 MLP

优点：

- 开发成本最低
- 训练稳定
- 容易控时延

缺点：

- 与论文最前沿版本有差距

### 方案 B：最接近 OneTrans / MixFormer

目标：尽量贴近比赛主题。

组合：

- 数据与实验：`FuxiCTR`
- 序列模块参考：`LongCTR`
- backbone：`xFormers` + 自写 unified encoder
- interaction 参考：`AutoInt` / `GDCN`

做法：

1. FuxiCTR 负责训练配置和对照实验
2. LongCTR 参考序列压缩或长序列处理
3. 自定义 unified tokenization 和 block
4. 用 xFormers 优化 attention

优点：

- 更接近论文路线
- 后续扩展空间更大

缺点：

- 工程成本明显更高

### 方案 C：后期冲榜扩展

目标：在已有 unified baseline 上继续做大模型化和高效化。

组合：

- backbone：自定义 TokenMixer / Transformer hybrid
- kernel：`FlashAttention`，高端卡可尝试 `GDPA`
- 扩容：`Tutel`
- embedding：必要时 `TorchRec`

优点：

- 最有可能贴近 TokenMixer-Large / InterFormer 的工业路线

缺点：

- 不适合早期 baseline
- 风险最高

---

## 6. 哪些部分不要浪费时间

下面这些点，**不建议在第一阶段投入太多时间**：

1. **严格复现论文中的特殊 kernel**
   `InterFormer` 的 `GDPA` 需要特定硬件，竞赛早期收益不高。
2. **过早上 MoE**
   `TokenMixer-Large` 的核心扩展点之一是 MoE，但在 baseline 没跑通之前引入 MoE 只会增加变量。
3. **追求论文同款 scaling law**
   `MixFormer` 和 `TokenMixer-Large` 的很多结论依赖工业级流量和算力，小规模竞赛环境很难复现。

---

## 7. 最终建议

如果目标是 **最短时间做出一套贴近比赛主题、但工程上可落地的方案**，建议路线如下：

1. **先用 DeepCTR-Torch + 自写 TokenPacker + PyTorch TransformerEncoderLayer** 做一个简化版 OneTrans。
2. **再把 DCNMix / AutoInt / DIEN 的思想嵌进去**，做 feature interaction 和 sequence modeling 的增强。
3. **等 baseline 稳定后，再尝试 xFormers / FlashAttention 优化时延**。
4. **只有在模型已经证明有效后，才考虑 Tutel、GDPA、TokenMixer 这类高复杂度增强项**。

对应到“哪些模块可以用开源库替代”的最终判断：

- **80% 的工程工作可以被开源库替代**
  - 输入编码
  - embedding
  - 序列建模基线
  - 特征交叉基线
  - attention 基础设施
  - 大规模 sparse 表示
- **20% 的核心创新必须自己实现**
  - 统一 tokenizer
  - 统一 block
  - 异构交互路径
  - token mixing / co-scaling / caching

也就是说，**真正要写的不是“整套推荐系统基础设施”，而是这 5 篇论文最核心的统一建模层**。

---

## 8. 参考的开源项目与文档

- DeepCTR-Torch: https://github.com/shenweichen/DeepCTR-Torch
- DeepCTR-Torch Features: https://deepctr-torch.readthedocs.io/en/release/Features.html
- AutoInt 官方实现: https://github.com/shichence/AutoInt
- DCNMix 文档: https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.dcnmix.html
- DIEN 文档: https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.dien.html
- RecBole: https://github.com/RUCAIBox/RecBole
- SASRec in RecBole: https://recbole.io/docs/v0.1.2/recbole/recbole.model.sequential_recommender.sasrec.html
- FuxiCTR: https://github.com/reczoo/FuxiCTR
- LongCTR: https://github.com/reczoo/LongCTR
- TorchRec: https://github.com/meta-pytorch/torchrec
- TorchRec Concepts: https://docs.pytorch.org/torchrec/concepts.html
- PyTorch TransformerEncoderLayer: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.TransformerEncoderLayer.html
- xFormers: https://github.com/facebookresearch/xformers
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- InterFormer 相关 GDPA kernel: https://github.com/facebookresearch/ads_model_kernel_library/tree/main/gdpa
- Tutel: https://github.com/microsoft/Tutel
