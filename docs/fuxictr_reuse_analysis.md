# FuxiCTR 用于 KDD Cup 2026：可复用 vs 必须自改

> 基于 FuxiCTR 源码（v2.x, PyTorch）深度分析，逐模块评估与比赛数据的兼容性。

---

## 📋 FuxiCTR 架构总览

```
FuxiCTR Pipeline
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  ① FeatureProcessor        → 数据预处理/编码                  │
│     ├── Tokenizer           (分类特征 → vocab ID)            │
│     ├── Normalizer          (数值特征归一化)                  │
│     └── fit/transform       (fit 建词表, transform 编码)      │
│                                                              │
│  ② RankDataLoader          → 数据加载                        │
│     ├── NpzDataLoader       (本地 npz)                       │
│     ├── ParquetDataLoader   (Parquet 格式)                   │
│     └── streaming/block     (大文件分块读取)                  │
│                                                              │
│  ③ FeatureEmbedding        → 特征嵌入                        │
│     ├── numeric  → nn.Linear(1, dim)                         │
│     ├── categorical → nn.Embedding(vocab, dim)               │
│     ├── sequence → nn.Embedding + Pooling/Encoder            │
│     └── embedding → nn.Identity + Linear projection          │
│                                                              │
│  ④ Model (model_zoo/)      → 模型定义                        │
│     ├── Feature Interaction: DCNv2/AutoInt/DeepFM/Wukong...  │
│     ├── Behavior Sequence:  DIN/DIEN/BST/TransAct/DMIN...    │
│     ├── Long Sequence:      SIM/ETA/SDIM/TWIN/MIRRN          │
│     └── Multi-task:         MMOE/PLE/ShareBottom             │
│                                                              │
│  ⑤ BaseModel               → 训练/评估循环                    │
│     ├── compile (loss, optimizer, metrics)                    │
│     ├── fit (train loop + early stopping)                    │
│     └── evaluate / predict                                   │
│                                                              │
│  ⑥ Config/Tuner            → YAML 配置 + 网格搜索            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## ✅ 可以直接复用的部分

### 1. 训练/评估框架 (BaseModel) — 复用度 95%

| 组件 | 可复用 | 说明 |
|------|:------:|------|
| `BaseModel.compile()` | ✅ | loss=binary_crossentropy, 评估=AUC, optimizer=Adam/Adagrad |
| `BaseModel.fit()` | ✅ | epoch 训练循环、early stopping、checkpoint 保存 |
| `BaseModel.evaluate()` | ✅ | 比赛 AUC 评估直接匹配 |
| `BaseModel.predict()` | ✅ | 输出概率值 |
| 学习率调度 | ✅ | warmup + decay 策略 |
| GPU 训练 | ✅ | 原生 CUDA 支持 |

```python
# 直接可用，无需修改
model.compile(optimizer="adam", loss="binary_crossentropy", lr=1e-3)
model.fit(train_gen, validation_data=valid_gen, epochs=10)
pred = model.predict(test_gen)
```

### 2. 特征交叉层 (layers/) — 复用度 90%

FuxiCTR 包含的所有交叉层可直接作为模块使用：

| 交叉层 | 来自模型 | 可复用方式 |
|-------|---------|----------|
| `CrossNetV2` | DCN V2 | 显式高阶特征交叉 |
| `MultiHeadSelfAttention` | AutoInt | 自注意力特征交互 |
| `BilinearInteraction` | FiBiNET | 双线性交互 |
| `CompressedInteraction` | xDeepFM/CIN | 压缩交互网络 |
| `InnerProductLayer` | PNN | 内积交互 |
| `GatedCrossNet` | GDCN | 门控交叉网络 |
| `FactorizationMachine` | Wukong/FM | 分解机交互 |
| `DNN` (MLP) | 所有模型 | 通用 MLP 层 |

```python
# 示例：直接取用 CrossNet 作为自定义模型的子模块
from fuxictr.pytorch.layers import CrossNetV2, DNN

cross_layer = CrossNetV2(input_dim=256, num_layers=3)
dnn_layer = DNN(input_dim=256, hidden_units=[512, 256, 128])
```

### 3. Embedding 层 (FeatureEmbedding) — 复用度 70%

| 功能 | 可复用 | 说明 |
|------|:------:|------|
| `nn.Embedding` (categorical) | ✅ | int_value 类型特征直接使用 |
| `nn.Linear` (numeric) | ✅ | float 特征投影 |
| `share_embedding` 机制 | ✅ | 序列中的 item_id 和 target item_id 共享 embedding |
| `padding_idx` 处理 | ✅ | 序列 padding |
| `pretrained_emb` 加载 | ✅ | 预训练向量接入 |

```python
# FeatureEmbedding 处理流程（可复用核心）:
# numeric    → nn.Linear(1, dim)        → [B, dim]
# categorical → nn.Embedding(vocab, dim) → [B, dim]  
# sequence   → nn.Embedding(vocab, dim) → [B, maxlen, dim] → Encoder → [B, dim]
```

### 4. 行为序列模型层 — 复用度 60%

这些模型的**注意力机制和序列编码器**可以复用：

| 序列模型 | 可复用组件 | 说明 |
|---------|----------|------|
| **DIN** | `DINAttention` | Target Attention：用 target item 查询历史行为 |
| **DIEN** | `AUGRUCell`, `AGRUCell` | 兴趣演化 GRU 单元 |
| **BST** | `TransformerEncoder` | 行为序列 Transformer |
| **TransAct** | `TransActLayer` | 实时行为序列建模 |
| **SIM** | `GeneralSearchUnit` (GSU) | 长序列检索单元（适合 action_seq 长达 4000 步） |
| **TWIN** | `TWINLayer` | 终身行为建模 |

### 5. 超参调优框架 — 复用度 100%

```yaml
# 直接可用的 YAML 配置驱动调参
tuner_space:
    model: DCNv2
    embedding_dim: [32, 64, 128]
    learning_rate: [1e-3, 5e-4, 1e-4]
    num_cross_layers: [2, 3, 4]
    batch_size: [512, 1024, 2048]
```

```bash
# 多 GPU 网格搜索
python run_param_tuner.py --config tuner_config.yaml --gpu 0 1 2 3
```

### 6. 多任务学习框架 — 复用度 80%

比赛虽然只要预测 CVR，但如果把 click + conversion 做多任务：

| 框架 | 可复用 | 用途 |
|------|:------:|------|
| **MMOE** | ✅ | 多门混合专家网络 |
| **PLE** | ✅ | 渐进式分层提取（腾讯自家方案） |
| **ShareBottom** | ✅ | 共享底层 |
| **ESMM** | ⚠️ | 需适配标签结构 |

---

## ⚠️ 需要修改适配的部分

### 1. 数据预处理层 — 改造量 🔴 大

**核心问题**：比赛数据是**深度嵌套的 Parquet**，FuxiCTR 预期**扁平表格**。

```
                    FuxiCTR 预期                    比赛实际
                    ──────────                     ──────────
                    扁平 CSV/Parquet               嵌套 struct Parquet
                    每列一个特征                    user_feature = array[struct]
                    序列用 "^" 分隔字符串           seq_feature = struct{action_seq: array[struct]}
                    固定列名                        动态 feature_id
```

**必须自写的预处理代码**：

```python
# ❌ FuxiCTR 不能直接读比赛 Parquet
# ✅ 需要先写转换脚本

def flatten_taac2026_parquet(df):
    """将比赛的嵌套 Parquet 拆平为 FuxiCTR 可接受的格式"""
    
    # 1. 提取 user_feature (array[struct]) → 扁平列
    #    每条记录的 user_feature 含 13~54 个不同 feature_id
    #    需展开为: uf_1, uf_3, uf_4, ..., uf_105
    for row in df['user_feature']:
        for feat in row:
            fid = feat['feature_id']
            # 根据 feature_value_type 取对应值
            if feat['feature_value_type'] == 'int_value':
                result[f'uf_{fid}'] = feat['int_value']
            elif feat['feature_value_type'] == 'int_array':
                result[f'uf_{fid}'] = '^'.join(map(str, feat['int_array']))
            elif feat['feature_value_type'] == 'float_array':
                # FuxiCTR 不支持 float_array 作为原始输入！
                result[f'uf_{fid}'] = feat['float_array']  # 需要特殊处理
    
    # 2. 提取 item_feature (类似)
    
    # 3. 提取 seq_feature — 最复杂的部分
    #    action_seq 含 10 个并行对齐的 int_array
    #    需要决定：拼接/分别处理/target attention 策略
    
    # 4. 提取 label
    #    action_type → binary label
    
    return flattened_df
```

**预估工作量**: 需要 **200-400 行** Python 代码的数据转换层。

### 2. 序列特征处理 — 改造量 🔴 大

**核心问题**：FuxiCTR 的 sequence 类型是**单个 ID 序列**，比赛是**多属性并行对齐序列**。

```
FuxiCTR sequence:     [item_1, item_2, item_3, ...]         # 单列 ID 序列
                       ↓ nn.Embedding ↓
                      [emb_1, emb_2, emb_3, ...]            # [B, L, D]

比赛 action_seq:      feature_id=19: [1, 1, 2, ...]          # 行为类型
                      feature_id=20: [252, 201, 136, ...]    # 一级类别
                      feature_id=21: [1192, 890, 559, ...]   # 二级类别
                      ...（共 10 个并行数组）
                      feature_id=28: [ts1, ts2, ts3, ...]    # 时间戳
                      
                      每个 position 是一个 10 维属性向量！
```

**三种适配策略（按复杂度递增）**：

#### 策略 A：多序列拆分（最简单，性能一般）
```python
# 把每个 feature_id 当作独立序列
feature_cols = [
    {"name": "action_type_seq",  "type": "sequence", "max_len": 50, ...},
    {"name": "action_cat1_seq",  "type": "sequence", "max_len": 50, ...},
    {"name": "action_item_seq",  "type": "sequence", "max_len": 50, ...},
    # ... 10 个序列
]
# 缺点: 丢失了位置对齐信息，各属性独立编码
```

#### 策略 B：多属性拼接嵌入（推荐，需改 Embedding 层）
```python
# 自定义: 每个事件的多个属性 embedding 拼接/相加后作为一个 token
class MultiAttrSequenceEmbedding(nn.Module):
    def __init__(self, attr_specs, embedding_dim):
        self.attr_embeddings = nn.ModuleDict({
            f"attr_{fid}": nn.Embedding(vocab_size, embedding_dim // num_attrs)
            for fid, vocab_size in attr_specs.items()
        })
    
    def forward(self, seq_dict):
        # seq_dict: {fid: [B, L]} × 10
        attr_embs = [self.attr_embeddings[f"attr_{fid}"](seq_dict[fid]) 
                     for fid in seq_dict]
        token_embs = torch.cat(attr_embs, dim=-1)  # [B, L, D]
        return token_embs
```

#### 策略 C：统一 Tokenizer（最贴近 OneTrans，需要大改）
```python
# 参考 OneTrans: 将每个事件的多个属性 group 起来统一 tokenize
# 需要自定义 SemanticGroupTokenizer，完全超出 FuxiCTR 框架
```

### 3. 混合类型特征 — 改造量 🟡 中

**核心问题**：比赛的 `int_array_and_float_array` 类型（用户特征 69-73, 83-85）在 FuxiCTR 中没有对应。

```python
# 比赛数据: feature_id=69
{
    "int_array":   [4, 2, 3, 5],             # 类别 ID
    "float_array": [60298., 25804., 3220., ...]  # 对应权重/频次
}

# FuxiCTR 没有 "带权重的多值分类" 这种类型！
```

**适配方案**：
```python
# 自定义 WeightedSequenceEmbedding
class WeightedMultiValueEmbedding(nn.Module):
    def forward(self, ids, weights):
        emb = self.embedding(ids)           # [B, N, D]
        w = weights.unsqueeze(-1)           # [B, N, 1]
        return (emb * w).sum(dim=1)         # [B, D]  加权聚合
```

### 4. float_array 稠密向量特征 — 改造量 🟡 中

用户特征 68、81 是 `float_array` 类型（可能是预训练 embedding 向量）。

```python
# FuxiCTR 有 "embedding" type 可以处理，但需要知道维度
# 在 feature_cols 中配置：
{"name": "uf_68", "type": "embedding", "pretrain_dim": 64, "embedding_dim": 32}
# 内部: nn.Identity() → nn.Linear(pretrain_dim, embedding_dim)
```

但问题是：**float_array 长度可能不固定**。如果长度不一，需要先 padding 再用 Linear 投影。

### 5. 超长序列处理 — 改造量 🟠 大

比赛 `action_seq` 中位数 **2,798 步**，最长 **3,998 步**。

| FuxiCTR 模型 | 最大序列长度 | 适用性 |
|-------------|:-----------:|:------:|
| DIN | 通常 ≤ 200 | ❌ 太慢 (O(L) attention) |
| DIEN | 通常 ≤ 200 | ❌ GRU 太慢 |
| BST | 通常 ≤ 100 | ❌ O(L²) self-attention |
| **SIM** | ✅ 千级+ | ⭕ 两阶段检索，适合长序列 |
| **ETA** | ✅ 千级+ | ⭕ 高效 hash 检索 |
| **SDIM** | ✅ 千级+ | ⭕ 采样策略 |
| **TWIN** | ✅ 终身序列 | ⭕ 两阶段兴趣网络 |

**建议**：使用 FuxiCTR 的 **SIM/ETA/TWIN** 长序列模块，先 truncate 到 ~200，后续再优化。

---

## ❌ 完全不能用、必须自己写的部分

### 1. Parquet 嵌套 Struct 解析器

FuxiCTR 的 `FeatureProcessor.read_data()` 使用 `polars.scan_parquet()` 读取**扁平 Parquet**。
比赛数据是嵌套 struct，需要 **完全自写解析层**。

```python
# 需要自写: taac2026_data_processor.py
class TAAC2026DataProcessor:
    """将比赛嵌套 Parquet 转换为 FuxiCTR 可用的扁平格式"""
    
    def parse_parquet(self, path):
        df = pd.read_parquet(path)
        flat_records = []
        for _, row in df.iterrows():
            record = {}
            record['label'] = 1 if row['label'][0]['action_type'] == 2 else 0
            record['user_id'] = row['user_id']
            record['item_id'] = row['item_id']
            record['timestamp'] = row['timestamp']
            
            # 展开 user_feature
            for feat in row['user_feature']:
                fid = feat['feature_id']
                vtype = feat['feature_value_type']
                if vtype == 'int_value':
                    record[f'uf_{fid}'] = feat['int_value']
                elif vtype == 'int_array':
                    record[f'uf_{fid}'] = '^'.join(map(str, feat['int_array']))
                elif vtype == 'float_array':
                    record[f'uf_{fid}_emb'] = feat['float_array']
                elif vtype == 'int_array_and_float_array':
                    record[f'uf_{fid}_ids'] = '^'.join(map(str, feat['int_array']))
                    record[f'uf_{fid}_wts'] = feat['float_array']
            
            # 展开 item_feature (类似)
            ...
            
            # 处理 seq_feature — 最复杂
            for seq_name in ['action_seq', 'content_seq', 'item_seq']:
                seq_data = row['seq_feature'][seq_name]
                for feat in seq_data:
                    fid = feat['feature_id']
                    record[f'{seq_name}_f{fid}'] = '^'.join(map(str, feat['int_array']))
            
            flat_records.append(record)
        return pd.DataFrame(flat_records)
```

### 2. 多属性并行序列 Token 编码

FuxiCTR 完全没有 "一个事件有多个属性" 的序列概念。

### 3. 统一 Tokenization 架构

比赛主题要求的 "统一 tokenization + 可堆叠 backbone" 在 FuxiCTR 中不存在。FuxiCTR 是传统的 "embedding → interaction" 两阶段范式。

### 4. 推理延迟优化

FuxiCTR 没有任何推理延迟优化（无 KV caching、无量化、无算子融合）。

---

## 📊 模块级复用度总结

| 模块 | 复用度 | 改造量 | 说明 |
|------|:------:|:------:|------|
| **训练循环 (BaseModel)** | 🟢 95% | 几乎不改 | compile/fit/evaluate 直接用 |
| **超参调优 (Tuner)** | 🟢 100% | 不改 | YAML 配置 + 网格搜索 |
| **交叉层 (CrossNet/Attention)** | 🟢 90% | 不改 | 当子模块直接 import |
| **模型损失/优化器** | 🟢 95% | 不改 | BCE loss + AUC metric |
| **多任务框架 (MMOE/PLE)** | 🟢 80% | 小改 | 需适配标签格式 |
| **分类特征 Embedding** | 🟡 70% | 小改 | int_value 类型直接用 |
| **序列注意力 (DIN/SIM)** | 🟡 60% | 中改 | 可复用 attention，需改输入 |
| **长序列检索 (SIM/ETA)** | 🟡 50% | 中改 | 检索逻辑可用，需改接口 |
| **数值特征处理** | 🟡 60% | 中改 | float_array 需自定义 |
| **混合类型特征** | 🟠 20% | 大改 | int_array_and_float_array 需自写 |
| **序列特征处理** | 🔴 10% | 大改 | 多属性并行序列需完全自写 |
| **数据加载 (DataLoader)** | 🔴 10% | 大改 | 嵌套 Parquet 解析需自写 |
| **统一 Tokenizer** | 🔴 0% | 全写 | 不存在，需参照 OneTrans 自实现 |
| **推理优化** | 🔴 0% | 全写 | 无 KV cache/量化/算子融合 |

---

## 🎯 推荐方案：分层复用策略

```
┌─────────────────────────────────────────────────────────────┐
│                    自写层 (比赛特有)                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ TAAC2026DataProcessor                                │   │
│  │  - 嵌套 Parquet → 扁平格式                            │   │
│  │  - 多属性序列编码                                      │   │
│  │  - int_array_and_float_array 处理                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ MultiAttrSequenceEncoder                             │   │
│  │  - 多属性 → 单 token embedding                       │   │
│  │  - Target item attention                             │   │
│  │  - 3 种序列融合策略                                   │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                   复用层 (来自 FuxiCTR)                       │
│  ┌──────────────────┐  ┌──────────────────────────────┐    │
│  │ FeatureEmbedding │  │ 交叉层: CrossNetV2 / AutoInt  │    │
│  │ (int_value 部分)  │  │        GDCN / FiBiNET        │    │
│  └──────────────────┘  └──────────────────────────────┘    │
│  ┌──────────────────┐  ┌──────────────────────────────┐    │
│  │ DINAttention     │  │ SIM GeneralSearchUnit (GSU)  │    │
│  │ (target attn)    │  │ (长序列检索)                   │    │
│  └──────────────────┘  └──────────────────────────────┘    │
│  ┌──────────────────┐  ┌──────────────────────────────┐    │
│  │ BaseModel        │  │ MMOE / PLE                   │    │
│  │ (训练/评估循环)    │  │ (多任务框架)                   │    │
│  └──────────────────┘  └──────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Tuner (YAML 配置 + 网格搜索 + 多 GPU 并行)            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## ⏱️ 预估工时

| 任务 | 工时 | 优先级 |
|------|:----:|:------:|
| **数据解析层** (嵌套 Parquet → 扁平) | 1-2 天 | P0 |
| **feature_cols YAML 配置** (57 user + 15 item + 序列) | 0.5 天 | P0 |
| **多属性序列编码器** | 2-3 天 | P0 |
| **混合类型特征处理** | 1 天 | P1 |
| **float_array embedding 投影** | 0.5 天 | P1 |
| **拼装 baseline 模型** (DIN/DCNv2 + 自定义序列) | 1-2 天 | P0 |
| **长序列处理** (SIM/ETA 适配) | 2-3 天 | P2 |
| ——————————————————— |——|——|
| **总计** | **~8-12 天** | — |

> [!TIP]
> **结论**: FuxiCTR 的核心价值在于其 **成熟的训练框架 + 丰富的交叉层/注意力层组件库 + 超参调优工具**。数据预处理和序列特征处理需要大量自定义工作，但模型层面的组件可以显著加速开发。建议先用 FuxiCTR 的 `DIN + DCNv2` 跑通 pipeline 拿到 baseline 分数，再逐步替换为自定义的统一架构。
