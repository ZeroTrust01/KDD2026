# 可用训练数据集参考

> 当前 sample 数据仅 1000 条，用于理解数据格式。**正式训练数据将在初赛阶段（4月24日）由官方发布**。
> 以下列出可用于前期模型开发和预训练的开源数据集。

---

## 一、关于竞赛正式数据

根据竞赛时间线：

| 阶段 | 时间 | 数据情况 |
|------|------|----------|
| Phase 1 | 3月15日 | ✅ 已发布 demo 数据集（1000 条） |
| Phase 2 | 3月19日–4月23日 | 报名期，无新数据 |
| **Phase 3** | **4月24日–5月23日** | **初赛，将发布正式训练/测试数据** |
| Phase 4 | 5月25日–6月24日 | 复赛，可能更新数据 |

> ⚠️ 正式的大规模训练数据需等到 **4月24日初赛开始时** 官方发布。当前阶段应利用开源数据集开发和验证模型架构。

---

## 二、推荐的开源数据集（按匹配度排序）

### ⭐⭐⭐⭐⭐ 第一梯队：广告 CVR/CTR + 用户行为序列

#### 1. Ali-CCP (Alibaba Click & Conversion Prediction)

| 项目 | 详情 |
|------|------|
| **来源** | 阿里巴巴淘宝广告，ESMM 论文配套数据集 |
| **规模** | **~8.4 亿条**样本 |
| **任务** | CTR + CVR 预测（多任务） |
| **特征** | 用户画像、物品属性、上下文特征、点击/转化标签 |
| **下载** | [天池](https://tianchi.aliyun.com/dataset/408)（需注册） |
| **匹配度** | ⭐⭐⭐⭐⭐ 电商广告 CVR 预测，包含多域特征，与竞赛最接近 |

#### 2. 淘宝广告展示/点击数据集 (Taobao Ad Display/Click)

| 项目 | 详情 |
|------|------|
| **来源** | 阿里巴巴淘宝广告系统 |
| **规模** | ~2600 万条展示记录，~100 万用户，~80 万广告 |
| **任务** | CTR 预测 |
| **特征** | 用户属性、广告特征、**用户历史行为序列** |
| **下载** | [天池](https://tianchi.aliyun.com/dataset/56)（需注册） |
| **匹配度** | ⭐⭐⭐⭐⭐ 包含用户行为序列 + 多域特征，DIN/DIEN 论文使用的数据集 |

#### 3. 淘宝用户行为数据集 (Taobao User Behavior)

| 项目 | 详情 |
|------|------|
| **来源** | 阿里巴巴淘宝 |
| **规模** | **~1 亿条**用户行为记录，~100 万用户 |
| **任务** | CTR 预测、序列推荐 |
| **特征** | 用户 ID、物品 ID、类别 ID、行为类型（点击/购买/收藏/加购）、时间戳 |
| **下载** | [天池](https://tianchi.aliyun.com/dataset/649)（需注册）/ [Kaggle](https://www.kaggle.com/datasets/heeraldedhia/taobao-user-behaviour-dataset) |
| **匹配度** | ⭐⭐⭐⭐ 大规模行为序列，行为类型与竞赛 action_type 类似，但特征维度较少 |

---

### ⭐⭐⭐⭐ 第二梯队：大规模 CTR 预测（多域特征，无序列）

#### 4. Criteo Terabyte Dataset

| 项目 | 详情 |
|------|------|
| **来源** | Criteo (法国广告技术公司) |
| **规模** | **~45 亿条**展示记录（1TB+） |
| **任务** | CTR 预测 |
| **特征** | 13 个连续特征 + 26 个类别特征（匿名化） |
| **下载** | [Criteo 官网](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) |
| **匹配度** | ⭐⭐⭐⭐ 超大规模，适合验证 scaling law，但**无用户行为序列** |

#### 5. Criteo (Kaggle Display Advertising)

| 项目 | 详情 |
|------|------|
| **来源** | Criteo |
| **规模** | ~4500 万条 |
| **任务** | CTR 预测 |
| **特征** | 13 个连续特征 + 26 个类别特征（匿名化） |
| **下载** | [Kaggle](https://www.kaggle.com/c/criteo-display-ad-challenge) / FuxiCTR 内置 |
| **匹配度** | ⭐⭐⭐⭐ CTR benchmark 标准数据集，适合快速验证特征交叉模型 |

#### 6. Avazu CTR Dataset

| 项目 | 详情 |
|------|------|
| **来源** | Avazu (移动广告平台) |
| **规模** | ~4000 万条 |
| **任务** | 移动广告 CTR 预测 |
| **特征** | 设备信息、广告位、时间等 22 个特征 |
| **下载** | [Kaggle](https://www.kaggle.com/c/avazu-ctr-prediction) / FuxiCTR 内置 |
| **匹配度** | ⭐⭐⭐ 移动广告场景，特征结构较简单，无序列 |

---

### ⭐⭐⭐ 第三梯队：电商/推荐（可构造行为序列）

#### 7. Amazon Product Reviews

| 项目 | 详情 |
|------|------|
| **来源** | Amazon |
| **规模** | ~2.3 亿条评论，多个品类子集可选 |
| **任务** | 序列推荐、评分预测 |
| **特征** | 用户 ID、物品 ID、评分、时间戳、品类 |
| **下载** | [jmcauley.ucsd.edu](https://jmcauley.ucsd.edu/data/amazon/) |
| **匹配度** | ⭐⭐⭐ 可构造行为序列，但非广告场景，特征维度少 |

#### 8. MovieLens 25M

| 项目 | 详情 |
|------|------|
| **来源** | GroupLens Research |
| **规模** | 2500 万条评分 |
| **任务** | 评分预测、序列推荐 |
| **下载** | [grouplens.org](https://grouplens.org/datasets/movielens/25m/) |
| **匹配度** | ⭐⭐ 经典基准，规模偏小，特征简单 |

---

## 三、数据集对比总结

| 数据集 | 规模 | 多域特征 | 行为序列 | CVR 标签 | 推荐用途 |
|--------|:----:|:--------:|:--------:|:--------:|----------|
| **Ali-CCP** | 8.4 亿 | ✅ | ❌ | ✅ | CVR 预测模型验证 |
| **淘宝广告展示** | 2600 万 | ✅ | ✅ | ❌ | 序列+特征交叉联合建模 |
| **淘宝用户行为** | 1 亿 | ⚠️ 少 | ✅ | ⚠️ 购买 | 长序列建模验证 |
| **Criteo TB** | 45 亿 | ✅ | ❌ | ❌ | Scaling law 验证 |
| **Criteo Kaggle** | 4500 万 | ✅ | ❌ | ❌ | 特征交叉 baseline |
| **Avazu** | 4000 万 | ✅ | ❌ | ❌ | 移动广告 CTR |
| **Amazon** | 2.3 亿 | ⚠️ 少 | ✅ | ⚠️ 购买 | 序列推荐 |

---

## 四、建议策略

### 模型开发阶段（当前 → 4月24日）

```
1. 用 Criteo Kaggle 数据集
   → 快速验证特征交叉组件（DCN V2, AutoInt, GDCN）
   → pip install fuxictr 一键跑 benchmark

2. 用淘宝广告展示数据集
   → 验证序列建模 + 特征交叉的联合架构
   → 包含用户行为序列，最匹配竞赛场景

3. 用 Ali-CCP 数据集
   → 验证 CVR 预测能力
   → 大规模训练，测试模型 scaling

4. 用竞赛 sample 1000 条
   → 确保数据解析 pipeline 正确
   → 适配竞赛数据格式
```

### 正式比赛阶段（4月24日之后）

```
→ 使用官方发布的正式训练数据
→ 将预训练/预验证的模型架构迁移到正式数据上
```

---

## 五、快速获取数据

```bash
# Criteo Kaggle（最快上手，FuxiCTR 内置下载）
pip install fuxictr
# FuxiCTR 会自动下载 Criteo 数据

# 淘宝广告展示数据（天池，需注册）
# https://tianchi.aliyun.com/dataset/56

# Ali-CCP（天池，需注册）
# https://tianchi.aliyun.com/dataset/408

# 淘宝用户行为（Kaggle 镜像）
# https://www.kaggle.com/datasets/heeraldedhia/taobao-user-behaviour-dataset

# Amazon Reviews
# https://jmcauley.ucsd.edu/data/amazon/
```
