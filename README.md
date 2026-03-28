# KDD Cup 2026 - Tencent Advertising Algorithm Competition

> **统一 Tokenization + 可堆叠 Backbone** 架构的广告转化率 (CVR) 预测

[![Competition](https://img.shields.io/badge/KDD%20Cup-2026-blue)](https://algo.qq.com)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)

## 📋 比赛简介

腾讯广告算法大赛 2026 (TAAC2026)，聚焦于广告 CVR 预测任务。要求构建 **统一 tokenization + 可堆叠同质 backbone** 的排序模型，在满足严格推理延迟约束下最大化 AUC。

- **任务**: CVR 预测（二分类，AUC 评估）
- **数据**: 异构多域特征 (user 57 维 + item 15 维) + 3 种用户行为序列 (最长 ~4000 步)
- **约束**: 严格推理延迟限制

## 📁 项目结构

```
KDD2026/
├── configs/                    # 实验配置文件
│   ├── baseline.yaml           # baseline 模型配置
│   └── ...
├── data/                       # 数据目录 (不上传 git)
│   ├── KDD_Cup2026/            # 比赛官方数据
│   └── TaobaoAd/               # 淘宝广告预训练数据
├── docs/                       # 研究文档与分析
│   ├── competition_overview.md
│   ├── data_overview.md
│   ├── relevant_papers_analysis.md
│   └── ...
├── notebooks/                  # 探索性分析 Notebook
├── scripts/                    # 数据处理与实用脚本
│   ├── preprocess_kdd.py       # 比赛数据预处理
│   ├── preprocess_taobao.py    # 淘宝数据预处理
│   └── explore_data.py         # 数据探索
├── src/                        # 核心模型代码
│   ├── __init__.py
│   ├── data/                   # 数据加载与处理
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset 定义
│   │   ├── feature_processor.py # 特征工程
│   │   └── collator.py         # Batch 组装
│   ├── models/                 # 模型定义
│   │   ├── __init__.py
│   │   ├── backbone.py         # 统一可堆叠 backbone
│   │   ├── tokenizer.py        # 统一 tokenizer
│   │   ├── layers/             # 可复用组件层
│   │   │   ├── __init__.py
│   │   │   ├── attention.py    # 注意力机制 (DIN/Target Attention)
│   │   │   ├── cross_net.py    # 特征交叉层 (DCN V2/GDCN)
│   │   │   ├── sequence.py     # 序列编码器
│   │   │   └── embedding.py    # 嵌入层
│   │   └── baselines/          # Baseline 模型
│   │       ├── __init__.py
│   │       ├── din_dcn.py      # DIN + DCN V2
│   │       └── unified.py      # 统一架构
│   ├── training/               # 训练逻辑
│   │   ├── __init__.py
│   │   ├── trainer.py          # 训练器
│   │   ├── optimizer.py        # 优化器配置
│   │   └── metrics.py          # 评估指标
│   └── inference/              # 推理与部署
│       ├── __init__.py
│       ├── predictor.py        # 推理服务
│       └── latency.py          # 延迟基准测试
├── tests/                      # 单元测试
│   ├── test_data.py
│   └── test_model.py
├── checkpoints/                # 模型权重 (不上传 git)
├── outputs/                    # 实验输出 (不上传 git)
├── Refs/                       # 参考论文 PDF (不上传 git)
├── .gitignore
├── README.md
├── requirements.txt
└── train.py                    # 训练入口
```

## 🚀 快速开始

### 环境安装

```bash
git clone https://github.com/ZeroTrust01/KDD2026.git
cd KDD2026
pip install -r requirements.txt
```

### 数据准备

1. 将比赛数据放入 `data/KDD_Cup2026/`
2. 将淘宝广告数据放入 `data/TaobaoAd/`

### 训练

```bash
python train.py --config configs/baseline.yaml
```

## 📚 参考论文

| 论文 | 机构 | 核心思想 |
|------|------|---------|
| OneTrans | Tencent | 统一 tokenizer + 同质可堆叠 block |
| MixFormer | ByteDance | 序列+特征混合建模 |
| TokenMixer-Large | ByteDance | Scaling up unified ranking |
| InterFormer | Meta | 异构特征交错交互 |
| HyFormer | Alibaba | 序列与特征交叉角色反思 |

## 📄 License

MIT
