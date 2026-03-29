"""
KDD Cup 2026 - Full Data Preprocessing on Kaggle
=================================================
在 Kaggle 上对全量淘宝广告数据进行预处理。

前置步骤:
1. 创建 Kaggle Notebook, 选 GPU T4 x2, Internet ON
2. Add data: 添加 "taobao-ad-display-click" (23GB 原始数据)
3. 运行本脚本, 处理完成后点 "Save Version" 保存输出
4. 用输出数据创建新 Dataset, 用于后续训练

预计耗时: 预处理 30-60 分钟, 训练 5-10 分钟
"""

import os
import sys
import subprocess
import glob
import shutil

# ─── Step 0: Clone code ───
print("=" * 60)
print("Step 0: Cloning code ...")
print("=" * 60)

if not os.path.exists("/kaggle/working/KDD2026"):
    subprocess.run(["git", "clone", "https://github.com/ZeroTrust01/KDD2026.git",
                    "/kaggle/working/KDD2026"], check=True)

os.chdir("/kaggle/working/KDD2026")
sys.path.insert(0, "/kaggle/working/KDD2026")

# ─── Step 1: Locate raw data ───
print("\n" + "=" * 60)
print("Step 1: Locating raw data ...")
print("=" * 60)

KAGGLE_INPUT = "/kaggle/input"
DATA_DIR = "data/TaobaoAd"
os.makedirs(DATA_DIR, exist_ok=True)

EXPECTED_CSVS = ["raw_sample.csv", "ad_feature.csv", "user_profile.csv", "behavior_log.csv"]

# Kaggle extracts tar.gz into nested dirs like: csv_name/csv_name
# Find actual files (not directories) for each expected CSV
for csv_name in EXPECTED_CSVS:
    target = os.path.join(DATA_DIR, csv_name)
    if os.path.isfile(target):
        size = os.path.getsize(target)
        label = f"{size / 1e6:.0f} MB" if size < 1e9 else f"{size / 1e9:.1f} GB"
        print(f"  ✓ {csv_name} already exists ({label})")
        continue

    # Remove stale symlink pointing to directory
    if os.path.islink(target):
        os.unlink(target)

    # Search for actual FILE (not directory) matching csv_name
    found = False
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        if csv_name in files:
            src = os.path.join(root, csv_name)
            if os.path.isfile(src) and not os.path.isdir(src):
                os.symlink(src, target)
                size = os.path.getsize(src)
                label = f"{size / 1e6:.0f} MB" if size < 1e9 else f"{size / 1e9:.1f} GB"
                print(f"  ✓ {csv_name} linked ({label})")
                found = True
                break
    if not found:
        print(f"  ✗ {csv_name} NOT FOUND")

# Verify all files present
print("\nData verification:")
all_found = True
for csv_name in EXPECTED_CSVS:
    fpath = os.path.join(DATA_DIR, csv_name)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        label = f"{size / 1e6:.0f} MB" if size < 1e9 else f"{size / 1e9:.1f} GB"
        print(f"  ✓ {csv_name}: {label}")
    else:
        print(f"  ✗ {csv_name}: MISSING")
        all_found = False

if not all_found:
    print("\n⚠️  Listing all available files:")
    for f in glob.glob(f"{KAGGLE_INPUT}/**/*", recursive=True):
        if os.path.isfile(f):
            print(f"    {f}")
    raise FileNotFoundError("Raw data files missing!")


# ─── Step 2: Preprocess (FULL data) ───
print("\n" + "=" * 60)
print("Step 2: Preprocessing FULL dataset ...")
print("  Using DuckDB for memory-efficient behavior_log processing")
print("=" * 60)

# Install DuckDB if not available
try:
    import duckdb
    print(f"  DuckDB {duckdb.__version__} available")
except ImportError:
    print("  Installing DuckDB ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "duckdb"], check=True)
    import duckdb
    print(f"  DuckDB {duckdb.__version__} installed")

PROCESSED_DIR = "data/TaobaoAd/processed"
VOCAB_PATH = os.path.join(PROCESSED_DIR, "feature_vocab.json")

if os.path.exists(VOCAB_PATH):
    print("  Processed data already exists, skipping.")
else:
    # FULL data: sample_users=0, max_behavior_rows=0
    cmd = [sys.executable, "scripts/preprocess_taobao.py",
           "--sample_users", "0",
           "--max_behavior_rows", "0",
           "--backend", "duckdb"]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# Show stats
import json
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
print(f"\n  Vocab features: {len(vocab)}")
for feat, v in vocab.items():
    print(f"    {feat}: {len(v)} tokens")

# Copy processed data to Kaggle output for saving
OUTPUT_DIR = "/kaggle/working/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)
for fname in ["train.parquet", "valid.parquet", "test.parquet", "feature_vocab.json"]:
    src = os.path.join(PROCESSED_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(src):
        shutil.copy(src, dst)
        size = os.path.getsize(src)
        print(f"  Copied {fname} to output ({size / 1e6:.1f} MB)")


# ─── Step 3: Train ───
print("\n" + "=" * 60)
print("Step 3: Training DIN + DCN V2 on FULL data ...")
print("=" * 60)

import torch
import yaml
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from src.data.dataset import create_dataloaders, get_feature_config
from src.models.baselines.din_dcn import DIN_DCN
from src.training.trainer import Trainer

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

with open("configs/baseline.yaml") as f:
    config = yaml.safe_load(f)

# Full data config
config["data"]["num_workers"] = 2
config["data"]["batch_size"] = 4096      # larger batch for full data
config["training"]["epochs"] = 20
config["training"]["early_stopping_patience"] = 5
config["training"]["learning_rate"] = 5e-4   # slightly smaller LR for full data
config["model"]["dnn_dropout"] = 0.2         # more regularization

data_cfg = config["data"]
model_cfg = config["model"]
train_cfg = config["training"]

with open(VOCAB_PATH) as f:
    feature_vocab = json.load(f)

max_seq_len = data_cfg.get("max_seq_len", 50)
loaders = create_dataloaders(
    PROCESSED_DIR, feature_vocab,
    batch_size=data_cfg["batch_size"],
    max_seq_len=max_seq_len,
    num_workers=data_cfg["num_workers"],
)

feature_config = get_feature_config(feature_vocab, emb_dim=model_cfg["embedding_dim"], max_seq_len=max_seq_len)
model_cfg["din_pairs"] = [tuple(p) for p in model_cfg.get("din_pairs", [])]
model = DIN_DCN(feature_config, model_cfg)
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

trainer = Trainer(model, device=device, config=train_cfg)
result = trainer.fit(
    train_loader=loaders["train"],
    valid_loader=loaders.get("valid"),
    epochs=train_cfg["epochs"],
    patience=train_cfg["early_stopping_patience"],
)

# ─── Step 4: Test ───
print("\n" + "=" * 60)
print("Step 4: Test Results")
print("=" * 60)

if "test" in loaders:
    test_metrics = trainer.evaluate(loaders["test"])
    print(f"  Test AUC:     {test_metrics['auc']:.4f}")
    print(f"  Test LogLoss: {test_metrics['logloss']:.4f}")

print(f"\n  Best valid AUC = {result['best_auc']:.4f} at epoch {result['best_epoch']}")

if os.path.exists("checkpoints/best_model.pt"):
    shutil.copy("checkpoints/best_model.pt", "/kaggle/working/best_model.pt")
    print("  Model saved to /kaggle/working/best_model.pt")

print("\n" + "=" * 60)
print("Done! 保存此 Notebook 的输出, 可从 /kaggle/working/processed/ 获取处理好的数据")
print("=" * 60)
