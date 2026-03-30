"""
KDD Cup 2026 - DIN + DCN V2 Baseline Training on Kaggle
========================================================
使用预处理好的数据直接训练

前置步骤:
1. 创建 Kaggle Notebook, 选择 GPU T4 x2, Internet ON
2. Add data: 搜索 "taobao-ad-processed" 并添加
3. 在 Cell 中运行本脚本
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# ─── Step 0: Clone code ───
print("=" * 60)
print("Step 0: Cloning code from GitHub ...")
print("=" * 60)

if not os.path.exists("/kaggle/working/KDD2026"):
    subprocess.run(["git", "clone", "https://github.com/ZeroTrust01/KDD2026.git",
                    "/kaggle/working/KDD2026"], check=True)

os.chdir("/kaggle/working/KDD2026")
sys.path.insert(0, "/kaggle/working/KDD2026")

# ─── Step 1: Locate preprocessed data ───
print("\n" + "=" * 60)
print("Step 1: Locating preprocessed data ...")
print("=" * 60)

KAGGLE_INPUT = "/kaggle/input"
EXPECTED_FILES = ["train.parquet", "valid.parquet", "test.parquet", "feature_vocab.json"]
PREFERRED_DATASET = os.environ.get("KAGGLE_PROCESSED_DATASET", "").strip()


def _has_expected_files(base_dir):
    return all((Path(base_dir) / fname).exists() for fname in EXPECTED_FILES)


def _find_processed_dir():
    if PREFERRED_DATASET:
        candidate = Path(KAGGLE_INPUT) / PREFERRED_DATASET
        if _has_expected_files(candidate):
            return candidate
        raise FileNotFoundError(
            f"KAGGLE_PROCESSED_DATASET={PREFERRED_DATASET!r} 不存在或缺少预处理文件"
        )

    kaggle_input = Path(KAGGLE_INPUT)
    direct_matches = [p for p in kaggle_input.iterdir() if p.is_dir() and _has_expected_files(p)]
    if direct_matches:
        return sorted(direct_matches)[0]

    recursive_matches = []
    for vocab_path in kaggle_input.rglob("feature_vocab.json"):
        candidate = vocab_path.parent
        if _has_expected_files(candidate):
            recursive_matches.append(candidate)
    if recursive_matches:
        return sorted(recursive_matches)[0]
    return None


processed_root = _find_processed_dir()
if processed_root is None:
    print("\n⚠️  Missing files! Available in Kaggle input:")
    for path in sorted(Path(KAGGLE_INPUT).rglob("*")):
        if path.is_file():
            print(f"    {path}")
    raise FileNotFoundError("Missing preprocessed data files. Did you add the processed dataset?")

PROCESSED_DIR = str(processed_root)
print(f"  Using processed dataset: {PROCESSED_DIR}")
for fname in EXPECTED_FILES:
    src = processed_root / fname
    if src.is_dir():
        part_count = sum(1 for _ in src.rglob("*.parquet"))
        print(f"  ✓ {fname} dataset found ({part_count} parquet parts)")
    else:
        size_mb = src.stat().st_size / 1e6
        print(f"  ✓ {fname} found ({size_mb:.1f} MB)")

# ─── Step 2: Check dependencies ───
print("\n" + "=" * 60)
print("Step 2: Checking dependencies ...")
print("=" * 60)

import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

try:
    import yaml, sklearn, tqdm as _tqdm
    print("  All dependencies OK")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "pyyaml", "scikit-learn", "tqdm"], check=True)

# ─── Step 3: Train ───
print("\n" + "=" * 60)
print("Step 3: Training DIN + DCN V2 ...")
print("=" * 60)

import json
import yaml
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from src.data.dataset import create_dataloaders, get_feature_config
from src.models.baselines.din_dcn import DIN_DCN
from src.training.trainer import Trainer

# Seed
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# Config
with open("configs/baseline.yaml") as f:
    config = yaml.safe_load(f)

# Kaggle overrides
config["data"]["num_workers"] = 2
config["data"]["batch_size"] = 2048
config["training"]["epochs"] = 15
config["training"]["early_stopping_patience"] = 4

data_cfg = config["data"]
model_cfg = config["model"]
train_cfg = config["training"]

# Vocab
VOCAB_PATH = os.path.join(PROCESSED_DIR, "feature_vocab.json")
with open(VOCAB_PATH) as f:
    feature_vocab = json.load(f)

print(f"  Vocab features: {len(feature_vocab)}")
print("  Training will read parquet directly from /kaggle/input without copying.")

# DataLoaders
max_seq_len = data_cfg.get("max_seq_len", 50)
loaders = create_dataloaders(
    PROCESSED_DIR, feature_vocab,
    batch_size=data_cfg["batch_size"],
    max_seq_len=max_seq_len,
    num_workers=data_cfg["num_workers"],
)

# Model
feature_config = get_feature_config(feature_vocab, emb_dim=model_cfg["embedding_dim"], max_seq_len=max_seq_len)
model_cfg["din_pairs"] = [tuple(p) for p in model_cfg.get("din_pairs", [])]
model = DIN_DCN(feature_config, model_cfg)
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
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

# Save model
if os.path.exists("checkpoints/best_model.pt"):
    shutil.copy("checkpoints/best_model.pt", "/kaggle/working/best_model.pt")
    print("  Model saved to /kaggle/working/best_model.pt")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
