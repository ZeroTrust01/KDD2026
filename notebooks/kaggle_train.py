"""
KDD Cup 2026 - DIN + DCN V2 Baseline Training on Kaggle
========================================================
Kaggle Notebook 训练脚本

使用方法:
1. 在 Kaggle 上创建 Dataset，上传以下文件:
   - data/TaobaoAd/raw_sample.csv.tar.gz
   - data/TaobaoAd/ad_feature.csv.tar.gz
   - data/TaobaoAd/user_profile.csv.tar.gz
   - data/TaobaoAd/behavior_log.csv.tar.gz

2. 创建 Kaggle Notebook，选择 GPU T4 x2
3. 添加上面创建的 Dataset
4. 复制本脚本到 Notebook cell 中运行

注意: Kaggle 有 30h/周 GPU 限额
"""

import os
import sys
import subprocess

# ─── Step 0: Clone code from GitHub ───
print("=" * 60)
print("Step 0: Cloning code from GitHub ...")
print("=" * 60)

if not os.path.exists("/kaggle/working/KDD2026"):
    subprocess.run(["git", "clone", "https://github.com/ZeroTrust01/KDD2026.git",
                    "/kaggle/working/KDD2026"], check=True)

os.chdir("/kaggle/working/KDD2026")
sys.path.insert(0, "/kaggle/working/KDD2026")

# ─── Step 1: Locate and extract data ───
print("\n" + "=" * 60)
print("Step 1: Locating and extracting data ...")
print("=" * 60)

# Kaggle datasets are mounted at /kaggle/input/<dataset-name>/
# Find the data files
import glob

KAGGLE_INPUT = "/kaggle/input"
DATA_DIR = "data/TaobaoAd"
os.makedirs(DATA_DIR, exist_ok=True)

# Search for tar.gz files in all kaggle input directories
tar_files = glob.glob(f"{KAGGLE_INPUT}/**/*.tar.gz", recursive=True)
print(f"Found {len(tar_files)} tar.gz files:")
for f in tar_files:
    print(f"  {f} ({os.path.getsize(f) / 1e6:.1f} MB)")

# Extract to data directory
import tarfile

for tar_path in tar_files:
    fname = os.path.basename(tar_path)
    csv_name = fname.replace(".tar.gz", "")
    target_path = os.path.join(DATA_DIR, csv_name)

    if os.path.exists(target_path):
        print(f"  ✓ {csv_name} already extracted")
        continue

    print(f"  Extracting {fname} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    print(f"  ✓ {csv_name} extracted")

# Also check for uncompressed CSV files
csv_files = glob.glob(f"{KAGGLE_INPUT}/**/*.csv", recursive=True)
for csv_path in csv_files:
    fname = os.path.basename(csv_path)
    target_path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(target_path):
        print(f"  Linking {fname} ...")
        os.symlink(csv_path, target_path)

# Verify
print("\nData files:")
for f in sorted(os.listdir(DATA_DIR)):
    fpath = os.path.join(DATA_DIR, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        print(f"  {f}: {size / 1e6:.1f} MB" if size < 1e9 else f"  {f}: {size / 1e9:.1f} GB")


# ─── Step 2: Install dependencies ───
print("\n" + "=" * 60)
print("Step 2: Checking dependencies ...")
print("=" * 60)

try:
    import torch
    import sklearn
    import yaml
    import tqdm
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("  All dependencies OK")
except ImportError as e:
    print(f"  Missing: {e}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch", "scikit-learn", "pyyaml", "tqdm"], check=True)


# ─── Step 3: Preprocess data ───
print("\n" + "=" * 60)
print("Step 3: Preprocessing data ...")
print("=" * 60)

PROCESSED_DIR = "data/TaobaoAd/processed"
VOCAB_PATH = os.path.join(PROCESSED_DIR, "feature_vocab.json")

# Configuration - adjust these for your training run
SAMPLE_USERS = 0        # 0 = ALL users (for full training)
MAX_BEHAVIOR_ROWS = 0   # 0 = ALL rows (for full training)
# For quick test, set:
# SAMPLE_USERS = 10000
# MAX_BEHAVIOR_ROWS = 10000000

if os.path.exists(VOCAB_PATH):
    print("  Preprocessed data already exists, skipping.")
else:
    cmd = [
        sys.executable, "scripts/preprocess_taobao.py",
        "--sample_users", str(SAMPLE_USERS),
        "--max_behavior_rows", str(MAX_BEHAVIOR_ROWS),
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# Show preprocessed data stats
import json
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
print(f"\n  Vocab stats:")
for feat, v in vocab.items():
    print(f"    {feat}: {len(v)} tokens")


# ─── Step 4: Train ───
print("\n" + "=" * 60)
print("Step 4: Training DIN + DCN V2 ...")
print("=" * 60)

import torch
import yaml
import json
import numpy as np
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.data.dataset import create_dataloaders, get_feature_config
from src.models.baselines.din_dcn import DIN_DCN
from src.training.trainer import Trainer

# Set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# Load config
with open("configs/baseline.yaml") as f:
    config = yaml.safe_load(f)

# Override config for Kaggle (stronger training)
config["data"]["num_workers"] = 2  # Kaggle has limited CPU cores
config["data"]["batch_size"] = 2048  # Larger batch with GPU
config["training"]["epochs"] = 15
config["training"]["early_stopping_patience"] = 4

data_cfg = config["data"]
model_cfg = config["model"]
train_cfg = config["training"]

# Load vocab
with open(VOCAB_PATH) as f:
    feature_vocab = json.load(f)

# DataLoaders
max_seq_len = data_cfg.get("max_seq_len", 50)
loaders = create_dataloaders(
    data_cfg["data_dir"], feature_vocab,
    batch_size=data_cfg["batch_size"],
    max_seq_len=max_seq_len,
    num_workers=data_cfg["num_workers"],
)

# Build model
feature_config = get_feature_config(feature_vocab, emb_dim=model_cfg["embedding_dim"], max_seq_len=max_seq_len)
model_cfg["din_pairs"] = [tuple(p) for p in model_cfg.get("din_pairs", [])]
model = DIN_DCN(feature_config, model_cfg)
num_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {num_params:,}")

# Train
trainer = Trainer(model, device=device, config=train_cfg)
result = trainer.fit(
    train_loader=loaders["train"],
    valid_loader=loaders.get("valid"),
    epochs=train_cfg["epochs"],
    patience=train_cfg["early_stopping_patience"],
)

# ─── Step 5: Test ───
print("\n" + "=" * 60)
print("Step 5: Testing ...")
print("=" * 60)

if "test" in loaders:
    test_metrics = trainer.evaluate(loaders["test"])
    print(f"  Test AUC:     {test_metrics['auc']:.4f}")
    print(f"  Test LogLoss: {test_metrics['logloss']:.4f}")

print("\n" + "=" * 60)
print(f"Training complete! Best valid AUC = {result['best_auc']:.4f} at epoch {result['best_epoch']}")
print("=" * 60)

# Save model to Kaggle output
import shutil
if os.path.exists("checkpoints/best_model.pt"):
    shutil.copy("checkpoints/best_model.pt", "/kaggle/working/best_model.pt")
    print("  Model saved to /kaggle/working/best_model.pt")
