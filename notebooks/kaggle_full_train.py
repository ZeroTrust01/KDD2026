"""
KDD Cup 2026 - Full Data Preprocessing on Kaggle
=================================================
在 Kaggle 上对全量淘宝广告数据进行预处理并导出 processed dataset。

前置步骤:
1. 创建 Kaggle Notebook, 选 GPU T4 x2, Internet ON
2. Add data: 添加 "taobao-ad-display-click" (23GB 原始数据)
3. 运行本脚本, 处理完成后点 "Save Version" 保存输出
4. 用 /kaggle/working/processed 创建新 Dataset
5. 后续训练请使用 notebooks/kaggle_train.py

预计耗时: 预处理 30-60 分钟
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
           "--backend", "duckdb",
           "--duckdb_memory_limit", "24GB",
           "--duckdb_num_shards", "12"]
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
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        elif os.path.exists(dst):
            os.remove(dst)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            total_size = 0
            for root, _, files in os.walk(src):
                for file_name in files:
                    total_size += os.path.getsize(os.path.join(root, file_name))
            print(f"  Copied {fname} dataset to output ({total_size / 1e6:.1f} MB)")
        else:
            shutil.copy(src, dst)
            size = os.path.getsize(src)
            print(f"  Copied {fname} to output ({size / 1e6:.1f} MB)")


print("\n" + "=" * 60)
print("Done! 处理好的数据位于 /kaggle/working/processed/")
print("下一步: 用该输出创建 Kaggle Dataset, 然后运行 notebooks/kaggle_train.py")
print("=" * 60)
