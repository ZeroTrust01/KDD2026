"""
TAAC2026 / KDD Cup 2026 — Sample Data Explorer
================================================
Reads data/sample_data.parquet (1000 samples) and prints a comprehensive overview
of the dataset structure, feature distributions, and sequence statistics.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict

DATA_PATH = "data/sample_data.parquet"

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def parse_features(feat_list):
    """Parse a list of feature dicts, return {feature_id: value}."""
    result = {}
    if feat_list is None:
        return result
    for f in feat_list:
        fid = f["feature_id"]
        ftype = f["feature_value_type"]
        if ftype == "int_value":
            result[fid] = f.get("int_value")
        elif ftype == "float_value":
            result[fid] = f.get("float_value")
        elif ftype == "int_array":
            result[fid] = f.get("int_array")
        elif ftype == "float_array":
            result[fid] = f.get("float_array")
        elif ftype == "int_array_and_float_array":
            result[fid] = {
                "int_array": f.get("int_array"),
                "float_array": f.get("float_array"),
            }
        else:
            result[fid] = None
    return result


# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
sep("1. Basic Dataset Info")
df = pd.read_parquet(DATA_PATH)
print(f"Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns        : {list(df.columns)}")
print(f"Column dtypes  :")
for col in df.columns:
    print(f"  {col:20s} → {df[col].dtype}")
print(f"\nMemory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ──────────────────────────────────────────────
# 2. ID statistics
# ──────────────────────────────────────────────
sep("2. ID & Timestamp Statistics")
print(f"Unique user_id : {df['user_id'].nunique()}")
print(f"Unique item_id : {df['item_id'].nunique()}")

from datetime import datetime
ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
print(f"Timestamp range: {ts_min} ~ {ts_max}")
print(f"  → {datetime.fromtimestamp(ts_min)} ~ {datetime.fromtimestamp(ts_max)}")
print(f"  → Span: {(ts_max - ts_min) / 3600:.1f} hours")

# ──────────────────────────────────────────────
# 3. Label analysis
# ──────────────────────────────────────────────
sep("3. Label Analysis")
action_type_counter = Counter()
action_count_per_sample = []
for labels in df["label"]:
    if isinstance(labels, (list, np.ndarray)):
        action_count_per_sample.append(len(labels))
        for lbl in labels:
            action_type_counter[lbl["action_type"]] += 1
    else:
        action_count_per_sample.append(0)

print(f"Actions per sample — min: {min(action_count_per_sample)}, "
      f"max: {max(action_count_per_sample)}, "
      f"mean: {np.mean(action_count_per_sample):.2f}")
print(f"\nAction type distribution:")
for atype, cnt in sorted(action_type_counter.items()):
    print(f"  action_type={atype} : {cnt} ({cnt/sum(action_type_counter.values())*100:.1f}%)")

# ──────────────────────────────────────────────
# 4. Item Features
# ──────────────────────────────────────────────
sep("4. Item Features (item_feature)")
item_feat_ids = set()
item_feat_types = defaultdict(set)
item_feat_counts = []

for feats in df["item_feature"]:
    if feats is None:
        item_feat_counts.append(0)
        continue
    item_feat_counts.append(len(feats))
    for f in feats:
        fid = f["feature_id"]
        item_feat_ids.add(fid)
        item_feat_types[fid].add(f["feature_value_type"])

print(f"Features per sample — min: {min(item_feat_counts)}, "
      f"max: {max(item_feat_counts)}, unique count: {len(item_feat_ids)}")
print(f"\nItem feature details:")
print(f"  {'feature_id':>12s}  {'value_type':>30s}")
print(f"  {'─'*12}  {'─'*30}")
for fid in sorted(item_feat_ids):
    types = ", ".join(sorted(item_feat_types[fid]))
    print(f"  {fid:>12d}  {types:>30s}")

# Show sample values for each item feature
print(f"\nSample values (first record):")
sample_item = parse_features(df["item_feature"].iloc[0])
for fid in sorted(sample_item.keys()):
    val = sample_item[fid]
    val_str = str(val)
    if len(val_str) > 80:
        val_str = val_str[:80] + "..."
    print(f"  feature_id={fid:>3d} : {val_str}")

# ──────────────────────────────────────────────
# 5. User Features
# ──────────────────────────────────────────────
sep("5. User Features (user_feature)")
user_feat_ids = set()
user_feat_types = defaultdict(set)
user_feat_counts = []

for feats in df["user_feature"]:
    if feats is None:
        user_feat_counts.append(0)
        continue
    user_feat_counts.append(len(feats))
    for f in feats:
        fid = f["feature_id"]
        user_feat_ids.add(fid)
        user_feat_types[fid].add(f["feature_value_type"])

print(f"Features per sample — min: {min(user_feat_counts)}, "
      f"max: {max(user_feat_counts)}, unique count: {len(user_feat_ids)}")
print(f"\nUser feature details:")
print(f"  {'feature_id':>12s}  {'value_type':>30s}")
print(f"  {'─'*12}  {'─'*30}")
for fid in sorted(user_feat_ids):
    types = ", ".join(sorted(user_feat_types[fid]))
    print(f"  {fid:>12d}  {types:>30s}")

# Show sample values for each user feature
print(f"\nSample values (first record):")
sample_user = parse_features(df["user_feature"].iloc[0])
for fid in sorted(sample_user.keys()):
    val = sample_user[fid]
    val_str = str(val)
    if len(val_str) > 80:
        val_str = val_str[:80] + "..."
    print(f"  feature_id={fid:>3d} : {val_str}")

# ──────────────────────────────────────────────
# 6. Sequence Features
# ──────────────────────────────────────────────
sep("6. Sequence Features (seq_feature)")
seq_keys_counter = Counter()
seq_sub_feat_info = defaultdict(lambda: {"feature_ids": set(), "lengths": []})

for seq in df["seq_feature"]:
    if seq is None:
        continue
    for key in seq.keys():
        seq_keys_counter[key] += 1
        sub_feats = seq[key]
        if sub_feats is not None and hasattr(sub_feats, "__len__"):
            for sf in sub_feats:
                fid = sf["feature_id"]
                seq_sub_feat_info[key]["feature_ids"].add(fid)
                arr = sf.get("int_array")
                if arr is not None and hasattr(arr, "__len__"):
                    seq_sub_feat_info[key]["lengths"].append(len(arr))

print(f"Seq feature keys and coverage:")
for key, cnt in sorted(seq_keys_counter.items()):
    print(f"  {key}: present in {cnt}/{len(df)} samples")

print(f"\nSeq feature sub-structure details:")
for key in sorted(seq_sub_feat_info.keys()):
    info = seq_sub_feat_info[key]
    fids = sorted(info["feature_ids"])
    lengths = info["lengths"]
    print(f"\n  [{key}]")
    print(f"    feature_ids  : {fids}")
    print(f"    num features : {len(fids)}")
    if lengths:
        print(f"    seq lengths  : min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.1f}, median={int(np.median(lengths))}")

# Show one complete sequence sample
print(f"\nSample seq_feature (first record):")
sample_seq = df["seq_feature"].iloc[0]
for key in sample_seq.keys():
    sub = sample_seq[key]
    print(f"\n  [{key}] — {len(sub)} sub-features")
    for sf in sub:
        fid = sf["feature_id"]
        arr = sf.get("int_array")
        if arr is not None and hasattr(arr, "__len__"):
            arr_str = str(arr[:10].tolist()) if hasattr(arr, "tolist") else str(list(arr)[:10])
            print(f"    feature_id={fid:>3d} : len={len(arr):>4d}, first_10={arr_str}")
        else:
            print(f"    feature_id={fid:>3d} : {str(sf)[:80]}")

# ──────────────────────────────────────────────
# 7. Summary
# ──────────────────────────────────────────────
sep("7. Data Summary for Competition")
print(f"""
┌─────────────────────────────────────────────────────┐
│  TAAC2026 / KDD Cup 2026 — Sample Data Overview     │
├─────────────────────────────────────────────────────┤
│  Total samples       : {df.shape[0]:>6d}                       │
│  Unique users        : {df['user_id'].nunique():>6d}                       │
│  Unique items        : {df['item_id'].nunique():>6d}                       │
│                                                     │
│  Item features       : {len(item_feat_ids):>6d} feature IDs              │
│  User features       : {len(user_feat_ids):>6d} feature IDs              │
│  Seq feature keys    : {len(seq_keys_counter):>6d} (action/content/item)  │
│                                                     │
│  Feature value types :                              │
│    - int_value (sparse categorical)                 │
│    - int_array (multi-value / sequence)             │
│    - float_array (dense embedding)                  │
│    - int_array_and_float_array (hybrid)             │
│                                                     │
│  Label               : action_type (CVR prediction) │
│  Evaluation metric   : AUC of ROC                   │
└─────────────────────────────────────────────────────┘
""")
