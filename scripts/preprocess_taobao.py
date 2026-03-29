"""
Taobao Ad Dataset Preprocessing Script (DuckDB version).

Uses DuckDB for behavior_log processing — disk-spill-capable,
handles 700M+ rows within Kaggle's 13GB RAM.

Usage:
    # 快速跑通 (1000 用户, 100万行行为日志)
    python3 scripts/preprocess_taobao.py --sample_users 1000 --max_behavior_rows 1000000

    # 全量预处理 (Kaggle 上跑)
    python3 scripts/preprocess_taobao.py --sample_users 0
"""
import os
import sys
import json
import logging
import argparse
from collections import Counter

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────── Configuration ───────────────────
RAW_DIR = "data/TaobaoAd"
OUT_DIR = "data/TaobaoAd/processed"
MAX_SEQ_LEN = 50
MIN_FREQ = 2

CATEGORICAL_FEATURES = [
    "userid", "adgroup_id", "pid",
    "cms_segid", "cms_group_id", "final_gender_code",
    "age_level", "pvalue_level", "shopping_level",
    "occupation", "new_user_class_level",
    "cate_id", "campaign_id", "customer", "brand",
]
NUMERIC_FEATURES = ["price"]
SEQUENCE_FEATURES = ["cate_seq", "brand_seq"]


def load_raw_tables(sample_users=0):
    """Load the 3 small tables, optionally sample users."""
    logger.info("Loading raw_sample.csv ...")
    raw_sample = pd.read_csv(os.path.join(RAW_DIR, "raw_sample.csv"))
    raw_sample.rename(columns={"user": "userid"}, inplace=True)
    logger.info(f"  raw_sample: {len(raw_sample):,} rows")

    if sample_users > 0:
        all_users = raw_sample["userid"].unique()
        np.random.seed(42)
        sampled = np.random.choice(
            all_users, size=min(sample_users, len(all_users)), replace=False
        )
        raw_sample = raw_sample[raw_sample["userid"].isin(sampled)].reset_index(drop=True)
        logger.info(f"  Sampled to {len(sampled):,} users, {len(raw_sample):,} rows")

    logger.info("Loading ad_feature.csv ...")
    ad_feature = pd.read_csv(os.path.join(RAW_DIR, "ad_feature.csv"))
    logger.info(f"  ad_feature: {len(ad_feature):,} rows")

    logger.info("Loading user_profile.csv ...")
    user_profile = pd.read_csv(os.path.join(RAW_DIR, "user_profile.csv"))
    logger.info(f"  user_profile: {len(user_profile):,} rows")

    return raw_sample, ad_feature, user_profile


def build_sequences_duckdb(user_ids, max_rows=0, max_len=50):
    """
    Use DuckDB to build behavior sequences. Disk-spill capable,
    handles 700M+ rows without OOM.

    Returns: pd.DataFrame with columns [userid, cate_seq, brand_seq]
    """
    import duckdb

    behavior_path = os.path.join(RAW_DIR, "behavior_log.csv")
    logger.info(f"Building behavior sequences with DuckDB (max_len={max_len}) ...")

    con = duckdb.connect()

    # Configure DuckDB for memory-constrained environment
    # Use temp directory for spilling
    tmp_dir = os.path.join(OUT_DIR, "_duckdb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA threads=4")

    # Register user_ids as a table for filtering
    user_df = pd.DataFrame({"userid": list(user_ids)})
    con.register("user_filter", user_df)

    # Build the query
    row_limit = f"LIMIT {max_rows}" if max_rows > 0 else ""

    query = f"""
    WITH behavior AS (
        SELECT
            "user" AS userid,
            time_stamp,
            CAST(cate AS VARCHAR) AS cate,
            CAST(brand AS VARCHAR) AS brand
        FROM read_csv_auto('{behavior_path}')
        {row_limit}
    ),
    matched AS (
        SELECT
            b.userid,
            b.time_stamp,
            b.cate,
            b.brand,
            ROW_NUMBER() OVER (
                PARTITION BY b.userid
                ORDER BY b.time_stamp DESC
            ) AS rn
        FROM behavior b
        INNER JOIN user_filter u ON b.userid = u.userid
    ),
    recent AS (
        SELECT userid, time_stamp, cate, brand
        FROM matched
        WHERE rn <= {max_len}
    )
    SELECT
        userid,
        string_agg(cate, '^' ORDER BY time_stamp) AS cate_seq,
        string_agg(brand, '^' ORDER BY time_stamp) AS brand_seq
    FROM recent
    GROUP BY userid
    """

    logger.info("  Executing DuckDB query (this may take 10-30 min for full data) ...")
    seq_df = con.execute(query).fetchdf()

    con.close()

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"  Built sequences for {len(seq_df):,} users")
    return seq_df


def build_sequences_pandas(user_set, max_rows=0, max_len=50, chunksize=2_000_000):
    """
    Fallback: pandas chunked processing for small datasets.
    Only safe when user_set is small (< 100K users).
    """
    behavior_path = os.path.join(RAW_DIR, "behavior_log.csv")
    logger.info(f"Loading behavior_log.csv with pandas (max_rows={max_rows or 'ALL'}) ...")

    chunks = []
    total_read = 0

    reader = pd.read_csv(behavior_path, chunksize=chunksize)
    for i, chunk in enumerate(reader):
        total_read += len(chunk)
        filtered = chunk[chunk["user"].isin(user_set)][["user", "time_stamp", "cate", "brand"]]
        if len(filtered) > 0:
            chunks.append(filtered)

        if (i + 1) % 5 == 0:
            kept = sum(len(c) for c in chunks)
            logger.info(f"  ... read {total_read:,} rows, kept {kept:,}")

        if max_rows > 0 and total_read >= max_rows:
            break

    if not chunks:
        return pd.DataFrame(columns=["userid", "cate_seq", "brand_seq"])

    behavior_df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Done: {total_read:,} rows read, {len(behavior_df):,} kept")
    del chunks

    # Build sequences
    logger.info(f"Building sequences (max_len={max_len}) ...")
    behavior_df = behavior_df.sort_values(["user", "time_stamp"])
    behavior_df["cate"] = behavior_df["cate"].astype(str)
    behavior_df["brand"] = behavior_df["brand"].astype(str)

    def agg_seq(group):
        recent = group.tail(max_len)
        return pd.Series({
            "cate_seq": "^".join(recent["cate"].values),
            "brand_seq": "^".join(recent["brand"].values),
        })

    seq_df = behavior_df.groupby("user").apply(agg_seq, include_groups=False).reset_index()
    seq_df.rename(columns={"user": "userid"}, inplace=True)
    logger.info(f"  Built sequences for {len(seq_df):,} users")
    return seq_df


def build_vocab(series, min_freq=2, name="unknown"):
    """Build vocabulary from a pandas Series."""
    counter = Counter()
    for val in series.dropna():
        val_str = str(val)
        if "^" in val_str:
            for token in val_str.split("^"):
                if token:
                    counter[token] += 1
        else:
            counter[val_str] += 1

    vocab = {"__PAD__": 0, "__OOV__": 1}
    idx = 2
    for token, freq in counter.most_common():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1

    logger.info(f"  Vocab [{name}]: {len(vocab)} tokens (min_freq={min_freq})")
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_users", type=int, default=1000)
    parser.add_argument("--max_behavior_rows", type=int, default=0,
                        help="Max rows to read from behavior_log. 0=ALL")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--min_freq", type=int, default=MIN_FREQ)
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "duckdb", "pandas"],
                        help="Backend for behavior_log processing")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ─── Step 1: Load tables ───
    raw_sample, ad_feature, user_profile = load_raw_tables(args.sample_users)

    # ─── Step 2: JOIN tables ───
    logger.info("Joining tables ...")
    df = raw_sample.merge(ad_feature, on="adgroup_id", how="left")
    df = df.merge(user_profile, on="userid", how="left")
    logger.info(f"  Joined: {len(df):,} rows, {len(df.columns)} cols")

    # ─── Step 3: Build behavior sequences ───
    user_set = set(df["userid"].unique())
    num_users = len(user_set)
    logger.info(f"  Users to process: {num_users:,}")

    # Choose backend
    backend = args.backend
    if backend == "auto":
        try:
            import duckdb
            backend = "duckdb"
            logger.info("  Backend: DuckDB (auto-detected)")
        except ImportError:
            backend = "pandas"
            logger.info("  Backend: pandas (DuckDB not available)")

    if backend == "duckdb":
        seq_df = build_sequences_duckdb(
            user_set, max_rows=args.max_behavior_rows, max_len=args.max_seq_len
        )
    else:
        if num_users > 100_000 and args.max_behavior_rows == 0:
            logger.warning("  ⚠️ pandas backend with >100K users and no row limit may OOM!")
            logger.warning("  Consider: pip install duckdb, or --sample_users 100000")
        seq_df = build_sequences_pandas(
            user_set, max_rows=args.max_behavior_rows, max_len=args.max_seq_len
        )

    if len(seq_df) > 0:
        df = df.merge(seq_df, on="userid", how="left")
        del seq_df
    else:
        df["cate_seq"] = ""
        df["brand_seq"] = ""

    df["cate_seq"] = df["cate_seq"].fillna("")
    df["brand_seq"] = df["brand_seq"].fillna("")

    # ─── Step 4: Fill missing + normalize ───
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("__MISSING__").astype(str)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(float)

    if "price" in df.columns:
        df["price"] = np.log1p(df["price"].clip(lower=0))
        mu, sigma = df["price"].mean(), df["price"].std()
        if sigma > 0:
            df["price"] = (df["price"] - mu) / sigma

    # ─── Step 5: Split by time_stamp (70/15/15) ───
    logger.info("Splitting by time_stamp ...")
    df = df.sort_values("time_stamp").reset_index(drop=True)
    n = len(df)
    t1, t2 = int(n * 0.7), int(n * 0.85)

    splits = {
        "train": df.iloc[:t1],
        "valid": df.iloc[t1:t2],
        "test": df.iloc[t2:],
    }
    for k, v in splits.items():
        logger.info(f"  {k}: {len(v):,} rows, clk_rate={v['clk'].mean():.4f}")

    # ─── Step 6: Build vocab (train only) ───
    logger.info("Building vocabs ...")
    feature_vocab = {}
    for feat in CATEGORICAL_FEATURES:
        if feat in splits["train"].columns:
            feature_vocab[feat] = build_vocab(
                splits["train"][feat], min_freq=args.min_freq, name=feat
            )
    for feat in SEQUENCE_FEATURES:
        if feat in splits["train"].columns:
            feature_vocab[feat] = build_vocab(
                splits["train"][feat], min_freq=args.min_freq, name=feat
            )

    # ─── Step 7: Save ───
    keep_cols = ["clk"] + CATEGORICAL_FEATURES + NUMERIC_FEATURES + SEQUENCE_FEATURES
    keep_cols = [c for c in keep_cols if c in df.columns]

    for name, sdf in splits.items():
        path = os.path.join(OUT_DIR, f"{name}.parquet")
        sdf[keep_cols].to_parquet(path, index=False)
        logger.info(f"  Saved {name}: {path}")

    vocab_path = os.path.join(OUT_DIR, "feature_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(feature_vocab, f, indent=2)
    logger.info(f"  Saved vocab: {vocab_path}")

    logger.info("=" * 50)
    logger.info("Preprocessing complete!")
    for feat, vocab in feature_vocab.items():
        logger.info(f"  {feat}: vocab_size={len(vocab)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
