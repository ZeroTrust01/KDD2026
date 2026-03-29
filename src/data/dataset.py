"""
TaobaoAd Dataset - PyTorch Dataset for preprocessed Taobao Ad data.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TaobaoAdDataset(Dataset):
    """PyTorch Dataset for preprocessed Taobao Ad click data."""

    # Feature column definitions
    CATEGORICAL_FEATURES = [
        "userid", "adgroup_id", "pid",
        "cms_segid", "cms_group_id", "final_gender_code",
        "age_level", "pvalue_level", "shopping_level",
        "occupation", "new_user_class_level",
        "cate_id", "campaign_id", "customer", "brand",
    ]
    NUMERIC_FEATURES = ["price"]
    SEQUENCE_FEATURES = ["cate_seq", "brand_seq"]  # behavior sequences
    LABEL_COL = "clk"

    def __init__(self, data_path, feature_vocab, max_seq_len=50):
        """
        Args:
            data_path: path to preprocessed parquet file
            feature_vocab: dict of {feature_name: {value: idx}} for encoding
            max_seq_len: max behavior sequence length
        """
        logger.info(f"Loading dataset from {data_path}")
        self.df = pd.read_parquet(data_path)
        self.feature_vocab = feature_vocab
        self.max_seq_len = max_seq_len
        logger.info(f"Loaded {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {}

        # Categorical features → vocab index
        for feat in self.CATEGORICAL_FEATURES:
            val = str(row.get(feat, ""))
            vocab = self.feature_vocab.get(feat, {})
            sample[feat] = vocab.get(val, vocab.get("__OOV__", 1))

        # Numeric features
        for feat in self.NUMERIC_FEATURES:
            val = row.get(feat, 0.0)
            sample[feat] = float(val) if pd.notna(val) else 0.0

        # Sequence features → padded index array
        for feat in self.SEQUENCE_FEATURES:
            seq_str = row.get(feat, "")
            if pd.isna(seq_str) or seq_str == "":
                seq_ids = []
            else:
                tokens = str(seq_str).split("^")
                vocab = self.feature_vocab.get(feat, {})
                oov_idx = vocab.get("__OOV__", 1)
                seq_ids = [vocab.get(t, oov_idx) for t in tokens]
            # Truncate and pad
            seq_ids = seq_ids[: self.max_seq_len]
            pad_len = self.max_seq_len - len(seq_ids)
            seq_ids = seq_ids + [0] * pad_len
            sample[feat] = seq_ids

        # Label
        sample["label"] = float(row.get(self.LABEL_COL, 0))

        return sample

    @staticmethod
    def collate_fn(batch):
        """Custom collate: convert list of dicts to dict of tensors."""
        keys = batch[0].keys()
        result = {}
        for key in keys:
            values = [item[key] for item in batch]
            if isinstance(values[0], list):
                result[key] = torch.LongTensor(values)
            elif isinstance(values[0], float):
                result[key] = torch.FloatTensor(values)
            else:
                result[key] = torch.LongTensor(values)
        return result


def get_feature_config(feature_vocab, emb_dim=16, max_seq_len=50):
    """Build feature_config dict for FeatureEmbedding from vocab."""
    config = {}
    for feat in TaobaoAdDataset.CATEGORICAL_FEATURES:
        vocab = feature_vocab.get(feat, {})
        config[feat] = {
            "type": "categorical",
            "vocab_size": max(vocab.values()) + 1 if vocab else 2,
            "embedding_dim": emb_dim,
        }
    for feat in TaobaoAdDataset.NUMERIC_FEATURES:
        config[feat] = {
            "type": "numeric",
            "embedding_dim": emb_dim,
        }
    for feat in TaobaoAdDataset.SEQUENCE_FEATURES:
        vocab = feature_vocab.get(feat, {})
        config[feat] = {
            "type": "sequence",
            "vocab_size": max(vocab.values()) + 1 if vocab else 2,
            "embedding_dim": emb_dim,
            "max_len": max_seq_len,
        }
    return config


def create_dataloaders(data_dir, feature_vocab, batch_size=1024,
                       max_seq_len=50, num_workers=4):
    """Create train/valid/test DataLoaders."""
    loaders = {}
    for split in ["train", "valid", "test"]:
        path = os.path.join(data_dir, f"{split}.parquet")
        if not os.path.exists(path):
            # Also check if it's a directory of parquet shards
            logger.warning(f"{path} not found, skipping {split}")
            continue
        ds = TaobaoAdDataset(path, feature_vocab, max_seq_len=max_seq_len)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=TaobaoAdDataset.collate_fn,
            pin_memory=True,
        )
    return loaders
