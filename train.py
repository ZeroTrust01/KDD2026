"""
KDD Cup 2026 - Training Entry Point
Usage:
    python train.py --config configs/baseline.yaml --gpu 0
"""
import os
import sys
import json
import argparse
import logging
import random

import numpy as np
import torch
import yaml

from src.data.dataset import TaobaoAdDataset, create_dataloaders, get_feature_config
from src.models.baselines.din_dcn import DIN_DCN
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="KDD Cup 2026 Training")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ─── Load config ───
    logger.info(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # ─── Device ───
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # ─── Check preprocessed data ───
    data_dir = data_cfg["data_dir"]
    vocab_path = os.path.join(data_dir, "feature_vocab.json")

    if not os.path.exists(vocab_path):
        logger.error(
            f"Preprocessed data not found at {data_dir}.\n"
            f"Run: python scripts/preprocess_taobao.py --sample_users 100000"
        )
        sys.exit(1)

    # ─── Load vocab ───
    logger.info(f"Loading vocab from {vocab_path}")
    with open(vocab_path, "r") as f:
        feature_vocab = json.load(f)

    # ─── Create DataLoaders ───
    max_seq_len = data_cfg.get("max_seq_len", 50)
    batch_size = data_cfg.get("batch_size", 1024)
    num_workers = data_cfg.get("num_workers", 4)

    loaders = create_dataloaders(
        data_dir, feature_vocab,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_workers=num_workers,
    )

    if "train" not in loaders:
        logger.error("Training data not found!")
        sys.exit(1)

    # ─── Build model ───
    feature_config = get_feature_config(
        feature_vocab,
        emb_dim=model_cfg.get("embedding_dim", 16),
        max_seq_len=max_seq_len,
    )

    # Convert din_pairs from list of lists to list of tuples
    din_pairs_raw = model_cfg.get("din_pairs", [])
    model_cfg["din_pairs"] = [tuple(p) for p in din_pairs_raw]

    model = DIN_DCN(feature_config, model_cfg)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_cfg['name']}")
    logger.info(f"Parameters: {num_params:,}")

    # ─── Train ───
    trainer = Trainer(model, device=device, config=train_cfg)
    result = trainer.fit(
        train_loader=loaders["train"],
        valid_loader=loaders.get("valid"),
        epochs=train_cfg.get("epochs", 10),
        patience=train_cfg.get("early_stopping_patience", 3),
    )

    # ─── Test ───
    if "test" in loaders:
        logger.info("=" * 60)
        logger.info("Evaluating on test set ...")
        test_metrics = trainer.evaluate(loaders["test"])
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Test LogLoss: {test_metrics['logloss']:.4f}")

    logger.info("=" * 60)
    logger.info(f"Training complete. Best valid AUC={result['best_auc']:.4f} "
                f"at epoch {result['best_epoch']}")


if __name__ == "__main__":
    main()
