"""
KDD Cup 2026 - Training Entry Point
"""
import argparse
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="KDD Cup 2026 Training")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Loading config from {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Config: {config}")
    logger.info(f"GPU: {args.gpu}, Seed: {args.seed}")

    # TODO: Implement training pipeline
    # 1. Load & preprocess data
    # 2. Build model
    # 3. Train
    # 4. Evaluate
    logger.info("Training pipeline not yet implemented. See src/ for modules.")


if __name__ == "__main__":
    main()
