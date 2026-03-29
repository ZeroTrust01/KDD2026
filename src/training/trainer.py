"""
Trainer - Training loop with early stopping.
"""
import os
import time
import logging
import torch
from torch import nn
from tqdm import tqdm
from src.training.metrics import compute_auc, compute_logloss

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop with validation, early stopping, and checkpoint saving."""

    def __init__(self, model, device="cpu", config=None):
        self.model = model.to(device)
        self.device = device
        self.config = config or {}

        # Optimizer
        lr = self.config.get("learning_rate", 1e-3)
        wd = self.config.get("weight_decay", 1e-5)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd
        )

        # Loss
        self.criterion = nn.BCELoss()

        # LR scheduler
        self.scheduler = None
        if self.config.get("lr_scheduler") == "cosine":
            epochs = self.config.get("epochs", 10)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )

        # Checkpoint
        self.checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, train_loader):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("label").float()

            # Forward
            output = self.model(batch)
            y_pred = output["y_pred"].squeeze(-1)
            loss = self.criterion(y_pred, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            grad_clip = self.config.get("gradient_clip", 0)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def evaluate(self, data_loader):
        """Evaluate on validation/test set."""
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("label").float()
            output = self.model(batch)
            y_pred = output["y_pred"].squeeze(-1)

            all_preds.extend(y_pred.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        auc = compute_auc(all_labels, all_preds)
        logloss = compute_logloss(all_labels, all_preds)
        return {"auc": auc, "logloss": logloss}

    def fit(self, train_loader, valid_loader, epochs=10, patience=3):
        """Full training with early stopping."""
        best_auc = 0
        patience_counter = 0
        best_epoch = 0

        logger.info(f"Start training for {epochs} epochs, patience={patience}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            valid_metrics = self.evaluate(valid_loader)

            # LR scheduler
            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"valid_auc={valid_metrics['auc']:.4f} | "
                f"valid_logloss={valid_metrics['logloss']:.4f} | "
                f"lr={lr:.2e} | "
                f"time={elapsed:.1f}s"
            )

            # Early stopping
            if valid_metrics["auc"] > best_auc:
                best_auc = valid_metrics["auc"]
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint("best_model.pt")
                logger.info(f"  ✓ New best AUC={best_auc:.4f}, model saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}. "
                                f"Best AUC={best_auc:.4f} at epoch {best_epoch}")
                    break

        # Load best model
        self.load_checkpoint("best_model.pt")
        return {"best_auc": best_auc, "best_epoch": best_epoch}

    def save_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded checkpoint from {path}")
