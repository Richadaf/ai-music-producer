"""
Training Loop
==============
Trains the BeatTransformer on tokenised beat data.
Features: gradient clipping, warmup scheduler, early stopping,
checkpoint saving, and validation loss tracking.
"""

import logging
import math
import os
import pickle
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import create_data_splits
from .model import BeatTransformer
from .tokenizer import Vocabulary

logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """Linear warmup → cosine decay learning rate scheduler."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        for base_lr, pg in zip(self.base_lrs, self.optimizer.param_groups):
            if self.step_count < self.warmup_steps:
                lr = base_lr * self.step_count / max(1, self.warmup_steps)
            else:
                progress = (self.step_count - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            pg["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def train(
    data_path: str | Path,
    vocab_path: str | Path,
    model_dir: str | Path,
    # Model config
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 1024,
    dropout: float = 0.1,
    max_seq_len: int = 2048,
    # Training config
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    max_epochs: int = 200,
    patience: int = 20,
    grad_clip: float = 1.0,
    save_every_n_epochs: int = 10,
    eval_split: float = 0.1,
    seed: int = 42,
    # Resume
    checkpoint_path: str | Path | None = None,
) -> Path:
    """
    Full training loop. Returns path to best checkpoint.
    """
    data_path = Path(data_path)
    vocab_path = Path(vocab_path)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # Load vocab
    vocab = Vocabulary.load(vocab_path)
    pad_id = vocab.encode("<PAD>")
    logger.info(f"Vocabulary: {len(vocab)} tokens")

    # Data
    train_dataset, val_dataset = create_data_splits(
        data_path, max_seq_len, mode="both", eval_split=eval_split, seed=seed, pad_token_id=pad_id
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Train: {len(train_dataset)} sequences, Val: {len(val_dataset)} sequences")

    # Model
    model = BeatTransformer(
        vocab_size=len(vocab),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        pad_token_id=pad_id,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    # Scheduler
    total_steps = max_epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        scheduler.step_count = ckpt.get("step_count", 0)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Training loop
    best_ckpt_path = model_dir / "best.pt"

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output = model(input_ids, attention_mask, targets=target_ids)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                output = model(input_ids, attention_mask, targets=target_ids)
                val_loss += output["loss"].item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"lr={scheduler.get_lr():.2e} | "
            f"time={elapsed:.1f}s"
        )

        # --- Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "step_count": scheduler.step_count,
                    "config": {
                        "vocab_size": len(vocab),
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "n_layers": n_layers,
                        "d_ff": d_ff,
                        "max_seq_len": max_seq_len,
                        "dropout": dropout,
                    },
                },
                best_ckpt_path,
            )
            logger.info(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Periodic save
        if (epoch + 1) % save_every_n_epochs == 0:
            periodic_path = model_dir / f"epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "step_count": scheduler.step_count,
                },
                periodic_path,
            )

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    # Save latest as well
    latest_path = model_dir / "latest.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "step_count": scheduler.step_count,
            "config": {
                "vocab_size": len(vocab),
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "d_ff": d_ff,
                "max_seq_len": max_seq_len,
                "dropout": dropout,
            },
        },
        latest_path,
    )

    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    logger.info(f"Best checkpoint: {best_ckpt_path}")
    return best_ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train the beat generation model")
    parser.add_argument("-c", "--config", default="configs/default.yaml", help="Config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed = Path(cfg["paths"]["processed_data"])
    train(
        data_path=processed / "tokenized_data.pkl",
        vocab_path=processed / "vocab.pkl",
        model_dir=cfg["paths"]["model_dir"],
        **cfg["model"],
        **cfg["training"],
        checkpoint_path=cfg["paths"].get("checkpoint"),
    )
