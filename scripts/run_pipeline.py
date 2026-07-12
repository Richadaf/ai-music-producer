#!/usr/bin/env python3
"""
Run Pipeline
=============
End-to-end orchestrator: ZIP/FLP → parse → tokenize → train → generate.

Usage:
    python scripts/run_pipeline.py                    # Full pipeline
    python scripts/run_pipeline.py --stage parse      # Only parse
    python scripts/run_pipeline.py --stage tokenize   # Only tokenize
    python scripts/run_pipeline.py --stage train      # Only train
    python scripts/run_pipeline.py --stage generate   # Only generate
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.flp_parser import process_raw_directory
from src.tokenizer import TokenizerConfig, process_all_projects
from src.train import train
from src.generate import generate_beats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)


def stage_parse(cfg: dict) -> None:
    """Stage 1: Parse ZIP/FLP files into structured JSON."""
    logger.info("=" * 60)
    logger.info("STAGE 1: PARSING FLP FILES")
    logger.info("=" * 60)

    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_data"]
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_data"]
    drum_keywords = cfg.get("parser", {}).get("drum_keywords", None)

    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        logger.info("Drop your ZIP/FLP files into data/raw/ and try again.")
        sys.exit(1)

    files = list(raw_dir.glob("*.zip")) + list(raw_dir.glob("*.flp"))
    if not files:
        logger.error(f"No ZIP or FLP files found in {raw_dir}")
        sys.exit(1)

    logger.info(f"Found {len(files)} file(s) in {raw_dir}")
    json_dir = processed_dir / "json"
    outputs = process_raw_directory(raw_dir, json_dir, drum_keywords)
    logger.info(f"Parsed {len(outputs)} project(s) → {json_dir}")


def stage_tokenize(cfg: dict) -> None:
    """Stage 2: Tokenize parsed JSON into training sequences."""
    logger.info("=" * 60)
    logger.info("STAGE 2: TOKENIZING")
    logger.info("=" * 60)

    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_data"]
    json_dir = processed_dir / "json"

    if not json_dir.exists():
        logger.error(f"No parsed data found at {json_dir}. Run parse stage first.")
        sys.exit(1)

    tok_cfg = cfg.get("tokenizer", {})
    config = TokenizerConfig(
        ticks_per_bar=cfg.get("parser", {}).get("ticks_per_bar", 48),
        velocity_bins=tok_cfg.get("velocity_bins", 8),
        duration_bins=tok_cfg.get("duration_bins", 16),
        tempo_min=tok_cfg.get("tempo_min", 60),
        tempo_max=tok_cfg.get("tempo_max", 200),
        tempo_step=tok_cfg.get("tempo_step", 2),
        max_pattern_length=tok_cfg.get("max_pattern_length", 512),
        max_arrangement_length=tok_cfg.get("max_arrangement_length", 128),
    )

    vocab, data = process_all_projects(json_dir, processed_dir, config)
    logger.info(f"Tokenized {len(data)} projects, vocab size {len(vocab)}")


def stage_train(cfg: dict) -> None:
    """Stage 3: Train the model."""
    logger.info("=" * 60)
    logger.info("STAGE 3: TRAINING")
    logger.info("=" * 60)

    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_data"]
    model_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]

    data_path = processed_dir / "tokenized_data.pkl"
    vocab_path = processed_dir / "vocab.pkl"

    if not data_path.exists() or not vocab_path.exists():
        logger.error("Tokenized data not found. Run tokenize stage first.")
        sys.exit(1)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    best_ckpt = train(
        data_path=data_path,
        vocab_path=vocab_path,
        model_dir=model_dir,
        d_model=model_cfg.get("d_model", 256),
        n_heads=model_cfg.get("n_heads", 8),
        n_layers=model_cfg.get("n_layers", 6),
        d_ff=model_cfg.get("d_ff", 1024),
        dropout=model_cfg.get("dropout", 0.1),
        max_seq_len=model_cfg.get("max_seq_len", 2048),
        batch_size=train_cfg.get("batch_size", 16),
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_steps=train_cfg.get("warmup_steps", 500),
        max_epochs=train_cfg.get("max_epochs", 200),
        patience=train_cfg.get("patience", 20),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        save_every_n_epochs=train_cfg.get("save_every_n_epochs", 10),
        eval_split=train_cfg.get("eval_split", 0.1),
        seed=train_cfg.get("seed", 42),
        checkpoint_path=cfg["paths"].get("checkpoint"),
    )

    logger.info(f"Best model saved at: {best_ckpt}")


def stage_generate(cfg: dict) -> None:
    """Stage 4: Generate new beats."""
    logger.info("=" * 60)
    logger.info("STAGE 4: GENERATING BEATS")
    logger.info("=" * 60)

    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_data"]
    model_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]
    output_dir = PROJECT_ROOT / cfg["paths"]["generated_output"]

    model_path = model_dir / "best.pt"
    vocab_path = processed_dir / "vocab.pkl"

    if not model_path.exists():
        logger.warning("best.pt not found, trying latest.pt")
        model_path = model_dir / "latest.pt"

    if not model_path.exists():
        logger.error("No model checkpoint found. Run train stage first.")
        sys.exit(1)

    gen_cfg = cfg.get("generation", {})

    paths = generate_beats(
        model_path=model_path,
        vocab_path=vocab_path,
        output_dir=output_dir,
        count=gen_cfg.get("count", 5),
        temperature=gen_cfg.get("temperature", 0.85),
        top_k=gen_cfg.get("top_k", 50),
        top_p=gen_cfg.get("top_p", 0.92),
        max_length=gen_cfg.get("max_length", 1024),
        genre_hint=gen_cfg.get("genre_hint"),
        tempo=gen_cfg.get("tempo"),
        ticks_per_bar=cfg.get("parser", {}).get("ticks_per_bar", 48),
    )

    logger.info(f"Generated {len(paths)} MIDI file(s):")
    for p in paths:
        logger.info(f"  → {p}")


def main():
    parser = argparse.ArgumentParser(description="RichieDaGenius Beat Generation Pipeline")
    parser.add_argument(
        "--stage",
        choices=["parse", "tokenize", "train", "generate", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument("-c", "--config", default="configs/default.yaml", help="Config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    stages = {
        "parse": stage_parse,
        "tokenize": stage_tokenize,
        "train": stage_train,
        "generate": stage_generate,
    }

    if args.stage == "all":
        for name, fn in stages.items():
            fn(cfg)
    else:
        stages[args.stage](cfg)

    logger.info("Done.")


if __name__ == "__main__":
    main()
