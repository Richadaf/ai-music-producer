# RichieDaGenius Beat Generation Model

A pipeline that ingests FL Studio projects (FLP files inside ZIPs), extracts MIDI patterns,
arrangement data, and audio features, then trains a generative model to produce new beats
in your signature style.

## Architecture

```
ZIP (FLP + audio stems)
        │
        ▼
┌──────────────────┐
│  Stage 1: INGEST │   Extract MIDI patterns, tempo, time sigs,
│  flp_parser.py   │   channel rack info, arrangement from FLP
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  Stage 2: FEATURE│   Convert raw MIDI → tokenised sequences
│  tokenizer.py    │   (pitch, velocity, duration, time-shift)
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  Stage 3: TRAIN  │   Transformer-based sequence model learns
│  train.py        │   your rhythmic/melodic/arrangement patterns
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  Stage 4: GENERATE│  Sample from trained model → new MIDI
│  generate.py      │  patterns → export as .mid files
└───────┬───────────┘
        │
        ▼
   Import .mid back
   into FL Studio
```

## Requirements

```
Python >= 3.10
pyflp >= 2.2.0        # FLP binary parser
mido >= 1.3.0         # MIDI file I/O
torch >= 2.0.0        # Model training
numpy >= 1.24.0
tqdm
pyyaml
librosa >= 0.10.0     # Audio feature extraction (optional, for stems)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Drop your ZIPs into data/raw/
cp ~/beats/*.zip data/raw/

# 3. Run the full pipeline
python scripts/run_pipeline.py

# 4. Generate new beats
python src/generate.py --model models/latest.pt --count 5 --output data/generated/
```

## Project Structure

```
beat-model/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # All hyperparams and paths
├── src/
│   ├── __init__.py
│   ├── flp_parser.py         # Stage 1: FLP/ZIP ingestion
│   ├── tokenizer.py          # Stage 2: MIDI → token sequences
│   ├── dataset.py            # PyTorch Dataset for training
│   ├── model.py              # Transformer architecture
│   ├── train.py              # Training loop
│   └── generate.py           # Inference / beat generation
├── scripts/
│   └── run_pipeline.py       # End-to-end orchestrator
├── data/
│   ├── raw/                  # Drop ZIP files here
│   ├── processed/            # Extracted + tokenised data
│   └── generated/            # Output MIDI files
└── models/                   # Saved checkpoints
```

## How It Works (Detail)

### FLP Parsing
PyFLP reverse-engineers the FL Studio binary format. We extract:
- **Channel Rack**: instrument names, plugin IDs, volumes, pans
- **Patterns**: MIDI note events (pitch, velocity, position, length)
- **Arrangement**: playlist clips — which patterns play when
- **Tempo & Time Signature**: global BPM and meter

Audio samples referenced in the FLP are extracted from the ZIP
and optionally analysed with librosa for timbral features.

### Tokenisation (REMI-style)
Each pattern is converted to a sequence of tokens:
- `BAR_N`         — bar boundary
- `POS_N`         — position within bar (quantised to 1/48)
- `PITCH_N`       — MIDI note number (0-127) or DRUM_KIT mapping
- `VEL_N`         — velocity bucket (8 levels)
- `DUR_N`         — duration bucket (16 levels)
- `INST_N`        — instrument/channel index
- `TEMPO_N`       — tempo change token
- `<PAT_START>`   — pattern boundary
- `<PAT_END>`     — pattern boundary
- `<SONG_START>`  — arrangement sequence start
- `<SONG_END>`    — arrangement sequence end

### Model
A GPT-style decoder-only Transformer that learns:
- Drum patterns (kick, snare, hat, perc placement + velocity)
- Melodic sequences (note selection, intervals)
- Arrangement structure (intro → verse → hook → bridge patterns)
- Genre-specific timing (Trap triplet swing vs Afrobeats groove)

### Generation
Temperature-controlled autoregressive sampling produces new
token sequences → decoded back to MIDI → import into FL Studio.

## Notes for Richie

- **Trap triplet feel**: The tokenizer preserves your 1/6 time shift (≈0.1667)
  by using high-resolution position quantisation (1/48 per bar)
- **Afrobeats groove**: Velocity variation is captured via 8-level bucketing,
  so the subtle ghost notes and accent patterns are learnable
- **Minimum dataset**: Aim for 50+ FLPs to get decent results; 200+ is ideal
- **Fine-tuning**: You can start from a pretrained music model and fine-tune
  on your beats — massively reduces the data requirement
