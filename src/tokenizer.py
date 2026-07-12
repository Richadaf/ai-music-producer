"""
Stage 2: MIDI Tokenizer
========================
Converts extracted FL Studio project data (JSON from flp_parser)
into REMI-style token sequences suitable for Transformer training.

Token vocabulary:
  BAR_0 .. BAR_N          — Bar boundary markers
  POS_0 .. POS_47         — Position within bar (1/48 resolution)
  PITCH_0 .. PITCH_127    — MIDI note number
  DRUM_0 .. DRUM_127      — Drum hit (same as PITCH but flagged)
  VEL_0 .. VEL_7          — Velocity bucket (8 levels)
  DUR_0 .. DUR_15         — Duration bucket (16 levels)
  INST_0 .. INST_N        — Instrument/channel index
  TEMPO_60 .. TEMPO_200   — Tempo tokens (step=2)
  <PAD>                   — Padding
  <BOS>                   — Beginning of sequence
  <EOS>                   — End of sequence
  <PAT_START>             — Pattern boundary
  <PAT_END>               — Pattern boundary
  <SONG_START>            — Arrangement start
  <SONG_END>              — Arrangement end
  <SEP>                   — Separator between patterns in arrangement
  GENRE_TRAP              — Genre conditioning token
  GENRE_AFROBEATS         — Genre conditioning token
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = [
    "<PAD>", "<BOS>", "<EOS>",
    "<PAT_START>", "<PAT_END>",
    "<SONG_START>", "<SONG_END>",
    "<SEP>",
    "GENRE_TRAP", "GENRE_AFROBEATS",
]


@dataclass
class Vocabulary:
    """Maps tokens ↔ indices."""
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    def encode(self, token: str) -> int:
        return self.token_to_id[token]

    def decode(self, idx: int) -> str:
        return self.id_to_token[idx]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"t2i": self.token_to_id, "i2t": self.id_to_token}, f)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path, "rb") as f:
            data = pickle.load(f)
        v = cls()
        v.token_to_id = data["t2i"]
        v.id_to_token = data["i2t"]
        return v


def build_vocabulary(
    ticks_per_bar: int = 48,
    velocity_bins: int = 8,
    duration_bins: int = 16,
    tempo_min: int = 60,
    tempo_max: int = 200,
    tempo_step: int = 2,
    max_instruments: int = 64,
    max_bars: int = 256,
) -> Vocabulary:
    """Build the full token vocabulary."""
    vocab = Vocabulary()

    # Special tokens
    for t in SPECIAL_TOKENS:
        vocab.add(t)

    # Bar markers
    for i in range(max_bars):
        vocab.add(f"BAR_{i}")

    # Position within bar
    for i in range(ticks_per_bar):
        vocab.add(f"POS_{i}")

    # Pitch (melodic)
    for i in range(128):
        vocab.add(f"PITCH_{i}")

    # Drum hits
    for i in range(128):
        vocab.add(f"DRUM_{i}")

    # Velocity buckets
    for i in range(velocity_bins):
        vocab.add(f"VEL_{i}")

    # Duration buckets
    for i in range(duration_bins):
        vocab.add(f"DUR_{i}")

    # Instruments
    for i in range(max_instruments):
        vocab.add(f"INST_{i}")

    # Tempo tokens
    for t in range(tempo_min, tempo_max + 1, tempo_step):
        vocab.add(f"TEMPO_{t}")

    logger.info(f"Vocabulary built: {len(vocab)} tokens")
    return vocab


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def quantise_position(position: int, ppq: int, ticks_per_bar: int, time_sig_num: int = 4) -> tuple[int, int]:
    """
    Convert absolute tick position → (bar_number, position_within_bar).

    Args:
        position: Absolute position in PPQ ticks
        ppq: Pulses per quarter note from the FLP
        ticks_per_bar: Target resolution (e.g., 48)
        time_sig_num: Beats per bar (e.g., 4 for 4/4)

    Returns:
        (bar_index, quantised_position_in_bar)
    """
    ticks_per_bar_source = ppq * time_sig_num  # e.g., 96 * 4 = 384
    scale = ticks_per_bar / ticks_per_bar_source

    scaled_pos = position * scale
    bar = int(scaled_pos // ticks_per_bar)
    pos_in_bar = int(round(scaled_pos % ticks_per_bar)) % ticks_per_bar

    return bar, pos_in_bar


def quantise_velocity(velocity: int, bins: int = 8) -> int:
    """Bucket velocity (0-127) into N bins."""
    return min(int(velocity * bins / 128), bins - 1)


def quantise_duration(duration: int, ppq: int, bins: int = 16, ticks_per_bar: int = 48, time_sig_num: int = 4) -> int:
    """
    Bucket note duration into N bins.
    Bins are logarithmically spaced from 1/48 of a bar to 4 bars.
    """
    ticks_per_bar_source = ppq * time_sig_num
    scale = ticks_per_bar / ticks_per_bar_source
    scaled_dur = max(duration * scale, 0.5)

    # Log-scale bins from 1 tick to 4 bars
    max_dur = ticks_per_bar * 4
    log_min = np.log(1)
    log_max = np.log(max_dur)
    log_dur = np.log(min(scaled_dur, max_dur))

    bin_idx = int((log_dur - log_min) / (log_max - log_min) * (bins - 1))
    return max(0, min(bin_idx, bins - 1))


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

@dataclass
class TokenizerConfig:
    ticks_per_bar: int = 48
    velocity_bins: int = 8
    duration_bins: int = 16
    tempo_min: int = 60
    tempo_max: int = 200
    tempo_step: int = 2
    max_pattern_length: int = 512
    max_arrangement_length: int = 128


def tokenize_pattern(
    pattern: dict,
    channels: list[dict],
    ppq: int,
    time_sig_num: int,
    config: TokenizerConfig,
    vocab: Vocabulary,
) -> list[int]:
    """
    Convert a single pattern's notes into a token sequence.

    Token order per note event:
        POS_X → INST_X → PITCH_X/DRUM_X → VEL_X → DUR_X
    Events are sorted by (bar, position, channel).
    Bar boundary tokens are inserted.
    """
    notes = pattern.get("notes", [])
    if not notes:
        return []

    # Build channel lookup for drum detection
    drum_channels = set()
    for ch in channels:
        if ch.get("is_drum", False):
            drum_channels.add(ch["index"])

    # Quantise all notes
    quantised = []
    for note in notes:
        bar, pos = quantise_position(
            note["position"], ppq, config.ticks_per_bar, time_sig_num
        )
        vel = quantise_velocity(note["velocity"], config.velocity_bins)
        dur = quantise_duration(
            note["length"], ppq, config.duration_bins, config.ticks_per_bar, time_sig_num
        )
        ch = note["channel"]
        pitch = note["pitch"]
        is_drum = ch in drum_channels

        quantised.append({
            "bar": bar,
            "pos": pos,
            "channel": ch,
            "pitch": pitch,
            "velocity": vel,
            "duration": dur,
            "is_drum": is_drum,
        })

    # Sort by bar → position → channel
    quantised.sort(key=lambda x: (x["bar"], x["pos"], x["channel"]))

    # Build token sequence
    tokens = [vocab.encode("<PAT_START>")]
    current_bar = -1

    for q in quantised:
        # Bar boundary
        if q["bar"] != current_bar:
            current_bar = q["bar"]
            bar_token = f"BAR_{min(current_bar, 255)}"
            if bar_token in vocab.token_to_id:
                tokens.append(vocab.encode(bar_token))

        # Position
        tokens.append(vocab.encode(f"POS_{q['pos']}"))

        # Instrument
        inst_token = f"INST_{min(q['channel'], 63)}"
        tokens.append(vocab.encode(inst_token))

        # Pitch or Drum
        if q["is_drum"]:
            tokens.append(vocab.encode(f"DRUM_{q['pitch']}"))
        else:
            tokens.append(vocab.encode(f"PITCH_{q['pitch']}"))

        # Velocity
        tokens.append(vocab.encode(f"VEL_{q['velocity']}"))

        # Duration
        tokens.append(vocab.encode(f"DUR_{q['duration']}"))

    tokens.append(vocab.encode("<PAT_END>"))

    # Truncate if needed
    if len(tokens) > config.max_pattern_length:
        tokens = tokens[: config.max_pattern_length - 1] + [vocab.encode("<PAT_END>")]

    return tokens


def tokenize_project(
    project_data: dict,
    config: TokenizerConfig,
    vocab: Vocabulary,
    genre_hint: Optional[str] = None,
) -> dict:
    """
    Tokenize an entire FL Studio project.

    Returns:
        {
            "pattern_sequences": [[int, ...], ...],  # One sequence per pattern
            "arrangement_sequence": [int, ...],       # Full arrangement as token seq
            "metadata": {...}
        }
    """
    ppq = project_data.get("ppq", 96)
    time_sig_num = project_data.get("time_sig_num", 4)
    tempo = project_data.get("tempo", 140.0)
    channels = project_data.get("channels", [])
    patterns = project_data.get("patterns", [])
    arrangement = project_data.get("arrangement", [])

    # --- Tokenise each pattern ---
    pattern_sequences = []
    for pat in patterns:
        seq = tokenize_pattern(pat, channels, ppq, time_sig_num, config, vocab)
        if seq:
            pattern_sequences.append(seq)

    # --- Build arrangement sequence ---
    # This represents the song structure: which patterns play in what order
    arr_tokens = [vocab.encode("<SONG_START>")]

    # Genre conditioning token
    if genre_hint:
        genre_token = f"GENRE_{genre_hint.upper()}"
        if genre_token in vocab.token_to_id:
            arr_tokens.append(vocab.encode(genre_token))

    # Tempo token
    tempo_rounded = round(tempo / config.tempo_step) * config.tempo_step
    tempo_clamped = max(config.tempo_min, min(tempo_rounded, config.tempo_max))
    tempo_token = f"TEMPO_{tempo_clamped}"
    if tempo_token in vocab.token_to_id:
        arr_tokens.append(vocab.encode(tempo_token))

    # Sort arrangement clips by position
    arr_sorted = sorted(arrangement, key=lambda x: (x.get("position", 0), x.get("track", 0)))

    # Map pattern indices to the actual pattern data indices
    pattern_index_map = {pat["index"]: i for i, pat in enumerate(patterns)}

    for clip in arr_sorted:
        pat_idx = clip.get("pattern_index", 0)
        if pat_idx in pattern_index_map:
            local_idx = pattern_index_map[pat_idx]
            if local_idx < len(pattern_sequences) and pattern_sequences[local_idx]:
                arr_tokens.extend(pattern_sequences[local_idx])
                arr_tokens.append(vocab.encode("<SEP>"))

    arr_tokens.append(vocab.encode("<SONG_END>"))

    # Truncate
    max_arr = config.max_arrangement_length * config.max_pattern_length
    if len(arr_tokens) > max_arr:
        arr_tokens = arr_tokens[:max_arr - 1] + [vocab.encode("<SONG_END>")]

    return {
        "pattern_sequences": pattern_sequences,
        "arrangement_sequence": arr_tokens,
        "metadata": {
            "source": project_data.get("source_file", ""),
            "tempo": tempo,
            "ppq": ppq,
            "time_sig": f"{time_sig_num}/4",
            "num_patterns": len(pattern_sequences),
            "total_pattern_tokens": sum(len(s) for s in pattern_sequences),
            "arrangement_tokens": len(arr_tokens),
        },
    }


# ---------------------------------------------------------------------------
# Genre detection heuristic
# ---------------------------------------------------------------------------

def detect_genre(project_data: dict) -> Optional[str]:
    """
    Simple heuristic to guess if a beat is Trap or Afrobeats
    based on tempo and drum pattern characteristics.
    """
    tempo = project_data.get("tempo", 140.0)

    # Afrobeats: typically 95-115 BPM (or half-time 180-220)
    # Trap: typically 130-170 BPM (or half-time 65-85)
    if 90 <= tempo <= 120:
        return "afrobeats"
    elif 125 <= tempo <= 175:
        return "trap"
    elif 60 <= tempo <= 89:
        # Could be half-time trap
        return "trap"

    return None


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_all_projects(
    json_dir: str | Path,
    output_dir: str | Path,
    config: TokenizerConfig | None = None,
) -> tuple[Vocabulary, list[dict]]:
    """
    Process all extracted project JSONs into tokenised sequences.
    Builds vocabulary and saves everything.
    """
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = TokenizerConfig()

    vocab = build_vocabulary(
        ticks_per_bar=config.ticks_per_bar,
        velocity_bins=config.velocity_bins,
        duration_bins=config.duration_bins,
        tempo_min=config.tempo_min,
        tempo_max=config.tempo_max,
        tempo_step=config.tempo_step,
    )

    all_tokenized = []

    for json_path in sorted(json_dir.glob("*.json")):
        with open(json_path) as f:
            project_data = json.load(f)

        genre = detect_genre(project_data)
        tokenized = tokenize_project(project_data, config, vocab, genre)

        if tokenized["pattern_sequences"]:
            all_tokenized.append(tokenized)
            logger.info(
                f"  {json_path.stem}: {tokenized['metadata']['num_patterns']} patterns, "
                f"{tokenized['metadata']['arrangement_tokens']} arrangement tokens, "
                f"genre={genre}"
            )

    # Save
    vocab.save(output_dir / "vocab.pkl")

    with open(output_dir / "tokenized_data.pkl", "wb") as f:
        pickle.dump(all_tokenized, f)

    logger.info(
        f"Tokenized {len(all_tokenized)} projects. "
        f"Vocab size: {len(vocab)}. "
        f"Saved to {output_dir}"
    )

    return vocab, all_tokenized


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Tokenize extracted FLP data")
    parser.add_argument("input_dir", help="Directory of extracted project JSONs")
    parser.add_argument("-o", "--output", default="data/processed", help="Output directory")
    args = parser.parse_args()

    process_all_projects(args.input_dir, args.output)
