"""
Generator
==========
Samples new beat sequences from a trained BeatTransformer model
and converts them back to standard MIDI files for FL Studio import.

Output: .mid files that can be dragged directly into FL Studio's
playlist or channel rack.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token → MIDI decoding
# ---------------------------------------------------------------------------

def decode_tokens_to_midi_events(
    token_ids: list[int],
    vocab,
    ticks_per_bar: int = 48,
    default_tempo: float = 140.0,
) -> dict:
    """
    Convert a sequence of token IDs back into structured MIDI events.

    Returns:
        {
            "tempo": float,
            "genre": str | None,
            "tracks": {
                channel_idx: [
                    {"pitch": int, "velocity": int, "start_tick": int, "duration_ticks": int, "is_drum": bool},
                    ...
                ]
            }
        }
    """
    # Velocity bin → MIDI value (center of each bin)
    def vel_bin_to_midi(bin_idx: int, n_bins: int = 8) -> int:
        bin_width = 128 / n_bins
        return int(min(127, bin_idx * bin_width + bin_width / 2))

    # Duration bin → tick duration (inverse of log bucketing)
    def dur_bin_to_ticks(bin_idx: int, n_bins: int = 16, tpb: int = 48) -> int:
        max_dur = tpb * 4
        log_min = np.log(1)
        log_max = np.log(max_dur)
        log_dur = log_min + (bin_idx / (n_bins - 1)) * (log_max - log_min)
        return max(1, int(np.exp(log_dur)))

    tempo = default_tempo
    genre = None
    tracks: dict[int, list[dict]] = {}

    current_bar = 0
    current_pos = 0
    current_inst = 0
    is_drum = False

    i = 0
    tokens = [vocab.decode(tid) for tid in token_ids]

    while i < len(tokens):
        token = tokens[i]

        if token.startswith("TEMPO_"):
            tempo = float(token.split("_")[1])
        elif token.startswith("GENRE_"):
            genre = token.split("_", 1)[1].lower()
        elif token.startswith("BAR_"):
            current_bar = int(token.split("_")[1])
        elif token.startswith("POS_"):
            current_pos = int(token.split("_")[1])
        elif token.startswith("INST_"):
            current_inst = int(token.split("_")[1])
        elif token.startswith("DRUM_"):
            pitch = int(token.split("_")[1])
            is_drum = True
            # Look ahead for velocity and duration
            vel = 100
            dur = ticks_per_bar // 4  # Default 1/16 note
            if i + 1 < len(tokens) and tokens[i + 1].startswith("VEL_"):
                vel = vel_bin_to_midi(int(tokens[i + 1].split("_")[1]))
                i += 1
            if i + 1 < len(tokens) and tokens[i + 1].startswith("DUR_"):
                dur = dur_bin_to_ticks(int(tokens[i + 1].split("_")[1]), tpb=ticks_per_bar)
                i += 1

            start_tick = current_bar * ticks_per_bar + current_pos
            tracks.setdefault(current_inst, []).append({
                "pitch": pitch,
                "velocity": vel,
                "start_tick": start_tick,
                "duration_ticks": dur,
                "is_drum": True,
            })
        elif token.startswith("PITCH_"):
            pitch = int(token.split("_")[1])
            vel = 100
            dur = ticks_per_bar // 4
            if i + 1 < len(tokens) and tokens[i + 1].startswith("VEL_"):
                vel = vel_bin_to_midi(int(tokens[i + 1].split("_")[1]))
                i += 1
            if i + 1 < len(tokens) and tokens[i + 1].startswith("DUR_"):
                dur = dur_bin_to_ticks(int(tokens[i + 1].split("_")[1]), tpb=ticks_per_bar)
                i += 1

            start_tick = current_bar * ticks_per_bar + current_pos
            tracks.setdefault(current_inst, []).append({
                "pitch": pitch,
                "velocity": vel,
                "start_tick": start_tick,
                "duration_ticks": dur,
                "is_drum": False,
            })

        i += 1

    return {"tempo": tempo, "genre": genre, "tracks": tracks}


def events_to_midi_file(
    events: dict,
    output_path: str | Path,
    ticks_per_bar: int = 48,
) -> Path:
    """
    Convert decoded events to a standard MIDI file using mido.

    Creates one MIDI track per instrument channel.
    Drum tracks are placed on MIDI channel 10 (GM standard).
    """
    import mido

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tempo = events["tempo"]
    tracks_data = events["tracks"]

    mid = mido.MidiFile(ticks_per_beat=ticks_per_bar)

    # Tempo track
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    tempo_track.append(mido.MetaMessage("track_name", name="Tempo", time=0))
    mid.tracks.append(tempo_track)

    # One track per instrument channel
    for ch_idx in sorted(tracks_data.keys()):
        notes = tracks_data[ch_idx]
        if not notes:
            continue

        track = mido.MidiTrack()
        is_drum_track = any(n["is_drum"] for n in notes)
        midi_channel = 9 if is_drum_track else min(ch_idx, 15)  # Ch 10 = drums (0-indexed: 9)
        if midi_channel == 9 and not is_drum_track:
            midi_channel = min(ch_idx + 1, 15)

        track_name = f"{'Drums' if is_drum_track else 'Inst'} {ch_idx}"
        track.append(mido.MetaMessage("track_name", name=track_name, time=0))

        # Build note-on/note-off event list sorted by time
        events_list = []
        for note in notes:
            events_list.append(("on", note["start_tick"], note["pitch"], note["velocity"]))
            events_list.append(("off", note["start_tick"] + note["duration_ticks"], note["pitch"], 0))

        events_list.sort(key=lambda x: (x[1], 0 if x[0] == "off" else 1))

        # Convert to delta times
        current_tick = 0
        for event_type, tick, pitch, vel in events_list:
            delta = tick - current_tick
            current_tick = tick
            if event_type == "on":
                track.append(mido.Message("note_on", note=pitch, velocity=vel, time=delta, channel=midi_channel))
            else:
                track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta, channel=midi_channel))

        mid.tracks.append(track)

    mid.save(str(output_path))
    logger.info(f"Saved MIDI: {output_path} ({len(tracks_data)} tracks, {tempo} BPM)")
    return output_path


# ---------------------------------------------------------------------------
# Generation orchestration
# ---------------------------------------------------------------------------

def generate_beats(
    model_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    count: int = 5,
    temperature: float = 0.85,
    top_k: int = 50,
    top_p: float = 0.92,
    max_length: int = 1024,
    genre_hint: str | None = None,
    tempo: float | None = None,
    ticks_per_bar: int = 48,
    seed: int | None = None,
) -> list[Path]:
    """
    Generate N new beats from a trained model.

    Args:
        model_path: Path to trained checkpoint
        vocab_path: Path to vocabulary pickle
        output_dir: Where to save MIDI files
        count: Number of beats to generate
        temperature: Sampling temperature (lower = more conservative)
        top_k: Top-k sampling
        top_p: Nucleus sampling threshold
        max_length: Max tokens per generated sequence
        genre_hint: "trap" or "afrobeats" to condition generation
        tempo: Override tempo (BPM), or None for model's choice
        ticks_per_bar: Resolution (must match training)
        seed: Random seed for reproducibility

    Returns:
        List of generated MIDI file paths
    """
    from .model import BeatTransformer

    model_path = Path(model_path)
    vocab_path = Path(vocab_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    vocab = Vocabulary.load(vocab_path)
    logger.info(f"Loaded vocabulary: {len(vocab)} tokens")

    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_config = ckpt["config"]

    model = BeatTransformer(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        d_ff=model_config["d_ff"],
        max_seq_len=model_config["max_seq_len"],
        dropout=0.0,  # No dropout for inference
        pad_token_id=vocab.encode("<PAD>"),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded model from epoch {ckpt.get('epoch', '?')}")

    # Generate
    eos_id = vocab.encode("<SONG_END>")
    output_paths = []

    for i in range(count):
        # Build prompt
        prompt_tokens = [vocab.encode("<BOS>"), vocab.encode("<SONG_START>")]

        if genre_hint:
            genre_token = f"GENRE_{genre_hint.upper()}"
            if genre_token in vocab.token_to_id:
                prompt_tokens.append(vocab.encode(genre_token))

        if tempo:
            # Round to nearest step
            from .tokenizer import TokenizerConfig
            cfg = TokenizerConfig()
            t = round(tempo / cfg.tempo_step) * cfg.tempo_step
            t = max(cfg.tempo_min, min(t, cfg.tempo_max))
            tempo_token = f"TEMPO_{t}"
            if tempo_token in vocab.token_to_id:
                prompt_tokens.append(vocab.encode(tempo_token))

        prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        # Generate
        generated = model.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_id,
        )

        gen_ids = generated[0].cpu().tolist()
        logger.info(f"Beat {i + 1}/{count}: generated {len(gen_ids)} tokens")

        # Decode to MIDI events
        events = decode_tokens_to_midi_events(gen_ids, vocab, ticks_per_bar)

        # Save MIDI
        genre_tag = genre_hint or "beat"
        midi_path = output_dir / f"generated_{genre_tag}_{i + 1:03d}.mid"
        events_to_midi_file(events, midi_path, ticks_per_bar)
        output_paths.append(midi_path)

    logger.info(f"Generated {count} beats → {output_dir}")
    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate new beats")
    parser.add_argument("-m", "--model", default="models/best.pt", help="Model checkpoint")
    parser.add_argument("-v", "--vocab", default="data/processed/vocab.pkl", help="Vocabulary")
    parser.add_argument("-o", "--output", default="data/generated", help="Output directory")
    parser.add_argument("-n", "--count", type=int, default=5, help="Number of beats")
    parser.add_argument("-t", "--temperature", type=float, default=0.85)
    parser.add_argument("-g", "--genre", choices=["trap", "afrobeats"], default=None)
    parser.add_argument("--tempo", type=float, default=None, help="Override BPM")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    generate_beats(
        model_path=args.model,
        vocab_path=args.vocab,
        output_dir=args.output,
        count=args.count,
        temperature=args.temperature,
        genre_hint=args.genre,
        tempo=args.tempo,
        seed=args.seed,
    )
