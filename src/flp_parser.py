"""
Stage 1: FLP Parser
====================
Extracts MIDI patterns, arrangement data, tempo, and channel info
from FL Studio project files (FLP) bundled inside ZIP archives.

Uses PyFLP for binary FLP parsing.
Falls back to raw binary parsing for unsupported FLP versions.
"""

import json
import logging
import os
import shutil
import struct
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    """A single MIDI note event extracted from an FLP pattern."""
    pitch: int           # 0-127
    velocity: int        # 0-127
    position: int        # Absolute position in ticks (PPQ-based)
    length: int          # Duration in ticks
    channel: int         # Channel rack index
    fine_pitch: int = 0  # Pitch fine-tuning (cents)
    release: int = 64    # Note release value
    pan: int = 64        # Note panning

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PatternData:
    """All notes within a single FL Studio pattern."""
    index: int
    name: str
    notes: list[NoteEvent] = field(default_factory=list)
    color: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "color": self.color,
            "note_count": len(self.notes),
            "notes": [n.to_dict() for n in self.notes],
        }


@dataclass
class ChannelInfo:
    """Metadata about a channel rack instrument."""
    index: int
    name: str
    plugin: Optional[str] = None
    volume: float = 1.0
    pan: float = 0.0
    is_drum: bool = False
    sample_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArrangementClip:
    """A pattern placement on the FL Studio playlist/arrangement."""
    pattern_index: int
    track: int            # Playlist track number
    position: int         # Start position in ticks
    length: int           # Duration in ticks
    offset: int = 0       # Start offset within the pattern

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProjectData:
    """Complete extracted data from one FL Studio project."""
    source_file: str
    tempo: float = 140.0
    ppq: int = 96             # Pulses per quarter note
    time_sig_num: int = 4
    time_sig_den: int = 4
    channels: list[ChannelInfo] = field(default_factory=list)
    patterns: list[PatternData] = field(default_factory=list)
    arrangement: list[ArrangementClip] = field(default_factory=list)
    audio_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "tempo": self.tempo,
            "ppq": self.ppq,
            "time_sig_num": self.time_sig_num,
            "time_sig_den": self.time_sig_den,
            "channel_count": len(self.channels),
            "pattern_count": len(self.patterns),
            "arrangement_clip_count": len(self.arrangement),
            "audio_file_count": len(self.audio_files),
            "channels": [c.to_dict() for c in self.channels],
            "patterns": [p.to_dict() for p in self.patterns],
            "arrangement": [a.to_dict() for a in self.arrangement],
            "audio_files": self.audio_files,
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved project data → {path}")


# ---------------------------------------------------------------------------
# Drum detection
# ---------------------------------------------------------------------------

DEFAULT_DRUM_KEYWORDS = [
    "kick", "snare", "hat", "hihat", "hi-hat", "clap", "perc",
    "808", "rim", "tom", "crash", "shaker", "cymbal", "open hat",
    "closed hat",
]


def is_drum_channel(name: str, keywords: list[str] | None = None) -> bool:
    """Heuristic: check if a channel name suggests a drum/percussion sound."""
    kw = keywords or DEFAULT_DRUM_KEYWORDS
    name_lower = name.lower()
    return any(k in name_lower for k in kw)


# ---------------------------------------------------------------------------
# PyFLP-based parser (preferred)
# ---------------------------------------------------------------------------

def parse_flp_pyflp(flp_path: str | Path, drum_keywords: list[str] | None = None) -> ProjectData:
    """
    Parse an FLP file using the PyFLP library.
    This is the preferred method — gives structured access to patterns,
    channels, arrangement, and global settings.
    """
    import pyflp

    flp_path = Path(flp_path)
    logger.info(f"Parsing with PyFLP: {flp_path.name}")

    project = pyflp.parse(flp_path)
    data = ProjectData(source_file=str(flp_path))

    # --- Global settings ---
    data.tempo = float(project.tempo) if hasattr(project, "tempo") else 140.0
    data.ppq = int(project.ppq) if hasattr(project, "ppq") else 96

    if hasattr(project, "time_sig_num"):
        data.time_sig_num = int(project.time_sig_num)
    if hasattr(project, "time_sig_den"):
        data.time_sig_den = int(project.time_sig_den)

    # --- Channels ---
    if hasattr(project, "channels"):
        for idx, ch in enumerate(project.channels):
            name = getattr(ch, "name", f"Channel {idx}")
            plugin_name = None
            if hasattr(ch, "plugin") and ch.plugin:
                plugin_name = getattr(ch.plugin, "name", str(type(ch.plugin).__name__))

            sample = None
            if hasattr(ch, "sample_path"):
                sample = str(ch.sample_path) if ch.sample_path else None

            volume = float(getattr(ch, "volume", 1.0))
            pan = float(getattr(ch, "pan", 0.0))

            ci = ChannelInfo(
                index=idx,
                name=name,
                plugin=plugin_name,
                volume=volume,
                pan=pan,
                is_drum=is_drum_channel(name, drum_keywords),
                sample_path=sample,
            )
            data.channels.append(ci)

    # --- Patterns ---
    if hasattr(project, "patterns"):
        for pat in project.patterns:
            pat_idx = getattr(pat, "index", 0)
            pat_name = getattr(pat, "name", f"Pattern {pat_idx}")
            pd = PatternData(index=pat_idx, name=pat_name)

            if hasattr(pat, "color"):
                pd.color = getattr(pat, "color", None)

            # Extract notes from the pattern
            if hasattr(pat, "notes"):
                for note in pat.notes:
                    ne = NoteEvent(
                        pitch=int(getattr(note, "key", 60)),
                        velocity=int(getattr(note, "velocity", 100)),
                        position=int(getattr(note, "position", 0)),
                        length=int(getattr(note, "length", 96)),
                        channel=int(getattr(note, "channel", 0)),
                        fine_pitch=int(getattr(note, "fine_pitch", 0)),
                        release=int(getattr(note, "release", 64)),
                        pan=int(getattr(note, "pan", 64)),
                    )
                    pd.notes.append(ne)

            if pd.notes:
                data.patterns.append(pd)

    # --- Arrangement ---
    if hasattr(project, "arrangements"):
        for arr in project.arrangements:
            if hasattr(arr, "tracks"):
                for track_idx, track in enumerate(arr.tracks):
                    if hasattr(track, "items"):
                        for item in track.items:
                            pat_idx = getattr(item, "pattern", None)
                            if pat_idx is not None:
                                clip = ArrangementClip(
                                    pattern_index=int(pat_idx),
                                    track=track_idx,
                                    position=int(getattr(item, "position", 0)),
                                    length=int(getattr(item, "length", 0)),
                                    offset=int(getattr(item, "start_offset", 0)),
                                )
                                data.arrangement.append(clip)

    logger.info(
        f"  Extracted: {len(data.channels)} channels, "
        f"{len(data.patterns)} patterns ({sum(len(p.notes) for p in data.patterns)} notes), "
        f"{len(data.arrangement)} arrangement clips"
    )
    return data


# ---------------------------------------------------------------------------
# Fallback: raw binary FLP parser
# ---------------------------------------------------------------------------

# FLP binary format constants
FLP_HEADER_MAGIC = b"FLhd"
FLP_DATA_MAGIC = b"FLdt"

# Event ID ranges
EVENT_BYTE = 0
EVENT_WORD = 64
EVENT_DWORD = 128
EVENT_TEXT = 192
EVENT_DATA = 208

# Known event IDs
EV_TEMPO = EVENT_WORD + 2          # 66 — BPM (x1000 in older versions)
EV_PATTERN_NAME = EVENT_TEXT + 1   # 193
EV_CHANNEL_NAME = EVENT_TEXT + 11  # 203
EV_NOTE = EVENT_DATA + 1          # 209 — Note data block
EV_PPQ = EVENT_WORD + 0           # 64


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Read a variable-length integer from FLP data stream."""
    result = 0
    shift = 0
    while offset < len(data):
        b = data[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break
    return result, offset


def parse_flp_raw(flp_path: str | Path, drum_keywords: list[str] | None = None) -> ProjectData:
    """
    Fallback raw binary parser for FLP files.
    Less complete than PyFLP but handles more FLP versions.
    Extracts: tempo, PPQ, pattern names, channel names, note events.
    """
    flp_path = Path(flp_path)
    logger.info(f"Parsing with raw binary parser: {flp_path.name}")

    with open(flp_path, "rb") as f:
        raw = f.read()

    data = ProjectData(source_file=str(flp_path))

    # --- Validate header ---
    if raw[:4] != FLP_HEADER_MAGIC:
        raise ValueError(f"Not a valid FLP file: {flp_path}")

    # Header: 4 bytes magic + 4 bytes header_len + header data
    header_len = struct.unpack_from("<I", raw, 4)[0]
    header_data = raw[8 : 8 + header_len]

    # Format, num_channels (not always reliable)
    if len(header_data) >= 6:
        _format = struct.unpack_from("<H", header_data, 0)[0]
        _num_channels = struct.unpack_from("<H", header_data, 2)[0]
        ppq = struct.unpack_from("<H", header_data, 4)[0]
        data.ppq = ppq

    # --- Find FLdt chunk ---
    fldt_offset = raw.find(FLP_DATA_MAGIC)
    if fldt_offset < 0:
        raise ValueError("FLdt chunk not found")

    fldt_len = struct.unpack_from("<I", raw, fldt_offset + 4)[0]
    events_data = raw[fldt_offset + 8 : fldt_offset + 8 + fldt_len]

    # --- Parse events ---
    pos = 0
    current_pattern_idx = 0
    pattern_names: dict[int, str] = {}
    channel_names: dict[int, str] = {}
    current_channel_idx = 0
    pattern_notes: dict[int, list[NoteEvent]] = {}

    while pos < len(events_data):
        event_id = events_data[pos]
        pos += 1

        if event_id < EVENT_WORD:
            # Byte-sized value
            value = events_data[pos]
            pos += 1

        elif event_id < EVENT_DWORD:
            # Word-sized value (2 bytes)
            value = struct.unpack_from("<H", events_data, pos)[0]
            pos += 2

            if event_id == EV_PPQ:
                data.ppq = value
            elif event_id == EV_TEMPO:
                # In older FLP versions, tempo is stored as BPM * 1000
                # In newer ones it's a direct value via DWORD event
                if value > 1000:
                    data.tempo = value / 1000.0
                else:
                    data.tempo = float(value)

        elif event_id < EVENT_TEXT:
            # DWORD-sized value (4 bytes)
            value = struct.unpack_from("<I", events_data, pos)[0]
            pos += 4

            # Tempo can also appear as DWORD (event 156 in newer FLPs)
            if event_id == 156:  # NewTempo
                data.tempo = value / 1000.0

        elif event_id < EVENT_DATA:
            # Text event (variable length)
            text_len, pos = _read_varint(events_data, pos)
            text_bytes = events_data[pos : pos + text_len]
            pos += text_len

            try:
                text = text_bytes.decode("utf-16-le").rstrip("\x00")
            except UnicodeDecodeError:
                try:
                    text = text_bytes.decode("utf-8").rstrip("\x00")
                except UnicodeDecodeError:
                    text = ""

            if event_id == EV_PATTERN_NAME:
                pattern_names[current_pattern_idx] = text
                current_pattern_idx += 1
            elif event_id == EV_CHANNEL_NAME:
                channel_names[current_channel_idx] = text
                current_channel_idx += 1

        else:
            # Data event (variable length)
            data_len, pos = _read_varint(events_data, pos)
            event_data = events_data[pos : pos + data_len]
            pos += data_len

            if event_id == EV_NOTE and len(event_data) >= 24:
                # Note events are packed in 24-byte or 32-byte blocks
                note_size = 24 if len(event_data) % 24 == 0 else 32
                if len(event_data) % 32 == 0 and note_size == 24:
                    # Prefer 32 if both divide evenly and data > 24
                    if len(event_data) > 24:
                        note_size = 32

                for i in range(0, len(event_data), note_size):
                    block = event_data[i : i + note_size]
                    if len(block) < 24:
                        break

                    note_pos = struct.unpack_from("<I", block, 0)[0]
                    _flags = struct.unpack_from("<H", block, 4)[0]
                    _rack = struct.unpack_from("<H", block, 6)[0]
                    note_dur = struct.unpack_from("<I", block, 8)[0]
                    note_key = struct.unpack_from("<H", block, 12)[0]
                    fine_pitch = struct.unpack_from("<h", block, 14)[0]
                    note_release = block[16] if len(block) > 16 else 64
                    _midi_ch = block[17] if len(block) > 17 else 0
                    note_pan = block[18] if len(block) > 18 else 64
                    note_vel = block[19] if len(block) > 19 else 100
                    pat_num = struct.unpack_from("<H", block, 20)[0]

                    ne = NoteEvent(
                        pitch=note_key & 0x7F,
                        velocity=min(note_vel, 127),
                        position=note_pos,
                        length=note_dur,
                        channel=_rack,
                        fine_pitch=fine_pitch,
                        release=note_release,
                        pan=note_pan,
                    )
                    pattern_notes.setdefault(pat_num, []).append(ne)

    # --- Assemble patterns ---
    for pat_idx, notes in pattern_notes.items():
        name = pattern_names.get(pat_idx, f"Pattern {pat_idx}")
        pd = PatternData(index=pat_idx, name=name, notes=notes)
        data.patterns.append(pd)

    # --- Assemble channels ---
    for ch_idx, ch_name in channel_names.items():
        ci = ChannelInfo(
            index=ch_idx,
            name=ch_name,
            is_drum=is_drum_channel(ch_name, drum_keywords),
        )
        data.channels.append(ci)

    logger.info(
        f"  Raw parse: {len(data.channels)} channels, "
        f"{len(data.patterns)} patterns ({sum(len(p.notes) for p in data.patterns)} notes)"
    )
    return data


# ---------------------------------------------------------------------------
# ZIP extraction + orchestration
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aif", ".aiff"}


def extract_zip(zip_path: str | Path, extract_to: str | Path) -> tuple[list[Path], list[Path]]:
    """
    Extract a ZIP file. Returns (flp_files, audio_files).
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    flp_files = []
    audio_files = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
        for member in zf.namelist():
            full_path = extract_to / member
            if not full_path.is_file():
                continue
            ext = full_path.suffix.lower()
            if ext == ".flp":
                flp_files.append(full_path)
            elif ext in AUDIO_EXTENSIONS:
                audio_files.append(full_path)

    logger.info(f"Extracted ZIP: {len(flp_files)} FLP(s), {len(audio_files)} audio file(s)")
    return flp_files, audio_files


def parse_zip(
    zip_path: str | Path,
    drum_keywords: list[str] | None = None,
    work_dir: str | Path | None = None,
) -> list[ProjectData]:
    """
    Main entry point: extract ZIP → parse each FLP → return project data.
    """
    zip_path = Path(zip_path)
    cleanup = False

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="flp_parse_"))
        cleanup = True
    else:
        work_dir = Path(work_dir)

    try:
        flp_files, audio_files = extract_zip(zip_path, work_dir)
        audio_names = [f.name for f in audio_files]

        results = []
        for flp_path in flp_files:
            # Try PyFLP first, fall back to raw parser
            try:
                project = parse_flp_pyflp(flp_path, drum_keywords)
            except Exception as e:
                logger.warning(f"PyFLP failed for {flp_path.name}: {e}")
                logger.info("Falling back to raw binary parser...")
                try:
                    project = parse_flp_raw(flp_path, drum_keywords)
                except Exception as e2:
                    logger.error(f"Raw parser also failed: {e2}")
                    continue

            project.audio_files = audio_names
            results.append(project)

        return results

    finally:
        if cleanup and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def parse_single_flp(
    flp_path: str | Path,
    drum_keywords: list[str] | None = None,
) -> ProjectData:
    """Parse a standalone FLP file (not inside a ZIP)."""
    flp_path = Path(flp_path)
    try:
        return parse_flp_pyflp(flp_path, drum_keywords)
    except Exception:
        return parse_flp_raw(flp_path, drum_keywords)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_raw_directory(
    raw_dir: str | Path,
    output_dir: str | Path,
    drum_keywords: list[str] | None = None,
) -> list[Path]:
    """
    Process all ZIP and FLP files in a directory.
    Saves extracted JSON for each project to output_dir.
    Returns list of output JSON paths.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    for file_path in sorted(raw_dir.iterdir()):
        ext = file_path.suffix.lower()

        if ext == ".zip":
            projects = parse_zip(file_path, drum_keywords)
            for i, proj in enumerate(projects):
                stem = file_path.stem
                suffix = f"_{i}" if len(projects) > 1 else ""
                out_path = output_dir / f"{stem}{suffix}.json"
                proj.save_json(out_path)
                output_paths.append(out_path)

        elif ext == ".flp":
            proj = parse_single_flp(file_path, drum_keywords)
            out_path = output_dir / f"{file_path.stem}.json"
            proj.save_json(out_path)
            output_paths.append(out_path)

    logger.info(f"Processed {len(output_paths)} project(s) → {output_dir}")
    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Extract MIDI data from FL Studio projects")
    parser.add_argument("input", help="ZIP file, FLP file, or directory of ZIPs/FLPs")
    parser.add_argument("-o", "--output", default="data/processed", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        process_raw_directory(input_path, args.output)
    elif input_path.suffix.lower() == ".zip":
        projects = parse_zip(input_path)
        for proj in projects:
            proj.save_json(Path(args.output) / f"{input_path.stem}.json")
    elif input_path.suffix.lower() == ".flp":
        proj = parse_single_flp(input_path)
        proj.save_json(Path(args.output) / f"{input_path.stem}.json")
    else:
        print(f"Unsupported file type: {input_path.suffix}")
