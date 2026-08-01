"""
Stage 1: FLP Parser
====================
Extracts MIDI patterns, arrangement data, tempo, and channel info
from FL Studio project files (FLP) bundled inside ZIP archives.

Uses PyFLP for binary FLP parsing (with a Python 3.12+ compatibility patch).
Falls back to raw binary parsing for unsupported FLP versions.

Can export one MIDI file per pattern into a per-beat folder:
  BEAT_EXAMPLE.zip @ 140 BPM → BEAT_EXAMPLE_140/*.mid
"""

from __future__ import annotations

import json
import logging
import re
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
    channel: int         # Channel rack index / IID
    fine_pitch: int = 0  # Pitch fine-tuning
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

    @property
    def bpm_label(self) -> int:
        """Integer BPM used in export folder names."""
        return max(1, int(round(self.tempo)))


# ---------------------------------------------------------------------------
# Drum detection / helpers
# ---------------------------------------------------------------------------

DEFAULT_DRUM_KEYWORDS = [
    "kick", "snare", "hat", "hihat", "hi-hat", "clap", "perc",
    "808", "rim", "tom", "crash", "shaker", "cymbal", "open hat",
    "closed hat", "drum",
]

_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def is_drum_channel(name: str, keywords: list[str] | None = None) -> bool:
    """Heuristic: check if a channel name suggests a drum/percussion sound."""
    kw = keywords or DEFAULT_DRUM_KEYWORDS
    name_lower = name.lower()
    return any(k in name_lower for k in kw)


def safe_filename(name: str, fallback: str = "untitled") -> str:
    """Make a string safe for use as a file/folder name."""
    cleaned = _UNSAFE_FILENAME.sub("_", name).strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or fallback


def beat_export_dirname(source_stem: str, tempo: float) -> str:
    """
    Build the per-beat export folder name.

    Example: BEAT_EXAMPLE.zip @ 140 BPM → BEAT_EXAMPLE_140
    """
    stem = safe_filename(source_stem, fallback="beat")
    bpm = max(1, int(round(tempo)))
    return f"{stem}_{bpm}"


def _note_key_to_midi(note) -> int:
    """Convert a PyFLP Note key (int or name like 'D#6') to MIDI pitch 0-127."""
    try:
        return int(note["key"]) & 0x7F
    except Exception:
        pass

    key = getattr(note, "key", 60)
    if isinstance(key, int):
        return key & 0x7F
    if isinstance(key, str):
        names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        for i, n in enumerate(names):
            if key.startswith(n) and (len(key) == len(n) or key[len(n)].isdigit() or key[len(n)] == "-"):
                try:
                    octave = int(key[len(n):])
                    return max(0, min(127, octave * 12 + i))
                except ValueError:
                    continue
    return 60


# ---------------------------------------------------------------------------
# PyFLP Python 3.12+ compatibility
# ---------------------------------------------------------------------------

_PYFLP_PATCHED = False


def _patch_pyflp_for_python312() -> None:
    """
    PyFLP 2.2.1's EventEnum base has no members. On Python 3.12+, calling an
    empty Enum raises TypeError before _missing_ can create pseudo-members.

    Inject a dummy member so EventEnum(id) works again.
    See: https://github.com/demberto/PyFLP/issues/183
    """
    global _PYFLP_PATCHED
    if _PYFLP_PATCHED:
        return

    try:
        from pyflp._events import EventEnum
    except ImportError:
        return

    if EventEnum._member_map_:
        _PYFLP_PATCHED = True
        return

    dummy = int.__new__(EventEnum, -1)
    dummy._name_ = "_PYFLP_PY312_PATCH"
    dummy._value_ = -1
    setattr(dummy, "type", None)
    EventEnum._member_map_[dummy._name_] = dummy
    EventEnum._value2member_map_[-1] = dummy
    EventEnum._member_names_.append(dummy._name_)
    _PYFLP_PATCHED = True
    logger.debug("Applied PyFLP Python 3.12+ EventEnum compatibility patch")


# ---------------------------------------------------------------------------
# PyFLP-based parser (preferred)
# ---------------------------------------------------------------------------

def parse_flp_pyflp(flp_path: str | Path, drum_keywords: list[str] | None = None) -> ProjectData:
    """
    Parse an FLP file using the PyFLP library.
    This is the preferred method — gives structured access to patterns,
    channels, arrangement, and global settings.
    """
    _patch_pyflp_for_python312()
    import pyflp

    flp_path = Path(flp_path)
    logger.info(f"Parsing with PyFLP: {flp_path.name}")

    project = pyflp.parse(flp_path)
    data = ProjectData(source_file=str(flp_path))

    # --- Global settings ---
    data.tempo = float(project.tempo) if hasattr(project, "tempo") else 140.0
    data.ppq = int(project.ppq) if hasattr(project, "ppq") else 96

    if hasattr(project, "time_sig_num") and project.time_sig_num is not None:
        data.time_sig_num = int(project.time_sig_num)
    if hasattr(project, "time_sig_den") and project.time_sig_den is not None:
        data.time_sig_den = int(project.time_sig_den)

    # --- Channels ---
    if hasattr(project, "channels"):
        for idx, ch in enumerate(project.channels):
            name = getattr(ch, "name", None) or f"Channel {idx}"
            plugin_name = None
            if hasattr(ch, "plugin") and ch.plugin:
                plugin_name = getattr(ch.plugin, "name", str(type(ch.plugin).__name__))

            sample = None
            if hasattr(ch, "sample_path"):
                sample = str(ch.sample_path) if ch.sample_path else None

            volume = float(getattr(ch, "volume", 1.0) or 1.0)
            pan = float(getattr(ch, "pan", 0.0) or 0.0)

            # Prefer channel IID when available (notes reference rack by IID)
            ch_index = getattr(ch, "iid", None)
            if ch_index is None:
                ch_index = idx

            ci = ChannelInfo(
                index=int(ch_index),
                name=str(name),
                plugin=plugin_name,
                volume=volume,
                pan=pan,
                is_drum=is_drum_channel(str(name), drum_keywords),
                sample_path=sample,
            )
            data.channels.append(ci)

    # --- Patterns ---
    if hasattr(project, "patterns"):
        for pat in project.patterns:
            pat_idx = getattr(pat, "iid", None)
            if pat_idx is None:
                pat_idx = getattr(pat, "index", 0)
            pat_name = getattr(pat, "name", None) or f"Pattern {pat_idx}"
            pd = PatternData(index=int(pat_idx), name=str(pat_name))

            if hasattr(pat, "color") and pat.color is not None:
                try:
                    pd.color = int(pat.color) if not hasattr(pat.color, "red") else None
                except (TypeError, ValueError):
                    pd.color = None

            if hasattr(pat, "notes"):
                for note in pat.notes:
                    rack = getattr(note, "rack_channel", None)
                    if rack is None:
                        rack = getattr(note, "channel", 0)
                    ne = NoteEvent(
                        pitch=_note_key_to_midi(note),
                        velocity=min(127, max(0, int(getattr(note, "velocity", 100) or 100))),
                        position=int(getattr(note, "position", 0) or 0),
                        length=int(getattr(note, "length", data.ppq) or data.ppq),
                        channel=int(rack or 0),
                        fine_pitch=int(getattr(note, "fine_pitch", 0) or 0),
                        release=int(getattr(note, "release", 64) or 64),
                        pan=int(getattr(note, "pan", 64) or 64),
                    )
                    pd.notes.append(ne)

            if pd.notes:
                data.patterns.append(pd)

    # --- Arrangement / playlist ---
    if hasattr(project, "arrangements"):
        for arr in project.arrangements:
            tracks = getattr(arr, "tracks", None)
            if tracks is None:
                continue
            for track_idx, track in enumerate(tracks):
                try:
                    items = list(track)
                except TypeError:
                    items = list(getattr(track, "items", []) or [])
                for item in items:
                    pattern = getattr(item, "pattern", None)
                    if pattern is None:
                        continue
                    pat_idx = getattr(pattern, "iid", None)
                    if pat_idx is None:
                        pat_idx = getattr(pattern, "index", 0)
                    offset = 0
                    offsets = getattr(item, "offsets", None)
                    if offsets is not None:
                        try:
                            offset = int(offsets[0])
                        except (TypeError, ValueError, IndexError):
                            offset = 0
                    clip = ArrangementClip(
                        pattern_index=int(pat_idx),
                        track=track_idx,
                        position=int(getattr(item, "position", 0) or 0),
                        length=int(getattr(item, "length", 0) or 0),
                        offset=offset,
                    )
                    data.arrangement.append(clip)

    logger.info(
        f"  Extracted: {len(data.channels)} channels, "
        f"{len(data.patterns)} patterns ({sum(len(p.notes) for p in data.patterns)} notes), "
        f"{len(data.arrangement)} arrangement clips, "
        f"{data.tempo:.2f} BPM"
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

# Known event IDs (aligned with PyFLP where possible)
EV_TEMPO = EVENT_WORD + 2          # 66 — BPM (x1000 in older versions)
EV_PATTERN_ID = EVENT_WORD + 1     # 65 — current pattern (New)
EV_PATTERN_NAME = EVENT_TEXT + 1   # 193
EV_CHANNEL_NAME = EVENT_TEXT + 11  # 203
EV_NOTE_OLD = EVENT_DATA + 1       # 209 — legacy note data
EV_NOTES = EVENT_DATA + 16         # 224 — modern NotesEvent (24-byte notes)
EV_PPQ = EVENT_WORD + 0            # 64
EV_NEW_TEMPO = 156                 # DWORD tempo * 1000

# Modern note: 24 bytes (PyFLP NotesEvent)
# position u32, flags u16, rack u16, length u32, key u16, group u16,
# fine_pitch u8, _u1 u8, release u8, midi_channel u8, pan u8, velocity u8, mod_x u8, mod_y u8
_NOTE24 = struct.Struct("<I H H I H H B B B B B B B B")


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


def _parse_notes_24(event_data: bytes) -> list[NoteEvent]:
    """Parse modern 24-byte note blocks (event 224)."""
    notes: list[NoteEvent] = []
    if len(event_data) < 24 or len(event_data) % 24 != 0:
        return notes
    for i in range(0, len(event_data), 24):
        (
            position, _flags, rack, length, key, _group,
            fine, _u1, release, _midich, pan, velocity, _modx, _mody,
        ) = _NOTE24.unpack_from(event_data, i)
        notes.append(
            NoteEvent(
                pitch=key & 0x7F,
                velocity=min(127, velocity),
                position=position,
                length=length,
                channel=rack,
                fine_pitch=fine,
                release=release,
                pan=pan,
            )
        )
    return notes


def _parse_notes_legacy(event_data: bytes) -> list[NoteEvent]:
    """Parse legacy note blocks (event 209) — 20/24/32-byte variants."""
    notes: list[NoteEvent] = []
    if len(event_data) < 20:
        return notes

    if len(event_data) % 24 == 0:
        note_size = 24
    elif len(event_data) % 32 == 0:
        note_size = 32
    elif len(event_data) % 20 == 0:
        note_size = 20
    else:
        note_size = 24 if len(event_data) >= 24 else 20

    for i in range(0, len(event_data), note_size):
        block = event_data[i : i + note_size]
        if len(block) < 20:
            break
        if len(block) >= 24:
            # Same layout as modern notes when 24+
            parsed = _parse_notes_24(block[:24])
            notes.extend(parsed)
            continue

        # 20-byte compact layout (older)
        note_pos = struct.unpack_from("<I", block, 0)[0]
        _flags = struct.unpack_from("<H", block, 4)[0]
        rack = struct.unpack_from("<H", block, 6)[0]
        note_dur = struct.unpack_from("<I", block, 8)[0]
        note_key = struct.unpack_from("<H", block, 12)[0]
        fine_pitch = struct.unpack_from("<h", block, 14)[0]
        note_release = block[16] if len(block) > 16 else 64
        note_pan = block[18] if len(block) > 18 else 64
        note_vel = block[19] if len(block) > 19 else 100
        notes.append(
            NoteEvent(
                pitch=note_key & 0x7F,
                velocity=min(note_vel, 127),
                position=note_pos,
                length=note_dur,
                channel=rack,
                fine_pitch=fine_pitch,
                release=note_release,
                pan=note_pan,
            )
        )
    return notes


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

    if len(header_data) >= 6:
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
            pos += 1

        elif event_id < EVENT_DWORD:
            value = struct.unpack_from("<H", events_data, pos)[0]
            pos += 2

            if event_id == EV_PPQ:
                data.ppq = value
            elif event_id == EV_TEMPO:
                if value > 1000:
                    data.tempo = value / 1000.0
                else:
                    data.tempo = float(value)
            elif event_id == EV_PATTERN_ID:
                current_pattern_idx = value

        elif event_id < EVENT_TEXT:
            value = struct.unpack_from("<I", events_data, pos)[0]
            pos += 4

            if event_id == EV_NEW_TEMPO:
                data.tempo = value / 1000.0

        elif event_id < EVENT_DATA:
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
            elif event_id == EV_CHANNEL_NAME:
                channel_names[current_channel_idx] = text
                current_channel_idx += 1

        else:
            data_len, pos = _read_varint(events_data, pos)
            event_data = events_data[pos : pos + data_len]
            pos += data_len

            if event_id == EV_NOTES:
                notes = _parse_notes_24(event_data)
                if notes:
                    pattern_notes[current_pattern_idx] = notes
            elif event_id == EV_NOTE_OLD:
                notes = _parse_notes_legacy(event_data)
                if notes and current_pattern_idx not in pattern_notes:
                    pattern_notes[current_pattern_idx] = notes

    # --- Assemble patterns ---
    for pat_idx, notes in sorted(pattern_notes.items()):
        name = pattern_names.get(pat_idx, f"Pattern {pat_idx}")
        pd = PatternData(index=pat_idx, name=name, notes=notes)
        data.patterns.append(pd)

    # Patterns that only have names (no notes) are skipped for MIDI export,
    # but we still record channels fully.
    for ch_idx, ch_name in channel_names.items():
        ci = ChannelInfo(
            index=ch_idx,
            name=ch_name,
            is_drum=is_drum_channel(ch_name, drum_keywords),
        )
        data.channels.append(ci)

    logger.info(
        f"  Raw parse: {len(data.channels)} channels, "
        f"{len(data.patterns)} patterns ({sum(len(p.notes) for p in data.patterns)} notes), "
        f"{data.tempo:.2f} BPM"
    )
    return data


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------

def export_pattern_to_midi(
    pattern: PatternData,
    project: ProjectData,
    output_path: str | Path,
) -> Path:
    """
    Write a single FL Studio pattern as a multi-track MIDI file.

    One MIDI track per channel that has notes in the pattern.
    Drum channels use GM channel 10 (0-indexed: 9).
    """
    import mido

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ppq = project.ppq or 96
    mid = mido.MidiFile(ticks_per_beat=ppq)

    # Tempo / meta track
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(project.tempo), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=project.time_sig_num,
                                 denominator=project.time_sig_den, time=0))
    meta.append(mido.MetaMessage("track_name", name=pattern.name, time=0))

    channel_map = {c.index: c for c in project.channels}

    # Group notes by rack channel
    by_channel: dict[int, list[NoteEvent]] = {}
    for note in pattern.notes:
        by_channel.setdefault(note.channel, []).append(note)

    for ch_idx in sorted(by_channel.keys()):
        notes = by_channel[ch_idx]
        ch_info = channel_map.get(ch_idx)
        track_name = ch_info.name if ch_info else f"Channel {ch_idx}"
        is_drum = ch_info.is_drum if ch_info else is_drum_channel(track_name)

        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=track_name, time=0))

        midi_channel = 9 if is_drum else min(ch_idx % 16, 15)
        if midi_channel == 9 and not is_drum:
            midi_channel = (ch_idx % 15)  # keep non-drums off channel 10
            if midi_channel == 9:
                midi_channel = 0

        # Build absolute-time note on/off events
        events: list[tuple[int, int, int, int]] = []  # (tick, order, pitch, velocity)
        # order: 0=off, 1=on so offs at same tick process first
        for n in notes:
            pitch = max(0, min(127, n.pitch))
            vel = max(1, min(127, n.velocity if n.velocity > 0 else 100))
            start = max(0, n.position)
            end = max(start + 1, start + max(1, n.length))
            events.append((start, 1, pitch, vel))
            events.append((end, 0, pitch, 0))

        events.sort(key=lambda e: (e[0], e[1], e[2]))

        current_tick = 0
        for tick, order, pitch, vel in events:
            delta = max(0, tick - current_tick)
            current_tick = tick
            if order == 1:
                track.append(mido.Message(
                    "note_on", note=pitch, velocity=vel, time=delta, channel=midi_channel
                ))
            else:
                track.append(mido.Message(
                    "note_off", note=pitch, velocity=0, time=delta, channel=midi_channel
                ))

    mid.save(str(output_path))
    return output_path


def export_project_midis(
    project: ProjectData,
    output_dir: str | Path,
    source_stem: str | None = None,
) -> Path:
    """
    Export all patterns from a project into one beat folder.

    Layout:
        {output_dir}/{stem}_{BPM}/
            01_Pattern_Name.mid
            02_Another.mid
            project.json

    Returns the beat folder path.
    """
    output_dir = Path(output_dir)
    if source_stem is None:
        source_stem = Path(project.source_file).stem
    beat_dir = output_dir / beat_export_dirname(source_stem, project.tempo)
    beat_dir.mkdir(parents=True, exist_ok=True)

    # Always save structured JSON alongside midis
    project.save_json(beat_dir / "project.json")

    if not project.patterns:
        logger.warning(f"No patterns with notes to export for {source_stem}")
        return beat_dir

    used_names: dict[str, int] = {}
    exported = 0

    # Sort by pattern index for stable ordering
    for pattern in sorted(project.patterns, key=lambda p: p.index):
        if not pattern.notes:
            continue
        base = safe_filename(pattern.name, fallback=f"Pattern_{pattern.index}")
        # Prefix with zero-padded index for uniqueness / ordering
        name = f"{pattern.index:02d}_{base}"
        if name in used_names:
            used_names[name] += 1
            name = f"{name}_{used_names[name]}"
        else:
            used_names[name] = 1

        midi_path = beat_dir / f"{name}.mid"
        export_pattern_to_midi(pattern, project, midi_path)
        exported += 1
        logger.info(f"  MIDI → {midi_path.name} ({len(pattern.notes)} notes)")

    logger.info(
        f"Exported {exported} MIDI(s) → {beat_dir} "
        f"({project.bpm_label} BPM)"
    )
    return beat_dir


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
        # Guard against zip-slip
        extract_root = extract_to.resolve()
        for member in zf.namelist():
            target = (extract_to / member).resolve()
            if not str(target).startswith(str(extract_root)):
                logger.warning(f"Skipping unsafe zip member: {member}")
                continue
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


def parse_single_flp(
    flp_path: str | Path,
    drum_keywords: list[str] | None = None,
) -> ProjectData:
    """Parse a standalone FLP file (not inside a ZIP)."""
    flp_path = Path(flp_path)
    try:
        return parse_flp_pyflp(flp_path, drum_keywords)
    except Exception as e:
        logger.warning(f"PyFLP failed for {flp_path.name}: {e}")
        logger.info("Falling back to raw binary parser...")
        return parse_flp_raw(flp_path, drum_keywords)


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

        if not flp_files:
            logger.warning(f"No FLP files found in {zip_path.name}")
            return []

        results = []
        for flp_path in flp_files:
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
            # Prefer zip stem as logical project name source
            project.source_file = str(zip_path)
            results.append(project)

        return results

    finally:
        if cleanup and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_raw_directory(
    raw_dir: str | Path,
    output_dir: str | Path,
    drum_keywords: list[str] | None = None,
    export_midi: bool = True,
) -> list[Path]:
    """
    Process all ZIP and FLP files in a directory (non-recursive).

    For each beat:
      - Saves project.json
      - If export_midi: writes all pattern MIDIs into {stem}_{BPM}/

    Returns list of output paths (beat folders or JSON files).
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    files = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".zip", ".flp"}
    )

    if not files:
        logger.warning(f"No .zip or .flp files found in {raw_dir}")
        return output_paths

    for file_path in files:
        ext = file_path.suffix.lower()
        try:
            if ext == ".zip":
                projects = parse_zip(file_path, drum_keywords)
                for i, proj in enumerate(projects):
                    stem = file_path.stem
                    if len(projects) > 1:
                        stem = f"{stem}_{i}"
                    if export_midi:
                        out = export_project_midis(proj, output_dir, source_stem=stem)
                    else:
                        out = output_dir / f"{stem}.json"
                        proj.save_json(out)
                    output_paths.append(out)

            elif ext == ".flp":
                proj = parse_single_flp(file_path, drum_keywords)
                stem = file_path.stem
                if export_midi:
                    out = export_project_midis(proj, output_dir, source_stem=stem)
                else:
                    out = output_dir / f"{stem}.json"
                    proj.save_json(out)
                output_paths.append(out)
        except zipfile.BadZipFile:
            logger.error(f"Corrupt ZIP, skipping: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    logger.info(f"Processed {len(output_paths)} project(s) → {output_dir}")
    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Extract MIDI data from FL Studio projects (.flp / .zip)"
    )
    parser.add_argument("input", help="ZIP file, FLP file, or directory of ZIPs/FLPs")
    parser.add_argument("-o", "--output", default="data/processed", help="Output directory")
    parser.add_argument(
        "--no-midi",
        action="store_true",
        help="Only write JSON (skip per-pattern MIDI export)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help=argparse.SUPPRESS,  # alias
    )
    args = parser.parse_args()

    export_midi = not (args.no_midi or args.json_only)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        process_raw_directory(input_path, output_path, export_midi=export_midi)
    elif input_path.suffix.lower() == ".zip":
        projects = parse_zip(input_path)
        if not projects:
            raise SystemExit(f"No FLP projects found in {input_path}")
        for i, proj in enumerate(projects):
            stem = input_path.stem if len(projects) == 1 else f"{input_path.stem}_{i}"
            if export_midi:
                export_project_midis(proj, output_path, source_stem=stem)
            else:
                proj.save_json(output_path / f"{stem}.json")
    elif input_path.suffix.lower() == ".flp":
        proj = parse_single_flp(input_path)
        if export_midi:
            export_project_midis(proj, output_path, source_stem=input_path.stem)
        else:
            proj.save_json(output_path / f"{input_path.stem}.json")
    else:
        raise SystemExit(f"Unsupported file type: {input_path.suffix}")
