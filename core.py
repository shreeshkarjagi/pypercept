from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


# ── frequency bands (quick-access dict) ──────────────────────────────────────

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta1": (13, 20),
    "beta2": (20, 30),
    "gamma": (30, 50),
}


# ── channel helpers ──────────────────────────────────────────────────────────

def get_hemisphere(channel: str) -> str:
    ch = channel.upper()
    if "LEFT" in ch:
        return "left"
    elif "RIGHT" in ch:
        return "right"
    return "unknown"


def get_channel_label(channel: str) -> str:
    """Convert device channel name like ZERO_THREE_LEFT --> L 0-3."""
    labels = {
        "ZERO_THREE_LEFT": "L 0-3", "ONE_THREE_LEFT": "L 1-3",
        "ZERO_TWO_LEFT": "L 0-2",
        "ZERO_THREE_RIGHT": "R 0-3", "ONE_THREE_RIGHT": "R 1-3",
        "ZERO_TWO_RIGHT": "R 0-2",
    }
    normalized = channel.replace("_AND_", "_")
    return labels.get(normalized, channel)


# ── input containers ─────────────────────────────────────────────────────────

@dataclass
class Recording:
    """A single channel recording (from IndefiniteStreaming or BrainSenseTimeDomain)."""

    channel: str
    data: np.ndarray
    fs: float = 250.0
    datetime: str = ""

    @property
    def hemisphere(self) -> str:
        return get_hemisphere(self.channel)

    @property
    def label(self) -> str:
        return get_channel_label(self.channel)

    @property
    def duration(self) -> float:
        return len(self.data) / self.fs

    @property
    def time(self) -> np.ndarray:
        return np.arange(len(self.data)) / self.fs


@dataclass
class Session:
    """All recordings from a single JSON file. This is the batch-processing input."""

    filepath: Path
    date: str
    recordings: List[Recording] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    timeline: List[dict] = field(default_factory=list)

    @property
    def channels(self) -> List[str]:
        return [r.channel for r in self.recordings]

    def by_hemisphere(self) -> Dict[str, List[Recording]]:
        result = {"left": [], "right": []}
        for rec in self.recordings:
            if rec.hemisphere in result:
                result[rec.hemisphere].append(rec)
        return result

    def __repr__(self):
        return f"Session({self.filepath.name}, {len(self.recordings)} channels)"


# ── output containers ────────────────────────────────────────────────────────

@dataclass
class ChannelResult:
    """Processed results for one channel: PSD, band powers, optional spectrogram."""

    channel: str
    hemisphere: str
    data: np.ndarray
    fs: float
    freqs: np.ndarray
    psd: np.ndarray
    band_powers: Dict[str, float]

    # spectrogram (only populated if you ask for it)
    spec_t: Optional[np.ndarray] = None
    spec_f: Optional[np.ndarray] = None
    spec_Sxx: Optional[np.ndarray] = None


# backward compat
TrialResult = ChannelResult


@dataclass
class SessionResult:
    """Processed results for a full session (all channels)."""

    filepath: Path
    date: str
    channels: List[ChannelResult] = field(default_factory=list)

    def by_hemisphere(self) -> Dict[str, List[ChannelResult]]:
        result = {"left": [], "right": []}
        for ch in self.channels:
            if ch.hemisphere in result:
                result[ch.hemisphere].append(ch)
        return result

    def mean_band_powers(self, hemisphere: str = None) -> Dict[str, float]:
        chs = self.channels
        if hemisphere:
            chs = [ch for ch in chs if ch.hemisphere == hemisphere]
        if not chs:
            return {}
        bands = list(chs[0].band_powers.keys())
        return {band: np.mean([ch.band_powers[band] for ch in chs]) for band in bands}

    def mean_psd(self, hemisphere: str = None) -> tuple:
        """Returns (freqs, mean_psd) averaged across channels."""
        chs = self.channels
        if hemisphere:
            chs = [ch for ch in chs if ch.hemisphere == hemisphere]
        if not chs:
            return None, None
        freqs = chs[0].freqs
        psds = np.array([ch.psd for ch in chs])
        return freqs, np.mean(psds, axis=0)
