from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import numpy as np


# ── frequency bands ──────────────────────────────────────────────────────────
# matches the LfpFrequencySnapshotEvents bands from the Percept device

@dataclass
class FrequencyBands:
    delta: Tuple[float, float] = (1.0, 4.0)
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta1: Tuple[float, float] = (13.0, 20.0)   # low beta
    beta2: Tuple[float, float] = (20.0, 30.0)    # high beta
    beta: Tuple[float, float] = (13.0, 30.0)     # full beta
    gamma: Tuple[float, float] = (30.0, 42.0)    # low gamma
    high_gamma: Tuple[float, float] = (60.0, 90.0)  # only useful for time-domain data

    def get_band(self, name: str) -> Tuple[float, float]:
        return getattr(self, name)

    def get_all(self) -> Dict[str, Tuple[float, float]]:
        return {
            "delta": self.delta, "theta": self.theta, "alpha": self.alpha,
            "beta1": self.beta1, "beta2": self.beta2, "beta": self.beta,
            "gamma": self.gamma, "high_gamma": self.high_gamma,
        }

    def get_standard(self) -> Dict[str, Tuple[float, float]]:
        """The 6-band decomposition we use most of the time."""
        return {
            "delta": self.delta, "theta": self.theta, "alpha": self.alpha,
            "beta1": self.beta1, "beta2": self.beta2, "gamma": self.gamma,
        }

    def get_indices(self, freqs: np.ndarray, band_name: str) -> np.ndarray:
        low, high = self.get_band(band_name)
        return np.where((freqs > low) & (freqs <= high))[0]


# ── processing params ────────────────────────────────────────────────────────

@dataclass
class ProcessingConfig:
    sample_rate: int = 250      # Percept PC/RC native rate
    nyquist: float = 125.0

    # filtering
    notch_freq: float = 60.0    # 60 Hz for US, swap to 50 for EU
    notch_q: float = 30.0
    highpass: float = 1.0
    lowpass: float = 100.0

    # welch PSD
    nperseg: int = 256
    noverlap: int = 128
    nfft: int = 512

    # spectrogram
    spec_nperseg: int = 250     # 1 sec at 250 Hz
    spec_noverlap: int = 125

    # artifact rejection
    amplitude_threshold: float = 500.0  # µV


# ── plot style ───────────────────────────────────────────────────────────────
# targeting Nature Medicine / clean publication style

@dataclass
class PlotConfig:
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 300
    style: str = "default"

    # fonts
    font_family: str = "Arial"
    font_size_title: int = 10
    font_size_label: int = 9
    font_size_tick: int = 8
    font_size_legend: int = 7

    # line widths
    line_width_data: float = 1.0
    line_width_axis: float = 0.8
    line_width_tick: float = 0.6

    # hemisphere colors
    left_color: str = "#2171B5"   # blue
    right_color: str = "#CB181D"  # red

    channel_colors: List[str] = field(default_factory=lambda: [
        "#2171B5", "#CB181D", "#238B45", "#6A51A3", "#D94801", "#525252",
    ])

    band_colors: Dict[str, str] = field(default_factory=lambda: {
        "delta": "#FEE5D9", "theta": "#FCBBA1", "alpha": "#FC9272",
        "beta1": "#FB6A4A", "beta2": "#DE2D26", "beta": "#A50F15",
        "gamma": "#67000D", "high_gamma": "#3F007D",
    })

    freq_range: Tuple[float, float] = (1.0, 100.0)
    spec_freq_range: Tuple[float, float] = (1.0, 50.0)


# ── channel mapping ──────────────────────────────────────────────────────────

@dataclass
class ChannelConfig:
    """Maps Percept device channel names --> readable labels."""

    labels: Dict[str, str] = field(default_factory=lambda: {
        # ZERO_THREE format
        "ZERO_THREE_LEFT": "L 0-3", "ONE_THREE_LEFT": "L 1-3",
        "ZERO_TWO_LEFT": "L 0-2",
        "ZERO_THREE_RIGHT": "R 0-3", "ONE_THREE_RIGHT": "R 1-3",
        "ZERO_TWO_RIGHT": "R 0-2",
        # ZERO_AND_THREE format (some firmware versions)
        "ZERO_AND_THREE_LEFT": "L 0-3", "ONE_AND_THREE_LEFT": "L 1-3",
        "ZERO_AND_TWO_LEFT": "L 0-2",
        "ZERO_AND_THREE_RIGHT": "R 0-3", "ONE_AND_THREE_RIGHT": "R 1-3",
        "ZERO_AND_TWO_RIGHT": "R 0-2",
    })

    def get_hemisphere(self, channel: str) -> str:
        ch = channel.upper()
        if "LEFT" in ch:
            return "left"
        elif "RIGHT" in ch:
            return "right"
        return "unknown"

    def get_label(self, channel: str) -> str:
        return self.labels.get(channel, channel)


# ── default instances ────────────────────────────────────────────────────────

FREQ_BANDS = FrequencyBands()
PROC_CONFIG = ProcessingConfig()
PLOT_CONFIG = PlotConfig()
CHANNEL_CONFIG = ChannelConfig()
