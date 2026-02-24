"""
Signal processing for LFP data.

Core functions:
    preprocess(data, fs) --> filtered data
    compute_psd(data, fs) --> (freqs, psd)
    compute_spectrogram(data, fs) --> (t, f, Sxx)
    extract_band_power(psd, freqs, band) --> float
"""

import numpy as np
from scipy import signal as sig
from typing import Dict, Tuple, Optional, Union

from .core import BANDS


# ── filtering ────────────────────────────────────────────────────────────────

def notch_filter(data: np.ndarray, fs: float = 250, freq: float = 60, Q: float = 30) -> np.ndarray:
    """Remove line noise. Use freq=60 for US, freq=50 for EU."""
    b, a = sig.iirnotch(freq, Q, fs)
    return sig.filtfilt(b, a, data)


def bandpass_filter(data: np.ndarray, fs: float = 250, low: float = 1, high: float = 100) -> np.ndarray:
    """4th order Butterworth bandpass."""
    nyq = fs / 2
    low_norm = max(0.001, low / nyq)
    high_norm = min(0.999, high / nyq)
    b, a = sig.butter(4, [low_norm, high_norm], btype="band")
    return sig.filtfilt(b, a, data)


def preprocess(data: np.ndarray, fs: float = 250, notch: bool = True, bandpass: bool = True) -> np.ndarray:
    """Standard preprocessing: notch + bandpass. Returns a copy."""
    result = data.copy()
    if notch:
        result = notch_filter(result, fs)
    if bandpass:
        result = bandpass_filter(result, fs)
    return result


# ── PSD ──────────────────────────────────────────────────────────────────────

def compute_psd(
    data: np.ndarray,
    fs: float = 250,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD. Returns (freqs, psd) in Hz and µV²/Hz.

    >>> freqs, psd = compute_psd(data, fs=250)
    >>> plt.semilogy(freqs, psd)
    """
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = sig.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="linear")
    return freqs, psd


def compute_spectrogram(
    data: np.ndarray,
    fs: float = 250,
    nperseg: int = 250,
    noverlap: Optional[int] = None,
    freq_range: Tuple[float, float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spectrogram via scipy. Returns (t, f, Sxx).

    >>> t, f, Sxx = compute_spectrogram(data, fs=250)
    >>> plt.pcolormesh(t, f, 10*np.log10(Sxx))
    """
    if noverlap is None:
        noverlap = nperseg // 2

    f, t, Sxx = sig.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)

    if freq_range:
        mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f, Sxx = f[mask], Sxx[mask, :]

    return t, f, Sxx


# ── band power extraction ───────────────────────────────────────────────────

def extract_band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    band: Union[str, Tuple[float, float]],
) -> float:
    """
    Mean power in a frequency band.

    band can be a name ('beta1') or a tuple ((13, 20)).

    >>> beta = extract_band_power(psd, freqs, 'beta1')
    >>> custom = extract_band_power(psd, freqs, (8, 12))
    """
    if isinstance(band, str):
        low, high = BANDS[band]
    else:
        low, high = band

    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return np.nan
    return np.mean(psd[mask])


def extract_all_bands(psd: np.ndarray, freqs: np.ndarray) -> Dict[str, float]:
    """Extract power for all 6 standard bands. Returns dict."""
    return {name: extract_band_power(psd, freqs, name) for name in BANDS}


def psd_to_db(psd: np.ndarray) -> np.ndarray:
    """Convert PSD to dB. Values < 1 µV²/Hz --> negative dB (that's correct)."""
    return 10 * np.log10(np.maximum(psd, 1e-10))


# ── epoching ─────────────────────────────────────────────────────────────────

def epoch(data: np.ndarray, fs: float, duration: float = 1.0, overlap: float = 0) -> np.ndarray:
    """
    Chop continuous data into fixed-length epochs.

    Returns array of shape (n_epochs, epoch_samples).
    """
    epoch_samples = int(duration * fs)
    step = int(epoch_samples * (1 - overlap))
    n_epochs = (len(data) - epoch_samples) // step + 1

    epochs = np.zeros((n_epochs, epoch_samples))
    for i in range(n_epochs):
        start = i * step
        epochs[i] = data[start : start + epoch_samples]
    return epochs


def reject_bad_epochs(epochs: np.ndarray, threshold: float = 500) -> np.ndarray:
    """Drop epochs where any sample exceeds ±threshold µV."""
    max_amp = np.max(np.abs(epochs), axis=1)
    return epochs[max_amp <= threshold]
