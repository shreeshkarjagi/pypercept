"""
Visualization for LFP analysis.

PSD plots, spectrograms, band power bars, timeline, event snapshots,
session dashboards, longitudinal PSDs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from .config import FREQ_BANDS, PLOT_CONFIG, CHANNEL_CONFIG
from .signal import psd_to_db


# ── style setup ──────────────────────────────────────────────────────────────

def set_style():
    """Set matplotlib defaults for clean publication-style figures."""
    import matplotlib as mpl

    plt.style.use("default")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["font.size"] = PLOT_CONFIG.font_size_tick
    mpl.rcParams["axes.titlesize"] = PLOT_CONFIG.font_size_title
    mpl.rcParams["axes.labelsize"] = PLOT_CONFIG.font_size_label
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["xtick.labelsize"] = PLOT_CONFIG.font_size_tick
    mpl.rcParams["ytick.labelsize"] = PLOT_CONFIG.font_size_tick
    mpl.rcParams["legend.fontsize"] = PLOT_CONFIG.font_size_legend
    mpl.rcParams["axes.linewidth"] = PLOT_CONFIG.line_width_axis
    mpl.rcParams["xtick.major.width"] = PLOT_CONFIG.line_width_tick
    mpl.rcParams["ytick.major.width"] = PLOT_CONFIG.line_width_tick
    mpl.rcParams["xtick.major.size"] = 3
    mpl.rcParams["ytick.major.size"] = 3
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.grid"] = False
    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["lines.linewidth"] = PLOT_CONFIG.line_width_data
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = PLOT_CONFIG.dpi
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.02


def _add_band_shading(ax: plt.Axes, freq_range: Tuple[float, float], alpha: float = 0.12):
    """Shade standard frequency band regions behind data."""
    bands = FREQ_BANDS.get_standard()
    colors = PLOT_CONFIG.band_colors
    for name, (low, high) in bands.items():
        low, high = max(low, freq_range[0]), min(high, freq_range[1])
        if low < high:
            ax.axvspan(low, high, alpha=alpha, color=colors.get(name, "gray"), zorder=0)


# ── PSD ──────────────────────────────────────────────────────────────────────

def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    ax: Optional[plt.Axes] = None,
    channel_names: Optional[List[str]] = None,
    freq_range: Tuple[float, float] = (1, 100),
    log_scale: bool = True,
    show_bands: bool = True,
    title: Optional[str] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot PSD. Handles single channel (1D) or multi-channel (2D) arrays.

    >>> freqs, psd = compute_psd(data, fs=250)
    >>> plot_psd(freqs, psd, show_bands=True, freq_range=(1, 50))
    """
    if ax is None:
        _, ax = plt.subplots(figsize=PLOT_CONFIG.figsize)

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_plot = freqs[mask]

    if psd.ndim == 1:
        psd_plot = psd_to_db(psd[mask]) if log_scale else psd[mask]
        ax.plot(freqs_plot, psd_plot, linewidth=1.5, **kwargs)
    else:
        colors = PLOT_CONFIG.channel_colors
        for i, ch_psd in enumerate(psd):
            psd_plot = psd_to_db(ch_psd[mask]) if log_scale else ch_psd[mask]
            label = CHANNEL_CONFIG.get_label(channel_names[i]) if channel_names else None
            ax.plot(freqs_plot, psd_plot, color=colors[i % len(colors)],
                    linewidth=1.5, label=label, **kwargs)
        if channel_names:
            ax.legend(fontsize=9, loc="upper right")

    if show_bands:
        _add_band_shading(ax, freq_range)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)" if log_scale else "Power (µV²/Hz)")
    ax.set_xlim(freq_range)
    if title:
        ax.set_title(title, fontsize=11)
    return ax


def plot_psd_by_hemisphere(
    freqs: np.ndarray,
    psds: Dict[str, np.ndarray],
    channel_names: Dict[str, List[str]],
    freq_range: Tuple[float, float] = (1, 50),
    log_scale: bool = True,
    show_mean: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Left/right hemisphere PSD panels side by side."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    hemi_labels = {"right": "Right Hemisphere", "left": "Left Hemisphere"}

    for ax, hemi in zip(axes, ["right", "left"]):
        if hemi not in psds or psds[hemi] is None or len(psds[hemi]) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(hemi_labels[hemi])
            continue

        hemi_psds = psds[hemi]
        ch_names = channel_names.get(hemi, [])
        colors = PLOT_CONFIG.channel_colors
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs_plot = freqs[mask]

        for i, ch_name in enumerate(ch_names):
            psd_plot = hemi_psds[i][mask] if hemi_psds.ndim > 1 else hemi_psds[mask]
            if log_scale:
                psd_plot = psd_to_db(psd_plot)
            ax.plot(freqs_plot, psd_plot, color=colors[i % len(colors)],
                    linewidth=1.5, alpha=0.7, label=CHANNEL_CONFIG.get_label(ch_name))

        if show_mean and hemi_psds.ndim > 1 and hemi_psds.shape[0] > 1:
            mean_psd = np.mean(hemi_psds[:, mask], axis=0)
            if log_scale:
                mean_psd = psd_to_db(mean_psd)
            ax.plot(freqs_plot, mean_psd, color="black", linewidth=2, linestyle="--", label="Mean")

        ax.set_title(hemi_labels[hemi], fontsize=11)
        ax.set_xlabel("Frequency (Hz)")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(freq_range)
        _add_band_shading(ax, freq_range)

    axes[0].set_ylabel("Power (dB)" if log_scale else "Power (µV²/Hz)")
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ── spectrogram ──────────────────────────────────────────────────────────────

def plot_spectrogram(
    times: np.ndarray,
    freqs: np.ndarray,
    Sxx: np.ndarray,
    ax: Optional[plt.Axes] = None,
    freq_range: Tuple[float, float] = (1, 50),
    cmap: str = "viridis",
    log_scale: bool = True,
    colorbar: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot spectrogram (time-frequency)."""
    if ax is None:
        _, ax = plt.subplots(figsize=PLOT_CONFIG.figsize)

    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_plot = freqs[freq_mask]
    Sxx_plot = Sxx[freq_mask, :] if Sxx.ndim == 2 else Sxx[:, freq_mask, :]
    if log_scale:
        Sxx_plot = psd_to_db(Sxx_plot)

    pcm = ax.pcolormesh(times, freqs_plot, Sxx_plot, shading="gouraud", cmap=cmap)
    if colorbar:
        cb = plt.colorbar(pcm, ax=ax)
        cb.set_label("Power (dB)" if log_scale else "Power")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(freq_range)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


# ── band powers ──────────────────────────────────────────────────────────────

def plot_band_powers(
    band_powers: Union[Dict[str, float], List[Dict[str, float]]],
    labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = False,
    title: Optional[str] = None,
) -> plt.Axes:
    """Bar chart of band powers. Pass a list of dicts to compare conditions."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    if isinstance(band_powers, dict):
        band_powers = [band_powers]
        labels = [""]

    band_names = list(band_powers[0].keys())
    n_bands = len(band_names)
    n_cond = len(band_powers)
    x = np.arange(n_bands)
    width = 0.8 / n_cond
    colors = PLOT_CONFIG.channel_colors

    for i, (bp, label) in enumerate(zip(band_powers, labels or [""] * n_cond)):
        powers = [bp[name] for name in band_names]
        if log_scale:
            powers = [psd_to_db(p) for p in powers]
        offset = (i - n_cond / 2 + 0.5) * width
        ax.bar(x + offset, powers, width, label=label if label else None,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Power (dB re: 1 µV²/Hz)" if log_scale else "Power (µV²/Hz)")
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    if labels and any(labels):
        ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    return ax


# ── timeline ─────────────────────────────────────────────────────────────────

def plot_timeline(
    datetimes: np.ndarray,
    lfp_power: np.ndarray,
    amplitude: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot chronic LFP power over time. Optional stim amplitude on secondary axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    ax.plot(datetimes, lfp_power, "o-", markersize=3, linewidth=1,
            color=PLOT_CONFIG.right_color, alpha=0.7, label="LFP Power")
    ax.set_xlabel("Date/Time")
    ax.set_ylabel("LFP Power")

    if amplitude is not None:
        ax2 = ax.twinx()
        ax2.plot(datetimes, amplitude, "s-", markersize=2, linewidth=1,
                 color=PLOT_CONFIG.left_color, alpha=0.7, label="Amplitude")
        ax2.set_ylabel("Amplitude (mA)", color=PLOT_CONFIG.left_color)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=30)
    if title:
        ax.set_title(title)
    return ax


# ── event snapshots ──────────────────────────────────────────────────────────

def plot_events(
    events: List[Dict],
    freq_range: Tuple[float, float] = (0, 50),
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot FFT snapshots from patient-triggered events."""
    n_events = len(events)
    if n_events == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No event data", ha="center", va="center")
        return fig

    n_cols = min(3, n_events)
    n_rows = (n_events + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False, sharex=True, sharey=True)

    for i, event in enumerate(events):
        ax = axes[i // n_cols, i % n_cols]
        ax.plot(event["frequency"], event["fft_data"], linewidth=1.5)
        dt_str = str(event.get("datetime", ""))[:10]
        ax.set_title(f"{event.get('event_name', f'Event {i}')}\n{dt_str}", fontsize=9)
        ax.set_xlim(freq_range)
        _add_band_shading(ax, freq_range)

    # hide empty subplots
    for i in range(n_events, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Power (µVp)")
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ── session dashboard ────────────────────────────────────────────────────────

def plot_session_dashboard(
    session_result,
    freq_range: Tuple[float, float] = (1, 50),
    time_window: float = 10.0,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Multi-panel dashboard for a processed session.

    Shows: per-channel PSDs, mean PSD, band powers, spectrograms (if computed),
    and time series snippet.
    """
    trials = session_result.channels
    n_channels = len(trials)

    if n_channels == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No channels in session", ha="center", va="center", fontsize=14)
        ax.set_title(f"Session: {session_result.filepath.name}")
        return fig

    n_psd_cols = min(n_channels + 1, 4)
    has_spec = trials[0].spec_Sxx is not None
    colors = PLOT_CONFIG.channel_colors

    if has_spec:
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 0.8, 0.6], hspace=0.35, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.6], hspace=0.35, wspace=0.3)

    # row 1: individual PSDs + mean
    for i in range(min(n_channels, 3)):
        ax = fig.add_subplot(gs[0, i])
        trial = trials[i]
        mask = (trial.freqs >= freq_range[0]) & (trial.freqs <= freq_range[1])
        freqs_plot = trial.freqs[mask]
        psd_plot = psd_to_db(trial.psd[mask])
        ax.plot(freqs_plot, psd_plot, color=colors[i % len(colors)], linewidth=1.5)
        _add_band_shading(ax, freq_range)
        ax.set_title(CHANNEL_CONFIG.get_label(trial.channel), fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        if i == 0:
            ax.set_ylabel("Power (dB)")
        ax.set_xlim(freq_range)

    # mean PSD
    ax_mean = fig.add_subplot(gs[0, 3])
    freqs = trials[0].freqs
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_plot = freqs[mask]
    all_psds = np.array([t.psd for t in trials])

    for i, trial in enumerate(trials):
        ax_mean.plot(freqs_plot, psd_to_db(trial.psd[mask]), color=colors[i % len(colors)],
                     alpha=0.3, linewidth=1, label=CHANNEL_CONFIG.get_label(trial.channel))
    ax_mean.plot(freqs_plot, psd_to_db(np.mean(all_psds[:, mask], axis=0)),
                 color="black", linewidth=2.5, label="Mean")
    _add_band_shading(ax_mean, freq_range)
    ax_mean.set_title("Mean PSD (all channels)", fontsize=10)
    ax_mean.set_xlabel("Frequency (Hz)")
    ax_mean.set_xlim(freq_range)
    ax_mean.legend(fontsize=7, loc="upper right")

    # row 2: extra PSDs if >3 channels + band powers
    if n_channels > 3:
        for i in range(3, min(n_channels, 6)):
            ax = fig.add_subplot(gs[1, i - 3])
            trial = trials[i]
            mask = (trial.freqs >= freq_range[0]) & (trial.freqs <= freq_range[1])
            psd_plot = psd_to_db(trial.psd[mask])
            ax.plot(freqs_plot, psd_plot, color=colors[i % len(colors)], linewidth=1.5)
            _add_band_shading(ax, freq_range)
            ax.set_title(CHANNEL_CONFIG.get_label(trial.channel), fontsize=10)
            ax.set_xlabel("Frequency (Hz)")
            if i == 3:
                ax.set_ylabel("Power (dB)")
            ax.set_xlim(freq_range)

    # band power bar chart
    bp_pos = gs[1, 3] if n_channels > 3 else (gs[0, 2] if n_channels <= 2 else gs[1, 0])
    ax_bands = fig.add_subplot(bp_pos)
    mean_bp = session_result.mean_band_powers()
    if mean_bp:
        bands = list(mean_bp.keys())
        powers = [mean_bp[b] for b in bands]
        band_colors = [PLOT_CONFIG.band_colors.get(b, "gray") for b in bands]
        ax_bands.bar(bands, powers, color=band_colors, alpha=0.8)
        ax_bands.set_ylabel("Power (µV²/Hz)")
        ax_bands.set_title("Mean Band Powers", fontsize=10)
        ax_bands.tick_params(axis="x", rotation=45)

    # spectrograms (if computed)
    if has_spec:
        for i in range(min(n_channels, 3)):
            ax = fig.add_subplot(gs[2, i])
            trial = trials[i]
            freq_mask = (trial.spec_f >= freq_range[0]) & (trial.spec_f <= freq_range[1])
            Sxx_plot = psd_to_db(trial.spec_Sxx[freq_mask, :])
            ax.pcolormesh(trial.spec_t, trial.spec_f[freq_mask], Sxx_plot,
                          shading="gouraud", cmap="viridis")
            ax.set_ylabel("Freq (Hz)" if i == 0 else "")
            ax.set_xlabel("Time (s)")
            ax.set_title(f"Spectrogram: {CHANNEL_CONFIG.get_label(trial.channel)}", fontsize=9)
            ax.set_ylim(freq_range)

        # mean spectrogram
        ax_ms = fig.add_subplot(gs[2, 3])
        all_specs = np.array([t.spec_Sxx for t in trials if t.spec_Sxx is not None])
        if len(all_specs) > 0:
            mean_spec = np.mean(all_specs, axis=0)
            freq_mask = (trials[0].spec_f >= freq_range[0]) & (trials[0].spec_f <= freq_range[1])
            pcm = ax_ms.pcolormesh(trials[0].spec_t, trials[0].spec_f[freq_mask],
                                   psd_to_db(mean_spec[freq_mask, :]),
                                   shading="gouraud", cmap="viridis")
            ax_ms.set_xlabel("Time (s)")
            ax_ms.set_title("Mean Spectrogram", fontsize=9)
            ax_ms.set_ylim(freq_range)
            plt.colorbar(pcm, ax=ax_ms, label="dB")

    # time series row
    row_idx = 3 if has_spec else 1
    ax_ts = fig.add_subplot(gs[row_idx, :])
    for i, trial in enumerate(trials):
        n_samples = int(time_window * trial.fs)
        t = np.arange(min(n_samples, len(trial.data))) / trial.fs
        ax_ts.plot(t, trial.data[: len(t)] + i * 50, color=colors[i % len(colors)],
                   linewidth=0.5, alpha=0.8, label=CHANNEL_CONFIG.get_label(trial.channel))
    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("Amplitude (µV, offset)")
    ax_ts.set_title(f"Time Series (first {time_window}s)", fontsize=10)
    ax_ts.legend(loc="upper right", fontsize=8, ncol=min(n_channels, 6))
    ax_ts.set_xlim(0, time_window)

    date_str = session_result.date[:10] if session_result.date else "Unknown"
    fig.suptitle(f"Session: {session_result.filepath.name}\nDate: {date_str}",
                 fontsize=12, y=0.98)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")
    if show:
        plt.show()
    return fig


# ── batch dashboards ─────────────────────────────────────────────────────────

def generate_all_dashboards(data_dir, output_dir, pattern="*.json",
                            compute_spec=True, verbose=True):
    """Generate a dashboard PNG for every session JSON in a directory."""
    from pathlib import Path
    from .io import find_json_files, load_session_simple
    from .batch import process_session

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = find_json_files(data_dir, pattern)

    if verbose:
        print(f"Found {len(json_files)} JSON files")

    output_paths = []
    for filepath in json_files:
        try:
            session = load_session_simple(filepath)
            if not session.recordings:
                if verbose:
                    print(f"  skip {filepath.name}: no recordings")
                continue

            result = process_session(session, compute_spec=compute_spec)
            date_str = result.date[:10].replace("-", "") if result.date else "unknown"
            out = output_dir / f"dashboard_{date_str}_{filepath.stem}.png"
            plot_session_dashboard(result, output_path=str(out), show=False)
            output_paths.append(out)

            if verbose:
                print(f"  saved: {out.name}")
            plt.close("all")
        except Exception as e:
            if verbose:
                print(f"  ERROR {filepath.name}: {e}")

    if verbose:
        print(f"\nGenerated {len(output_paths)} dashboards in {output_dir}")
    return output_paths


# ── longitudinal PSD ─────────────────────────────────────────────────────────

def plot_longitudinal_psd(
    results,
    freq_range: Tuple[float, float] = (1, 50),
    colormap: str = "viridis",
    color_by: str = "date",
    log_scale: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 5),
    linewidth: float = 1.2,
    alpha: float = 0.9,
    show_grid: bool = True,
    legend_annotations: Optional[Dict[str, str]] = None,
    shade_bands: bool = False,
) -> plt.Figure:
    """
    Longitudinal PSDs — one subplot per channel, color = recording date.

    Great for tracking spectral changes over time across sessions.

    >>> results = lfp.process_directory("./data/")
    >>> fig = lfp.plot_longitudinal_psd(results, title="Longitudinal PSDs")
    """
    from datetime import datetime
    from matplotlib.colors import Normalize

    if not results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No results to plot", ha="center", va="center")
        return fig

    sorted_results = sorted(results, key=lambda r: r.date if r.date else "")
    n_sessions = len(sorted_results)

    # collect all unique channels
    all_channels = set()
    for result in sorted_results:
        for ch in result.channels:
            all_channels.add(ch.channel)

    def _channel_sort_key(ch):
        ch_upper = ch.upper()
        hemi = 0 if "LEFT" in ch_upper else (1 if "RIGHT" in ch_upper else 2)
        if "ZERO_THREE" in ch_upper:
            pair = 0
        elif "ONE_THREE" in ch_upper:
            pair = 1
        elif "ZERO_ONE" in ch_upper or "ZERO_TWO" in ch_upper:
            pair = 2
        else:
            pair = 3
        return (hemi, pair, ch)

    channels = sorted(all_channels, key=_channel_sort_key)
    n_channels = len(channels)

    if n_channels == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No channels found", ha="center", va="center")
        return fig

    # parse dates for coloring
    dates = []
    for r in sorted_results:
        try:
            dates.append(datetime.fromisoformat(r.date[:10]) if r.date else datetime.now())
        except (ValueError, TypeError):
            dates.append(datetime.now())

    cmap = plt.get_cmap(colormap)
    if color_by == "date" and dates:
        day_nums = [(d - min(dates)).days for d in dates]
        max_days = max(day_nums) if max(day_nums) > 0 else 1
        norm = Normalize(vmin=0, vmax=max_days)
        colors = [cmap(norm(d)) for d in day_nums]
    elif color_by == "week" and dates:
        weeks = [(d - min(dates)).days // 7 for d in dates]
        max_weeks = max(weeks) if max(weeks) > 0 else 1
        norm = Normalize(vmin=0, vmax=max_weeks)
        colors = [cmap(norm(w)) for w in weeks]
    else:
        norm = Normalize(vmin=0, vmax=n_sessions - 1 if n_sessions > 1 else 1)
        colors = [cmap(norm(i)) for i in range(n_sessions)]

    # layout: group by hemisphere if both present
    left_ch = [ch for ch in channels if "LEFT" in ch.upper()]
    right_ch = [ch for ch in channels if "RIGHT" in ch.upper()]
    has_hemispheres = bool(left_ch) and bool(right_ch)

    if has_hemispheres:
        n_cols = max(len(left_ch), len(right_ch))
        n_rows = 2
        ordered = left_ch + right_ch
    else:
        import math
        n_cols = min(n_channels, 3)
        n_rows = math.ceil(n_channels / n_cols)
        ordered = channels

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    # map channels --> subplot positions
    ch_to_ax = {}
    if has_hemispheres:
        for i, ch in enumerate(left_ch):
            ch_to_ax[ch] = axes_flat[i]
        for i, ch in enumerate(right_ch):
            ch_to_ax[ch] = axes_flat[n_cols + i]
    else:
        for i, ch in enumerate(ordered):
            ch_to_ax[ch] = axes_flat[i]

    # plot each channel
    for channel in channels:
        if channel not in ch_to_ax:
            continue
        ax = ch_to_ax[channel]

        for i, (result, color, dt) in enumerate(zip(sorted_results, colors, dates)):
            ch_result = next((ch for ch in result.channels if ch.channel == channel), None)
            if ch_result is None:
                continue

            freqs = ch_result.freqs
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            psd_plot = psd_to_db(ch_result.psd[mask]) if log_scale else ch_result.psd[mask]
            ax.plot(freqs[mask], psd_plot, color=color, linewidth=linewidth, alpha=alpha)

        if shade_bands:
            for band_range, color_hex in [((4, 8), "#E8F4FD"), ((8, 13), "#FFF3E0"), ((13, 30), "#F3E5F5")]:
                if band_range[0] >= freq_range[0] and band_range[1] <= freq_range[1]:
                    ax.axvspan(band_range[0], band_range[1], alpha=0.3, color=color_hex, zorder=0)

        if show_grid:
            ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, zorder=0)

        ax.set_title(CHANNEL_CONFIG.get_label(channel), fontweight="semibold")
        ax.set_xlim(freq_range)
        ax.set_xlabel("Frequency (Hz)")

    for row in range(n_rows):
        axes[row, 0].set_ylabel("Power (dB)" if log_scale else "Power (µV²/Hz)")

    # hide unused subplots
    for i in range(len(ch_to_ax), len(axes_flat)):
        axes_flat[i].set_visible(False)

    if has_hemispheres:
        fig.text(0.01, 0.73, "Left", fontsize=PLOT_CONFIG.font_size_label,
                 fontweight="bold", rotation=90, va="center")
        fig.text(0.01, 0.27, "Right", fontsize=PLOT_CONFIG.font_size_label,
                 fontweight="bold", rotation=90, va="center")

    # legend
    legend_width = 0.22 if legend_annotations else 0.07
    legend_left = 0.79 if legend_annotations else 0.91
    legend_ax = fig.add_axes([legend_left, 0.12, legend_width, 0.76])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, n_sessions + 1)
    legend_ax.axis("off")

    for i, (color, dt) in enumerate(zip(colors, dates)):
        y = n_sessions - i
        date_str = dt.strftime("%m/%d")
        legend_ax.scatter(0.05, y, color=color, s=30, edgecolors="none")
        legend_ax.text(0.12, y, date_str, fontsize=PLOT_CONFIG.font_size_legend, va="center")
        if legend_annotations and date_str in legend_annotations:
            legend_ax.text(0.32, y, legend_annotations[date_str],
                           fontsize=PLOT_CONFIG.font_size_legend, va="center",
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="#E0E0E0",
                                     edgecolor="none", alpha=0.9))

    if title:
        fig.suptitle(title, fontsize=PLOT_CONFIG.font_size_title, fontweight="bold", y=0.98)

    right_margin = 0.77 if legend_annotations else 0.89
    plt.subplots_adjust(left=0.09, right=right_margin, top=0.93, bottom=0.10, hspace=0.35, wspace=0.08)
    return fig


# set style on import
set_style()
