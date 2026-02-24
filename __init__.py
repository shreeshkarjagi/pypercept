"""
perceptlfp — analysis toolkit for Medtronic Percept PC/RC LFP data.

Load Percept JSON session files, run spectral analysis, generate figures.

Quick start:
    >>> import perceptlfp as lfp
    >>> session = lfp.load_session("report.json")
    >>> result = lfp.process_session(session)
    >>> lfp.plot_session_dashboard(result)
"""

__version__ = "0.2.0"

# core data structures
from .core import (
    BANDS, Recording, Session, ChannelResult, TrialResult,
    SessionResult, get_hemisphere, get_channel_label,
)

# I/O
from .io import (
    load_session_simple as load_session,   # default: returns Session for batch compat
    load_session as load_session_full,     # full: returns PerceptSession
    load_json, find_json_files, percept_to_session,
    PerceptSession, StreamingData, BrainSenseLfpData,
    TimelineData, EventSnapshot, align_by_ticks,
)

# signal processing
from .signal import (
    preprocess, notch_filter, bandpass_filter,
    compute_psd, compute_spectrogram,
    extract_band_power, extract_all_bands, psd_to_db,
    epoch, reject_bad_epochs,
)

# batch processing
from .batch import (
    process_recording, process_session, process_directory,
    results_to_dataframe, aggregate_by_session, aggregate_psds,
    compute_group_stats, compare_groups,
)

# viz
from .viz import (
    plot_psd, plot_psd_by_hemisphere, plot_spectrogram,
    plot_band_powers, plot_timeline, plot_events,
    plot_session_dashboard, generate_all_dashboards,
    plot_longitudinal_psd, set_style,
)

# config
from .config import (
    FrequencyBands, ProcessingConfig, PlotConfig, ChannelConfig,
    FREQ_BANDS, PROC_CONFIG, PLOT_CONFIG, CHANNEL_CONFIG,
)

__all__ = [
    "__version__",
    # core
    "BANDS", "Recording", "Session", "ChannelResult", "TrialResult",
    "SessionResult", "get_hemisphere", "get_channel_label",
    # io
    "load_session", "load_session_full", "load_json", "find_json_files",
    "percept_to_session", "PerceptSession", "StreamingData",
    "BrainSenseLfpData", "TimelineData", "EventSnapshot", "align_by_ticks",
    # signal
    "preprocess", "notch_filter", "bandpass_filter",
    "compute_psd", "compute_spectrogram",
    "extract_band_power", "extract_all_bands", "psd_to_db",
    "epoch", "reject_bad_epochs",
    # batch
    "process_recording", "process_session", "process_directory",
    "results_to_dataframe", "aggregate_by_session", "aggregate_psds",
    "compute_group_stats", "compare_groups",
    # viz
    "plot_psd", "plot_psd_by_hemisphere", "plot_spectrogram",
    "plot_band_powers", "plot_timeline", "plot_events",
    "plot_session_dashboard", "generate_all_dashboards",
    "plot_longitudinal_psd", "set_style",
    # config
    "FrequencyBands", "ProcessingConfig", "PlotConfig", "ChannelConfig",
    "FREQ_BANDS", "PROC_CONFIG", "PLOT_CONFIG", "CHANNEL_CONFIG",
]
