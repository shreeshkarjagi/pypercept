"""
I/O for loading Percept PC/RC JSON session files.

Supports all BrainSense modalities:
    - IndefiniteStreaming: raw time-domain, 6 channels, stim OFF
    - BrainSenseTimeDomain + BrainSenseLfp: streaming with stim, 1-2 channels
    - Timeline (LFPTrendLogs): chronic 10-min power averages
    - Events (LfpFrequencySnapshotEvents): patient-triggered FFT snapshots

Key difference: IndefiniteStreaming gives you all 6 sensing pairs at once (stim OFF),
while BrainSenseTimeDomain gives 1-2 channels but can have stim ON and comes with
companion BrainSenseLfp device-computed power data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import CHANNEL_CONFIG, PROC_CONFIG

logger = logging.getLogger(__name__)

# Percept tick counter rolls over at this value (~54.6 min)
TICK_ROLLOVER = 3276750


# ── data containers ──────────────────────────────────────────────────────────

@dataclass
class StreamingData:
    """One channel of time-domain data (IndefiniteStreaming or BrainSenseTimeDomain)."""

    channel: str
    hemisphere: str
    data: np.ndarray
    sample_rate: float
    first_packet_datetime: str
    global_sequences: Optional[List[int]] = None
    global_packet_sizes: Optional[List[int]] = None
    ticks_in_ms: Optional[List[int]] = None
    gain: Optional[float] = None
    source: str = "unknown"          # 'indefinite' or 'brainsense'
    trial_number: Optional[int] = None  # for multi-trial BrainSenseTimeDomain

    @property
    def duration_seconds(self) -> float:
        return len(self.data) / self.sample_rate

    @property
    def n_samples(self) -> int:
        return len(self.data)

    @property
    def time_vector(self) -> np.ndarray:
        return np.arange(len(self.data)) / self.sample_rate

    def __repr__(self) -> str:
        return (f"StreamingData('{self.channel}', {self.n_samples} samples, "
                f"{self.duration_seconds:.1f}s, source='{self.source}')")


@dataclass
class BrainSenseLfpData:
    """Device-computed LFP power snapshots (companion to BrainSenseTimeDomain).

    One instance per (session, hemisphere). A session here = one entry in the
    top-level `BrainSenseLfp` list, identified by its `FirstPacketDateTime`.
    """

    hemisphere: str
    ticks_in_ms: np.ndarray
    lfp_power: np.ndarray
    stim_amplitude: np.ndarray
    sample_rate: float = 2.0                 # BrainSenseLfp is typically 2 Hz
    frequency_band: Optional[float] = None
    therapy_snapshot: Optional[Dict] = None
    first_packet_datetime: Optional[str] = None
    trial_number: Optional[int] = None

    @property
    def n_samples(self) -> int:
        return len(self.lfp_power)

    @property
    def time_vector(self) -> np.ndarray:
        """Seconds since the first sample, derived from TicksInMs (rollover-corrected)."""
        if self.ticks_in_ms is None or len(self.ticks_in_ms) == 0:
            return np.array([])
        t = _handle_tick_rollover(np.asarray(self.ticks_in_ms, dtype=np.float64))
        return (t - t[0]) / 1000.0

    def __repr__(self) -> str:
        return f"BrainSenseLfpData('{self.hemisphere}', {self.n_samples} samples)"


@dataclass
class TimelineData:
    """Chronic LFP tracking data (LFPTrendLogs). 10-min averages over days/weeks."""

    hemisphere: str
    datetime: List[str]
    lfp_power: List[float]
    amplitude_ma: List[float]
    group_id: Optional[str] = None

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "datetime": pd.to_datetime(self.datetime),
            "lfp_power": self.lfp_power,
            "amplitude_ma": self.amplitude_ma,
            "hemisphere": self.hemisphere,
        })

    def __repr__(self) -> str:
        return f"TimelineData('{self.hemisphere}', {len(self.datetime)} records)"


@dataclass
class EventSnapshot:
    """Patient-triggered FFT snapshot (LfpFrequencySnapshotEvents)."""

    datetime: str
    event_name: str
    hemisphere: str
    sense_id: str
    group_id: str
    fft_bin_data: np.ndarray
    frequency: np.ndarray

    @property
    def freq_resolution(self) -> float:
        if len(self.frequency) > 1:
            return self.frequency[1] - self.frequency[0]
        return 0.98  # default Percept resolution

    def __repr__(self) -> str:
        return (f"EventSnapshot('{self.event_name}', "
                f"'{self.hemisphere}', {self.datetime[:10]})")


@dataclass
class PerceptSession:
    """
    Everything from a single Percept JSON session file.

    Use load_session() to get one of these, then access whichever modalities
    are available via the attributes below.
    """

    filepath: Path
    session_date: str
    device_info: Dict[str, Any]
    utc_offset: str

    indefinite_streaming: List[StreamingData] = field(default_factory=list)
    brainsense_timedomain: List[StreamingData] = field(default_factory=list)
    brainsense_lfp: List[BrainSenseLfpData] = field(default_factory=list)
    timeline_data: List[TimelineData] = field(default_factory=list)
    event_snapshots: List[EventSnapshot] = field(default_factory=list)

    groups: Optional[Dict] = None
    impedance: Optional[List] = None

    # ── convenience accessors ────────────────────────────────────────────

    def get_streaming_channels(self, modality: str = "indefinite") -> Dict[str, StreamingData]:
        if modality == "indefinite":
            data_list = self.indefinite_streaming
        elif modality == "brainsense":
            data_list = self.brainsense_timedomain
        else:
            data_list = self.indefinite_streaming + self.brainsense_timedomain
        return {d.channel: d for d in data_list}

    def get_channels_by_hemisphere(self, modality: str = "indefinite") -> Dict[str, List[StreamingData]]:
        channels = self.get_streaming_channels(modality)
        result: Dict[str, List[StreamingData]] = {"left": [], "right": []}
        for data in channels.values():
            if data.hemisphere in result:
                result[data.hemisphere].append(data)
        return result

    def get_all_streaming(self) -> List[StreamingData]:
        return self.indefinite_streaming + self.brainsense_timedomain

    def get_aligned_streaming(self, hemisphere: str = "right") -> Optional[Tuple[StreamingData, BrainSenseLfpData, np.ndarray]]:
        """Get BrainSenseTimeDomain aligned with BrainSenseLfp via tick timestamps."""
        stream = next((s for s in self.brainsense_timedomain if s.hemisphere == hemisphere), None)
        lfp = next((l for l in self.brainsense_lfp if l.hemisphere == hemisphere), None)
        if stream is None or lfp is None:
            return None
        aligned_idx = align_by_ticks(stream.ticks_in_ms, lfp.ticks_in_ms)
        return stream, lfp, aligned_idx

    def get_sessions(self) -> List[Dict[str, Any]]:
        """
        Group BrainSenseTimeDomain + BrainSenseLfp into one dict per session.
        Each entry looks like:

            {"datetime": "...Z", "trial_number": 1,
             "td":  {"left": StreamingData|None, "right": StreamingData|None},
             "lfp": {"left": BrainSenseLfpData|None,
                     "right": BrainSenseLfpData|None}}

        Matching prefers `trial_number` (assigned by the 60 s-gap rule in
        `_assign_trial_numbers`) over raw timestamp equality, because TD and
        LFP blocks for the same trial can differ by ~1 s in their
        FirstPacketDateTime fields.
        """
        sessions: Dict[Any, Dict[str, Any]] = {}

        def _key(item, prefix):
            if item.trial_number is not None:
                return ("trial", item.trial_number)
            return (prefix, item.first_packet_datetime or id(item))

        def _bucket(key, dt, trial):
            return sessions.setdefault(key, {
                "datetime": dt,
                "trial_number": trial,
                "td": {"left": None, "right": None},
                "lfp": {"left": None, "right": None},
            })

        for s in self.brainsense_timedomain:
            sess = _bucket(_key(s, "td"), s.first_packet_datetime, s.trial_number)
            if s.hemisphere in sess["td"]:
                sess["td"][s.hemisphere] = s

        for l in self.brainsense_lfp:
            sess = _bucket(_key(l, "lfp"), l.first_packet_datetime, l.trial_number)
            if l.hemisphere in sess["lfp"]:
                sess["lfp"][l.hemisphere] = l

        return sorted(
            sessions.values(),
            key=lambda s: (s["trial_number"] is None, s["trial_number"], s["datetime"] or ""),
        )

    def has_indefinite_streaming(self) -> bool:
        return len(self.indefinite_streaming) > 0

    def has_brainsense_timedomain(self) -> bool:
        return len(self.brainsense_timedomain) > 0

    def has_brainsense_lfp(self) -> bool:
        return len(self.brainsense_lfp) > 0

    def has_timeline_data(self) -> bool:
        return len(self.timeline_data) > 0

    def has_event_snapshots(self) -> bool:
        return len(self.event_snapshots) > 0

    def summary(self) -> str:
        lines = [
            f"Percept Session: {self.filepath.name}",
            f"  Date: {self.session_date}",
            f"  UTC Offset: {self.utc_offset}",
            f"  Device: {self.device_info.get('Neurostimulator', 'Unknown')}",
            "",
        ]
        if self.has_indefinite_streaming():
            n = len(self.indefinite_streaming)
            dur = self.indefinite_streaming[0].duration_seconds
            lines.append(f"  IndefiniteStreaming: {n} channels, {dur:.1f}s (stim OFF)")
        if self.has_brainsense_timedomain():
            n = len(self.brainsense_timedomain)
            trials = len(set(s.trial_number for s in self.brainsense_timedomain if s.trial_number))
            lines.append(f"  BrainSenseTimeDomain: {n} recordings ({trials} trials)")
        if self.has_brainsense_lfp():
            n = len(self.brainsense_lfp)
            samples = sum(l.n_samples for l in self.brainsense_lfp)
            lines.append(f"  BrainSenseLfp: {n} hemispheres, {samples} total samples")
        if self.has_timeline_data():
            n = sum(len(t.datetime) for t in self.timeline_data)
            lines.append(f"  Timeline: {n} records")
        if self.has_event_snapshots():
            n = len(self.event_snapshots)
            lines.append(f"  Events: {n} FFT snapshots")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PerceptSession('{self.filepath.name}')"


# ── JSON loading ─────────────────────────────────────────────────────────────

def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    logger.info(f"Loading: {filepath.name}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ── internal helpers ─────────────────────────────────────────────────────────

def _parse_sequence_string(seq_string: str) -> List[int]:
    """Parse comma-separated sequence string --> list of ints."""
    if not seq_string:
        return []
    parts = [p.strip() for p in seq_string.split(",") if p.strip()]
    return [int(p) for p in parts]


def _handle_tick_rollover(ticks: np.ndarray) -> np.ndarray:
    """
    Correct tick counter rollover. The TicksInMs counter rolls over at
    TICK_ROLLOVER (~54.6 min). Detect large negative jumps and add offset.
    """
    if len(ticks) == 0:
        return ticks
    ticks = np.array(ticks, dtype=np.float64)
    corrected = ticks.copy()
    diffs = np.diff(ticks)
    rollover_idx = np.where(diffs < -TICK_ROLLOVER / 2)[0]
    for idx in rollover_idx:
        corrected[idx + 1 :] += TICK_ROLLOVER
    return corrected


def align_by_ticks(
    td_ticks: Optional[List[int]],
    lfp_ticks: np.ndarray,
    sample_rate: float = 250.0,
) -> np.ndarray:
    """
    Align time-domain sample indices with LFP power snapshots using tick timestamps.

    Returns array of time-domain indices corresponding to each LFP sample.
    """
    if td_ticks is None or len(td_ticks) == 0:
        logger.warning("No tick data for alignment")
        return np.array([])

    td_ticks = _handle_tick_rollover(np.array(td_ticks))
    lfp_ticks = _handle_tick_rollover(lfp_ticks)

    aligned = np.zeros(len(lfp_ticks), dtype=np.int64)
    for i, tick in enumerate(lfp_ticks):
        aligned[i] = np.argmin(np.abs(td_ticks - tick))
    return aligned


def _assign_trial_numbers(data_list: List[Dict], datetime_key: str = "FirstPacketDateTime") -> List[Dict]:
    """
    Group BrainSenseTimeDomain recordings into trials based on timestamps.
    Recordings starting within 60s of each other get the same trial number.
    """
    if not data_list:
        return data_list

    sorted_data = sorted(data_list, key=lambda x: x.get(datetime_key, ""))
    trial_num = 0
    prev_time = None

    for item in sorted_data:
        dt_str = item.get(datetime_key, "")
        if dt_str:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if prev_time is None or (dt - prev_time).total_seconds() > 60:
                    trial_num += 1
                prev_time = dt
            except (ValueError, TypeError):
                trial_num += 1
        item["_trial_number"] = trial_num

    return sorted_data


# ── extraction functions (one per modality) ──────────────────────────────────

def _extract_indefinite_streaming(data: Dict) -> List[StreamingData]:
    """Extract IndefiniteStreaming: all 6 channels, stim OFF, no LFP companion."""
    if "IndefiniteStreaming" not in data:
        return []

    streams = data["IndefiniteStreaming"]
    logger.info(f"Found {len(streams)} IndefiniteStreaming channels")

    result = []
    for stream in streams:
        channel = stream.get("Channel", "Unknown")
        result.append(StreamingData(
            channel=channel,
            hemisphere=CHANNEL_CONFIG.get_hemisphere(channel),
            data=np.array(stream.get("TimeDomainData", []), dtype=np.float64),
            sample_rate=float(stream.get("SampleRateInHz", PROC_CONFIG.sample_rate)),
            first_packet_datetime=stream.get("FirstPacketDateTime", ""),
            global_sequences=_parse_sequence_string(stream.get("GlobalSequences", "")) or None,
            global_packet_sizes=_parse_sequence_string(stream.get("GlobalPacketSizes", "")) or None,
            ticks_in_ms=_parse_sequence_string(stream.get("TicksInMses", "")) or None,
            gain=stream.get("Gain"),
            source="indefinite",
        ))
    return result


def _extract_brainsense_timedomain(data: Dict) -> List[StreamingData]:
    """Extract BrainSenseTimeDomain: 1-2 channels, can have stim ON."""
    if "BrainSenseTimeDomain" not in data:
        return []

    streams = data["BrainSenseTimeDomain"]
    logger.info(f"Found {len(streams)} BrainSenseTimeDomain recordings")
    streams = _assign_trial_numbers(streams, "FirstPacketDateTime")

    result = []
    for stream in streams:
        channel = stream.get("Channel", "Unknown")
        result.append(StreamingData(
            channel=channel,
            hemisphere=CHANNEL_CONFIG.get_hemisphere(channel),
            data=np.array(stream.get("TimeDomainData", []), dtype=np.float64),
            sample_rate=float(stream.get("SampleRateInHz", PROC_CONFIG.sample_rate)),
            first_packet_datetime=stream.get("FirstPacketDateTime", ""),
            global_sequences=_parse_sequence_string(stream.get("GlobalSequences", "")) or None,
            global_packet_sizes=_parse_sequence_string(stream.get("GlobalPacketSizes", "")) or None,
            ticks_in_ms=_parse_sequence_string(stream.get("TicksInMses", "")) or None,
            gain=stream.get("Gain"),
            source="brainsense",
            trial_number=stream.get("_trial_number"),
        ))
    return result


# LFP sentinel returned by the device for "no valid reading" (0xFFFFFFFF).
_LFP_INVALID_SENTINEL = 4294967295


def _extract_brainsense_lfp(data: Dict) -> List[BrainSenseLfpData]:
    """
    Extract BrainSenseLfp: device-computed LFP power + stim amplitude that
    accompanies BrainSenseTimeDomain.

    Each entry in the top-level `BrainSenseLfp` list is one *session* with
    its own `FirstPacketDateTime` and a per-tick `LfpData` list:

        { "FirstPacketDateTime": ..., "SampleRateInHz": 2,
          "TherapySnapshot": {...},
          "LfpData": [
              {"TicksInMs": 2430150,
               "Left":  {"LFP": 1053, "mA": 6},
               "Right": {"LFP": 3691, "mA": 6}},
              ... ] }

    We emit one BrainSenseLfpData per (session, hemisphere), preserving session
    identity via `first_packet_datetime` and `trial_number` so callers can
    pair it with the matching BrainSenseTimeDomain streams.
    """
    if "BrainSenseLfp" not in data:
        return []

    lfp_list = _assign_trial_numbers(list(data["BrainSenseLfp"]), "FirstPacketDateTime")
    logger.info(f"Found {len(lfp_list)} BrainSenseLfp entries")

    result: List[BrainSenseLfpData] = []
    for entry in lfp_list:
        records = entry.get("LfpData") or []
        if not isinstance(records, list) or not records:
            continue

        fs = float(entry.get("SampleRateInHz", 2.0))
        dt = entry.get("FirstPacketDateTime", "")
        therapy = entry.get("TherapySnapshot")
        trial = entry.get("_trial_number")

        for side, key in (("left", "Left"), ("right", "Right")):
            ticks, lfp, stim = [], [], []
            for rec in records:
                hd = rec.get(key)
                if not isinstance(hd, dict):
                    continue
                ticks.append(rec.get("TicksInMs", 0))
                lfp.append(hd.get("LFP", 0))
                stim.append(hd.get("mA", 0))

            if not ticks:
                continue

            lfp_arr = np.asarray(lfp, dtype=np.float64)
            lfp_arr[lfp_arr == _LFP_INVALID_SENTINEL] = np.nan

            hemi_therapy = therapy.get(key) if isinstance(therapy, dict) else None

            result.append(BrainSenseLfpData(
                hemisphere=side,
                ticks_in_ms=np.asarray(ticks, dtype=np.float64),
                lfp_power=lfp_arr,
                stim_amplitude=np.asarray(stim, dtype=np.float64),
                sample_rate=fs,
                therapy_snapshot=hemi_therapy,
                first_packet_datetime=dt,
                trial_number=trial,
            ))

    logger.info(
        "BrainSenseLfp parsed: "
        + ", ".join(f"{l.hemisphere}@{l.first_packet_datetime}={l.n_samples}" for l in result)
    )
    return result


def _extract_timeline_data(data: Dict) -> List[TimelineData]:
    """Extract LFPTrendLogs (chronic power tracking)."""
    diag = data.get("DiagnosticData", {})
    if "LFPTrendLogs" not in diag:
        return []

    lfp_trends = diag["LFPTrendLogs"]
    logger.info(f"Timeline data for: {list(lfp_trends.keys())}")

    result = []
    for hemi_key, groups in lfp_trends.items():
        hemisphere = "right" if "Right" in hemi_key else "left"
        for group_date, records in groups.items():
            result.append(TimelineData(
                hemisphere=hemisphere,
                datetime=[r.get("DateTime", "") for r in records],
                lfp_power=[float(r.get("LFP", 0)) for r in records],
                amplitude_ma=[float(r.get("AmplitudeInMilliAmps", 0)) for r in records],
                group_id=group_date,
            ))
    return result


def _extract_event_snapshots(data: Dict) -> List[EventSnapshot]:
    """Extract LfpFrequencySnapshotEvents (patient-triggered FFT snapshots)."""
    diag = data.get("DiagnosticData", {})
    if "LfpFrequencySnapshotEvents" not in diag:
        return []

    events = diag["LfpFrequencySnapshotEvents"]
    logger.info(f"Found {len(events)} event entries")

    result = []
    for event in events:
        if "LfpFrequencySnapshotEvents" not in event:
            continue

        lfp_snap = event["LfpFrequencySnapshotEvents"]
        event_dt = event.get("DateTime", "")
        event_name = event.get("EventName", "Unknown")

        for hemi_key, hemi_data in lfp_snap.items():
            hemisphere = "right" if "Right" in hemi_key else "left"
            result.append(EventSnapshot(
                datetime=hemi_data.get("DateTime", event_dt),
                event_name=event_name,
                hemisphere=hemisphere,
                sense_id=hemi_data.get("SenseID", ""),
                group_id=hemi_data.get("GroupId", ""),
                fft_bin_data=np.array(hemi_data.get("FFTBinData", []), dtype=np.float64),
                frequency=np.array(hemi_data.get("Frequency", []), dtype=np.float64),
            ))

    logger.info(f"Extracted {len(result)} events with FFT data")
    return result


# ── main loading functions ───────────────────────────────────────────────────

def load_session(filepath: Union[str, Path]) -> PerceptSession:
    """
    Load a Percept JSON file --> PerceptSession with all modalities.

    This is the full-access loader. For batch processing, use load_session_simple().

    >>> session = load_session("Report_Session_20251014.json")
    >>> print(session.summary())
    """
    filepath = Path(filepath)
    data = load_json(filepath)

    session = PerceptSession(
        filepath=filepath,
        session_date=data.get("SessionDate", ""),
        utc_offset=data.get("ProgrammerUtcOffset", "+00:00"),
        device_info=data.get("DeviceInformation", {}).get("Initial", {}),
        indefinite_streaming=_extract_indefinite_streaming(data),
        brainsense_timedomain=_extract_brainsense_timedomain(data),
        brainsense_lfp=_extract_brainsense_lfp(data),
        timeline_data=_extract_timeline_data(data),
        event_snapshots=_extract_event_snapshots(data),
        groups=data.get("Groups"),
        impedance=data.get("Impedance"),
    )
    logger.info(f"\n{session.summary()}")
    return session


def find_json_files(directory: Union[str, Path], pattern: str = "*.json") -> List[Path]:
    """Find all JSON files in a directory, sorted newest first."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    files = list(directory.glob(pattern))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")
    return files


def percept_to_session(percept_session: PerceptSession, source: str = "indefinite") -> "Session":
    """Convert PerceptSession --> Session for batch processing."""
    from .core import Recording, Session

    if source == "indefinite":
        streams = percept_session.indefinite_streaming
    elif source == "brainsense":
        streams = percept_session.brainsense_timedomain
    else:
        raise ValueError(f"source must be 'indefinite' or 'brainsense', got '{source}'")

    recordings = [
        Recording(
            channel=s.channel, data=s.data,
            fs=s.sample_rate, datetime=s.first_packet_datetime,
        )
        for s in streams
    ]
    return Session(filepath=percept_session.filepath, date=percept_session.session_date, recordings=recordings)


def load_session_simple(filepath: Union[str, Path], source: str = "indefinite") -> "Session":
    """
    Load JSON --> simplified Session object for batch processing.

    >>> session = load_session_simple("report.json")
    >>> result = process_session(session)
    """
    return percept_to_session(load_session(filepath), source=source)


# ── BrainSense export ──────────────────────────────────────────────────────

def export_brainsense(
    filepath: Union[str, Path],
    outdir: Union[str, Path, None] = None,
) -> List[Path]:
    """
    Load a Percept JSON and export per-trial BrainSense data to CSV + PKL.

    For each trial writes three files:
      - ``{stem}_trial{N}_td.csv``   — raw time-domain waveform at 250 Hz,
        columns: ``time_s``, ``left_uV``, ``right_uV``
      - ``{stem}_trial{N}_lfp.csv``  — device-computed LFP power + stim mA
        at 2 Hz, columns: ``time_s``, ``left_lfp``, ``left_stim_mA``,
        ``right_lfp``, ``right_stim_mA``
      - ``{stem}_trial{N}.pkl``      — full trial bundle dict with both
        streams, metadata, and tick arrays for alignment.

    Parameters
    ----------
    filepath : path to a Percept JSON that contains BrainSenseTimeDomain.
    outdir   : directory for output files.  Defaults to the same directory
               as the input JSON.

    Returns
    -------
    List of Path objects for all files written.

    Example
    -------
    >>> from pypercept.io import export_brainsense
    >>> written = export_brainsense("Report_BaselineEEG_20240403.json")
    """
    import pickle

    filepath = Path(filepath)
    session = load_session(filepath)
    trials = session.get_sessions()

    if not trials:
        logger.warning(f"No BrainSense sessions in {filepath.name}")
        return []

    if outdir is None:
        outdir = filepath.parent
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = filepath.stem

    written: List[Path] = []
    for sess in trials:
        trial_num = sess["trial_number"] or 0
        td_l, td_r = sess["td"]["left"], sess["td"]["right"]
        lfp_l, lfp_r = sess["lfp"]["left"], sess["lfp"]["right"]

        # ── CSV 1: raw time-domain waveform at 250 Hz ──────────────
        td_ref = td_l or td_r
        if td_ref is not None:
            td_data: Dict[str, Any] = {"time_s": td_ref.time_vector}
            if td_l is not None:
                td_data["left_uV"] = td_l.data
            if td_r is not None:
                td_data["right_uV"] = td_r.data

            df_td = pd.DataFrame(td_data)
            td_csv = outdir / f"{stem}_trial{trial_num}_td.csv"
            df_td.to_csv(td_csv, index=False)
            written.append(td_csv)
            logger.info(f"Wrote {td_csv.name} ({len(df_td)} rows @ {td_ref.sample_rate} Hz)")

        # ── CSV 2: LFP + stim at 2 Hz ─────────────────────────────
        lfp_ref = lfp_l or lfp_r
        if lfp_ref is not None:
            lfp_data: Dict[str, Any] = {"time_s": lfp_ref.time_vector}
            if lfp_l is not None:
                lfp_data["left_lfp"] = lfp_l.lfp_power
                lfp_data["left_stim_mA"] = lfp_l.stim_amplitude
            if lfp_r is not None:
                lfp_data["right_lfp"] = lfp_r.lfp_power
                lfp_data["right_stim_mA"] = lfp_r.stim_amplitude

            df_lfp = pd.DataFrame(lfp_data)
            lfp_csv = outdir / f"{stem}_trial{trial_num}_lfp.csv"
            df_lfp.to_csv(lfp_csv, index=False)
            written.append(lfp_csv)
            logger.info(f"Wrote {lfp_csv.name} ({len(df_lfp)} rows @ {lfp_ref.sample_rate} Hz)")

        # ── PKL: full trial bundle ─────────────────────────────────
        bundle = {
            "filepath": str(filepath),
            "session_date": session.session_date,
            "trial_number": trial_num,
            "datetime": sess["datetime"],
            "td": {
                "left": {
                    "data": td_l.data,
                    "time_s": td_l.time_vector,
                    "fs": td_l.sample_rate,
                    "channel": td_l.channel,
                } if td_l else None,
                "right": {
                    "data": td_r.data,
                    "time_s": td_r.time_vector,
                    "fs": td_r.sample_rate,
                    "channel": td_r.channel,
                } if td_r else None,
            },
            "lfp": {
                "left": {
                    "lfp_power": lfp_l.lfp_power,
                    "stim_mA": lfp_l.stim_amplitude,
                    "time_s": lfp_l.time_vector,
                    "fs": lfp_l.sample_rate,
                    "ticks_ms": lfp_l.ticks_in_ms,
                } if lfp_l else None,
                "right": {
                    "lfp_power": lfp_r.lfp_power,
                    "stim_mA": lfp_r.stim_amplitude,
                    "time_s": lfp_r.time_vector,
                    "fs": lfp_r.sample_rate,
                    "ticks_ms": lfp_r.ticks_in_ms,
                } if lfp_r else None,
            },
        }

        pkl_path = outdir / f"{stem}_trial{trial_num}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(bundle, f)
        written.append(pkl_path)
        logger.info(f"Wrote {pkl_path.name}")

    return written
