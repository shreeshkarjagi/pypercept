from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd

from .core import Recording, Session, ChannelResult, SessionResult, BANDS
from .io import load_session_simple as load_session, find_json_files
from .signal import preprocess, compute_psd, compute_spectrogram, extract_all_bands


# ── single recording / session processing ────────────────────────────────────

def process_recording(
    rec: Recording,
    do_preprocess: bool = True,
    compute_spec: bool = False,
) -> ChannelResult:
    #filter --> PSD --> band powers
    data = preprocess(rec.data, rec.fs) if do_preprocess else rec.data

    freqs, psd = compute_psd(data, rec.fs)
    band_powers = extract_all_bands(psd, freqs)

    spec_t, spec_f, spec_Sxx = None, None, None
    if compute_spec:
        spec_t, spec_f, spec_Sxx = compute_spectrogram(data, rec.fs)

    return ChannelResult(
        channel=rec.channel, hemisphere=rec.hemisphere,
        data=data, fs=rec.fs,
        freqs=freqs, psd=psd, band_powers=band_powers,
        spec_t=spec_t, spec_f=spec_f, spec_Sxx=spec_Sxx,
    )


def process_session(
    session: Session,
    do_preprocess: bool = True,
    compute_spec: bool = False,
) -> SessionResult:
    #all recordings in a session.
    channels = [process_recording(rec, do_preprocess, compute_spec) for rec in session.recordings]
    return SessionResult(filepath=session.filepath, date=session.date, channels=channels)


def process_directory(
    directory: Union[str, Path],
    pattern: str = "*.json",
    require_recordings: bool = True,
    do_preprocess: bool = True,
    verbose: bool = True,
    source: str = "indefinite",
) -> List[SessionResult]:
    #all JSON files in a directory
    json_files = find_json_files(directory, pattern)
    if verbose:
        print(f"Found {len(json_files)} JSON files")

    results = []
    for filepath in json_files:
        try:
            session = load_session(filepath, source=source)
            if require_recordings and not session.recordings:
                if verbose:
                    print(f"  skip {filepath.name}: no recordings")
                continue

            result = process_session(session, do_preprocess=do_preprocess)
            results.append(result)
            if verbose:
                print(f"  {filepath.name}: {len(result.channels)} channels")
        except Exception as e:
            if verbose:
                print(f"  ERROR {filepath.name}: {e}")

    results.sort(key=lambda r: r.date)
    if verbose:
        print(f"Processed {len(results)} sessions")
    return results


# ── aggregation ──────────────────────────────────────────────────────────────

def results_to_dataframe(results: List[SessionResult]) -> pd.DataFrame:
    #Flatten results --> one row per channel per session
    rows = []
    for result in results:
        for ch in result.channels:
            rows.append({
                "filepath": str(result.filepath),
                "date": result.date,
                "channel": ch.channel,
                "hemisphere": ch.hemisphere,
                **ch.band_powers,
            })
    return pd.DataFrame(rows)


def aggregate_by_session(results: List[SessionResult], hemisphere: str = None) -> pd.DataFrame:
    #Average band powers across channels within each session
    rows = []
    for result in results:
        bp = result.mean_band_powers(hemisphere=hemisphere)
        if bp:
            rows.append({
                "filepath": str(result.filepath),
                "date": result.date,
                "hemisphere": hemisphere or "both",
                **bp,
            })
    return pd.DataFrame(rows)


def aggregate_psds(results: List[SessionResult], hemisphere: str = None) -> Dict[str, np.ndarray]:
    #Collect all PSDs. Returns dict with 'freqs', 'psds' (n_sessions x n_freqs), 'dates'
    freqs = None
    psds, dates = [], []
    for result in results:
        f, p = result.mean_psd(hemisphere=hemisphere)
        if f is not None:
            freqs = f
            psds.append(p)
            dates.append(result.date)
    return {"freqs": freqs, "psds": np.array(psds) if psds else None, "dates": dates}


# ── stats ────────────────────────────────────────────────────────────────────

def compute_group_stats(df: pd.DataFrame, group_col: str, value_cols: List[str] = None) -> pd.DataFrame:
    #Mean ± std for each group.
    if value_cols is None:
        value_cols = list(BANDS.keys())
    value_cols = [c for c in value_cols if c in df.columns]
    return df.groupby(group_col)[value_cols].agg(["mean", "std", "count"])


def compare_groups(
    df: pd.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
    value_cols: List[str] = None,
) -> pd.DataFrame:
    #Compare two groups with independent t-test.
    from scipy import stats

    if value_cols is None:
        value_cols = list(BANDS.keys())
    value_cols = [c for c in value_cols if c in df.columns]

    g1 = df[df[group_col] == group1]
    g2 = df[df[group_col] == group2]

    rows = []
    for col in value_cols:
        v1 = g1[col].dropna().values
        v2 = g2[col].dropna().values
        t_stat, p_val = stats.ttest_ind(v1, v2) if len(v1) > 1 and len(v2) > 1 else (np.nan, np.nan)
        m1, m2 = np.mean(v1), np.mean(v2)
        rows.append({
            "band": col,
            f"mean_{group1}": m1,
            f"mean_{group2}": m2,
            "difference": m2 - m1,
            "percent_change": 100 * (m2 - m1) / m1 if m1 != 0 else np.nan,
            "t_statistic": t_stat,
            "p_value": p_val,
        })
    return pd.DataFrame(rows)
