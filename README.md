# pypercept

Analysis toolkit for Medtronic Percept PC/RC Local Field Potential (LFP) data.

Loads JSON session reports exported from the Percept programmer, runs standard spectral analysis, and generates publication-ready figures.

## Supported data modalities

| Modality | JSON key | What it is | Channels | Stim |
|---|---|---|---|---|
| **IndefiniteStreaming** | `IndefiniteStreaming` | Raw time-domain recording | 6 (all sensing pairs) | OFF |
| **BrainSenseTimeDomain** | `BrainSenseTimeDomain` | Streaming with optional stim | 1–2 | ON or OFF |
| **BrainSenseLfp** | `BrainSenseLfp` | Device-computed LFP power (companion to TimeDomain) | 1–2 | ON or OFF |
| **Timeline** | `DiagnosticData.LFPTrendLogs` | Chronic 10-min power averages | per hemisphere | ON |
| **Events** | `DiagnosticData.LfpFrequencySnapshotEvents` | Patient-triggered FFT snapshots | per hemisphere | ON |

## Install

```bash
pip install numpy scipy pandas matplotlib
```

Then either install as a package or just drop the `pypercept/` folder next to your scripts.

## Quick start

### Load and process a single session

```python
import pypercept as lfp

# simple load --> returns Session object ready for processing
session = lfp.load_session("Report_Session_20251014.json")

# process all channels: notch filter, bandpass, Welch PSD, band powers
result = lfp.process_session(session)

# look at the results
for ch in result.channels:
    print(f"{ch.channel} ({ch.hemisphere})")
    print(f"  beta1 power: {ch.band_powers['beta1']:.4f} µV²/Hz")
```

### Full access to all modalities

```python
# load_session_full gives you the full PerceptSession with all data types
ps = lfp.load_session_full("Report_Session_20251014.json")
print(ps.summary())

# access indefinite streaming channels
for stream in ps.indefinite_streaming:
    print(f"{stream.channel}: {stream.duration_seconds:.1f}s, {stream.n_samples} samples")

# access event snapshots
for event in ps.event_snapshots:
    print(f"{event.event_name} | {event.hemisphere} | {event.datetime[:10]}")
    print(f"  FFT bins: {len(event.fft_bin_data)}, freq range: {event.frequency[0]:.1f}-{event.frequency[-1]:.1f} Hz")

# access timeline (chronic tracking)
for tl in ps.timeline_data:
    df = tl.to_dataframe()
    print(f"{tl.hemisphere}: {len(df)} records")
```

### Batch process a directory

```python
# process all JSON files in a folder
results = lfp.process_directory("./data/", verbose=True)

# convert to a flat dataframe
df = lfp.results_to_dataframe(results)
print(df.head())

# aggregate by session (average across channels)
df_sessions = lfp.aggregate_by_session(results, hemisphere="left")
```

### Signal processing building blocks

```python
import numpy as np

# these work on any 1D array, not just Percept data
data = np.random.randn(5000)  # or your LFP signal
fs = 250

# preprocess: 60 Hz notch + 1-100 Hz bandpass
clean = lfp.preprocess(data, fs)

# PSD via Welch
freqs, psd = lfp.compute_psd(clean, fs)

# extract power in a specific band
beta_power = lfp.extract_band_power(psd, freqs, "beta1")
custom_band = lfp.extract_band_power(psd, freqs, (13, 20))

# all standard bands at once
bands = lfp.extract_all_bands(psd, freqs)
# {'delta': ..., 'theta': ..., 'alpha': ..., 'beta1': ..., 'beta2': ..., 'gamma': ...}

# spectrogram
t, f, Sxx = lfp.compute_spectrogram(clean, fs)

# epoching + artifact rejection
epochs = lfp.epoch(clean, fs, duration=2.0, overlap=0.5)
good_epochs = lfp.reject_bad_epochs(epochs, threshold=500)
```

### Plotting

```python
lfp.set_style()  # sets Nature Medicine-ish matplotlib defaults

# PSD plot
freqs, psd = lfp.compute_psd(clean, fs)
lfp.plot_psd(freqs, psd, show_bands=True, freq_range=(1, 50))

# session dashboard (PSDs + spectrograms + time series + band powers)
result = lfp.process_session(session, compute_spec=True)
lfp.plot_session_dashboard(result, output_path="dashboard.png")

# longitudinal PSDs across sessions (one subplot per channel, color = date)
results = lfp.process_directory("./data/")
lfp.plot_longitudinal_psd(
    results,
    title="Longitudinal PSDs",
    legend_annotations={"10/24": "10/24 (ECT started)", "11/13": "11/13 (DBS ON)"},
)

# batch dashboards for every session in a folder
lfp.generate_all_dashboards("./data/", "./figures/", compute_spec=True)
```

### Group comparison

```python
df = lfp.results_to_dataframe(results)
df["phase"] = ["pre" if d < "2025-01-01" else "post" for d in df["date"]]

stats = lfp.compare_groups(df, "phase", "pre", "post")
print(stats[["band", "mean_pre", "mean_post", "p_value"]])
```

## Frequency bands

Default band definitions (matching Percept event snapshot bands):

| Band | Range (Hz) |
|---|---|
| delta | 1–4 |
| theta | 4–8 |
| alpha | 8–13 |
| beta1 (low beta) | 13–20 |
| beta2 (high beta) | 20–30 |
| gamma (low gamma) | 30–50 |

## Percept JSON structure

The Medtronic programmer exports a single JSON file per session. Key top-level fields this package reads:

```
SessionDate
IndefiniteStreaming[]           --> raw time-domain, 6 channels
  .Channel, .TimeDomainData[], .SampleRateInHz, .FirstPacketDateTime
  .GlobalSequences, .GlobalPacketSizes, .TicksInMses
BrainSenseTimeDomain[]         --> streaming with stim
  (same fields as IndefiniteStreaming)
BrainSenseLfp[]                --> device-computed LFP power
  .LfpData.Left/.Right         --> .TicksInMs, .LFP, .mA, .Frequency
DiagnosticData
  .LFPTrendLogs                --> chronic power averages
  .LfpFrequencySnapshotEvents  --> patient-triggered FFT snapshots
Groups                         --> therapy group settings
DeviceInformation              --> neurostimulator metadata
```

## Channel naming

Percept uses contact pair names like `ZERO_THREE_LEFT` or `ONE_AND_THREE_RIGHT`. The library normalizes these to readable labels:

| Device name | Label |
|---|---|
| `ZERO_THREE_LEFT` | L 0-3 |
| `ONE_THREE_LEFT` | L 1-3 |
| `ZERO_TWO_LEFT` | L 0-2 |
| `ZERO_THREE_RIGHT` | R 0-3 |
| `ONE_THREE_RIGHT` | R 1-3 |
| `ZERO_TWO_RIGHT` | R 0-2 |

## Processing defaults

- **Sample rate**: 250 Hz (Percept PC/RC native rate)
- **Notch filter**: 60 Hz (Q=30)
- **Bandpass**: 1–100 Hz, 4th order Butterworth
- **PSD**: Welch's method, 256-sample segments, 50% overlap, linear detrend
- **Artifact rejection**: ±500 µV threshold on epochs
