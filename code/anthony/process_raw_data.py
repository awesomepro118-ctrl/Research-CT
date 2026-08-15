from __future__ import annotations

from pathlib import Path
import argparse
import re
from typing import Mapping, Sequence, Dict, List

import mne
import numpy as np
import pandas as pd


# Default standard configurations (fully overridable)
DEFAULT_EEG_CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]

DEFAULT_BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

DEFAULT_METADATA_COLUMNS = [
    "dataset",
    "subject_id",
    "recording_id",
    "label",
    "condition",
    "source_file",
    "absolute_load",
    "target",
    "event_index",
    "epoch_id",
    "epoch_uid",
    "row_in_epoch",
    "time_seconds",
    "epoch_start_seconds",
    "epoch_end_seconds",
    "trial_type",
    "event_value",
]


def _normalise_extension(extension: str) -> str:
    extension = str(extension).strip().lower()
    if not extension:
        raise ValueError("extension must not be empty")
    return extension if extension.startswith(".") else f".{extension}"


def _derive_output_directory(input_directory: Path, win_len: float) -> Path:
    """Derive ProcessedData directory following workflow conventions[cite: 3]."""
    input_directory = input_directory.resolve()
    parts = input_directory.parts
    raw_indices = [i for i, part in enumerate(parts) if part.lower() == "rawdata"]

    if raw_indices:
        raw_index = raw_indices[-1]
        project_root = Path(*parts[:raw_index])
        relative = Path(*parts[raw_index + 1 :])
        dataset_name = relative.name if relative.parts else input_directory.name
        return project_root / "ProcessedData" / f"{dataset_name}_{win_len:g}s"

    return input_directory.parent / "ProcessedData" / f"{input_directory.name}_{win_len:g}s"


def _parse_metadata(file_path: Path) -> tuple[str, str]:
    subject_match = re.search(r"(?:^|_)sub-([^_]+)", file_path.stem, flags=re.IGNORECASE)
    if subject_match:
        subject_id = subject_match.group(1)
    else:
        subject_dir = next(
            (part for part in reversed(file_path.parts) if part.lower().startswith("sub-")),
            None,
        )
        subject_id = subject_dir[4:] if subject_dir else file_path.stem

    task_match = re.search(r"_task-([^_]+)", file_path.stem, flags=re.IGNORECASE)
    label = task_match.group(1) if task_match else file_path.stem
    return subject_id, label


def _validate_bands(
    bands: Mapping[str, Sequence[float]],
    sampling_frequency: float,
) -> dict[str, tuple[float, float]]:
    if not bands:
        raise ValueError("At least one frequency band is required.")

    nyquist = float(sampling_frequency) / 2.0
    validated = {}

    for name, limits in bands.items():
        if len(limits) != 2:
            raise ValueError(f"Band {name!r} must be (low_hz, high_hz).")

        low_hz, high_hz = float(limits[0]), float(limits[1])

        if low_hz <= 0 or high_hz <= low_hz:
            raise ValueError(f"Invalid band {name!r}: {limits}")

        if high_hz >= nyquist:
            raise ValueError(
                f"Band {name!r} ends at {high_hz:g} Hz, but Nyquist is "
                f"{nyquist:g} Hz for freq={sampling_frequency:g}."
            )

        validated[str(name)] = (low_hz, high_hz)

    return validated


def _read_raw(file_path: Path):
    readers = {
        ".set": mne.io.read_raw_eeglab,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".fif": mne.io.read_raw_fif,
    }

    reader = readers.get(file_path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported EEG extension {file_path.suffix!r}. "
            "Supported: .set, .edf, .bdf, .fif"
        )

    return reader(file_path, preload=True, verbose="ERROR")


def _select_channels(raw, channels, strict_channels: bool):
    lookup = {str(ch).strip().lower(): ch for ch in raw.ch_names}
    actual = []
    wanted_present = []
    missing = []

    for wanted in channels:
        match = lookup.get(str(wanted).strip().lower())
        if match is None:
            missing.append(str(wanted))
        else:
            actual.append(match)
            wanted_present.append(str(wanted))

    if strict_channels and missing:
        raise ValueError(
            "Missing required EEG channels: "
            + ", ".join(missing)
            + "\nAvailable channels: "
            + ", ".join(map(str, raw.ch_names))
        )

    if not actual:
        raise ValueError("None of the requested EEG channels were found.")

    selected = raw.copy().pick(actual)

    rename_map = {old: new for old, new in zip(actual, wanted_present) if old != new}
    if rename_map:
        selected.rename_channels(rename_map)

    selected.set_channel_types(
        {channel: "eeg" for channel in selected.ch_names},
        verbose=False,
    )

    return selected


def _matching_events_file(file_path: Path) -> Path:
    replaced = re.sub(
        r"_eeg\.[^.]+$",
        "_events.tsv",
        file_path.name,
        flags=re.IGNORECASE,
    )
    if replaced != file_path.name:
        return file_path.with_name(replaced)

    return file_path.with_name(file_path.stem + "_events.tsv")


def _load_digit_events(
    file_path: Path,
    allowed_loads: set[int] | None,
    allowed_conditions: set[str] | None,
) -> list[dict]:
    event_file = _matching_events_file(file_path)
    if not event_file.exists():
        return []

    events = pd.read_csv(event_file, sep="\t")
    if not {"onset", "trial_type"}.issubset(events.columns):
        return []

    pattern = re.compile(
        r"^(memory|control)\s+(\d{1,3})/(\d{1,3})\b",
        flags=re.IGNORECASE,
    )

    digit_events = []

    for source_index, row in events.iterrows():
        trial_type = str(row["trial_type"]).strip()
        match = pattern.search(trial_type)

        if match is None:
            continue

        condition = match.group(1).lower()
        target = int(match.group(2))
        absolute_load = int(match.group(3))

        if allowed_conditions is not None and condition not in allowed_conditions:
            continue

        if allowed_loads is not None and absolute_load not in allowed_loads:
            continue

        if target < 1 or target > absolute_load:
            continue

        try:
            onset = float(row["onset"])
        except (TypeError, ValueError):
            continue

        if not np.isfinite(onset):
            continue

        digit_events.append(
            {
                "source_index": int(source_index),
                "onset": onset,
                "condition": condition,
                "absolute_load": absolute_load,
                "target": target,
                "trial_type": trial_type,
                "value": row.get("value", np.nan),
            }
        )

    digit_events.sort(key=lambda item: item["onset"])
    return digit_events


def _rms(segment: np.ndarray) -> np.ndarray:
    """Calculate RMS = sqrt(mean(x^2)) per workflow recommendation[cite: 3]."""
    squared_mean = np.mean(np.square(segment, dtype=np.float64), axis=1)
    return np.sqrt(squared_mean)


def _build_band_arrays(raw, band_map):
    arrays = {}
    for band_name, (low_hz, high_hz) in band_map.items():
        filtered = raw.copy().filter(
            l_freq=low_hz,
            h_freq=high_hz,
            picks="eeg",
            method="fir",
            fir_design="firwin",
            verbose="ERROR",
        )
        arrays[band_name] = filtered.get_data()
    return arrays


def _clean_channel_name(channel_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(channel_name)).strip("_")


def _row_bounds(
    start_sample: int,
    row_in_epoch: int,
    samples_per_epoch: int,
    rows_per_epoch: int,
) -> tuple[int, int]:
    left = start_sample + int(round(row_in_epoch * samples_per_epoch / rows_per_epoch))
    right = start_sample + int(round((row_in_epoch + 1) * samples_per_epoch / rows_per_epoch))
    return left, right


def _process_event_epochs(
    *,
    raw,
    band_arrays,
    digit_events,
    file_path: Path,
    subject_id: str,
    label: str,
    dataset_name: str,
    win_len: float,
    freq: float,
    rows_per_epoch: int,
    metadata_columns: list[str],
):
    feature_rows = []
    metadata_rows = []

    samples_per_epoch = int(round(win_len * freq))
    epoch_counter = 0
    skipped = 0

    for event in digit_events:
        start_sample = int(round(event["onset"] * freq))
        end_sample = start_sample + samples_per_epoch

        if start_sample < 0 or end_sample > raw.n_times:
            skipped += 1
            continue

        epoch_uid = f"{subject_id}__{file_path.stem}__event_{event['source_index']}"

        for row_in_epoch in range(rows_per_epoch):
            row_start, row_end = _row_bounds(
                start_sample,
                row_in_epoch,
                samples_per_epoch,
                rows_per_epoch,
            )

            if row_end <= row_start:
                raise ValueError("rows_per_epoch is too large for win_len * freq.")

            feature_row = {}

            for band_name, data in band_arrays.items():
                values = _rms(data[:, row_start:row_end])

                for channel_index, channel_name in enumerate(raw.ch_names):
                    feature_row[
                        f"{_clean_channel_name(channel_name)}_{band_name}"
                    ] = values[channel_index]

            feature_rows.append(feature_row)

            row_dict = {
                "dataset": dataset_name,
                "subject_id": subject_id,
                "recording_id": file_path.stem,
                "label": label,
                "condition": event["condition"],
                "source_file": file_path.name,
                "absolute_load": event["absolute_load"],
                "target": event["target"],
                "event_index": event["source_index"],
                "epoch_id": epoch_counter,
                "epoch_uid": epoch_uid,
                "row_in_epoch": row_in_epoch,
                "time_seconds": (event["onset"] + row_in_epoch * win_len / rows_per_epoch),
                "epoch_start_seconds": event["onset"],
                "epoch_end_seconds": event["onset"] + win_len,
                "trial_type": event["trial_type"],
                "event_value": event["value"],
            }
            # Keep only requested metadata columns
            metadata_rows.append({k: row_dict[k] for k in metadata_columns if k in row_dict})

        epoch_counter += 1

    if skipped:
        print(f"  skipped {skipped} event epoch(s) outside recording bounds")

    return feature_rows, metadata_rows


def _process_continuous_epochs(
    *,
    raw,
    band_arrays,
    file_path: Path,
    subject_id: str,
    label: str,
    dataset_name: str,
    win_len: float,
    freq: float,
    rows_per_epoch: int,
    metadata_columns: list[str],
):
    feature_rows = []
    metadata_rows = []

    samples_per_epoch = int(round(win_len * freq))
    n_epochs = raw.n_times // samples_per_epoch

    for epoch_id in range(n_epochs):
        start_sample = epoch_id * samples_per_epoch
        epoch_start = start_sample / freq
        epoch_uid = f"{subject_id}__{file_path.stem}__continuous_{epoch_id}"

        for row_in_epoch in range(rows_per_epoch):
            row_start, row_end = _row_bounds(
                start_sample,
                row_in_epoch,
                samples_per_epoch,
                rows_per_epoch,
            )

            feature_row = {}

            for band_name, data in band_arrays.items():
                values = _rms(data[:, row_start:row_end])

                for channel_index, channel_name in enumerate(raw.ch_names):
                    feature_row[
                        f"{_clean_channel_name(channel_name)}_{band_name}"
                    ] = values[channel_index]

            feature_rows.append(feature_row)

            row_dict = {
                "dataset": dataset_name,
                "subject_id": subject_id,
                "recording_id": file_path.stem,
                "label": label,
                "condition": "unknown",
                "source_file": file_path.name,
                "absolute_load": pd.NA,
                "target": pd.NA,
                "event_index": pd.NA,
                "epoch_id": epoch_id,
                "epoch_uid": epoch_uid,
                "row_in_epoch": row_in_epoch,
                "time_seconds": (epoch_start + row_in_epoch * win_len / rows_per_epoch),
                "epoch_start_seconds": epoch_start,
                "epoch_end_seconds": epoch_start + win_len,
                "trial_type": "",
                "event_value": np.nan,
            }
            metadata_rows.append({k: row_dict[k] for k in metadata_columns if k in row_dict})

    return feature_rows, metadata_rows


def validate_processed_data(
    data: pd.DataFrame,
    metadata_columns: list[str],
    rows_per_epoch: int = 24,
    expected_feature_count: int = 95,
) -> dict:
    missing_metadata = [c for c in metadata_columns if c not in data.columns]
    if missing_metadata:
        raise ValueError("Missing metadata columns: " + ", ".join(missing_metadata))

    features = [c for c in data.columns if c not in metadata_columns]
    if len(features) != expected_feature_count:
        raise ValueError(f"Expected {expected_feature_count} feature columns; found {len(features)}.")

    epoch_sizes = data.groupby("epoch_uid", sort=False).size()
    if not epoch_sizes.eq(rows_per_epoch).all():
        raise ValueError("Not every epoch has the required number of rows.")

    targets = []
    if "target" in data.columns:
        targets = sorted(pd.to_numeric(data["target"], errors="coerce").dropna().astype(int).unique().tolist())

    return {
        "rows": len(data),
        "epochs": int(data["epoch_uid"].nunique()),
        "rows_per_epoch": rows_per_epoch,
        "feature_columns": len(features),
        "sample_shape": (rows_per_epoch, len(features)),
        "targets": targets,
    }


def process_all_data(
    directory_name,
    extension,
    win_len,
    freq,
    *,
    output_directory=None,
    combined_output_file=None,
    bands: dict | None = None,
    channels: list[str] | None = None,
    metadata_columns: list[str] | None = None,
    strict_channels=True,
    rows_per_epoch=24,
    recursive=True,
    resample_if_needed=True,
    use_event_epochs=True,
    allowed_loads=(5, 9, 13),
    allowed_conditions=("memory", "control"),
    dataset_name="Kosachenko",
    save_per_file=True,
    overwrite=True,
    max_files=None,
    lowpass_hz=None,
):
    """
    Highly configurable Workflow 3a implementation matching required signature:
        process_all_data(directory_name, extension, win_len, freq)[cite: 3]
    """
    input_dir = Path(directory_name).expanduser()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Raw-data directory does not exist: {input_dir}")

    win_len = float(win_len)
    freq = float(freq)
    rows_per_epoch = int(rows_per_epoch)

    ext = _normalise_extension(extension)

    # Use default variables if not supplied
    active_channels = channels if channels is not None else DEFAULT_EEG_CHANNELS
    active_bands = bands if bands is not None else DEFAULT_BANDS
    active_metadata = metadata_columns if metadata_columns is not None else DEFAULT_METADATA_COLUMNS

    candidates = input_dir.rglob(f"*{ext}") if recursive else input_dir.glob(f"*{ext}")
    files = sorted(
        (path for path in candidates if path.is_file() and ".git" not in path.parts),
        key=lambda path: str(path).lower(),
    )

    if ext == ".set":
        bids_files = [path for path in files if path.name.lower().endswith("_eeg.set")]
        if bids_files:
            files = bids_files

    if max_files is not None:
        files = files[: int(max_files)]

    if not files:
        raise FileNotFoundError(f"No {ext} files found in {input_dir}")

    output_dir = (
        Path(output_directory).expanduser()
        if output_directory is not None
        else _derive_output_directory(input_dir, win_len)
    )

    if save_per_file:
        output_dir.mkdir(parents=True, exist_ok=True)

    band_map = _validate_bands(active_bands, sampling_frequency=freq)
    load_set = None if allowed_loads is None else {int(v) for v in allowed_loads}
    condition_set = None if allowed_conditions is None else {str(v).strip().lower() for v in allowed_conditions}

    print("=" * 72)
    print("WORKFLOW 3A - FULLY CONFIGURABLE EEG PREPROCESSING")
    print("=" * 72)
    print(f"Files found: {len(files)}")
    print(f"Window length: {win_len:g} s")
    print(f"Sampling frequency: {freq:g} Hz")
    print(f"Active Channels ({len(active_channels)}): {active_channels}")
    print(f"Active Bands: {list(active_bands.keys())}")
    print(f"Output directory: {output_dir}")

    frames = []
    failures = []

    for number, file_path in enumerate(files, start=1):
        print(f"\n[{number}/{len(files)}] Processing: {file_path.name}")

        try:
            raw = _read_raw(file_path)

            if lowpass_hz is not None:
                raw.filter(l_freq=None, h_freq=float(lowpass_hz), picks="eeg", verbose="ERROR")

            native_freq = float(raw.info["sfreq"])
            if not np.isclose(native_freq, freq, rtol=1e-7, atol=1e-7):
                if not resample_if_needed:
                    raise ValueError(f"Native frequency {native_freq:g} Hz does not match requested {freq:g} Hz.")
                raw.resample(freq, npad="auto", verbose="ERROR")

            selected = _select_channels(raw, active_channels, strict_channels)
            subject_id, label = _parse_metadata(file_path)
            band_arrays = _build_band_arrays(selected, band_map)

            digit_events = (
                _load_digit_events(file_path, allowed_loads=load_set, allowed_conditions=condition_set)
                if use_event_epochs
                else []
            )

            if digit_events:
                feature_rows, metadata_rows = _process_event_epochs(
                    raw=selected,
                    band_arrays=band_arrays,
                    digit_events=digit_events,
                    file_path=file_path,
                    subject_id=subject_id,
                    label=label,
                    dataset_name=dataset_name,
                    win_len=win_len,
                    freq=freq,
                    rows_per_epoch=rows_per_epoch,
                    metadata_columns=active_metadata,
                )
            else:
                feature_rows, metadata_rows = _process_continuous_epochs(
                    raw=selected,
                    band_arrays=band_arrays,
                    file_path=file_path,
                    subject_id=subject_id,
                    label=label,
                    dataset_name=dataset_name,
                    win_len=win_len,
                    freq=freq,
                    rows_per_epoch=rows_per_epoch,
                    metadata_columns=active_metadata,
                )

            metadata = pd.DataFrame(metadata_rows)
            features = pd.DataFrame(feature_rows)
            frame = pd.concat([metadata[active_metadata], features], axis=1)

            frames.append(frame)

            if save_per_file:
                destination = output_dir / f"{file_path.stem}.csv"
                if destination.exists() and not overwrite:
                    raise FileExistsError(f"Output already exists: {destination}")
                frame.to_csv(destination, index=False)
                print(f"  saved -> {destination}")

        except Exception as error:
            failures.append((file_path, str(error)))
            print(f"  SKIPPED: {error}")

    if not frames:
        raise RuntimeError("No files were processed successfully.")

    data = pd.concat(frames, ignore_index=True, sort=False)
    summary = validate_processed_data(
        data, 
        metadata_columns=active_metadata, 
        rows_per_epoch=rows_per_epoch, 
        expected_feature_count=len(active_channels) * len(band_map)
    )

    if combined_output_file is not None:
        combined_path = Path(combined_output_file).expanduser()
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(combined_path, index=False)
        print(f"\nCombined CSV saved -> {combined_path}")

    return data


def _main():
    parser = argparse.ArgumentParser(description="Configurable Workflow 3a EEG preprocessing.")
    parser.add_argument("directory_name", help="Path to raw data directory")
    parser.add_argument("extension", help="File extension (e.g. set, edf)")
    parser.add_argument("--win-len", type=float, default=2.0, help="Window length in seconds")
    parser.add_argument("--freq", type=float, default=256.0, help="Sampling frequency")
    parser.add_argument("--rows-per-epoch", type=int, default=24, help="Rows per epoch")
    parser.add_argument("--lowpass-hz", type=float, default=None, help="Optional pre-filtering lowpass cutoff")
    parser.add_argument("--output-directory", default=None, help="Custom output directory")
    parser.add_argument("--combined-output-file", default=None, help="Path for combined CSV output")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--memory-only", action="store_true", help="Filter for memory condition only")
    parser.add_argument("--no-save-per-file", action="store_true", help="Skip saving individual CSVs per file")

    args = parser.parse_args()

    conditions = ("memory",) if args.memory_only else ("memory", "control")

    process_all_data(
        args.directory_name,
        args.extension,
        args.win_len,
        args.freq,
        output_directory=args.output_directory,
        combined_output_file=args.combined_output_file,
        rows_per_epoch=args.rows_per_epoch,
        allowed_conditions=conditions,
        max_files=args.max_files,
        lowpass_hz=args.lowpass_hz,
        save_per_file=not args.no_save_per_file,
    )


if __name__ == "__main__":
    _main()
