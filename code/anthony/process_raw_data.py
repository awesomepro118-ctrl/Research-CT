from __future__ import annotations

from pathlib import Path
import argparse
import re
from typing import Mapping, Sequence

import mne
import numpy as np
import pandas as pd


# 19 EEG channels x 5 frequency bands = 95 features per row.
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

# These are metadata only. PCA / ML code should use the 95 feature columns,
# grouped by epoch_uid so one model sample is 24 x 95.
METADATA_COLUMNS = [
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
    """
    If input is EEGShared/RawData/Kosachenko, default output becomes
    EEGShared/ProcessedData/Kosachenko_2s (for win_len=2).
    """
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
    subject_match = re.search(
        r"(?:^|_)sub-([^_]+)", file_path.stem, flags=re.IGNORECASE
    )
    if subject_match:
        subject_id = subject_match.group(1)
    else:
        subject_dir = next(
            (part for part in reversed(file_path.parts) if part.lower().startswith("sub-")),
            None,
        )
        subject_id = subject_dir[4:] if subject_dir else file_path.stem

    task_match = re.search(
        r"_task-([^_]+)", file_path.stem, flags=re.IGNORECASE
    )
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

    rename_map = {
        old: new
        for old, new in zip(actual, wanted_present)
        if old != new
    }
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
    """
    Kosachenko event labels look like:
        memory 04/09 correct
        control 04/09 correct

    The model target is the serial position in the sequence:
        target = 4 in the example above

    Therefore target values span 1..13.
    """
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
    """
    RMS = sqrt(mean(x^2)), matching the workflow recommendation.
    Returns one value per EEG channel.
    """
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
    left = start_sample + int(
        round(row_in_epoch * samples_per_epoch / rows_per_epoch)
    )
    right = start_sample + int(
        round((row_in_epoch + 1) * samples_per_epoch / rows_per_epoch)
    )
    return left, right


def _feature_names(channels, band_map) -> list[str]:
    names = []
    for band_name in band_map:
        for channel_name in channels:
            names.append(f"{_clean_channel_name(channel_name)}_{band_name}")
    return names


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

        epoch_uid = (
            f"{subject_id}__{file_path.stem}__event_{event['source_index']}"
        )

        # IMPORTANT:
        # one model sample = this whole group of 24 rows x 95 features.
        for row_in_epoch in range(rows_per_epoch):
            row_start, row_end = _row_bounds(
                start_sample,
                row_in_epoch,
                samples_per_epoch,
                rows_per_epoch,
            )

            if row_end <= row_start:
                raise ValueError(
                    "rows_per_epoch is too large for win_len * freq."
                )

            feature_row = {}

            for band_name, data in band_arrays.items():
                values = _rms(data[:, row_start:row_end])

                for channel_index, channel_name in enumerate(raw.ch_names):
                    feature_row[
                        f"{_clean_channel_name(channel_name)}_{band_name}"
                    ] = values[channel_index]

            feature_rows.append(feature_row)

            metadata_rows.append(
                {
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
                    "time_seconds": (
                        event["onset"]
                        + row_in_epoch * win_len / rows_per_epoch
                    ),
                    "epoch_start_seconds": event["onset"],
                    "epoch_end_seconds": event["onset"] + win_len,
                    "trial_type": event["trial_type"],
                    "event_value": event["value"],
                }
            )

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
):
    """
    Generic fallback for non-Kosachenko files that do not have matching
    events.tsv workload labels. target is left blank.
    """
    feature_rows = []
    metadata_rows = []

    samples_per_epoch = int(round(win_len * freq))
    n_epochs = raw.n_times // samples_per_epoch

    for epoch_id in range(n_epochs):
        start_sample = epoch_id * samples_per_epoch
        epoch_start = start_sample / freq
        epoch_uid = (
            f"{subject_id}__{file_path.stem}__continuous_{epoch_id}"
        )

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

            metadata_rows.append(
                {
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
                    "time_seconds": (
                        epoch_start
                        + row_in_epoch * win_len / rows_per_epoch
                    ),
                    "epoch_start_seconds": epoch_start,
                    "epoch_end_seconds": epoch_start + win_len,
                    "trial_type": "",
                    "event_value": np.nan,
                }
            )

    return feature_rows, metadata_rows


def _process_one_file(
    file_path: Path,
    *,
    win_len: float,
    freq: float,
    band_map,
    channels,
    strict_channels: bool,
    rows_per_epoch: int,
    resample_if_needed: bool,
    use_event_epochs: bool,
    allowed_loads: set[int] | None,
    allowed_conditions: set[str] | None,
    dataset_name: str,
):
    print(f"\nProcessing: {file_path.name}")

    raw = _read_raw(file_path)
    native_freq = float(raw.info["sfreq"])

    if not np.isclose(native_freq, freq, rtol=1e-7, atol=1e-7):
        if not resample_if_needed:
            raise ValueError(
                f"{file_path.name} is {native_freq:g} Hz, "
                f"but freq={freq:g} was requested."
            )
        raw.resample(freq, npad="auto", verbose="ERROR")

    selected = _select_channels(raw, channels, strict_channels)

    samples_per_epoch_float = freq * win_len
    samples_per_epoch = int(round(samples_per_epoch_float))

    if not np.isclose(
        samples_per_epoch_float,
        samples_per_epoch,
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError(
            "freq * win_len must be an integer number of samples."
        )

    if rows_per_epoch > samples_per_epoch:
        raise ValueError(
            f"rows_per_epoch={rows_per_epoch} exceeds "
            f"samples_per_epoch={samples_per_epoch}."
        )

    subject_id, label = _parse_metadata(file_path)
    band_arrays = _build_band_arrays(selected, band_map)

    digit_events = (
        _load_digit_events(
            file_path,
            allowed_loads=allowed_loads,
            allowed_conditions=allowed_conditions,
        )
        if use_event_epochs
        else []
    )

    if digit_events:
        print(f"  workload digit events: {len(digit_events)}")
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
        )
    else:
        print("  no matching workload events; using continuous fallback")
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
        )

    if not feature_rows:
        raise ValueError("No output rows were produced.")

    metadata = pd.DataFrame(metadata_rows)
    features = pd.DataFrame(feature_rows)
    frame = pd.concat(
        [metadata[METADATA_COLUMNS], features],
        axis=1,
    )

    expected_features = len(selected.ch_names) * len(band_map)
    if len(features.columns) != expected_features:
        raise RuntimeError(
            f"Expected {expected_features} feature columns, "
            f"found {len(features.columns)}."
        )

    epoch_sizes = frame.groupby("epoch_uid", sort=False).size()
    if not epoch_sizes.eq(rows_per_epoch).all():
        raise RuntimeError(
            "Not every epoch has the required number of rows."
        )

    print(
        f"  features per row: {len(selected.ch_names)} channels x "
        f"{len(band_map)} bands = {len(features.columns)}"
    )
    print(
        f"  model sample shape: {rows_per_epoch} x "
        f"{len(features.columns)}"
    )
    print(
        f"  epochs: {frame['epoch_uid'].nunique()} | "
        f"output rows: {len(frame)}"
    )

    return frame


def feature_columns(data: pd.DataFrame) -> list[str]:
    """Return only EEG band-feature columns, in saved order."""
    return [column for column in data.columns if column not in METADATA_COLUMNS]


def validate_processed_data(
    data: pd.DataFrame,
    rows_per_epoch: int = 24,
    expected_feature_count: int = 95,
) -> dict:
    """
    Validate the shape required by the later PCA/classification workflow.
    """
    missing_metadata = [
        column for column in METADATA_COLUMNS if column not in data.columns
    ]
    if missing_metadata:
        raise ValueError(
            "Missing metadata columns: " + ", ".join(missing_metadata)
        )

    features = feature_columns(data)

    if len(features) != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} feature columns; "
            f"found {len(features)}."
        )

    epoch_sizes = data.groupby("epoch_uid", sort=False).size()
    if not epoch_sizes.eq(rows_per_epoch).all():
        bad = epoch_sizes[~epoch_sizes.eq(rows_per_epoch)]
        raise ValueError(
            f"{len(bad)} epoch(s) do not have {rows_per_epoch} rows."
        )

    targets = sorted(
        pd.to_numeric(data["target"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if targets and (min(targets) < 1 or max(targets) > 13):
        raise ValueError(
            f"target should be within 1..13; found {targets}."
        )

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
    bands=None,
    channels=DEFAULT_EEG_CHANNELS,
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
):
    """
    Workflow 3a implementation.

    Required workflow signature:
        process_all_data(directory_name, extension, win_len, freq)

    For the Kosachenko workload dataset, each workload digit event is one epoch.
    Each epoch is saved as 24 rows, and each row contains 95 EEG features
    (19 channels x 5 frequency bands). Therefore one later ML/PCA sample is
    the full 24 x 95 block identified by epoch_uid.

    The target column is the digit's serial position, with values 1..13.

    The function saves one CSV per raw EEG file, as requested in Workflow 3a,
    and returns one combined pandas DataFrame.
    """
    input_dir = Path(directory_name).expanduser()

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Raw-data directory does not exist: {input_dir}"
        )

    win_len = float(win_len)
    freq = float(freq)
    rows_per_epoch = int(rows_per_epoch)

    if win_len <= 0:
        raise ValueError("win_len must be > 0")
    if freq <= 0:
        raise ValueError("freq must be > 0")
    if rows_per_epoch <= 0:
        raise ValueError("rows_per_epoch must be > 0")

    ext = _normalise_extension(extension)

    candidates = (
        input_dir.rglob(f"*{ext}")
        if recursive
        else input_dir.glob(f"*{ext}")
    )

    files = sorted(
        (
            path
            for path in candidates
            if path.is_file() and ".git" not in path.parts
        ),
        key=lambda path: str(path).lower(),
    )

    # For BIDS/OpenNeuro .set data, prefer actual EEG files only.
    if ext == ".set":
        bids_files = [
            path
            for path in files
            if path.name.lower().endswith("_eeg.set")
        ]
        if bids_files:
            files = bids_files

    if max_files is not None:
        files = files[: int(max_files)]

    if not files:
        raise FileNotFoundError(
            f"No {ext} files found in {input_dir}"
        )

    output_dir = (
        Path(output_directory).expanduser()
        if output_directory is not None
        else _derive_output_directory(input_dir, win_len)
    )

    if save_per_file:
        output_dir.mkdir(parents=True, exist_ok=True)

    band_map = _validate_bands(
        bands or DEFAULT_BANDS,
        sampling_frequency=freq,
    )
    load_set = (
        None
        if allowed_loads is None
        else {int(value) for value in allowed_loads}
    )
    condition_set = (
        None
        if allowed_conditions is None
        else {str(value).strip().lower() for value in allowed_conditions}
    )

    print("=" * 72)
    print("WORKFLOW 3A - RAW EEG PREPROCESSING")
    print("=" * 72)
    print(f"Files found: {len(files)}")
    print(f"Window length: {win_len:g} s")
    print(f"Sampling frequency: {freq:g} Hz")
    print(f"Rows per epoch: {rows_per_epoch}")
    print(
        f"Expected model sample: {rows_per_epoch} x "
        f"{len(channels) * len(band_map)}"
    )
    print("Aggregation: RMS = sqrt(mean(x^2))")
    print(f"Output directory: {output_dir}")

    frames = []
    failures = []

    for number, file_path in enumerate(files, start=1):
        print(f"\n[{number}/{len(files)}]")

        try:
            frame = _process_one_file(
                file_path,
                win_len=win_len,
                freq=freq,
                band_map=band_map,
                channels=channels,
                strict_channels=strict_channels,
                rows_per_epoch=rows_per_epoch,
                resample_if_needed=resample_if_needed,
                use_event_epochs=use_event_epochs,
                allowed_loads=load_set,
                allowed_conditions=condition_set,
                dataset_name=dataset_name,
            )
        except Exception as error:
            failures.append((file_path, str(error)))
            print(f"  SKIPPED: {error}")
            continue

        frames.append(frame)

        if save_per_file:
            # Save one CSV per raw file. Avoid nested subdirectories because
            # Workflow 3b should be able to load the ProcessedData directory
            # directly and naturally sort the subject files.
            destination = output_dir / f"{file_path.stem}.csv"

            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"Output already exists: {destination}"
                )

            frame.to_csv(destination, index=False)
            print(f"  saved -> {destination}")

    if not frames:
        details = "\n".join(
            f"- {path.name}: {reason}"
            for path, reason in failures[:10]
        )
        raise RuntimeError(
            "No files were processed successfully.\n" + details
        )

    data = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    integer_columns = [
        "absolute_load",
        "target",
        "event_index",
        "epoch_id",
        "row_in_epoch",
    ]
    for column in integer_columns:
        if column in data.columns:
            data[column] = (
                pd.to_numeric(data[column], errors="coerce")
                .astype("Int64")
            )

    summary = validate_processed_data(
        data,
        rows_per_epoch=rows_per_epoch,
        expected_feature_count=len(channels) * len(band_map),
    )

    if combined_output_file is not None:
        combined_path = Path(combined_output_file).expanduser()
        combined_path.parent.mkdir(parents=True, exist_ok=True)

        if combined_path.exists() and not overwrite:
            raise FileExistsError(
                f"Combined output already exists: {combined_path}"
            )

        data.to_csv(combined_path, index=False)
        print(f"\nCombined CSV saved -> {combined_path}")

    print("\n" + "=" * 72)
    print("FINAL CHECK")
    print("=" * 72)
    print(f"Rows: {summary['rows']}")
    print(f"Epochs / model samples: {summary['epochs']}")
    print(f"Rows per epoch: {summary['rows_per_epoch']}")
    print(f"Feature columns: {summary['feature_columns']}")
    print(f"Model sample shape: {summary['sample_shape']}")
    print(f"Target values present: {summary['targets']}")
    print(f"Files skipped: {len(failures)}")

    return data


def _main():
    parser = argparse.ArgumentParser(
        description="Workflow 3a EEG preprocessing."
    )
    parser.add_argument("directory_name")
    parser.add_argument("extension")
    parser.add_argument("--win-len", type=float, default=2.0)
    parser.add_argument("--freq", type=float, default=256.0)
    parser.add_argument("--rows-per-epoch", type=int, default=24)
    parser.add_argument("--output-directory", default=None)
    parser.add_argument("--combined-output-file", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Use only memory-condition workload events.",
    )

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
    )


if __name__ == "__main__":
    _main()
