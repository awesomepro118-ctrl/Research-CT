
from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne


METADATA_COLUMNS = [
    "dataset",
    "subject_id",
    "recording_id",
    "label",
    "source_file",
    "epoch_id",
    "epoch_uid",
    "row_in_epoch",
    "time_seconds",
    "epoch_start_seconds",
    "event_count",
    "event_labels",
]


def _metadata_from_filename(file_path):
    name = file_path.name

    subject_match = re.search(r"sub-([A-Za-z0-9]+)", name)
    task_match = re.search(r"_task-([^_]+)", name)

    subject_id = (
        subject_match.group(1)
        if subject_match
        else file_path.parent.parent.name.replace("sub-", "")
    )

    label = task_match.group(1) if task_match else file_path.stem

    return subject_id, label


def _load_raw(file_path, extension, freq):
    extension = extension.lower().lstrip(".")

    if extension == "set":
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)

    elif extension == "edf":
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    elif extension == "bdf":
        raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)

    elif extension == "fif":
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

    else:
        raise ValueError(
            "Supported raw EEG extensions are: set, edf, bdf, fif"
        )

    # Make sure usable channels are treated as EEG.
    keep = []
    for channel in raw.ch_names:
        upper = channel.upper()
        if (
            "ECG" not in upper
            and "EMG" not in upper
            and "AUX" not in upper
        ):
            keep.append(channel)

    if keep:
        raw.pick(keep)

    raw.set_channel_types(
        {channel: "eeg" for channel in raw.ch_names},
        verbose=False,
    )

    return raw


def _events_from_annotations(raw):
    try:
        events, event_id = mne.events_from_annotations(
            raw,
            verbose=False,
        )
    except Exception:
        return [], {}

    reverse_event_id = {
        number: name
        for name, number in event_id.items()
    }

    output = []

    for event in events:
        sample = int(event[0])
        code = int(event[2])

        output.append(
            {
                "time_seconds": sample / raw.info["sfreq"],
                "label": reverse_event_id.get(code, str(code)),
            }
        )

    return output, event_id


def _process_one_file(
    file_path,
    extension,
    win_len,
    freq,
    rows_per_epoch,
    lowpass_hz,
):
    print("\n" + "=" * 70)
    print("FILE:", file_path.name)
    print("=" * 70)

    raw = _load_raw(file_path, extension, freq)

    original_sfreq = float(raw.info["sfreq"])
    original_rows = int(raw.n_times)
    original_channels = int(raw.info["nchan"])

    print("Original sampling frequency:", original_sfreq, "Hz")
    print("Original rows/samples:", original_rows)
    print("EEG channels:", original_channels)

    events_before, event_id = _events_from_annotations(raw)

    print("Events found:", len(events_before))
    print("Event types:", len(event_id))

    # Required first downsample.
    if abs(float(raw.info["sfreq"]) - float(freq)) > 1e-6:
        raw.resample(freq, npad="auto", verbose=False)

    # 15 Hz here is the low-pass cutoff.
    # 24 rows per 2-second epoch requires 12 output rows/second.
    raw.filter(
        l_freq=0.5,
        h_freq=lowpass_hz,
        picks="eeg",
        verbose=False,
    )

    processed_rows_256 = int(raw.n_times)

    print("After resampling:", raw.info["sfreq"], "Hz")
    print("Rows/samples after 256 Hz resampling:", processed_rows_256)
    print("Low-pass:", lowpass_hz, "Hz")

    rows_per_second = rows_per_epoch / win_len
    samples_per_output_row = float(freq) / rows_per_second

    number_output_rows = int(
        np.floor(raw.n_times / samples_per_output_row)
    )

    if number_output_rows < 1:
        raise ValueError("No output rows were produced.")

    bands = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 45.0),
    }

    feature_data = {}

    # Band-first column order:
    # all Delta channels, then Theta, Alpha, Beta, Gamma.
    for band_name, (low_hz, high_hz) in bands.items():
        filtered = raw.copy().filter(
            l_freq=low_hz,
            h_freq=high_hz,
            picks="eeg",
            verbose=False,
        )

        band_values = filtered.get_data().T
        rows_for_band = []

        for row_number in range(number_output_rows):
            start = int(round(row_number * samples_per_output_row))
            end = int(round((row_number + 1) * samples_per_output_row))

            if end <= start or end > len(band_values):
                continue

            segment = band_values[start:end]

            # RMS band amplitude for each EEG channel.
            rms = np.sqrt(
                np.mean(
                    np.square(segment),
                    axis=0,
                )
            )

            rows_for_band.append(rms)

        rows_for_band = np.asarray(rows_for_band)

        for channel_index, channel_name in enumerate(raw.ch_names):
            clean_channel = re.sub(
                r"[^A-Za-z0-9]+",
                "_",
                str(channel_name),
            ).strip("_")

            feature_data[
                f"{clean_channel}_{band_name}"
            ] = rows_for_band[:, channel_index]

    feature_frame = pd.DataFrame(feature_data)

    # Keep complete epochs only.
    complete_row_count = (
        len(feature_frame) // rows_per_epoch
    ) * rows_per_epoch

    feature_frame = (
        feature_frame
        .iloc[:complete_row_count]
        .reset_index(drop=True)
    )

    number_epochs = (
        complete_row_count // rows_per_epoch
    )

    subject_id, label = _metadata_from_filename(file_path)

    metadata = pd.DataFrame(
        {
            "dataset": "Kosachenko",
            "subject_id": subject_id,
            "recording_id": file_path.stem,
            "label": label,
            "source_file": file_path.name,
            "epoch_id": np.arange(complete_row_count) // rows_per_epoch,
            "row_in_epoch": np.arange(complete_row_count) % rows_per_epoch,
        }
    )

    metadata["epoch_uid"] = (
        metadata["subject_id"].astype(str)
        + "__"
        + metadata["recording_id"].astype(str)
        + "__epoch_"
        + metadata["epoch_id"].astype(str)
    )

    metadata["time_seconds"] = (
        np.arange(complete_row_count)
        / rows_per_second
    )

    metadata["epoch_start_seconds"] = (
        metadata["epoch_id"] * win_len
    )

    # Assign annotation events to their 2-second epoch.
    epoch_event_labels = {
        epoch_number: []
        for epoch_number in range(number_epochs)
    }

    events_processed = 0

    for event in events_before:
        epoch_number = int(
            event["time_seconds"] // win_len
        )

        if epoch_number in epoch_event_labels:
            epoch_event_labels[epoch_number].append(
                event["label"]
            )
            events_processed += 1

    metadata["event_count"] = metadata["epoch_id"].map(
        lambda epoch: len(
            epoch_event_labels.get(epoch, [])
        )
    )

    metadata["event_labels"] = metadata["epoch_id"].map(
        lambda epoch: "|".join(
            epoch_event_labels.get(epoch, [])
        )
    )

    output = pd.concat(
        [
            metadata[METADATA_COLUMNS],
            feature_frame,
        ],
        axis=1,
    )

    print("Output rows:", len(output))
    print("Complete 2-second epochs:", number_epochs)
    print("Rows per epoch:", rows_per_epoch)
    print("Events processed into complete epochs:", events_processed)
    print("Metadata columns written:", len(METADATA_COLUMNS))
    print("Feature/data columns:", len(feature_frame.columns))
    print("Total columns:", len(output.columns))
    print("Metadata columns:")
    print(METADATA_COLUMNS)

    return output


def process_all_data(
    directory_name,
    extension,
    win_len=2,
    freq=256,
    rows_per_epoch=24,
    lowpass_hz=15,
    output_file=None,
    max_files=None,
):
    """
    Load raw EEG files, process them, combine them into one dataframe,
    optionally save one combined CSV, and RETURN the dataframe.

    The first four arguments preserve the original project function style:
        process_all_data(directory_name, extension, win_len, freq)

    Parameters
    ----------
    directory_name : str or Path
        Folder containing raw files. Searched recursively.
    extension : str
        Raw EEG extension such as "set" or "edf".
    win_len : float
        Epoch length in seconds. Default = 2.
    freq : float
        First resampling frequency. Default = 256 Hz.
    rows_per_epoch : int
        Number of dataframe rows in each epoch. Default = 24.
    lowpass_hz : float
        Low-pass filter cutoff. Default = 15 Hz.
    output_file : str or Path or None
        If supplied, saves ONE combined CSV here.
    max_files : int or None
        Useful for testing one file. Use max_files=1.

    Returns
    -------
    pandas.DataFrame
        One combined dataframe containing every processed file.
    """

    folder = Path(directory_name)

    if not folder.exists():
        raise FileNotFoundError(
            "Directory does not exist: " + str(folder)
        )

    extension = extension.lower().lstrip(".")

    files = sorted(
        folder.rglob("*." + extension)
    )

    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise FileNotFoundError(
            "No ." + extension + " files found in " + str(folder)
        )

    print("Files found:", len(files))
    print("Epoch length:", win_len, "seconds")
    print("Resample frequency:", freq, "Hz")
    print("Low-pass:", lowpass_hz, "Hz")
    print("Rows per epoch:", rows_per_epoch)

    frames = []

    for number, file_path in enumerate(files, start=1):
        print(
            "\nProcessing file",
            number,
            "of",
            len(files),
        )

        try:
            frame = _process_one_file(
                file_path=file_path,
                extension=extension,
                win_len=win_len,
                freq=freq,
                rows_per_epoch=rows_per_epoch,
                lowpass_hz=lowpass_hz,
            )

            frames.append(frame)

        except Exception as error:
            print("SKIPPED:", file_path.name)
            print("Reason:", error)

    if len(frames) == 0:
        raise RuntimeError(
            "No files were processed successfully."
        )

    data = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data.to_csv(
            output_file,
            index=False,
        )

        print("\nSaved combined dataframe:")
        print(output_file)

    print("\n" + "=" * 70)
    print("FINAL COMBINED DATAFRAME")
    print("=" * 70)
    print("Rows:", len(data))
    print("Columns:", len(data.columns))
    print("Subjects:", data["subject_id"].nunique())
    print("Recordings:", data["recording_id"].nunique())
    print("Epochs:", data["epoch_uid"].nunique())
    print("Metadata columns:", len(METADATA_COLUMNS))
    print(
        "Feature/data columns:",
        len(data.columns) - len(METADATA_COLUMNS),
    )

    return data
