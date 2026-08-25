import re
import random
import time
from pathlib import Path
import pandas as pd


def natural_sort_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def find_processed_csvs(directory):
    return sorted((p for p in Path(directory).rglob("*_processed.csv") if p.is_file()), key=natural_sort_key)


def _read_csv_with_retries(file_path, retries=3, retry_wait_seconds=3):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return pd.read_csv(file_path)
        except OSError as error:
            last_error = error
            print(f"Read error {attempt}/{retries}: {file_path.name}")
            print("Reason:", error)
            if attempt < retries:
                time.sleep(retry_wait_seconds)
    raise last_error


def _subject_from_frame(frame, file_path):
    for column in ["subject_id", "subject"]:
        if column in frame.columns:
            values = frame[column].dropna()
            if len(values):
                return str(values.iloc[0])
    match = re.search(r"sub-([A-Za-z0-9]+)", str(file_path), flags=re.IGNORECASE)
    if match:
        return match.group(1)
    raise ValueError(f"Could not determine subject ID for {file_path.name}")



def _subject_from_filename(file_path):
    match = re.search(
        r"sub-([A-Za-z0-9]+)",
        file_path.stem,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            f"Could not parse subject ID from filename: {file_path.name}"
        )
    return match.group(1)


def _normalize_source_frame(frame, file_path, master_source_columns):
    """
    Normalize one source CSV before any concatenation.

    - forces one source schema/order
    - fills missing columns with NA
    - rejects unexpected columns
    - forces subject identity from filename
    - keeps epoch_uid consistently typed
    """
    frame = frame.copy()

    extra_columns = [
        column
        for column in frame.columns
        if column not in master_source_columns
    ]

    if extra_columns:
        raise ValueError(
            f"{file_path.name} has unexpected columns: {extra_columns}"
        )

    for column in master_source_columns:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[master_source_columns]

    filename_subject = _subject_from_filename(file_path)

    if "subject_id" in frame.columns:
        frame["subject_id"] = str(filename_subject)

    if "subject" in frame.columns:
        frame["subject"] = str(filename_subject)

    if "epoch_uid" not in frame.columns:
        raise ValueError(
            f"{file_path.name} has no epoch_uid column"
        )

    frame["epoch_uid"] = frame["epoch_uid"].astype("string")

    return frame


def _append_frame(frame, output_path, master_columns):
    """
    Append using one fixed column schema/order for every subject.
    This prevents values from shifting under the wrong CSV headers.
    """
    frame = frame.copy()

    for column in master_columns:
        if column not in frame.columns:
            frame[column] = pd.NA

    extra_columns = [
        column for column in frame.columns
        if column not in master_columns
    ]

    if extra_columns:
        raise ValueError(
            f"Unexpected columns before write: {extra_columns}"
        )

    frame = frame[master_columns]

    if "subject_id" in frame.columns:
        frame["subject_id"] = frame["subject_id"].astype("string")

    if "subject" in frame.columns:
        frame["subject"] = frame["subject"].astype("string")

    if "epoch_uid" in frame.columns:
        frame["epoch_uid"] = frame["epoch_uid"].astype("string")

    first_write = not output_path.exists()

    frame.to_csv(
        output_path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
    )


def _split_epoch_ids(epoch_ids, fractions, seed):
    epoch_ids = list(epoch_ids)
    rng = random.Random(seed)
    rng.shuffle(epoch_ids)

    names = list(fractions)
    values = [fractions[name] for name in names]
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError("Epoch split fractions must add to 1.0")

    n = len(epoch_ids)
    result = {}
    start = 0
    for name, frac in zip(names[:-1], values[:-1]):
        count = int(round(n * frac))
        count = min(count, n - start)
        result[name] = epoch_ids[start:start + count]
        start += count
    result[names[-1]] = epoch_ids[start:]
    return result


def build_exact_subject_split(
    directory_name,
    output_directory=None,
    development_subject_count=40,
    holdout_subject_count=25,
    development_train_fraction=0.50,
    development_validation_fraction=0.25,
    holdout_train_fraction=0.50,
    holdout_validation_fraction=0.25,
    random_state=1,
    overwrite=True,
    retries=3,
    retry_wait_seconds=3,
):
    """
    Exact hierarchical split for 65 subjects.

    Subject level:
      40 development subjects
      25 holdout subjects

    Inside every subject:
      50% complete epochs -> train
      25% complete epochs -> validation
      25% complete epochs -> holdout

    No epoch_uid is ever split across outputs.
    """
    directory = Path(directory_name)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    files = find_processed_csvs(directory)
    if not files:
        raise FileNotFoundError(f"No *_processed.csv files found under {directory}")

    # Lock one canonical column order before writing any split files.
    first_frame = _read_csv_with_retries(
        files[0],
        retries,
        retry_wait_seconds,
    )

    master_source_columns = list(first_frame.columns)
    master_columns = list(master_source_columns)

    for split_column in ["subject_split", "sample_split"]:
        if split_column not in master_columns:
            master_columns.append(split_column)

    del first_frame

    output_directory = Path(output_directory) if output_directory else directory / "3B_Exact_40_25_Split"
    dev_folder = output_directory / "subject_train"
    holdout_folder = output_directory / "subject_holdout"
    dev_folder.mkdir(parents=True, exist_ok=True)
    holdout_folder.mkdir(parents=True, exist_ok=True)

    outputs = {
        "subject_train": {
            "train": dev_folder / "train.csv",
            "validation": dev_folder / "validation.csv",
            "holdout": dev_folder / "holdout.csv",
        },
        "subject_holdout": {
            "train": holdout_folder / "train.csv",
            "validation": holdout_folder / "validation.csv",
            "holdout": holdout_folder / "holdout.csv",
        },
    }
    manifest_file = output_directory / "subject_assignments.csv"

    if overwrite:
        for group in outputs.values():
            for path in group.values():
                if path.exists():
                    path.unlink()
        if manifest_file.exists():
            manifest_file.unlink()

    # Pass 1: index subjects without loading all data into RAM at once.
    subject_to_files = {}
    print("Discovering subjects...")
    for i, file_path in enumerate(files, start=1):
        print(f"Indexing file {i}/{len(files)}: {file_path.name}")
        try:
            frame = _read_csv_with_retries(file_path, retries, retry_wait_seconds)
        except OSError as error:
            print("SKIPPED unreadable file:", file_path.name, error)
            continue
        subject_id = _subject_from_filename(file_path)

        frame = _normalize_source_frame(
            frame,
            file_path,
            master_source_columns,
        )

        subject_to_files.setdefault(subject_id, []).append(file_path)
        del frame

    subjects = sorted(subject_to_files, key=lambda x: int(x) if str(x).isdigit() else str(x))
    required_total = development_subject_count + holdout_subject_count
    if len(subjects) < required_total:
        raise ValueError(f"Need {required_total} subjects, but found only {len(subjects)}")

    rng = random.Random(random_state)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    selected = shuffled[:required_total]

    development_subjects = selected[:development_subject_count]
    holdout_subjects = selected[development_subject_count:required_total]
    development_set = set(development_subjects)
    holdout_set = set(holdout_subjects)

    if development_set & holdout_set:
        raise RuntimeError("Subject overlap between development and holdout groups")

    print("\nSubject split:")
    print("Development subjects:", len(development_subjects))
    print("Holdout subjects:", len(holdout_subjects))

    manifest_rows = []
    epoch_counts = {
        "subject_train": {
            "train": 0,
            "validation": 0,
            "holdout": 0,
        },
        "subject_holdout": {
            "train": 0,
            "validation": 0,
            "holdout": 0,
        },
    }

    # Pass 2: one subject at a time.
    ordered_subjects = development_subjects + holdout_subjects
    for subject_number, subject_id in enumerate(ordered_subjects, start=1):
        group = "subject_train" if subject_id in development_set else "subject_holdout"
        print(f"\nSubject {subject_number}/{required_total}: {subject_id} -> {group}")

        subject_frames = []
        for file_path in subject_to_files[subject_id]:
            try:
                frame = _read_csv_with_retries(file_path, retries, retry_wait_seconds)
            except OSError as error:
                print("  SKIPPED unreadable file:", file_path.name, error)
                continue
            frame = _normalize_source_frame(
                frame,
                file_path,
                master_source_columns,
            )

            parsed_subject = _subject_from_filename(file_path)

            if str(parsed_subject) != str(subject_id):
                raise RuntimeError(
                    f"Filename subject mismatch: expected {subject_id}, "
                    f"got {parsed_subject} from {file_path.name}"
                )

            subject_frames.append(frame)

        if not subject_frames:
            raise RuntimeError(f"No readable files for subject {subject_id}")

        subject_data = pd.concat(subject_frames, ignore_index=True)
        del subject_frames

        subject_col = (
            "subject_id"
            if "subject_id" in subject_data.columns
            else "subject"
        )

        observed_subjects = set(
            subject_data[subject_col]
            .dropna()
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )

        expected_subject = str(subject_id).replace(".0", "")

        if observed_subjects != {expected_subject}:
            raise RuntimeError(
                f"Source normalization failed for subject {subject_id}. "
                f"Observed IDs={sorted(observed_subjects)}"
            )

        counts = subject_data.groupby("epoch_uid").size()
        valid_epoch_ids = counts[counts == 24].index.tolist()
        incomplete = int((counts != 24).sum())
        if incomplete:
            print(f"  Dropping {incomplete} incomplete epochs")

        subject_data = subject_data[subject_data["epoch_uid"].isin(valid_epoch_ids)].copy()

        if group == "subject_train":
            fractions = {
                "train": development_train_fraction,
                "validation": development_validation_fraction,
                "holdout": 1.0 - development_train_fraction - development_validation_fraction,
            }
        else:
            fractions = {
                "train": holdout_train_fraction,
                "validation": holdout_validation_fraction,
                "holdout": 1.0 - holdout_train_fraction - holdout_validation_fraction,
            }

        split_ids = _split_epoch_ids(valid_epoch_ids, fractions, random_state + sum(ord(c) for c in str(subject_id)))

        assigned = [epoch for ids in split_ids.values() for epoch in ids]
        if len(assigned) != len(set(assigned)):
            raise RuntimeError(f"Epoch overlap detected for subject {subject_id}")
        if set(assigned) != set(valid_epoch_ids):
            raise RuntimeError(f"Not every complete epoch was assigned for subject {subject_id}")

        for split_name, ids in split_ids.items():
            split_frame = subject_data[subject_data["epoch_uid"].isin(ids)].copy()
            split_frame["subject_split"] = group
            split_frame["sample_split"] = split_name
            _append_frame(split_frame, outputs[group][split_name], master_columns)
            epoch_counts[group][split_name] += len(ids)
            print(f"  {split_name}: {len(ids)} epochs / {len(split_frame)} rows")

        manifest_rows.append({
            "subject_id": subject_id,
            "subject_split": group,
            "total_complete_epochs": len(valid_epoch_ids),
        })
        del subject_data

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_file, index=False)

    actual_dev = manifest.loc[manifest["subject_split"] == "subject_train", "subject_id"].nunique()
    actual_holdout = manifest.loc[manifest["subject_split"] == "subject_holdout", "subject_id"].nunique()

    if actual_dev != development_subject_count:
        raise RuntimeError(f"Expected {development_subject_count} development subjects, got {actual_dev}")
    if actual_holdout != holdout_subject_count:
        raise RuntimeError(f"Expected {holdout_subject_count} holdout subjects, got {actual_holdout}")

    print("\n" + "=" * 72)
    print("3B EXACT 40/25 SPLIT COMPLETE")
    print("=" * 72)
    print("Development subjects:", actual_dev)
    print("Holdout subjects:", actual_holdout)
    print("Verification passed: no subject overlap; epochs were split only as whole epoch_uid blocks.")

    print("\nVerifying output CSV schemas, subjects, and epoch sizes...")

    expected_subjects = {
        "subject_train": set(
            manifest.loc[
                manifest["subject_split"] == "subject_train",
                "subject_id",
            ].astype(str)
        ),
        "subject_holdout": set(
            manifest.loc[
                manifest["subject_split"] == "subject_holdout",
                "subject_id",
            ].astype(str)
        ),
    }

    for group_name, group_outputs in outputs.items():
        for split_name, file_path in group_outputs.items():
            dtype_map = {
                name: "string"
                for name in ["subject_id", "subject", "epoch_uid"]
                if name in master_columns
            }

            check = pd.read_csv(
                file_path,
                low_memory=False,
                dtype=dtype_map if dtype_map else None,
            )

            if list(check.columns) != master_columns:
                raise RuntimeError(
                    f"Column schema mismatch in {file_path}"
                )

            subject_col = (
                "subject_id"
                if "subject_id" in check.columns
                else "subject"
            )

            actual_subjects = set(
                check[subject_col]
                .dropna()
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
            )

            normalized_expected = set(
                str(x).replace(".0", "")
                for x in expected_subjects[group_name]
            )

            missing_subjects = normalized_expected - actual_subjects
            unexpected_subjects = actual_subjects - normalized_expected

            if missing_subjects or unexpected_subjects:
                raise RuntimeError(
                    f"Subject membership verification failed for "
                    f"{group_name}/{split_name}. "
                    f"Missing={sorted(missing_subjects)} "
                    f"Unexpected={sorted(unexpected_subjects)}"
                )

            epoch_sizes = check.groupby("epoch_uid").size()
            bad_epochs = epoch_sizes[epoch_sizes != 24]

            if len(bad_epochs) > 0:
                raise RuntimeError(
                    f"Epoch size verification failed in "
                    f"{group_name}/{split_name}: "
                    f"{len(bad_epochs)} epochs are not 24 rows."
                )

            print(
                f"  PASS {group_name}/{split_name}: "
                f"{len(actual_subjects)} subjects, "
                f"{len(epoch_sizes)} epochs"
            )

    print(
        "\nAll verification passed: fixed column order, "
        "correct subject membership, 24 rows per epoch."
    )

    return {
        "outputs": outputs,
        "manifest": manifest,
        "manifest_file": manifest_file,
        "epoch_counts": epoch_counts,
    }
