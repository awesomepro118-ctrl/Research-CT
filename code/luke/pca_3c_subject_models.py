from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

METADATA_COLUMNS = {
    "dataset","subject","subject_id","recording","recording_id","label","condition",
    "source_file","absolute_load","load_in_sequence","load_in_epoch","epoch_id",
    "epoch_uid","row_in_epoch","time_seconds","epoch_start_seconds","event_count",
    "event_labels","chunk_id","subject_split","sample_split"
}

def _subject_column(data):
    if "subject_id" in data.columns:
        return "subject_id"
    if "subject" in data.columns:
        return "subject"
    raise ValueError("No subject_id or subject column found.")

def _feature_columns(data):
    numeric = data.select_dtypes(include=np.number).columns
    features = [c for c in numeric if c not in METADATA_COLUMNS]
    if len(features) != 95:
        raise ValueError(f"Expected 95 EEG feature columns, found {len(features)}.")
    return features

def flatten_subject_epochs(data):
    features = _feature_columns(data)
    rows = []

    for epoch_uid, epoch in data.groupby("epoch_uid", sort=False):
        epoch = epoch.sort_values("row_in_epoch")

        if len(epoch) != 24:
            continue

        if epoch["row_in_epoch"].astype(int).tolist() != list(range(24)):
            continue

        matrix = epoch[features].to_numpy(dtype=np.float32)

        if matrix.shape != (24, 95):
            continue

        first = epoch.iloc[0]
        row = {}

        for c in METADATA_COLUMNS:
            if c in epoch.columns and c != "row_in_epoch":
                row[c] = first[c]

        flat = matrix.reshape(-1)

        i = 0
        for t in range(24):
            for feature in features:
                row[f"t{t:02d}__{feature}"] = flat[i]
                i += 1

        rows.append(row)

    output = pd.DataFrame(rows)
    flat_features = [
        c for c in output.columns
        if c.startswith("t") and "__" in c
    ]

    if len(output) > 0 and len(flat_features) != 2280:
        raise ValueError(
            f"Expected 2280 flattened features, found {len(flat_features)}."
        )

    return output, flat_features

def _fit_subject_pca(train_subject_data, eval_subject_data, n_components):
    train_flat, train_features = flatten_subject_epochs(train_subject_data)
    eval_flat, eval_features = flatten_subject_epochs(eval_subject_data)

    if len(train_flat) == 0:
        raise ValueError("No valid training epochs.")
    if len(eval_flat) == 0:
        raise ValueError("No valid evaluation epochs.")
    if train_features != eval_features:
        raise ValueError("Training and evaluation flattened features differ.")

    X_train = train_flat[train_features].to_numpy(dtype=np.float32)
    X_eval = eval_flat[eval_features].to_numpy(dtype=np.float32)

    n = min(
        int(n_components),
        X_train.shape[0],
        X_train.shape[1],
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_eval_pca = pca.transform(X_eval_scaled)

    pcs = [f"PC{i}" for i in range(1, n + 1)]

    train_meta = train_flat.drop(columns=train_features).reset_index(drop=True)
    eval_meta = eval_flat.drop(columns=eval_features).reset_index(drop=True)

    train_out = pd.concat(
        [train_meta, pd.DataFrame(X_train_pca, columns=pcs)],
        axis=1,
    )

    eval_out = pd.concat(
        [eval_meta, pd.DataFrame(X_eval_pca, columns=pcs)],
        axis=1,
    )

    variance = pd.DataFrame({
        "principal_component": np.arange(1, n + 1),
        "eigenvalue": pca.explained_variance_,
        "explained_variance_percent": pca.explained_variance_ratio_ * 100,
        "cumulative_variance_percent": np.cumsum(
            pca.explained_variance_ratio_
        ) * 100,
    })

    return train_out, eval_out, variance

def run_training_subject_pca(
    hierarchical_split_folder,
    output_directory,
    n_components=20,
):
    """
    Hyperparameter-selection stage.

    For every SUBJECT_TRAIN subject:
      fit scaler + PCA on that subject's TRAIN epochs only
      transform that same subject's VALIDATION epochs

    Holdout subjects are not loaded.
    """
    split = Path(hierarchical_split_folder)

    train = pd.read_csv(
        split / "subject_train" / "train.csv"
    )

    validation = pd.read_csv(
        split / "subject_train" / "validation.csv"
    )

    subject_col = _subject_column(train)

    subjects = sorted(
        set(train[subject_col].dropna().astype(str))
        & set(validation[subject_col].dropna().astype(str))
    )

    root = (
        Path(output_directory)
        / f"pc_{int(n_components)}"
        / "training_subjects"
    )

    root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for number, subject in enumerate(subjects, start=1):
        print(f"Training subject {number}/{len(subjects)}: {subject}")

        subject_train = train[
            train[subject_col].astype(str) == subject
        ].copy()

        subject_validation = validation[
            validation[subject_col].astype(str) == subject
        ].copy()

        try:
            train_pca, validation_pca, variance = _fit_subject_pca(
                subject_train,
                subject_validation,
                n_components,
            )
        except Exception as error:
            print("  SKIPPED:", error)
            continue

        folder = root / f"sub-{subject}"
        folder.mkdir(parents=True, exist_ok=True)

        train_pca.to_csv(
            folder / "train_pca.csv",
            index=False,
        )

        validation_pca.to_csv(
            folder / "validation_pca.csv",
            index=False,
        )

        variance.to_csv(
            folder / "pca_variance.csv",
            index=False,
        )

        summary_rows.append({
            "subject_id": subject,
            "requested_components": int(n_components),
            "actual_components": len(
                [c for c in train_pca.columns if c.startswith("PC")]
            ),
            "train_samples": len(train_pca),
            "validation_samples": len(validation_pca),
            "variance_retained_percent": (
                variance["explained_variance_percent"].sum()
            ),
        })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        root / "training_subject_pca_summary.csv",
        index=False,
    )

    return root, summary

def run_holdout_subject_pca(
    hierarchical_split_folder,
    output_directory,
    n_components=20,
):
    """
    FINAL holdout stage.

    Only after hyperparameters are chosen.

    For every SUBJECT_HOLDOUT subject:
      fit scaler + PCA on that subject's TRAIN epochs only
      transform that same subject's HOLDOUT epochs

    subject_holdout/validation.csv is intentionally unused.
    """
    split = Path(hierarchical_split_folder)

    train = pd.read_csv(
        split / "subject_holdout" / "train.csv"
    )

    holdout = pd.read_csv(
        split / "subject_holdout" / "holdout.csv"
    )

    subject_col = _subject_column(train)

    subjects = sorted(
        set(train[subject_col].dropna().astype(str))
        & set(holdout[subject_col].dropna().astype(str))
    )

    root = (
        Path(output_directory)
        / f"pc_{int(n_components)}"
        / "holdout_subjects"
    )

    root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for number, subject in enumerate(subjects, start=1):
        print(f"Holdout subject {number}/{len(subjects)}: {subject}")

        subject_train = train[
            train[subject_col].astype(str) == subject
        ].copy()

        subject_holdout = holdout[
            holdout[subject_col].astype(str) == subject
        ].copy()

        try:
            train_pca, holdout_pca, variance = _fit_subject_pca(
                subject_train,
                subject_holdout,
                n_components,
            )
        except Exception as error:
            print("  SKIPPED:", error)
            continue

        folder = root / f"sub-{subject}"
        folder.mkdir(parents=True, exist_ok=True)

        train_pca.to_csv(
            folder / "train_pca.csv",
            index=False,
        )

        holdout_pca.to_csv(
            folder / "holdout_pca.csv",
            index=False,
        )

        variance.to_csv(
            folder / "pca_variance.csv",
            index=False,
        )

        summary_rows.append({
            "subject_id": subject,
            "requested_components": int(n_components),
            "actual_components": len(
                [c for c in train_pca.columns if c.startswith("PC")]
            ),
            "train_samples": len(train_pca),
            "holdout_samples": len(holdout_pca),
            "variance_retained_percent": (
                variance["explained_variance_percent"].sum()
            ),
        })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        root / "holdout_subject_pca_summary.csv",
        index=False,
    )

    return root, summary
