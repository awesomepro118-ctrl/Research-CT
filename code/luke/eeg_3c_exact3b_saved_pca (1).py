from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DEFAULT_METADATA_COLUMNS = {
    'subject_id','subject','subject_split','sample_split','epoch_uid','epoch','epoch_id','epoch_index',
    'absolute_load','load_in_sequence','load_in_epoch','digit_count','digit','label','target','class',
    'time','timestamp','sample','sample_index','row','row_index','task','condition','trial','trial_id',
    'file','filename'
}


def normalize_subject_id(value):
    s = str(value).strip()
    if s.endswith('.0'):
        s = s[:-2]
    if s.isdigit():
        return str(int(s))
    return s


def get_exact_3b_paths(split_folder):
    root = Path(split_folder)

    return {
        "development_train": root / "subject_train" / "train.csv",
        "development_validation": root / "subject_train" / "validation.csv",
        "development_holdout": root / "subject_train" / "holdout.csv",

        "holdout_train": root / "subject_holdout" / "train.csv",
        "holdout_validation": root / "subject_holdout" / "validation.csv",
        "holdout_holdout": root / "subject_holdout" / "holdout.csv",
    }


def inspect_schema(csv_path, target_column='load_in_sequence', epoch_rows=24):
    csv_path = Path(csv_path)
    sample = pd.read_csv(csv_path, nrows=500, low_memory=False)
    subject_col = 'subject_id' if 'subject_id' in sample.columns else 'subject'
    if subject_col not in sample.columns:
        raise ValueError(f'{csv_path} has no subject_id or subject column')
    if 'epoch_uid' not in sample.columns:
        raise ValueError(f'{csv_path} has no epoch_uid column')
    if target_column not in sample.columns:
        target_like = [c for c in sample.columns if any(k in c.lower() for k in ['load','digit','target','label'])]
        raise ValueError(f'{csv_path} has no {target_column!r}. Target-like columns: {target_like}')

    metadata = set(DEFAULT_METADATA_COLUMNS)
    metadata.add(target_column)
    feature_columns = []
    for c in sample.columns:
        if c in metadata:
            continue
        numeric = pd.to_numeric(sample[c], errors='coerce')
        if numeric.notna().mean() >= 0.95:
            feature_columns.append(c)
    if not feature_columns:
        raise RuntimeError('No numeric EEG feature columns detected.')
    return {
        'subject_column': subject_col,
        'feature_columns': feature_columns,
        'features_per_row': len(feature_columns),
        'flattened_features_per_epoch': len(feature_columns) * epoch_rows,
    }


def _finalize_epoch(group, subject_col, target_column, feature_columns, epoch_rows):
    if len(group) != epoch_rows:
        return None
    values = group[feature_columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        return None
    targets = pd.to_numeric(group[target_column], errors='coerce').dropna().to_numpy(dtype=np.float32)
    if len(targets) == 0:
        return None
    unique_targets = np.unique(targets)
    if len(unique_targets) != 1:
        raise ValueError(
            f"Epoch {group['epoch_uid'].iloc[0]} subject {group[subject_col].iloc[0]} "
            f"has multiple {target_column} values: {unique_targets.tolist()}"
        )
    return (
        values.reshape(-1),
        float(unique_targets[0]),
        normalize_subject_id(group[subject_col].iloc[0]),
        str(group['epoch_uid'].iloc[0]),
    )


def flatten_split_csv(csv_path, target_column, feature_columns, epoch_rows=24, chunk_rows=50000):
    """Read one 3B CSV and convert each complete 24-row epoch into one flattened sample."""
    csv_path = Path(csv_path)
    sample = pd.read_csv(csv_path, nrows=5, low_memory=False)
    subject_col = 'subject_id' if 'subject_id' in sample.columns else 'subject'
    usecols = [subject_col, 'epoch_uid', target_column] + list(feature_columns)

    X_parts, y_parts, subject_parts, epoch_parts = [], [], [], []
    carry = pd.DataFrame()

    for chunk_i, chunk in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_rows, low_memory=False), 1):
        if len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = pd.DataFrame()

        chunk[subject_col] = chunk[subject_col].map(normalize_subject_id)
        chunk['epoch_uid'] = chunk['epoch_uid'].astype('string')

        last_subject = chunk[subject_col].iloc[-1]
        last_epoch = chunk['epoch_uid'].iloc[-1]
        last_mask = (chunk[subject_col] == last_subject) & (chunk['epoch_uid'] == last_epoch)
        carry = chunk[last_mask].copy()
        work = chunk[~last_mask].copy()

        for _, group in work.groupby([subject_col, 'epoch_uid'], sort=False, dropna=False):
            result = _finalize_epoch(group, subject_col, target_column, feature_columns, epoch_rows)
            if result is None:
                continue
            x, y, subject, epoch_uid = result
            X_parts.append(x)
            y_parts.append(y)
            subject_parts.append(subject)
            epoch_parts.append(epoch_uid)

        if chunk_i % 5 == 0:
            print(f'  {csv_path.name}: chunk {chunk_i}, usable epochs={len(y_parts)}')

    if len(carry):
        result = _finalize_epoch(carry, subject_col, target_column, feature_columns, epoch_rows)
        if result is not None:
            x, y, subject, epoch_uid = result
            X_parts.append(x)
            y_parts.append(y)
            subject_parts.append(subject)
            epoch_parts.append(epoch_uid)

    if not X_parts:
        raise RuntimeError(f'No usable 24-row epochs extracted from {csv_path}')

    return {
        'X': np.asarray(X_parts, dtype=np.float32),
        'y': np.asarray(y_parts, dtype=np.float32),
        'subjects': np.asarray(subject_parts, dtype=str),
        'epoch_uids': np.asarray(epoch_parts, dtype=str),
    }


def _subject_mask(subject_array, subject):
    wanted = normalize_subject_id(subject)
    normalized = np.asarray([normalize_subject_id(x) for x in subject_array], dtype=str)
    return normalized == wanted


def _save_one_subject_pca(subject, group_name, train, validation, holdout, output_root, pc_count):
    train_mask = _subject_mask(train['subjects'], subject)
    val_mask = _subject_mask(validation['subjects'], subject)
    hold_mask = _subject_mask(holdout['subjects'], subject)

    X_train = train['X'][train_mask]
    y_train = train['y'][train_mask]
    e_train = train['epoch_uids'][train_mask]
    X_val = validation['X'][val_mask]
    y_val = validation['y'][val_mask]
    e_val = validation['epoch_uids'][val_mask]
    X_hold = holdout['X'][hold_mask]
    y_hold = holdout['y'][hold_mask]
    e_hold = holdout['epoch_uids'][hold_mask]

    counts = (len(X_train), len(X_val), len(X_hold))
    if min(counts) == 0:
        raise RuntimeError(
            f'Cannot fit/save PCA for {group_name} subject {subject}: '
            f'train/validation/holdout epochs={counts}. All three must be nonzero.'
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_hold_scaled = scaler.transform(X_hold)

    n_components = min(int(pc_count), X_train_scaled.shape[0], X_train_scaled.shape[1])
    if n_components < 1:
        raise RuntimeError(f'No valid PCA components for subject {subject}')
    pca = PCA(n_components=n_components, random_state=0)
    Z_train = pca.fit_transform(X_train_scaled).astype(np.float32)
    Z_val = pca.transform(X_val_scaled).astype(np.float32)
    Z_hold = pca.transform(X_hold_scaled).astype(np.float32)

    subject_dir = Path(output_root) / group_name / f'sub-{normalize_subject_id(subject).zfill(3)}'
    subject_dir.mkdir(parents=True, exist_ok=True)
    npz_path = subject_dir / 'pca_outputs.npz'
    np.savez_compressed(
        npz_path,
        X_train_pca=Z_train,
        X_validation_pca=Z_val,
        X_holdout_pca=Z_hold,
        y_train=y_train.astype(np.float32),
        y_validation=y_val.astype(np.float32),
        y_holdout=y_hold.astype(np.float32),
        epoch_uid_train=np.asarray(e_train, dtype=str),
        epoch_uid_validation=np.asarray(e_val, dtype=str),
        epoch_uid_holdout=np.asarray(e_hold, dtype=str),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        subject_id=np.asarray(normalize_subject_id(subject)),
        subject_group=np.asarray(group_name),
    )
    joblib.dump(scaler, subject_dir / 'scaler.joblib')
    joblib.dump(pca, subject_dir / 'pca.joblib')

    return {
        'subject_id': normalize_subject_id(subject),
        'subject_group': group_name,
        'train_epochs': len(Z_train),
        'validation_epochs': len(Z_val),
        'holdout_epochs': len(Z_hold),
        'pca_components': n_components,
        'npz_path': str(npz_path),
    }


def _validate_group_membership(train, validation, holdout, group_name):
    tr = {normalize_subject_id(x) for x in train['subjects']}
    va = {normalize_subject_id(x) for x in validation['subjects']}
    ho = {normalize_subject_id(x) for x in holdout['subjects']}
    missing_val = sorted(tr - va, key=lambda x: int(x) if x.isdigit() else x)
    missing_hold = sorted(tr - ho, key=lambda x: int(x) if x.isdigit() else x)
    if missing_val or missing_hold:
        raise RuntimeError(
            f'{group_name} split membership is incomplete before PCA. '
            f'Missing from validation={missing_val}; missing from holdout={missing_hold}'
        )
    return sorted(tr, key=lambda x: int(x) if x.isdigit() else x)


def build_and_save_pca_from_exact3b(
    split_folder,
    pca_output_folder,
    target_column='load_in_sequence',
    epoch_rows=24,
    pc_count=20,
    chunk_rows=50000,
    include_holdout_subjects=False,
):
    """
    Fresh 3C pipeline. No old cache reuse.
    Reads current repaired 3B CSVs, flattens 24 rows -> 1 epoch, fits PCA per subject,
    and saves train/validation/holdout PCA arrays for later model notebooks.
    """
    paths = get_exact_3b_paths(split_folder)
    schema = inspect_schema(paths['development_train'], target_column=target_column, epoch_rows=epoch_rows)
    feature_columns = schema['feature_columns']
    print('Feature columns per row:', len(feature_columns))
    print('Flattened dimensions per epoch:', len(feature_columns) * epoch_rows)
    print('Target:', target_column)

    print('\nReading DEVELOPMENT train...')
    dev_train = flatten_split_csv(paths['development_train'], target_column, feature_columns, epoch_rows, chunk_rows)
    print('Reading DEVELOPMENT validation...')
    dev_val = flatten_split_csv(paths['development_validation'], target_column, feature_columns, epoch_rows, chunk_rows)
    print('Reading DEVELOPMENT holdout...')
    dev_hold = flatten_split_csv(paths['development_holdout'], target_column, feature_columns, epoch_rows, chunk_rows)

    dev_subjects = _validate_group_membership(dev_train, dev_val, dev_hold, 'development')
    print(f'\nDevelopment subjects verified: {len(dev_subjects)}')

    manifest_rows = []
    for i, subject in enumerate(dev_subjects, 1):
        row = _save_one_subject_pca(subject, 'development', dev_train, dev_val, dev_hold, pca_output_folder, pc_count)
        manifest_rows.append(row)
        print(
            f"development {i}/{len(dev_subjects)} sub-{subject.zfill(3)} saved | "
            f"train/val/hold={row['train_epochs']}/{row['validation_epochs']}/{row['holdout_epochs']} | "
            f"PCs={row['pca_components']}"
        )

    if include_holdout_subjects:
        print('\nReading HOLDOUT-SUBJECT train...')
        h_train = flatten_split_csv(paths['holdout_train'], target_column, feature_columns, epoch_rows, chunk_rows)
        print('Reading HOLDOUT-SUBJECT validation...')
        h_val = flatten_split_csv(paths['holdout_validation'], target_column, feature_columns, epoch_rows, chunk_rows)
        print('Reading HOLDOUT-SUBJECT holdout...')
        h_hold = flatten_split_csv(paths['holdout_holdout'], target_column, feature_columns, epoch_rows, chunk_rows)
        h_subjects = _validate_group_membership(h_train, h_val, h_hold, 'holdout_subjects')
        print(f'\nHoldout subjects verified: {len(h_subjects)}')
        for i, subject in enumerate(h_subjects, 1):
            row = _save_one_subject_pca(subject, 'holdout_subjects', h_train, h_val, h_hold, pca_output_folder, pc_count)
            manifest_rows.append(row)
            print(
                f"holdout_subjects {i}/{len(h_subjects)} sub-{subject.zfill(3)} saved | "
                f"train/val/hold={row['train_epochs']}/{row['validation_epochs']}/{row['holdout_epochs']} | "
                f"PCs={row['pca_components']}"
            )

    output_root = Path(pca_output_folder)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_root / 'pca_manifest.csv'
    manifest.to_csv(manifest_path, index=False)
    settings_path = output_root / 'pca_settings.json'
    settings_path.write_text(json.dumps({
        'split_folder': str(Path(split_folder)),
        'target_column': target_column,
        'epoch_rows': int(epoch_rows),
        'feature_columns': feature_columns,
        'features_per_row': len(feature_columns),
        'flattened_features_per_epoch': len(feature_columns) * int(epoch_rows),
        'requested_pc_count': int(pc_count),
        'include_holdout_subjects': bool(include_holdout_subjects),
    }, indent=2))
    print('\n3C COMPLETE — saved PCA outputs to:', output_root)
    print('Manifest:', manifest_path)
    return manifest


def load_saved_subject_pca(pca_output_folder, subject_group, subject_id):
    path = Path(pca_output_folder) / subject_group / f'sub-{normalize_subject_id(subject_id).zfill(3)}' / 'pca_outputs.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
