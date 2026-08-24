from pathlib import Path
import numpy as np
import pandas as pd
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
    paths = {
        'development_train': root / 'subject_train' / 'train.csv',
        'development_validation': root / 'subject_train' / 'validation.csv',
        'development_holdout': root / 'subject_train' / 'holdout.csv',
        'holdout_train': root / 'subject_holdout' / 'train.csv',
        'holdout_validation': root / 'subject_holdout' / 'validation.csv',
        'holdout_holdout': root / 'subject_holdout' / 'holdout.csv',
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing 3B split files:\n' + '\n'.join(missing))
    return paths


def inspect_schema(csv_path, target_column='load_in_sequence', epoch_rows=24, metadata_columns=None):
    csv_path = Path(csv_path)
    sample = pd.read_csv(csv_path, nrows=300, low_memory=False)
    subject_col = 'subject_id' if 'subject_id' in sample.columns else 'subject'
    if subject_col not in sample.columns:
        raise ValueError(f'{csv_path} has no subject_id or subject column')
    if 'epoch_uid' not in sample.columns:
        raise ValueError(f'{csv_path} has no epoch_uid column')
    if target_column not in sample.columns:
        target_like = [c for c in sample.columns if any(k in c.lower() for k in ['load','digit','target','label'])]
        raise ValueError(
            f'{csv_path} has no target column {target_column!r}. '
            f'Target-like columns found: {target_like}'
        )

    metadata = set(DEFAULT_METADATA_COLUMNS)
    if metadata_columns:
        metadata.update(metadata_columns)
    metadata.add(target_column)

    candidates = [c for c in sample.columns if c not in metadata]
    feature_columns = []
    for c in candidates:
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


def _finish_epoch_group(group, subject_col, target_column, feature_columns, epoch_rows):
    if len(group) != epoch_rows:
        return None
    vals = group[feature_columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    if not np.isfinite(vals).all():
        return None
    targets = pd.to_numeric(group[target_column], errors='coerce').dropna().to_numpy()
    if len(targets) == 0:
        return None
    # One epoch must have one target. If rows disagree, use the first but surface the problem.
    unique_targets = np.unique(targets)
    if len(unique_targets) != 1:
        raise ValueError(
            f"Epoch {group['epoch_uid'].iloc[0]} for subject {group[subject_col].iloc[0]} "
            f"contains multiple {target_column} values: {unique_targets.tolist()}"
        )
    subject = normalize_subject_id(group[subject_col].iloc[0])
    epoch_uid = str(group['epoch_uid'].iloc[0])
    target = float(unique_targets[0])
    return vals.reshape(-1), target, subject, epoch_uid


def flatten_split_csv(
    csv_path,
    cache_path,
    target_column='load_in_sequence',
    feature_columns=None,
    epoch_rows=24,
    chunk_rows=50000,
    force_rebuild=False,
):
    """Flatten every complete 24-row (subject, epoch_uid) block into one sample."""
    csv_path = Path(csv_path)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_rebuild:
        data = np.load(cache_path, allow_pickle=True)
        cached_target = str(data['target_column'].item()) if 'target_column' in data else ''
        if cached_target == target_column:
            return {
                'X': data['X'], 'y': data['y'], 'subjects': data['subjects'],
                'epoch_uids': data['epoch_uids'],
                'feature_columns': list(data['feature_columns'].astype(str)),
            }

    schema = inspect_schema(csv_path, target_column=target_column, epoch_rows=epoch_rows)
    subject_col = schema['subject_column']
    detected = schema['feature_columns']
    if feature_columns is None:
        feature_columns = detected

    usecols = [subject_col, 'epoch_uid', target_column] + list(feature_columns)
    X_parts, y_parts, subject_parts, epoch_parts = [], [], [], []
    carry = pd.DataFrame()

    for chunk_number, chunk in enumerate(
        pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_rows, low_memory=False), start=1
    ):
        if len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = pd.DataFrame()

        chunk[subject_col] = chunk[subject_col].map(normalize_subject_id)
        chunk['epoch_uid'] = chunk['epoch_uid'].astype('string')

        # Carry the final (subject, epoch_uid) group to the next chunk.
        last_subject = chunk[subject_col].iloc[-1]
        last_epoch = chunk['epoch_uid'].iloc[-1]
        last_mask = (chunk[subject_col] == last_subject) & (chunk['epoch_uid'] == last_epoch)
        carry = chunk[last_mask].copy()
        work = chunk[~last_mask].copy()

        for (_, _), g in work.groupby([subject_col, 'epoch_uid'], sort=False, dropna=False):
            result = _finish_epoch_group(g, subject_col, target_column, feature_columns, epoch_rows)
            if result is None:
                continue
            x, y, subject, epoch_uid = result
            X_parts.append(x)
            y_parts.append(y)
            subject_parts.append(subject)
            epoch_parts.append(epoch_uid)

        if chunk_number % 5 == 0:
            print(f'{csv_path.name}: chunks={chunk_number}, epochs={len(y_parts)}')

    if len(carry):
        result = _finish_epoch_group(carry, subject_col, target_column, feature_columns, epoch_rows)
        if result is not None:
            x, y, subject, epoch_uid = result
            X_parts.append(x)
            y_parts.append(y)
            subject_parts.append(subject)
            epoch_parts.append(epoch_uid)

    if not X_parts:
        raise RuntimeError(f'No complete usable epochs extracted from {csv_path}')

    X = np.asarray(X_parts, dtype=np.float32)
    y = np.asarray(y_parts, dtype=np.float32)
    subjects = np.asarray(subject_parts, dtype=str)
    epoch_uids = np.asarray(epoch_parts, dtype=str)

    np.savez_compressed(
        cache_path,
        X=X, y=y, subjects=subjects, epoch_uids=epoch_uids,
        feature_columns=np.asarray(feature_columns, dtype=str),
        target_column=np.asarray(target_column),
        epoch_rows=np.asarray(epoch_rows),
    )
    print(f'Saved {cache_path.name}: X={X.shape}, epochs={len(y)}, subjects={len(np.unique(subjects))}')
    return {
        'X': X, 'y': y, 'subjects': subjects, 'epoch_uids': epoch_uids,
        'feature_columns': list(feature_columns),
    }


def build_all_six_epoch_caches(
    split_folder,
    cache_folder,
    target_column='load_in_sequence',
    epoch_rows=24,
    chunk_rows=50000,
    force_rebuild=False,
):
    paths = get_exact_3b_paths(split_folder)
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)

    schema = inspect_schema(paths['development_train'], target_column=target_column, epoch_rows=epoch_rows)
    feature_columns = schema['feature_columns']
    print('Feature columns:', len(feature_columns))
    print('Flattened dimensions per epoch:', len(feature_columns) * epoch_rows)

    datasets = {}
    for name, csv_path in paths.items():
        datasets[name] = flatten_split_csv(
            csv_path,
            cache_folder / f'{name}.npz',
            target_column=target_column,
            feature_columns=feature_columns,
            epoch_rows=epoch_rows,
            chunk_rows=chunk_rows,
            force_rebuild=force_rebuild,
        )
    return datasets, feature_columns


def fit_subject_pca(X_train, X_other_list, pc_count=20):
    n_pc = min(pc_count, X_train.shape[0] - 1, X_train.shape[1])
    if n_pc < 1:
        raise ValueError('Not enough training samples for PCA.')
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    others_s = [scaler.transform(x) for x in X_other_list]
    pca = PCA(n_components=n_pc, random_state=1)
    Ztr = pca.fit_transform(Xtr_s)
    Zothers = [pca.transform(x) for x in others_s]
    return scaler, pca, Ztr, Zothers
