from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, accuracy_score

DEFAULT_HYPERPARAMETERS = {
    'logistic': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 3000},
    'knn': {'k': 31, 'weights': 'distance', 'p': 1},
    'svm': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'epsilon': 0.1},
}


def normalize_subject_id(value):
    s = str(value).strip()
    if s.endswith('.0'):
        s = s[:-2]
    if s.isdigit():
        return str(int(s))
    return s


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return np.nan
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return np.nan
    return float(spearmanr(y_true, y_pred).statistic)


def build_models(hp=None):
    hp = hp or DEFAULT_HYPERPARAMETERS
    return {
        'KNN': KNeighborsClassifier(
            n_neighbors=hp['knn']['k'], weights=hp['knn']['weights'], p=hp['knn']['p']
        ),
        'Logistic Regression': LogisticRegression(
            C=hp['logistic']['C'], class_weight=hp['logistic']['class_weight'],
            max_iter=hp['logistic']['max_iter']
        ),
        'SVM': SVR(
            kernel=hp['svm']['kernel'],
            C=hp['svm']['C'],
            gamma=hp['svm']['gamma'],
            epsilon=hp['svm']['epsilon'],
        ),
    }


def discover_saved_pca_subjects(pca_output_folder, subject_group):
    root = Path(pca_output_folder) / subject_group
    if not root.exists():
        return []
    subjects = []
    for p in root.glob('sub-*/pca_outputs.npz'):
        sid = p.parent.name.replace('sub-', '')
        subjects.append(normalize_subject_id(sid))
    return sorted(set(subjects), key=lambda x: int(x) if x.isdigit() else x)


def load_subject_pca(pca_output_folder, subject_group, subject):
    path = Path(pca_output_folder) / subject_group / f'sub-{normalize_subject_id(subject).zfill(3)}' / 'pca_outputs.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    return {
        'train': (d['X_train_pca'], d['y_train'], d['epoch_uid_train']),
        'validation': (d['X_validation_pca'], d['y_validation'], d['epoch_uid_validation']),
        'holdout': (d['X_holdout_pca'], d['y_holdout'], d['epoch_uid_holdout']),
    }


def _metric_row(subject, subject_group, model_name, split_name, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # SVR produces continuous predictions.
    # For accuracy only, round to the nearest valid target class 1-13.
    if model_name == 'SVM':
        accuracy_pred = np.clip(np.rint(y_pred), 1, 13).astype(int)
    else:
        accuracy_pred = y_pred

    return {
        'subject_id': normalize_subject_id(subject),
        'subject_group': subject_group,
        'model': model_name,
        'split': split_name,
        'n_epochs': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, accuracy_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'spearman': safe_spearman(y_true, y_pred),
    }


def _prediction_frame(subject, subject_group, model_name, split_name, epoch_uids, y_true, y_pred):
    return pd.DataFrame({
        'subject_id': normalize_subject_id(subject),
        'subject_group': subject_group,
        'split': split_name,
        'epoch_uid': np.asarray(epoch_uids, dtype=str),
        'target': np.asarray(y_true),
        'prediction': np.asarray(y_pred),
        'model': model_name,
    })


def run_models_from_saved_pca(
    pca_output_folder,
    output_folder,
    subject_group='development',
    hyperparameters=None,
):
    subjects = discover_saved_pca_subjects(pca_output_folder, subject_group)
    if not subjects:
        raise FileNotFoundError(f'No saved PCA subjects found under {Path(pca_output_folder) / subject_group}')

    metric_rows = []
    prediction_frames = []
    for i, subject in enumerate(subjects, 1):
        splits = load_subject_pca(pca_output_folder, subject_group, subject)
        Xtr, ytr, _ = splits['train']
        if len(Xtr) == 0:
            raise RuntimeError(f'Empty PCA training set for subject {subject}')

        models = build_models(hyperparameters)
        for model_name, model in models.items():
            model.fit(Xtr, ytr)
            for split_name in ['train', 'validation', 'holdout']:
                X, y, epoch_uids = splits[split_name]
                if len(X) == 0:
                    raise RuntimeError(f'Empty saved PCA split: {subject_group} subject {subject} {split_name}')
                pred = model.predict(X)
                metric_rows.append(_metric_row(subject, subject_group, model_name, split_name, y, pred))
                prediction_frames.append(_prediction_frame(subject, subject_group, model_name, split_name, epoch_uids, y, pred))

        print(f'{subject_group}: subject {i}/{len(subjects)} sub-{subject.zfill(3)} complete')

    metrics = pd.DataFrame(metric_rows)
    long_predictions = pd.concat(prediction_frames, ignore_index=True)
    wide_predictions = long_predictions.pivot_table(
        index=['subject_id','subject_group','split','epoch_uid','target'],
        columns='model', values='prediction', aggfunc='first'
    ).reset_index().rename(columns={
        'KNN': 'knn_prediction',
        'Logistic Regression': 'logistic_prediction',
        'SVM': 'svm_prediction',
    })

    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / f'{subject_group}_metrics.csv'
    long_path = out / f'{subject_group}_epoch_predictions_long.csv'
    wide_path = out / f'{subject_group}_epoch_predictions_wide.csv'
    summary_path = out / f'{subject_group}_summary.csv'
    metrics.to_csv(metrics_path, index=False)
    long_predictions.to_csv(long_path, index=False)
    wide_predictions.to_csv(wide_path, index=False)
    summary = metrics.groupby(['model','split'])[['accuracy','mae','spearman']].agg(['mean','std']).round(4)
    summary.to_csv(summary_path)

    return metrics, wide_predictions, {
        'metrics': metrics_path,
        'predictions_long': long_path,
        'predictions_wide': wide_path,
        'summary': summary_path,
    }


def combine_all_six_predictions(output_folder):
    out = Path(output_folder)
    dev = out / 'development_epoch_predictions_wide.csv'
    hold = out / 'holdout_subjects_epoch_predictions_wide.csv'
    missing = [str(p) for p in [dev, hold] if not p.exists()]
    if missing:
        raise FileNotFoundError('Need both development and holdout-subject predictions first:\n' + '\n'.join(missing))
    combined = pd.concat([pd.read_csv(dev), pd.read_csv(hold)], ignore_index=True)
    path = out / 'all_six_datasets_epoch_predictions.csv'
    combined.to_csv(path, index=False)
    return path, combined
