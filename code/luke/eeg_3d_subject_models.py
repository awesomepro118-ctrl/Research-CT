from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, accuracy_score

from eeg_3c_epoch_pca import fit_subject_pca

DEFAULT_HYPERPARAMETERS = {
    'logistic': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 3000},
    'knn': {'k': 31, 'weights': 'distance', 'p': 1},
    'svm': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'class_weight': 'balanced'},
}


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
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
        'SVM': SVC(
            kernel=hp['svm']['kernel'], C=hp['svm']['C'], gamma=hp['svm']['gamma'],
            class_weight=hp['svm']['class_weight']
        ),
    }


def _metric_row(subject, subject_group, model_name, split_name, y_true, y_pred):
    return {
        'subject_id': subject,
        'subject_group': subject_group,
        'model': model_name,
        'split': split_name,
        'n_epochs': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'spearman': safe_spearman(y_true, y_pred),
    }


def _prediction_frame(subject, subject_group, model_name, split_name, epoch_uids, y_true, y_pred):
    return pd.DataFrame({
        'subject_id': subject,
        'subject_group': subject_group,
        'split': split_name,
        'epoch_uid': np.asarray(epoch_uids, dtype=str),
        'target': np.asarray(y_true),
        'prediction': np.asarray(y_pred),
        'model': model_name,
    })


def fit_subject_group_models(
    train_data,
    validation_data,
    holdout_data,
    subject_group,
    pc_count=20,
    hyperparameters=None,
):
    """Fit a separate scaler + PCA + KNN/Logistic/SVM for each subject."""
    subjects = sorted(set(train_data['subjects']), key=lambda x: int(x) if str(x).isdigit() else str(x))
    metric_rows = []
    prediction_frames = []

    for i, subject in enumerate(subjects, start=1):
        masks = {
            'train': train_data['subjects'] == subject,
            'validation': validation_data['subjects'] == subject,
            'holdout': holdout_data['subjects'] == subject,
        }
        split_data = {
            'train': train_data,
            'validation': validation_data,
            'holdout': holdout_data,
        }
        Xtr = train_data['X'][masks['train']]
        ytr = train_data['y'][masks['train']]
        if len(Xtr) == 0:
            continue

        Xva = validation_data['X'][masks['validation']]
        Xho = holdout_data['X'][masks['holdout']]
        _, _, Ztr, [Zva, Zho] = fit_subject_pca(Xtr, [Xva, Xho], pc_count=pc_count)
        Zs = {'train': Ztr, 'validation': Zva, 'holdout': Zho}

        models = build_models(hyperparameters)
        for model_name, model in models.items():
            model.fit(Ztr, ytr)
            for split_name in ['train', 'validation', 'holdout']:
                mask = masks[split_name]
                data = split_data[split_name]
                y_true = data['y'][mask]
                epoch_uids = data['epoch_uids'][mask]
                if len(y_true) == 0:
                    continue
                y_pred = model.predict(Zs[split_name])
                metric_rows.append(_metric_row(
                    subject, subject_group, model_name, split_name, y_true, y_pred
                ))
                prediction_frames.append(_prediction_frame(
                    subject, subject_group, model_name, split_name, epoch_uids, y_true, y_pred
                ))
        print(f'{subject_group}: subject {i}/{len(subjects)} = {subject} complete')

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return metrics, predictions


def save_group_results(metrics, predictions, output_folder, prefix):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    metrics_path = output_folder / f'{prefix}_metrics.csv'
    predictions_path = output_folder / f'{prefix}_epoch_predictions_long.csv'
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    # One row per epoch with all three model predictions side by side.
    if len(predictions):
        wide = predictions.pivot_table(
            index=['subject_id','subject_group','split','epoch_uid','target'],
            columns='model', values='prediction', aggfunc='first'
        ).reset_index()
        rename = {
            'KNN': 'knn_prediction',
            'Logistic Regression': 'logistic_prediction',
            'SVM': 'svm_prediction',
        }
        wide = wide.rename(columns=rename)
        wide_path = output_folder / f'{prefix}_epoch_predictions_wide.csv'
        wide.to_csv(wide_path, index=False)
    else:
        wide_path = None

    summary = (
        metrics.groupby(['model','split'])[['accuracy','mae','spearman']]
        .agg(['mean','std']).round(4)
    ) if len(metrics) else pd.DataFrame()
    summary_path = output_folder / f'{prefix}_summary.csv'
    summary.to_csv(summary_path)
    return {
        'metrics': metrics_path,
        'predictions_long': predictions_path,
        'predictions_wide': wide_path,
        'summary': summary_path,
    }


def run_development_models(datasets, output_folder, pc_count=20, hyperparameters=None):
    metrics, predictions = fit_subject_group_models(
        datasets['development_train'], datasets['development_validation'], datasets['development_holdout'],
        subject_group='development', pc_count=pc_count, hyperparameters=hyperparameters
    )
    paths = save_group_results(metrics, predictions, output_folder, 'development')
    return metrics, predictions, paths


def run_holdout_subject_models(datasets, output_folder, pc_count=20, hyperparameters=None):
    metrics, predictions = fit_subject_group_models(
        datasets['holdout_train'], datasets['holdout_validation'], datasets['holdout_holdout'],
        subject_group='holdout_subjects', pc_count=pc_count, hyperparameters=hyperparameters
    )
    paths = save_group_results(metrics, predictions, output_folder, 'holdout_subjects')
    return metrics, predictions, paths


def combine_all_six_epoch_predictions(output_folder):
    """Combine development + holdout-subject wide files into one CSV covering all 6 datasets."""
    output_folder = Path(output_folder)
    parts = []
    for name in ['development_epoch_predictions_wide.csv', 'holdout_subjects_epoch_predictions_wide.csv']:
        p = output_folder / name
        if p.exists():
            parts.append(pd.read_csv(p, low_memory=False))
    if not parts:
        raise FileNotFoundError('No epoch prediction files found to combine.')
    all_predictions = pd.concat(parts, ignore_index=True)
    path = output_folder / 'all_six_datasets_epoch_predictions.csv'
    all_predictions.to_csv(path, index=False)
    return path, all_predictions
