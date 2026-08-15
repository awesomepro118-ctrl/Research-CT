import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random

# -------------------------------------------------
# Configuration
# -------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CSV_PATH = r"C:\Users\abiga\Downloads\research\test.csv"

ELECTRODES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T7','C3','Cz','C4','T8',
    'P7','P3','Pz','P4','P8','O1','O2'
]
BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

LABEL_NAMES = {0: 'Low', 1: 'High'}

# -------------------------------------------------
# Label function – ONLY Low vs High
# -------------------------------------------------
def get_activity_label(row):
    if str(row['label']).lower() == 'rest':
        return -1  # ignore Rest

    pos = row.get('load_in_sequence', None)
    if pd.isna(pos):
        return -1

    pos = int(pos)
    if pos <= 3:
        return 0  # Low
    elif pos >= 8:
        return 1  # High
    else:
        return -1  # ignore Medium (4-7) and position 4


# -------------------------------------------------
# Dataset with per-subject normalization
# -------------------------------------------------
class EEGEpochDataset(Dataset):
    def __init__(self, csv_path):
        print(f"Loading {csv_path} ...")
        df = pd.read_csv(csv_path)
        print(f"Total rows: {len(df)}")

        subject_data = defaultdict(list)

        for epoch_uid, group in df.groupby('epoch_uid'):
            group = group.sort_values('row_in_epoch')

            try:
                feats = []
                for band in BANDS:
                    cols = [f'{e}_{band}' for e in ELECTRODES]
                    feats.append(group[cols].values.astype(np.float32))
                x = np.stack(feats, axis=-1)  # (T, 19, 5)
            except Exception:
                continue

            y = get_activity_label(group.iloc[0])
            if y == -1:
                continue

            subject = str(group['subject_id'].iloc[0])
            subject_data[subject].append((x, y))

        # Per-subject log + z-score normalization
        self.samples = []
        for subject, epochs in subject_data.items():
            all_x = np.concatenate([ep[0] for ep in epochs], axis=0)
            all_x = np.log10(all_x + 1e-15)

            mean = all_x.mean(axis=0, keepdims=True)
            std  = all_x.std(axis=0, keepdims=True) + 1e-8

            for x, y in epochs:
                x = np.log10(x + 1e-15)
                x = (x - mean) / std
                self.samples.append((torch.from_numpy(x.astype(np.float32)), y, subject))

        print(f"Usable epochs (Low + High only): {len(self.samples)}")
        labels = [s[1] for s in self.samples]
        print("Class distribution:", {LABEL_NAMES[k]: v for k, v in Counter(labels).items()})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -------------------------------------------------
# Continuous 70/30 split per subject
# -------------------------------------------------
def subject_aware_split(dataset, train_ratio=0.7):
    subject_to_indices = defaultdict(list)
    for idx, (_, _, subj) in enumerate(dataset.samples):
        subject_to_indices[subj].append(idx)

    train_indices, val_indices = [], []

    for subj, indices in subject_to_indices.items():
        n_train = int(len(indices) * train_ratio)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])

    print(f"\nNumber of subjects: {len(subject_to_indices)}")
    print(f"Train epochs: {len(train_indices)} | Val epochs: {len(val_indices)}")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# -------------------------------------------------
# CNN + Pooling model
# -------------------------------------------------
class EEGPoolingCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # (B, T, 19, 5) → (B, 5, T, 19)
        x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        return self.classifier(x)


# -------------------------------------------------
# Training
# -------------------------------------------------
def train():
    dataset = EEGEpochDataset(CSV_PATH)
    train_set, val_set = subject_aware_split(dataset, train_ratio=0.7)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'xpu')
    print(f"Using device: {device}")

    model = EEGPoolingCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Class weights
    label_counts = Counter([s[1] for s in dataset.samples])
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / max(label_counts[i], 1) for i in range(2)],
        dtype=torch.float32
    ).to(device)
    print("Class weights:", np.round(class_weights.cpu().numpy(), 2))

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0

    for epoch in range(1, 41):
        # ---- Train ----
        model.train()
        correct, total = 0, 0
        running_loss = 0.0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        # ---- Validation ----
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(1)

                val_correct += (preds == y).sum().item()
                val_total += x.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        val_acc = val_correct / val_total
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Per-class accuracy
        low_acc  = (all_preds[all_labels == 0] == 0).float().mean().item() if (all_labels == 0).sum() > 0 else float('nan')
        high_acc = (all_preds[all_labels == 1] == 1).float().mean().item() if (all_labels == 1).sum() > 0 else float('nan')

        print(f"\nEpoch {epoch:02d}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
        print(f"  Val Acc: {val_acc:.3f}")
        print(f"  Per-class Val Acc → Low: {low_acc:.3f} | High: {high_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "eeg_low_vs_high_best.pt")
            print("  → Best model saved")

    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    train()