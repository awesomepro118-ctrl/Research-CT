import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from pathlib import Path

# -------------------------------------------------
# Configuration
# -------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# >>> Your consolidated file <<<
CSV_PATH = r"e:\research\combined_processed.csv"

ELECTRODES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T7','C3','Cz','C4','T8',
    'P7','P3','Pz','P4','P8','O1','O2'
]
BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

LABEL_NAMES = {0: 'Rest', 1: 'Low', 2: 'Medium', 3: 'High'}

# -------------------------------------------------
# 1. Label function
# -------------------------------------------------
def get_activity_label(row):
    if str(row['label']).lower() == 'rest':
        return 0  # Rest

    pos = row['load_in_sequence']
    if pd.isna(pos):
        return -1

    pos = int(pos)
    if pos <= 3:
        return 1  # Low
    elif 5 <= pos <= 7:
        return 2  # Medium
    elif pos >= 8:
        return 3  # High
    else:
        return -1  # skip position 4


# -------------------------------------------------
# 2. Dataset (single consolidated file)
# -------------------------------------------------
class EEGEpochDataset(Dataset):
    def __init__(self, csv_path):
        print(f"Loading {csv_path} ...")
        df = pd.read_csv(csv_path)
        print(f"Total rows: {len(df)}")

        self.samples = []

        for epoch_uid, group in df.groupby('epoch_uid'):
            group = group.sort_values('row_in_epoch')

            # Build (T, 19, 5) tensor
            try:
                feats = []
                for band in BANDS:
                    cols = [f'{e}_{band}' for e in ELECTRODES]
                    feats.append(group[cols].values.astype(np.float32))
                x = np.stack(feats, axis=-1)   # (T, 19, 5)
            except Exception as e:
                print(f"Skipping epoch {epoch_uid}: {e}")
                continue

            y = get_activity_label(group.iloc[0])
            if y == -1:
                continue

            subject = str(group['subject_id'].iloc[0])
            self.samples.append((torch.from_numpy(x), y, subject))

        print(f"Usable epochs: {len(self.samples)}")

        # Quick class distribution
        from collections import Counter
        labels = [s[1] for s in self.samples]
        print("Class distribution:", {LABEL_NAMES[k]: v for k, v in Counter(labels).items()})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -------------------------------------------------
# 3. 70/30 split – every subject appears in both sets
# -------------------------------------------------
def subject_aware_split(dataset, train_ratio=0.7):
    subject_to_indices = defaultdict(list)
    for idx, (_, _, subj) in enumerate(dataset.samples):
        subject_to_indices[subj].append(idx)

    train_indices, val_indices = [], []

    for subj, indices in subject_to_indices.items():
        random.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])

    print(f"\nNumber of subjects: {len(subject_to_indices)}")
    print(f"Train epochs: {len(train_indices)} | Val epochs: {len(val_indices)}")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# -------------------------------------------------
# 4. Vector Quantizer (Soft / Hard)
# -------------------------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=64, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z, soft=True, temperature=1.0):
        dist = (z.pow(2).sum(1, keepdim=True)
                - 2 * z @ self.embeddings.weight.t()
                + self.embeddings.weight.pow(2).sum(1))

        if soft:
            weights = F.softmax(-dist / temperature, dim=1)
            z_q = weights @ self.embeddings.weight
            indices = weights.argmax(dim=1)
        else:
            indices = dist.argmin(dim=1)
            z_q = self.embeddings(indices)
            z_q = z + (z_q - z).detach()  # straight-through estimator

        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(z_q.detach(), z)
        return z_q, loss, indices


# -------------------------------------------------
# 5. Model
# -------------------------------------------------
class EEGPoolingVQ(nn.Module):
    def __init__(self, num_classes=4, codebook_size=64):
        super().__init__()

        self.spatial = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((12, 8))
        )

        self.temporal = nn.Sequential(
            nn.Conv1d(64 * 8, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.vq = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, soft_vq=True, temperature=1.0):
        # x: (B, T, 19, 5) → (B, 5, T, 19)
        x = x.permute(0, 3, 1, 2)

        x = self.spatial(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C * W, H)
        x = self.temporal(x).squeeze(-1)

        z_q, vq_loss, indices = self.vq(x, soft=soft_vq, temperature=temperature)
        logits = self.classifier(z_q)
        return logits, vq_loss, indices


# -------------------------------------------------
# 6. Training loop
# -------------------------------------------------
def train():
    dataset = EEGEpochDataset(CSV_PATH)
    train_set, val_set = subject_aware_split(dataset, train_ratio=0.7)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = EEGPoolingVQ(num_classes=4, codebook_size=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 31):
        model.train()
        soft = epoch <= 20
        temperature = max(0.5, 1.0 - (epoch - 1) * 0.025)

        total_loss, correct, total = 0, 0, 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)

            logits, vq_loss, _ = model(x, soft_vq=soft, temperature=temperature)
            cls_loss = F.cross_entropy(logits, y)
            loss = cls_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        train_acc = correct / total

        # Validation (always Hard VQ)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _, _ = model(x, soft_vq=False)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch:02d} | Soft={soft} | Temp={temperature:.2f} | "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    torch.save(model.state_dict(), "eeg_pooling_vq_4class.pt")
    print("\nModel saved as eeg_pooling_vq_4class.pt")


if __name__ == "__main__":
    train()