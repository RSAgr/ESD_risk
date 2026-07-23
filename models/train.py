"""
=============================================================
Training Pipeline  (Member 2)
=============================================================
Covers:
  1. Dataset loading + windowing
  2. Label encoding for categoricals
  3. Class-imbalance handling (weighted CrossEntropy)
  4. Training loop with LR scheduler + early stopping
  5. Validation metrics (Accuracy, F1, AUC-ROC)
  6. Baseline comparisons (LSTM, SVM, Random Forest)
  7. Ablation study helpers
  8. Model checkpoint saving
=============================================================
"""

import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, f1_score,
    roc_auc_score, accuracy_score, confusion_matrix
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from models.attention_model import build_model, ESDAttentionModel

# ── config ────────────────────────────────────────────────────
WINDOW_SIZE  = 50      # 50 time steps = 1 second at 50Hz
STRIDE       = 10      # sliding window stride
BATCH_SIZE   = 64
EPOCHS       = 40
LR           = 3e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH    = PROJECT_ROOT / "data" / "esd_dataset.csv"
CKPT_PATH    = PROJECT_ROOT / "models" / "best_model.pt"

NUMERICAL_COLS   = ["humidity_pct", "temperature_c", "efield_vm", "contact_voltage_v", "movement_g"]
CATEGORICAL_COLS = ["activity", "environment", "fabric_type"]
LABEL_COL        = "risk_label"


# ── dataset ───────────────────────────────────────────────────
class ESDWindowDataset(Dataset):
    """Sliding-window dataset from multi-modal sensor CSV."""

    def __init__(self, df, encoders, scaler, window=50, stride=10):
        self.windows = []
        self.cat_windows = []
        self.labels = []

        # Scale numerical
        num_scaled = scaler.transform(df[NUMERICAL_COLS].values)

        # Encode categoricals (take mode over window)
        cat_encoded = {}
        for col in CATEGORICAL_COLS:
            cat_encoded[col] = encoders[col].transform(df[col].astype(str).values)

        labels = df[LABEL_COL].values
        N = len(df)

        for start in range(0, N - window, stride):
            end = start + window
            self.windows.append(num_scaled[start:end])

            cat_dict = {}
            for col in CATEGORICAL_COLS:
                # Use the most frequent category in the window
                vals = cat_encoded[col][start:end]
                mode_val = int(np.bincount(vals).argmax())
                cat_dict[col] = mode_val
            self.cat_windows.append(cat_dict)

            # Label = most frequent in window
            window_labels = labels[start:end]
            self.labels.append(int(np.bincount(window_labels).argmax()))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        num  = torch.tensor(self.windows[idx], dtype=torch.float32)
        cats = {k: torch.tensor(v, dtype=torch.long)
                for k, v in self.cat_windows[idx].items()}
        lbl  = torch.tensor(self.labels[idx], dtype=torch.long)
        return num, cats, lbl


def collate_fn(batch):
    nums, cats, lbls = zip(*batch)
    num_tensor = torch.stack(nums)
    lbl_tensor = torch.stack(lbls)
    cat_tensor = {
        key: torch.stack([c[key] for c in cats])
        for key in cats[0]
    }
    return num_tensor, cat_tensor, lbl_tensor


# ── data preparation ──────────────────────────────────────────
def prepare_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Fit label encoders
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le

    # Train / val / test split  (70 / 15 / 15)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df[LABEL_COL])
    val_df,   test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df[LABEL_COL])

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERICAL_COLS].values)

    train_ds = ESDWindowDataset(train_df, encoders, scaler, WINDOW_SIZE, STRIDE)
    val_ds   = ESDWindowDataset(val_df,   encoders, scaler, WINDOW_SIZE, STRIDE)
    test_ds  = ESDWindowDataset(test_df,  encoders, scaler, WINDOW_SIZE, STRIDE)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Class weights for imbalance
    label_counts = np.bincount(train_ds.labels, minlength=3)
    label_counts = np.maximum(label_counts, 1)
    class_weights = torch.tensor(
        1.0 / (label_counts / label_counts.sum()), dtype=torch.float32
    ).to(DEVICE)

    print(f"  Train windows : {len(train_ds):,}")
    print(f"  Val windows   : {len(val_ds):,}")
    print(f"  Test windows  : {len(test_ds):,}")

    return train_loader, val_loader, test_loader, class_weights, scaler, encoders, test_ds


# ── training loop ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for num, cats, lbl in loader:
        num  = num.to(DEVICE)
        cats = {k: v.to(DEVICE) for k, v in cats.items()}
        lbl  = lbl.to(DEVICE)

        optimizer.zero_grad()
        logits = model(num, cats)
        loss   = criterion(logits, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * lbl.size(0)
        correct    += (logits.argmax(1) == lbl).sum().item()
        total      += lbl.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for num, cats, lbl in loader:
        num  = num.to(DEVICE)
        cats = {k: v.to(DEVICE) for k, v in cats.items()}
        logits = model(num, cats)
        probs  = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(lbl.numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(preds, labels, probs, split="Test"):
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="macro")
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0
    print(f"\n{'─'*40}")
    print(f"  {split} Results")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n{classification_report(labels, preds, labels=[0, 1, 2], target_names=['Low','Medium','High'], zero_division=0)}")
    return {"accuracy": acc, "f1": f1, "auc": auc}


# ── full training run ─────────────────────────────────────────
def train(epochs=EPOCHS):
    train_loader, val_loader, test_loader, class_weights, scaler, encoders, test_ds = prepare_data()

    model     = build_model(window_size=WINDOW_SIZE).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Device          : {DEVICE}")
    print(f"\nStarting training for {epochs} epochs...")

    best_f1, patience, patience_count = 0.0, 7, 0
    history = {"train_loss": [], "train_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_preds, val_labels, val_probs = evaluate(model, val_loader)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_f1"].append(val_f1)

        print(f"  Epoch {epoch:02d}/{epochs} | loss {tr_loss:.4f} | acc {tr_acc:.4f} | val_f1 {val_f1:.4f} | {time.time()-t0:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model_state": model.state_dict(), "config": {"window_size": WINDOW_SIZE}}, CKPT_PATH)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # Load best model and evaluate on test set
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    test_preds, test_labels, test_probs = evaluate(model, test_loader)
    results = compute_metrics(test_preds, test_labels, test_probs, "Test")
    results["history"] = history

    # Save results
    results_path = PROJECT_ROOT / "models" / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items() if k != "history"}, f, indent=2)
    print(f"\n  Checkpoint : {CKPT_PATH}")
    print(f"  Results    : {results_path}")

    return model, results, scaler, encoders


# ── baseline comparisons ──────────────────────────────────────
def run_baselines(test_ds):
    """Compare against LSTM, SVM, Random Forest on flattened windows."""
    print("\n" + "="*50)
    print("  Baseline Model Comparisons")
    print("="*50)

    X = np.array(test_ds.windows).reshape(len(test_ds.windows), -1)
    y = np.array(test_ds.labels)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        only_class = int(classes[0]) if len(classes) else "none"
        print(f"  Skipping baselines: only one class is present in the windowed data ({only_class}).")
        return {}

    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify
    )

    if len(np.unique(y_train)) < 2:
        print("  Skipping baselines: training split contains fewer than two classes.")
        return {}

    baselines = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM (RBF)":     SVC(kernel="rbf", probability=True, random_state=42),
    }

    baseline_results = {}
    for name, clf in baselines.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="macro", zero_division=0)
        print(f"  {name:20s} → Acc: {acc:.4f}  F1: {f1:.4f}")
        baseline_results[name] = {"accuracy": acc, "f1": f1}

    return baseline_results


# ── ablation study ────────────────────────────────────────────
def ablation_study(train_loader, val_loader):
    """Test impact of removing key components."""
    print("\n" + "="*50)
    print("  Ablation Study")
    print("="*50)

    configs = {
        "Full model (n_heads=4, n_layers=2)":  {"n_heads": 4, "n_layers": 2},
        "No context (n_heads=4, no cats)":     {"n_heads": 4, "n_layers": 2, "no_cats": True},
        "Single head (n_heads=1)":             {"n_heads": 1, "n_layers": 2},
        "Single layer (n_layers=1)":           {"n_heads": 4, "n_layers": 1},
    }

    label_counts = np.bincount([item for _, _, item in val_loader.dataset])
    class_weights = torch.tensor(1.0 / (label_counts / label_counts.sum()), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for config_name, cfg in configs.items():
        no_cats = cfg.pop("no_cats", False)
        from models.attention_model import ESDAttentionModel
        cat_vocab = {} if no_cats else {"activity": 4, "environment": 4, "fabric_type": 4}
        m = ESDAttentionModel(
            n_numerical=5, cat_vocab=cat_vocab,
            window_size=WINDOW_SIZE, **cfg
        ).to(DEVICE)
        opt = AdamW(m.parameters(), lr=LR)
        for _ in range(5):   # quick 5-epoch warmup
            train_epoch(m, train_loader, opt, criterion)
        preds, labels, _ = evaluate(m, val_loader)
        f1 = f1_score(labels, preds, average="macro")
        print(f"  {config_name:45s} → Val F1: {f1:.4f}")


if __name__ == "__main__":
    model, results, scaler, encoders = train()
