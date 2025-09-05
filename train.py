import os
import json
import math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    precision_recall_fscore_support, confusion_matrix, accuracy_score
)

from utils.dataset import MultimodalDFDataset
from models.assemble_model import MultiModalDeepfakeModel

warnings.filterwarnings("ignore", category=UserWarning)

# ============================== CONFIG ==============================
BASE_DIR = r"D:\WORK\PycharmProjects\DYNAMAMBAU++\data"
METADATA_PATH = os.path.join(BASE_DIR, "metadata_filtered.json")
RUNS_DIR = "runs"
BATCH_SIZE = 4              # <- as requested
NUM_EPOCHS = 100            # <- as requested
PATIENCE = 5                # <- early stopping patience
RESUME = True               # auto-resume from runs/latest.pt
SEED = 42


# ============================== UTIL: discover valid IDs on disk ==============================
def get_valid_video_ids(base_dir: str) -> set:
    faces_dir = os.path.join(base_dir, 'faces')
    lips_dir = os.path.join(base_dir, 'lips')
    audio_dir = os.path.join(base_dir, 'audio')
    headpose_dir = os.path.join(base_dir, 'headpose')

    face_ids     = set(os.listdir(faces_dir)) if os.path.isdir(faces_dir) else set()
    lip_ids      = set(os.listdir(lips_dir))  if os.path.isdir(lips_dir)  else set()
    audio_ids    = set(f.replace('.wav', '') for f in os.listdir(audio_dir)) if os.path.isdir(audio_dir) else set()
    headpose_ids = set(f.replace('.json', '') for f in os.listdir(headpose_dir)) if os.path.isdir(headpose_dir) else set()

    return face_ids & lip_ids & audio_ids & headpose_ids


# ============================== LOAD & STRATIFIED 70/15/15 + BALANCE ==============================
def stratified_balanced_70_15_15(
    metadata_path: str,
    base_dir: str,
    rng: np.random.Generator
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    1) Read metadata → collect (video_id, label) for all videos present in ALL 4 modalities.
       label: 0=REAL, 1=FAKE
    2) Ignore metadata 'split'. Do a fresh stratified 70/15/15 split.
    3) Balance classes within each split by downsampling the larger class.

    Returns: train_ids, train_labels, val_ids, val_labels, test_ids, test_labels
    """
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    valid_ids = get_valid_video_ids(base_dir)

    real_all, fake_all = [], []
    for fname, info in meta.items():
        vid = fname[:-4] if fname.endswith(".mp4") else fname
        if vid not in valid_ids:
            continue
        label = info.get("label", "").upper()
        if label == "REAL":
            real_all.append(vid)
        elif label == "FAKE":
            fake_all.append(vid)

    # Shuffle deterministically
    real_all = np.array(real_all)
    fake_all = np.array(fake_all)
    rng.shuffle(real_all)
    rng.shuffle(fake_all)

    # Desired counts per split for each class (approx 70/15/15)
    def split_counts(n):
        n_train = int(round(0.070 * n))
        n_val = int(round(0.015 * n))
        n_test = int(round(0.015 * n))
        return n_train, n_val, n_test

    r_train, r_val, r_test = split_counts(len(real_all))
    f_train, f_val, f_test = split_counts(len(fake_all))

    # Slice per class
    real_train = real_all[:r_train]
    real_val = real_all[r_train:r_train + r_val]
    real_test = real_all[r_train + r_val:r_train + r_val + r_test]

    fake_train = fake_all[:f_train]
    fake_val = fake_all[f_train:f_train + f_val]
    fake_test = fake_all[f_train + f_val:f_train + f_val + f_test]

    # Balance within each split by downsampling larger class
    def balance_pair(a_ids, b_ids):
        n = min(len(a_ids), len(b_ids))
        return a_ids[:n], b_ids[:n]

    real_train, fake_train = balance_pair(real_train, fake_train)
    real_val,   fake_val   = balance_pair(real_val,   fake_val)
    real_test,  fake_test  = balance_pair(real_test,  fake_test)

    # Build final ids/labels, shuffle each split
    def pack_and_shuffle(reals, fakes):
        ids = np.concatenate([reals, fakes])
        labels = np.array([0]*len(reals) + [1]*len(fakes))
        idx = np.arange(len(ids))
        rng.shuffle(idx)
        return ids[idx].tolist(), labels[idx].tolist()

    train_ids, train_labels = pack_and_shuffle(real_train, fake_train)
    val_ids,   val_labels   = pack_and_shuffle(real_val,   fake_val)
    test_ids,  test_labels  = pack_and_shuffle(real_test,  fake_test)

    print(f"70/15/15 balanced → "
          f"train R/F: {sum(np.array(train_labels)==0)}/{sum(np.array(train_labels)==1)} | "
          f"val R/F: {sum(np.array(val_labels)==0)}/{sum(np.array(val_labels)==1)} | "
          f"test R/F: {sum(np.array(test_labels)==0)}/{sum(np.array(test_labels)==1)}")

    return train_ids, train_labels, val_ids, val_labels, test_ids, test_labels


# ============================== METRICS HELPERS ==============================
def compute_basic_metrics(y_true: np.ndarray, probs_pos: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (probs_pos >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    # AUC (guard for single-class)
    try:
        auc = roc_auc_score(y_true, probs_pos)
    except ValueError:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    far = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    frr = fn / (fn + tp) if (fn + tp) > 0 else float('nan')
    hter = (far + frr) / 2 if not (math.isnan(far) or math.isnan(frr)) else float('nan')

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "auc": auc, "far": far, "frr": frr, "hter": hter,
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)
    }


def compute_eer(y_true: np.ndarray, probs_pos: np.ndarray):
    try:
        fpr, tpr, thr = roc_curve(y_true, probs_pos)
    except ValueError:
        return float('nan'), float('nan')
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    thr_eer = thr[idx] if idx < len(thr) else float('nan')
    return float(eer), float(thr_eer)


def best_threshold_by_f1(y_true: np.ndarray, probs_pos: np.ndarray):
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, probs_pos)
    except ValueError:
        return float('nan'), float('nan')
    f1s = (2 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-8, None)
    if len(f1s) == 0:
        return float('nan'), float('nan')
    idx = int(np.argmax(f1s))
    return float(thresholds[idx]), float(f1s[idx])


# ============================== PLOTS & CSVs ==============================
def save_history_plots_and_csv(history: Dict[str, list], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # Loss curves
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss_curves.png")); plt.close()

    # Accuracy curves
    plt.figure()
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"],   label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "accuracy_curves.png")); plt.close()


def save_roc_curve(y_true: np.ndarray, probs_pos: np.ndarray, out_base: str):
    try:
        fpr, tpr, thr = roc_curve(y_true, probs_pos)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(out_base + ".csv", index=False)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle='--', label="Chance")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(out_base + ".png"); plt.close()
    except ValueError:
        pd.DataFrame({"fpr": [], "tpr": [], "threshold": []}).to_csv(out_base + ".csv", index=False)


def save_pr_curve(y_true: np.ndarray, probs_pos: np.ndarray, out_base: str):
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, probs_pos)
        pd.DataFrame({"precision": precision, "recall": recall}).to_csv(out_base + ".csv", index=False)
        plt.figure()
        plt.plot(recall, precision, label="PR")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(out_base + ".png"); plt.close()
    except ValueError:
        pd.DataFrame({"precision": [], "recall": []}).to_csv(out_base + ".csv", index=False)


def save_confusion_matrix(cm: np.ndarray, out_path: str):
    labels = ["Real (0)", "Fake (1)"]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(labels))
    ax.set(xticks=tick_marks, yticks=tick_marks, xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label', title="Confusion Matrix")
    thresh = cm.max() / 2. if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================== LOOPS ==============================
def run_one_epoch(model, loader, device, optimizer=None, criterion=None, pbar_desc=None):
    """
    If optimizer is provided → training step, else eval step.
    Returns: avg_loss, avg_acc, all_labels (np), all_probs_pos (np)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_labels, all_probs = [], []

    iterable = loader
    if pbar_desc is not None:
        iterable = tqdm(loader, desc=pbar_desc)

    with torch.set_grad_enabled(is_train):
        for batch in iterable:
            face         = batch['face'].to(device)
            audio_mel    = batch['audio'].to(device)
            lips         = batch['lips'].to(device)
            audio_mel_lp = batch['audio_lip'].to(device)
            head_pose    = batch['headpose'].to(device)
            labels       = batch['label'].to(device)

            logits = model(face, audio_mel, lips, audio_mel_lp, head_pose)  # [B, 2]
            probs  = torch.softmax(logits, dim=-1)[:, 1]                    # P(class=1)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

            total_loss += loss.item()
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

            # live progress (train)
            if pbar_desc is not None and is_train and hasattr(iterable, "set_postfix"):
                curr_acc = total_correct / max(total_samples, 1)
                curr_loss = total_loss / max(len(all_labels), 1)
                iterable.set_postfix(train_loss=f"{curr_loss:.4f}", train_acc=f"{curr_acc:.4f}")

    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    all_probs  = np.concatenate(all_probs) if all_probs else np.array([])
    avg_loss = total_loss / max(len(loader), 1)
    avg_acc  = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc, all_labels, all_probs


def make_loader(dataset, ids: List[str], id_to_index: Dict[str, int], batch_size: int, shuffle: bool, device):
    indices = [id_to_index[vid] for vid in ids if vid in id_to_index]
    subset = Subset(dataset, indices)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=(device.type == "cuda")
    )


# ============================== CHECKPOINTS ==============================
def save_checkpoint(path, epoch, model, optimizer, history, best_val_acc):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "best_val_acc": best_val_acc,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt.get("history", None), ckpt.get("best_val_acc", -1.0)


# ============================== MAIN ==============================
def train():
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}")
    os.makedirs(RUNS_DIR, exist_ok=True)

    # ===== 70/15/15 stratified & balanced splits (ignoring metadata 'split') =====
    train_ids, train_labels, val_ids, val_labels, test_ids, test_labels = stratified_balanced_70_15_15(
        METADATA_PATH, BASE_DIR, rng
    )

    # Build a unified dataset that covers all ids (train+val+test)
    union_ids = train_ids + val_ids + test_ids
    union_labels = train_labels + val_labels + test_labels
    dataset = MultimodalDFDataset(union_ids, union_labels, base_dir=BASE_DIR)

    # index map
    id_to_index = {vid: i for i, vid in enumerate(dataset.video_list)}

    # DataLoaders
    train_loader = make_loader(dataset, train_ids, id_to_index, BATCH_SIZE, shuffle=True,  device=device)
    val_loader   = make_loader(dataset, val_ids,   id_to_index, BATCH_SIZE, shuffle=False, device=device)
    test_loader  = make_loader(dataset, test_ids,  id_to_index, BATCH_SIZE, shuffle=False, device=device)

    print(f"Final split sizes → train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}")

    # ===== Model / Loss / Optim =====
    model = MultiModalDeepfakeModel(embed_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ===== Resume (optional) =====
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    latest_ckpt = os.path.join(RUNS_DIR, "latest.pt")
    best_model_path = os.path.join(RUNS_DIR, "best_val_acc.pt")
    start_epoch = 1

    if RESUME and os.path.exists(latest_ckpt):
        print(f"🔁 Resuming from: {latest_ckpt}")
        last_epoch, prev_hist, best_val_acc_ckpt = load_checkpoint(latest_ckpt, model, optimizer, device)
        start_epoch = last_epoch + 1
        if prev_hist:
            history = prev_hist
        best_val_acc = best_val_acc_ckpt

    # ===== Training loop =====
    epochs_without_improve = 0
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        tr_loss, tr_acc, _, _ = run_one_epoch(
            model, train_loader, device, optimizer=optimizer, criterion=criterion,
            pbar_desc=f"Epoch {epoch} [train]"
        )
        val_loss, val_acc, y_val, p_val = run_one_epoch(
            model, val_loader, device, optimizer=None, criterion=criterion,
            pbar_desc=f"Epoch {epoch} [val]"
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save per-epoch ROC/PR (validation)
        save_roc_curve(y_val, p_val, out_base=os.path.join(RUNS_DIR, f"val_epoch{epoch}_roc"))
        save_pr_curve (y_val, p_val, out_base=os.path.join(RUNS_DIR, f"val_epoch{epoch}_pr"))

        # Console summary
        try:
            val_auc = roc_auc_score(y_val, p_val)
        except ValueError:
            val_auc = float('nan')
        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

        # Early stopping on Val Acc + save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improve += 1

        # Rolling checkpoint and per-epoch ckpt
        save_checkpoint(latest_ckpt, epoch, model, optimizer, history, best_val_acc)
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()},
            os.path.join(RUNS_DIR, f"checkpoint_epoch{epoch}.pt")
        )

        if epochs_without_improve >= PATIENCE:
            print(f"⏹ Early stopping: no val acc improvement for {PATIENCE} epochs.")
            break

    # Save curves & CSV history
    save_history_plots_and_csv(history, RUNS_DIR)

    # Load best model by val acc for final test
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best-by-val-acc model: {best_model_path}")

    # ===== Test evaluation =====
    test_loss, test_acc, y_test, p_test = run_one_epoch(
        model, test_loader, device, optimizer=None, criterion=criterion, pbar_desc="Testing"
    )

    metrics_05 = compute_basic_metrics(y_test, p_test, threshold=0.5)
    eer, thr_eer = compute_eer(y_test, p_test)
    thr_f1, best_f1 = best_threshold_by_f1(y_test, p_test)
    metrics_eer = compute_basic_metrics(y_test, p_test, threshold=thr_eer) if not math.isnan(thr_eer) else {}
    metrics_f1  = compute_basic_metrics(y_test, p_test, threshold=thr_f1)  if not math.isnan(thr_f1)  else {}

    # Save ROC & PR for test
    save_roc_curve(y_test, p_test, out_base=os.path.join(RUNS_DIR, "test_roc"))
    save_pr_curve (y_test, p_test, out_base=os.path.join(RUNS_DIR, "test_pr"))

    # Confusion matrix @ 0.5
    cm_05 = confusion_matrix(y_test, (p_test >= 0.5).astype(int), labels=[0, 1])
    save_confusion_matrix(cm_05, out_path=os.path.join(RUNS_DIR, "test_confusion_matrix_0p5.png"))

    # Save all key metrics & thresholds to CSV
    rows = [{"split": "test@0.5", "threshold": 0.5, **metrics_05}]
    if metrics_eer:
        rows.append({"split": "test@EER", "threshold": thr_eer, **metrics_eer})
    if metrics_f1:
        rows.append({"split": "test@bestF1", "threshold": thr_f1, "bestF1": best_f1, **metrics_f1})
    pd.DataFrame(rows).to_csv(os.path.join(RUNS_DIR, "test_metrics_summary.csv"), index=False)

    print("\n===== TEST METRICS (threshold=0.5) =====")
    for k, v in metrics_05.items():
        print(f"{k:>10}: {v:.4f}" if isinstance(v, float) and not math.isnan(v) else f"{k:>10}: {v}")
    if not math.isnan(eer):
        print(f"\nEER: {eer:.4f} at threshold {thr_eer:.4f}")
    if not math.isnan(thr_f1):
        print(f"Best-F1 threshold: {thr_f1:.4f} (F1={best_f1:.4f})")
    print("========================================\n")


if __name__ == '__main__':
    train()
