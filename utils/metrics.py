# printerid/utils/metrics.py
from __future__ import annotations
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd


def batch_accuracy(logits, labels):
    preds = logits.argmax(1)
    return (preds == labels).float().mean().item()


def report(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    clf_rep = classification_report(y_true, y_pred, target_names=class_names)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion": pd.DataFrame(cm, index=class_names, columns=class_names),
        "report": clf_rep,
    }
