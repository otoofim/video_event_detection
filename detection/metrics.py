"""Frame- and event-level metrics (numpy only)."""
from __future__ import annotations

import numpy as np

from common.schema import CLASSES, Event


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int = len(CLASSES)) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_prf(cm: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        p = tp / max(cm[:, i].sum(), 1)
        r = tp / max(cm[i, :].sum(), 1)
        out[name] = {"precision": float(p), "recall": float(r),
                     "f1": float(2 * p * r / (p + r)) if p + r else 0.0}
    return out


def event_prf(pred: list[Event], gt: list[Event], tol: float = 0.5) -> dict[str, dict[str, float]]:
    """Greedy one-to-one matching of events of the same type within ``tol`` seconds."""
    res = {}
    for etype in sorted({e.event_type for e in gt} | {e.event_type for e in pred}):
        p = sorted((e for e in pred if e.event_type == etype), key=lambda e: e.timestamp)
        g = sorted((e for e in gt if e.event_type == etype), key=lambda e: e.timestamp)
        used: set[int] = set()
        tp = 0
        for pe in p:
            best, best_d = None, tol + 1e-9
            for j, ge in enumerate(g):
                d = abs(pe.timestamp - ge.timestamp)
                if j not in used and d <= best_d:
                    best, best_d = j, d
            if best is not None:
                used.add(best)
                tp += 1
        prec = tp / max(len(p), 1)
        rec = tp / max(len(g), 1)
        res[etype] = {"tp": tp, "n_pred": len(p), "n_gt": len(g), "precision": prec, "recall": rec,
                      "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    return res
