from __future__ import annotations
from collections import Counter
from typing import Dict, List, Literal, Tuple

Sentiment = Literal["positive", "negative", "neutral"]
LABELS: Tuple[Sentiment, ...] = ("positive", "negative", "neutral")

def _pr_rc_f1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1

def _build_map(items: List[dict]) -> Dict[tuple, str]:
    """
    items: [{"id": "...", "labels":[{"aspect":"battery","sentiment":"negative"}, ...]}, ...]
    Returns: { (id, aspect_lower): sentiment }
    """
    out: Dict[tuple, str] = {}
    for row in items:
        rid = row["id"]
        for lab in row.get("labels", []):
            key = (rid, str(lab["aspect"]).strip().lower())
            out[key] = lab["sentiment"]
    return out

def compute_label_metrics(pred_items: List[dict], gold_items: List[dict]) -> dict:
    pred_map = _build_map(pred_items)
    gold_map = _build_map(gold_items)

    keys = set(pred_map) | set(gold_map)

    tp = Counter(); fp = Counter(); fn = Counter()

    for k in keys:
        g = gold_map.get(k)
        p = pred_map.get(k)
        if g is None and p is not None:
            fp[p] += 1
        elif g is not None and p is None:
            fn[g] += 1
        else:
            if p == g:
                tp[g] += 1
            else:
                fp[p] += 1
                fn[g] += 1

    per_label = {}
    for lab in LABELS:
        precision, recall, f1 = _pr_rc_f1(tp[lab], fp[lab], fn[lab])
        per_label[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp[lab] + fn[lab],
        }

    T = sum(tp.values()); P = sum(fp.values()); F = sum(fn.values())
    micro_p, micro_r, micro_f1 = _pr_rc_f1(T, P, F)
    acc = T / (T + P + F) if (T + P + F) else 0.0

    macro_p = sum(per_label[l]["precision"] for l in LABELS) / len(LABELS)
    macro_r = sum(per_label[l]["recall"]    for l in LABELS) / len(LABELS)
    macro_f = sum(per_label[l]["f1"]        for l in LABELS) / len(LABELS)

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1, "accuracy": acc},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        "per_label": per_label
    }
