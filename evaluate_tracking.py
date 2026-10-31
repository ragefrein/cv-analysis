"""Identity metrics on independently annotated frames, never on self-generated GT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from identity_review import visible_players
from storage import load_session


def box_ious(first, second):
    first, second = np.asarray(first, float).reshape(-1, 4), np.asarray(second, float).reshape(-1, 4)
    lo = np.maximum(first[:, None, :2], second[None, :, :2])
    hi = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.prod(np.maximum(0, hi - lo), axis=2)
    areas_a = np.prod(np.maximum(0, first[:, 2:] - first[:, :2]), axis=1)
    areas_b = np.prod(np.maximum(0, second[:, 2:] - second[:, :2]), axis=1)
    return intersection / np.maximum(1e-9, areas_a[:, None] + areas_b[None, :] - intersection)


def evaluate(tracks, annotations, threshold=.5):
    """Detection and global identity F1 on ALL players in labelled frames.

    Identity switches are counted between successive annotated observations,
    so sparse annotation cannot prove absence of switches between its samples.
    """
    if annotations.get("complete_frame_labels") is not True:
        raise ValueError("Label semua pemain di setiap frame acuan dan tetapkan complete_frame_labels=true.")
    labelled = annotations.get("frames", [])
    if not labelled:
        raise ValueError("Belum ada frame berlabel manual; akurasi identitas belum dapat dihitung.")
    indices = [int(fr["index"]) for fr in labelled]
    if len(set(indices)) != len(indices):
        raise ValueError("Frame acuan tidak boleh duplikat.")
    counts, previous = {}, {}
    total_gt = total_pred = tp = switches = 0
    for sample in sorted(labelled, key=lambda fr: int(fr["index"])):
        index = int(sample["index"])
        if not 0 <= index < len(tracks["frames"]):
            raise ValueError(f"Frame {index} belum dianalisis.")
        gt = sample["players"]
        predictions = visible_players(tracks, index)
        ids = [str(player["id"]) for player in gt]
        if len(ids) != len(set(ids)):
            raise ValueError("Satu identitas acuan muncul dua kali dalam satu frame.")
        total_gt += len(gt)
        total_pred += len(predictions)
        if not gt or not predictions:
            continue
        overlaps = box_ious([player["box"] for player in gt], [p[1:] for p in predictions])
        # Every potential per-identity overlap contributes to GLOBAL matching.
        for a, b in zip(*np.where(overlaps >= threshold)):
            pair = (str(gt[a]["id"]), int(predictions[b][0]))
            counts[pair] = counts.get(pair, 0) + 1
        cost = np.where(overlaps >= threshold, 1 - overlaps, 1000)
        rows, cols = linear_sum_assignment(cost)
        for a, b in zip(rows, cols):
            if overlaps[a, b] < threshold:
                continue
            tp += 1
            identity, predicted_id = str(gt[a]["id"]), int(predictions[b][0])
            if identity in previous and previous[identity] != predicted_id:
                switches += 1
            previous[identity] = predicted_id
    gt_ids = sorted({pair[0] for pair in counts})
    pred_ids = sorted({pair[1] for pair in counts})
    matches = np.zeros((len(gt_ids), len(pred_ids)), int)
    for (gt_id, pred_id), count in counts.items():
        matches[gt_ids.index(gt_id), pred_ids.index(pred_id)] = count
    idtp = 0
    if matches.size:
        rows, cols = linear_sum_assignment(-matches)
        idtp = int(matches[rows, cols].sum())
    return {"labelled_frames": len(labelled), "iou_threshold": threshold,
            "ground_truth_observations": total_gt, "prediction_observations": total_pred,
            "detection_precision": tp / total_pred if total_pred else 0.,
            "detection_recall": tp / total_gt if total_gt else 0.,
            "identity_f1_on_labelled_frames": 2 * idtp / (total_gt + total_pred) if total_gt + total_pred else 0.,
            "identity_switches_between_labelled_observations": switches,
            "identity_true_positive": idtp, "identity_false_positive": total_pred - idtp,
            "identity_false_negative": total_gt - idtp,
            "scope": "Only independently annotated frames. Unlabelled intervals are not evaluated."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--video", default="1031(2).mp4")
    parser.add_argument("--session", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/reference/identity-metrics.json"))
    args = parser.parse_args()
    tracks = load_session(args.video, args.session)
    if tracks is None:
        parser.error("Analisis untuk video ini belum tersedia atau tidak cocok.")
    result = evaluate(tracks, json.loads(args.annotations.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
