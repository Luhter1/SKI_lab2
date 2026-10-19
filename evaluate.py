#!/usr/bin/env python3
"""
Оценка точности трекинга: IoU между рамкой алгоритма и истинным положением
объекта (ground truth) для каждого кадра видео, где объект полностью в кадре.
"""

import cv2
import numpy as np

from generate_test_videos import OBJ, W, H, rotate_points
from tracker import KeypointTracker


def build_transforms(kind):
    """Воспроизводит последовательность преобразований из генератора."""
    N = 60
    if kind == "test1":
        return [(W // 2, H // 2, 0.0, 1.0)] + [
            (W // 2 + int(150 * np.sin(2 * np.pi * k / N)),
             H // 2 + int(80 * np.sin(4 * np.pi * k / N)),
             15 * np.sin(2 * np.pi * k / N), 1.0) for k in range(1, N)
        ]
    if kind == "test2":
        return [(W // 2, H // 2, 0.0, 0.4)] + [
            (W // 2, H // 2, 360 * k / N, 0.4 + 0.6 * k / N)
            for k in range(1, N)
        ]
    return [(W // 2, H // 2, 0.0, 0.7)] + [
        (W // 2 + int(220 * np.sin(2 * np.pi * k / N)),
         H // 2 - int(160 * np.sin(2 * np.pi * k / N)),
         -30 * np.sin(2 * np.pi * k / N), 0.7) for k in range(1, N)
    ]


def true_quad(cx, cy, ang, s):
    half = (OBJ * s) / 2.0
    corners = np.float32(
        [[cx - half, cy - half], [cx + half, cy - half],
         [cx + half, cy + half], [cx - half, cy + half]]
    )
    return rotate_points(corners, cx, cy, ang).astype(np.int32)


def quad_mask(quad, shape=(H, W)):
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(m, np.int32(quad), 255)
    return m


def iou(box, gt):
    mb, mg = quad_mask(box), quad_mask(gt)
    inter = np.logical_and(mb, mg).sum()
    union = np.logical_or(mb, mg).sum()
    return inter / union if union else 0.0


def evaluate(video, kind):
    cap = cv2.VideoCapture(video)
    template = cap.read()[1]
    tracker = KeypointTracker(template)
    transforms = build_transforms(kind)  # по одному преобразованию на кадр сцены

    ious = []
    k = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, found = tracker.track(frame)
        if found:
            gt = true_quad(*transforms[k])
            ious.append(iou(tracker.last_box, gt))
        k += 1
    cap.release()
    mean_iou = np.mean(ious) if ious else 0.0
    print(f"{kind}: кадров={k}, найдено={len(ious)}, "
          f"средний IoU={mean_iou:.3f}")


if __name__ == "__main__":
    evaluate("videos/test1_translation.mp4", "test1")
    evaluate("videos/test2_scale_rotation.mp4", "test2")
    evaluate("videos/own1_object.mp4", "own1")