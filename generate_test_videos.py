#!/usr/bin/env python3
"""
Генерация синтетических тестовых видео для ЛР2.

Формат каждого видео согласован с алгоритмом трекинга:
  - Кадр 0: крупное изображение отслеживаемого объекта (эталон),
    заполняющее весь кадр.
  - Кадры 1..N: сцена того же размера, где этот текстурированный плоский
    объект перемещается / вращается / масштабируется на фоне.

Все кадры одного разрешения (640x480), кратны 16 — требование кодеков.
Сгенерировано 3 тестовых видео + 1 «своё» видео с похожим объектом.
"""

import os

import cv2
import numpy as np

W, H = 640, 480   # размер сцены и всех кадров (кратен 16)
OBJ = 300         # размер объекта в сцене
N = 60            # число кадров сцены


def make_object(seed, size=OBJ):
    """Создаёт текстурированное плоское изображение объекта (эталон)."""
    rng = np.random.default_rng(seed)
    # Мелкий шум — основа текстуры.
    noise = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (5, 5), 0)
    # Крупные цветные пятна для устойчивых ORB-дескрипторов.
    for _ in range(40):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(8, 35)
        color = rng.integers(0, 255, 3, dtype=int)
        cv2.circle(noise, (int(cx), int(cy)), int(r), color.tolist(), -1)
        cv2.circle(noise, (int(cx), int(cy)), int(r), (0, 0, 0), 1)
    # Сетка — геометрические ключевые точки (углы).
    step = size // 8
    for i in range(0, size, step):
        cv2.line(noise, (i, 0), (i, size), (0, 0, 0), 1)
        cv2.line(noise, (0, i), (size, i), (0, 0, 0), 1)
    # Чёткая рамка объекта.
    cv2.rectangle(noise, (0, 0), (size - 1, size - 1), (255, 255, 255), 3)
    return noise


def make_background(seed):
    """Создаёт фон сцены (другой шум, чтобы не совпадал с объектом)."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(0, 90, (H, W, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (9, 9), 0)
    return bg


def template_frame(obj):
    """Кадр 0: крупное изображение объекта, заполняющее весь кадр."""
    return cv2.resize(obj, (W, H), interpolation=cv2.INTER_LINEAR)


def rotate_points(pts, cx, cy, angle_deg):
    """Поворот точек вокруг (cx, cy) на угол в градусах."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    out = []
    for p in pts:
        x, y = p[0] - cx, p[1] - cy
        out.append([ca * x - sa * y + cx, sa * x + ca * y + cy])
    return np.float32(out)


def place_object(scene, obj, corners_dst):
    """Встраивает прямоугольный объект в сцену по целевому четырёхугольнику."""
    w, h = scene.shape[1], scene.shape[0]
    oh, ow = obj.shape[:2]
    src = np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]])
    dst = np.float32(corners_dst)
    homography = cv2.getPerspectiveTransform(src, dst)
    mask = np.ones((oh, ow), dtype=np.uint8) * 255
    warped_obj = cv2.warpPerspective(
        obj, homography, (w, h), flags=cv2.INTER_LINEAR
    )
    warped_mask = cv2.warpPerspective(
        mask, homography, (w, h), flags=cv2.INTER_NEAREST
    )
    mask3 = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR).astype(bool)
    scene[mask3] = warped_obj[mask3]
    return scene


def scene_frames(obj, bg, transforms):
    """Строит кадры сцены из списка преобразований (cx, cy, угол, масштаб)."""
    frames = []
    for cx, cy, ang, s in transforms:
        half = (OBJ * s) / 2.0
        corners = np.float32(
            [[cx - half, cy - half],
             [cx + half, cy - half],
             [cx + half, cy + half],
             [cx - half, cy + half]]
        )
        corners = rotate_points(corners, cx, cy, ang)
        scene = bg.copy()
        scene = place_object(scene, obj, corners)
        frames.append(scene)
    return frames


def write_video(path, frames, fps=25.0):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"Создано: {path} ({len(frames)} кадров, {w}x{h})")


def main():
    os.makedirs("videos", exist_ok=True)

    # --- Тест 1: поступательное движение + лёгкий поворот ---
    obj = make_object(seed=1)
    bg = make_background(seed=99)
    transforms = [(W // 2, H // 2, 0.0, 1.0)] + [
        (W // 2 + int(150 * np.sin(2 * np.pi * k / N)),
         H // 2 + int(80 * np.sin(4 * np.pi * k / N)),
         15 * np.sin(2 * np.pi * k / N), 1.0)
        for k in range(1, N)
    ]
    frames = [template_frame(obj)] + scene_frames(obj, bg, transforms)
    write_video("videos/test1_translation.mp4", frames)

    # --- Тест 2: масштабирование (приближение) + вращение ---
    transforms = [(W // 2, H // 2, 0.0, 0.4)] + [
        (W // 2, H // 2, 360 * k / N, 0.4 + 0.6 * k / N)
        for k in range(1, N)
    ]
    frames = [template_frame(obj)] + scene_frames(obj, bg, transforms)
    write_video("videos/test2_scale_rotation.mp4", frames)

    # --- Тест 3: объект частично выходит из кадра и возвращается ---
    transforms = [(W // 2, H // 2, 0.0, 1.0)] + [
        (W // 2 + int(0.5 * W * np.sin(np.pi * k / N)),
         H // 2 - int(0.4 * H * np.sin(np.pi * k / N)),
         20 * np.sin(np.pi * k / N), 1.0)
        for k in range(1, N)
    ]
    frames = [template_frame(obj)] + scene_frames(obj, bg, transforms)
    write_video("videos/test3_offscreen.mp4", frames)

    # --- Своё видео: похожий объект, диагональное движение + поворот ---
    obj_own = make_object(seed=7)  # другой, но похожий объект
    bg_own = make_background(seed=555)
    transforms = [(W // 2, H // 2, 0.0, 0.7)] + [
        (W // 2 + int(220 * np.sin(2 * np.pi * k / N)),
         H // 2 - int(160 * np.sin(2 * np.pi * k / N)),
         -30 * np.sin(2 * np.pi * k / N), 0.7)
        for k in range(1, N)
    ]
    frames = [template_frame(obj_own)] + scene_frames(obj_own, bg_own, transforms)
    write_video("videos/own1_object.mp4", frames)


if __name__ == "__main__":
    main()