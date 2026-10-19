#!/usr/bin/env python3
"""
ЛР2. Трекинг объекта на видео на основе ключевых точек.

Алгоритм:
  1. Первый кадр видео считается крупным изображением отслеживаемого объекта
     и используется как эталон (шаблон).
  2. Для эталона вычисляются ключевые точки ORB и их дескрипторы.
  3. На каждом последующем кадре:
       - вычисляются ключевые точки ORB и дескрипторы;
       - дескрипторы сопоставляются с эталоном (BFMatcher, метрика HAMMING);
       - применяется ratio-тест Лоу (отсев ложных сопоставлений);
       - по парам сопоставлений оценивается гомография методом RANSAC;
       - углы эталона проецируются через гомографию — полученный
         четырёхугольник является рамкой объекта;
       - если гомография устойчива (достаточно inlier'ов), рамка и подпись
         рисуются на кадре.
  4. Кадры с рамками сохраняются в выходное видео.
"""

import argparse
import os

import cv2
import numpy as np


class KeypointTracker:
    """Трекинг плоского текстурированного объекта по ключевым точкам ORB."""

    LABEL = "object"

    def __init__(
        self,
        template_img,
        n_features=1000,
        ratio=0.75,
        min_inliers=15,
        min_matches=10,
        scale_factor=1.2,
        nlevels=12,
    ):
        # Расширенная пирамида масштабов: 1.2^11 ~ 7.4x покрытия по масштабу,
        # что позволяет находить объект при сильном уменьшении (приближении/удалении).
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=nlevels,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.min_matches = min_matches

        # Эталоном является само переданное изображение объекта.
        self.template = template_img
        self.template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        self.template_keypoints, self.template_descriptors = self.orb.detectAndCompute(
            self.template_gray, None
        )

        # Углы эталона (границы объекта), которые проецируются на кадр.
        h, w = self.template_gray.shape[:2]
        self.template_corners = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]]
        ).reshape(-1, 1, 2)

        self.raw_matches = 0
        self.good_matches = 0
        self.inliers = 0
        self.last_box = None  # четырёхугольник рамки в координатах кадра

    def _match(self, frame_gray):
        """Поиск сопоставлений между эталоном и кадром."""
        frame_keypoints, frame_descriptors = self.orb.detectAndCompute(
            frame_gray, None
        )
        if (
            self.template_descriptors is None
            or frame_descriptors is None
            or len(frame_descriptors) < 2
        ):
            return None, None, 0, 0

        self.raw_matches = len(self.template_descriptors)
        # k=2 — нужно для ratio-теста (два ближайших соседа).
        matches = self.matcher.knnMatch(self.template_descriptors, frame_descriptors, k=2)

        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        self.good_matches = len(good)
        return frame_keypoints, frame_descriptors, good, self.good_matches

    def track(self, frame):
        """Обрабатывает кадр, возвращает копию с нарисованной рамкой (или без неё)."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_keypoints, _, good, n_good = self._match(frame_gray)
        result = frame.copy()
        found = False

        if n_good >= self.min_matches:
            src_pts = np.float32(
                [self.template_keypoints[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [frame_keypoints[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, 5.0
            )
            if homography is not None:
                self.inliers = int(mask.sum())
                if self.inliers >= self.min_inliers:
                    projected = cv2.perspectiveTransform(
                        self.template_corners, homography
                    )
                    self.last_box = projected.reshape(4, 2)
                    result = self._draw_box(result, projected, (0, 255, 0))
                    found = True

        if not found:
            # Объект не найден — рисуем информационную подпись.
            self.inliers = 0
            self.last_box = None
            cv2.putText(
                result,
                self.LABEL + ": not found",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        return result, found

    def _draw_box(self, frame, projected, color):
        """Рисует рамку и подпись вокруг найденного объекта."""
        cv2.polylines(
            frame, [np.int32(projected)], True, color, 3, cv2.LINE_AA
        )
        # Подпись над верхним левым углом рамки.
        x, y = int(projected[0][0][0]), int(projected[0][0][1])
        cv2.putText(
            frame,
            self.LABEL,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
        return frame


def process_video(
    input_path,
    output_path,
    show=False,
    stats=True,
):
    """Обрабатывает видео: первый кадр — эталон, далее трекинг."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if output_path:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    ret, first = cap.read()
    if not ret:
        cap.release()
        raise IOError("Видео пустое (нет первого кадра).")

    # Первый кадр — эталонный объект (крупное изображение объекта).
    template = first
    tracker = KeypointTracker(template)

    found_flags = []
    inliers_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, found = tracker.track(frame)
        inliers_list.append(tracker.inliers)
        if writer:
            writer.write(result)
        if show:
            cv2.imshow("tracker", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        found_flags.append(found)

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    if stats:
        total = len(found_flags)
        tracked = sum(found_flags)
        print(f"Кадров обработано: {total}")
        print(f"Кадров с найденным объектом: {tracked} "
              f"({100.0 * tracked / total:.1f}%)")
        print(f"Среднее число inlier'ов: {np.mean(inliers_list):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Трекинг объекта по ключевым точкам ORB."
    )
    parser.add_argument("video", help="Путь к входному видео")
    parser.add_argument("-o", "--output", default=None,
                        help="Путь для сохранения результата (mp4)")
    parser.add_argument("--show", action="store_true",
                        help="Показывать окно с результатом")
    args = parser.parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.video)
        args.output = f"{base}_tracked.mp4"

    print(f"Вход:  {args.video}")
    print(f"Выход: {args.output}")
    process_video(args.video, args.output, show=args.show)


if __name__ == "__main__":
    main()