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
        n_features=3000,
        ratio=0.85,
        min_inliers=6,
        min_matches=4,
        scale_factor=1.2,
        nlevels=12,
        crosscheck=True,
        ransac_thresh=5.0,
    ):
        # Расширенная пирамида масштабов: 1.2^11 ~ 7.4x покрытия по масштабу,
        # что позволяет находить объект при сильном уменьшении (приближении/удалении).
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=nlevels,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crosscheck)
        self.crosscheck = crosscheck
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.min_matches = min_matches
        self.ransac_thresh = ransac_thresh

        # Матчер для взаимного сопоставления соседних кадров (временной трекинг).
        self._cc_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Состояние временного трекинга: предыдущий кадр и последняя гомография.
        self._prev_des = None
        self._prev_kp = None
        self._H_ref = None

        # Эталоном является само переданное изображение объекта.
        self.template = template_img
        self.template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        self.template_keypoints, self.template_descriptors = self.orb.detectAndCompute(
            self.template_gray, None
        )

        # Углы эталона (границы объекта), которые проецируются на кадр.
        h, w = self.template_gray.shape[:2]
        self.frame_area = h * w
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
        if self.crosscheck:
            # Взаимный поиск ближайшего соседа уже отсеял ложные совпадения,
            # ratio-тест не требуется.
            matches = self.matcher.match(self.template_descriptors, frame_descriptors)
            good = list(matches)
            self.good_matches = len(good)
            return frame_keypoints, frame_descriptors, good, self.good_matches
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

    def _area_ok(self, projected):
        """Рамка не должна резко менять размер между соседними кадрами."""
        if self.last_box is None:
            return True
        a = abs(cv2.contourArea(np.int32(projected)))
        pa = abs(cv2.contourArea(np.int32(self.last_box)))
        if pa > 0:
            ratio = a / pa
            if ratio > 3.0 or ratio < 1.0 / 3.0:
                return False
        return True

    def _temporal_ok(self, projected):
        """Временная согласованность рамки с предыдущим кадром: похожий размер
        и не слишком большой скачок центра. Отсекает ложные детекции, «прыгающие»
        на другие напечатанные изображения в кадре."""
        if self.last_box is None:
            return True
        if not self._area_ok(projected):
            return False
        c = projected.reshape(4, 2).mean(axis=0)
        pc = np.array(self.last_box).mean(axis=0)
        diag = np.linalg.norm(
            np.array(self.last_box[0]) - np.array(self.last_box[2])
        )
        if diag <= 0:
            return True
        return np.hypot(c[0] - pc[0], c[1] - pc[1]) <= 1.0 * diag

    def _propagate(self, frame_kp, frame_des):
        """Временная пропагация рамки через гомографию между соседними кадрами.

        Если прямое сопоставление с эталоном слабое (объект мал/размыт), но в
        предыдущем кадре рамка уже была построена, переносим её на текущий кадр
        по гомографии предыдущий->текущий кадр. Чтобы рамка не «раздувалась»
        из-за преобладающих на фоне точек, учитываются только совпадения внутри
        предыдущей рамки объекта.
        """
        if (
            self._H_ref is None
            or self.last_box is None
            or self._prev_des is None
            or frame_des is None
            or len(self._prev_des) < 4
            or len(frame_des) < 4
        ):
            return None, None, 0
        matches = self._cc_matcher.match(self._prev_des, frame_des)
        poly = np.int32(self.last_box)
        kept = [
            m for m in matches
            if cv2.pointPolygonTest(poly, self._prev_kp[m.queryIdx].pt, False) >= 0
        ]
        if len(kept) < 6:
            return None, None, 0
        src = np.float32(
            [self._prev_kp[m.queryIdx].pt for m in kept]
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [frame_kp[m.trainIdx].pt for m in kept]
        ).reshape(-1, 1, 2)
        h_prev_cur, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if h_prev_cur is None:
            return None, None, 0
        h_cur = h_prev_cur @ self._H_ref  # эталон -> текущий кадр
        projected = cv2.perspectiveTransform(self.template_corners, h_cur)
        if not self._box_valid(projected) or not self._temporal_ok(projected):
            return None, None, 0
        return h_cur, projected, int(mask.sum())

    def track(self, frame):
        """Обрабатывает кадр, возвращает копию с нарисованной рамкой (или без неё)."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors, good, n_good = self._match(frame_gray)
        result = frame.copy()
        found = False
        H = None

        if n_good >= self.min_matches:
            src_pts = np.float32(
                [self.template_keypoints[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [frame_keypoints[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh
            )
            if homography is not None:
                self.inliers = int(mask.sum())
                projected = cv2.perspectiveTransform(
                    self.template_corners, homography
                )
                if self.inliers >= self.min_inliers and self._box_valid(projected) \
                        and self._temporal_ok(projected):
                    self.last_box = projected.reshape(4, 2)
                    result = self._draw_box(result, projected, (0, 255, 0))
                    found = True
                    H = homography

        if not found:
            # Прямое сопоставление не дало устойчивой рамки — пробуем перенести
            # её из предыдущего кадра (временной трекинг).
            H, projected, n_inl = self._propagate(
                frame_keypoints, frame_descriptors
            )
            if H is not None:
                self.inliers = n_inl
                self.last_box = projected.reshape(4, 2)
                result = self._draw_box(result, projected, (0, 255, 0))
                found = True

        # Обновляем состояние для следующего кадра.
        self._prev_des = frame_descriptors
        self._prev_kp = frame_keypoints
        if found:
            self._H_ref = H

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

    def _box_valid(self, projected):
        """Геометрическая проверка рамки: выпуклый четырёхугольник разумного
        размера и пропорций. Отсекает вырожденные/ложные гомографии (точка,
        вытянутая полоса, гигантский или крошечный четырёхугольник)."""
        pts = projected.reshape(4, 2)
        if not cv2.isContourConvex(np.int32(pts)):
            return False
        area = abs(cv2.contourArea(np.int32(pts)))
        if area < 800 or area > 0.6 * self.frame_area:
            return False
        sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
        if min(sides) < 15:
            return False
        # Не допускаем слишком вытянутые (почти вырожденные) четырёхугольники.
        return max(sides) / min(sides) <= 5.0

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