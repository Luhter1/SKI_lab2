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
    """Трекинг плоского текстурированного объекта по ключевым точкам ORB.

    Основная идея: объект описывается набором
    ключевых точек и дескрипторов. На каждом кадре мы ищем те же точки,
    сопоставляем их с эталоном и по соответствиям восстанавливаем рамку.
    """

    # Подпись, которой подписываем рамку на кадре.
    LABEL = "object"

    def __init__(
        self,
        template_img,        # эталон: изображение объекта (первый кадр видео)
        n_features=3000,     # макс. число ключевых точек ORB
        ratio=0.85,          # порог ratio-теста Лоу (см. TEORIA.md, раздел 4)
        min_inliers=6,       # мин. число inlier'ов для «уверенной» гомографии
        min_matches=4,       # мин. число хороших пар до попытки построить H
        scale_factor=1.2,    # коэффициент уменьшения между уровнями пирамиды ORB
        nlevels=12,          # число уровней пирамиды масштабов ORB
        crosscheck=True,     # взаимная проверка ближайшего соседа в BFMatcher
        ransac_thresh=5.0,   # порог расстояния для RANSAC (пиксели)
    ):
        # --- ORB (Oriented FAST + Rotated BRIEF)
        # Детектор FAST ищет яркостные углы; BRIEF даёт бинарный дескриптор,
        # поэтому сравнение идёт по метрике Хэмминга.
        # Пирамида масштабов: изображение уменьшается scale_factor раз между
        # уровнями; 1.2^11 ~ 7.4x покрытия по масштабу. Благодаря этому объект
        # находится даже при сильном уменьшении (приближении/удалении камеры).
        self.orb = cv2.ORB_create(
            nfeatures=n_features,      # сколько максимум точек выделять
            scaleFactor=scale_factor,  # шаг пирамиды
            nlevels=nlevels,           # глубина пирамиды
        )
        # --- Матчер дескрипторов. NORM_HAMMING — метрика Хэмминга (число
        # различающихся битов) для бинарных дескрипторов BRIEF (см. раздел 3).
        # crossCheck=True — «взаимный» поиск: пара берётся только если точка
        # эталона — ближайший сосед для точки кадра И наоборот. Это уже
        # отсеивает ложные пары, поэтому ratio-тест тогда не нужен.
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crosscheck)
        self.crosscheck = crosscheck      # флаг режима взаимной проверки
        self.ratio = ratio                # порог для ratio-теста Лоу
        self.min_inliers = min_inliers    # порог «уверенности» в гомографии
        self.min_matches = min_matches    # мин. число пар для RANSAC
        self.ransac_thresh = ransac_thresh  # пиксельный порог inlier'ов

        # Матчер для временного (межкадрового) трекинга: сопоставляет точки
        # предыдущего кадра с текущим, чтобы переносить рамку, когда прямое
        # сопоставление с эталоном слабое. Всегда со взаимной проверкой.
        self._cc_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Состояние временного трекинга: дескрипторы/точки предыдущего кадра
        # и гомография «эталон -> последний кадр» (используется для пропагации).
        self._prev_des = None
        self._prev_kp = None
        self._H_ref = None

        # --- Эталоном является само переданное изображение объекта
        # (первый кадр видео). Сразу переводим в оттенки серого, т.к. ORB
        # работает по яркости, и вычисляем ключевые точки с дескрипторами.
        self.template = template_img
        self.template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        self.template_keypoints, self.template_descriptors = self.orb.detectAndCompute(
            self.template_gray, None
        )

        # --- Углы эталона: 4 вершины прямоугольника объекта в координатах
        # эталона. Проецируя их через гомографию на кадр, получаем рамку
        # Формат Nx1x2 нужен для
        # cv2.perspectiveTransform.
        h, w = self.template_gray.shape[:2]   # высота и ширина эталона
        self.frame_area = h * w               # площадь кадра-эталона (для проверки рамки)
        self.template_corners = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]]   # углы: ЛВ, ПВ, ПН, ЛН
        ).reshape(-1, 1, 2)

        # --- Счётчики для статистики и отладки.
        self.raw_matches = 0   # всего пар после матчинга
        self.good_matches = 0  # пар после ratio-теста / взаимной проверки
        self.inliers = 0       # inlier'ов после RANSAC
        self.last_box = None   # четырёхугольник рамки в координатах кадра

    def _match(self, frame_gray):
        """Поиск сопоставлений между эталоном и кадром.

        Возвращает ключевые точки и дескрипторы кадра, список «хороших» пар
        и их количество. См. TEORIA.md, разделы 4 (ratio-тест) и 3 (Хэмминг).
        """
        # Детектируем ключевые точки текущего кадра и вычисляем их дескрипторы.
        frame_keypoints, frame_descriptors = self.orb.detectAndCompute(
            frame_gray, None
        )
        # Защита от вырожденного случая: если у эталона или кадра нет
        # дескрипторов (или их слишком мало), сопоставлять нечего.
        if (
            self.template_descriptors is None
            or frame_descriptors is None
            or len(frame_descriptors) < 2
        ):
            return None, None, 0, 0

        self.raw_matches = len(self.template_descriptors)
        if self.crosscheck:
            # Режим взаимной проверки: парные соответствия уже «хорошие»,
            # т.к. взаимный поиск ближайшего соседа отсеял ложные пары.
            # Метод .match возвращает по одному лучшему совпадению на точку.
            matches = self.matcher.match(self.template_descriptors, frame_descriptors)
            good = list(matches)
            self.good_matches = len(good)
            return frame_keypoints, frame_descriptors, good, self.good_matches
        # --- Иначе — режим ratio-теста Лоу. k=2: ищем ДВА ближайших соседа,
        # чтобы судить об однозначности пары.
        matches = self.matcher.knnMatch(self.template_descriptors, frame_descriptors, k=2)

        good = []
        for pair in matches:
            if len(pair) < 2:   # у точки нет двух соседей — пара неоднозначна
                continue
            m, n = pair   # m — ближайший сосед, n — второй ближайший
            # Ratio-тест Лоу: если первый сосед почти так же близок, как второй,
            # точка неоднозначна — это, скорее всего, ложное совпадение.
            # Оставляем только пары с запасом: d_m < ratio * d_n.
            if m.distance < self.ratio * n.distance:
                good.append(m)
        self.good_matches = len(good)
        return frame_keypoints, frame_descriptors, good, self.good_matches

    def _area_ok(self, projected):
        """Рамка не должна резко менять размер между соседними кадрами.

        Эвристика временной согласованности: площадь текущего четырёхугольника
        должна отличаться от площади предыдущей рамки не более чем в 3 раза.
        Иначе — это ложная гомография на посторонний объект.
        """
        if self.last_box is None:
            return True   # предыдущей рамки нет — не с чем сравнивать
        a = abs(cv2.contourArea(np.int32(projected)))   # площадь новой рамки
        pa = abs(cv2.contourArea(np.int32(self.last_box)))  # площадь прошлой
        if pa > 0:
            ratio = a / pa
            if ratio > 3.0 or ratio < 1.0 / 3.0:   # сильный скачок размера
                return False
        return True

    def _temporal_ok(self, projected):
        """Временная согласованность рамки с предыдущим кадром.

        Помимо похожего размера (см. _area_ok), центр рамки не должен «прыгать»
        дальше, чем на длину диагонали рамки. Это отсекает ложные детекции,
        когда алгоритм перескакивает на другое напечатанное изображение в кадре.
        """
        if self.last_box is None:
            return True
        if not self._area_ok(projected):   # размер изменился слишком резко
            return False
        c = projected.reshape(4, 2).mean(axis=0)   # центр новой рамки
        pc = np.array(self.last_box).mean(axis=0)  # центр прошлой рамки
        diag = np.linalg.norm(                     # длина диагонали прошлой рамки
            np.array(self.last_box[0]) - np.array(self.last_box[2])
        )
        if diag <= 0:
            return True   # вырожденная рамка — не можем оценить скачок
        # Расстояние между центрами не должно превышать длину диагонали.
        return np.hypot(c[0] - pc[0], c[1] - pc[1]) <= 1.0 * diag

    def _propagate(self, frame_kp, frame_des):
        """Временная пропагация рамки через гомографию между соседними кадрами.

        Если прямое сопоставление с эталоном слабое (объект мал/размыт), но в
        предыдущем кадре рамка уже построена, переносим её на текущий кадр:
           H(эталон->текущий) = H(предыдущий->текущий) @ H(эталон->предыдущий).
        Чтобы рамка не «раздувалась» из-за преобладающих на фоне точек,
        учитываются только совпадения внутри предыдущей рамки объекта.
        """
        # Нужны состояние временного трекинга и достаточно точек в обоих кадрах.
        if (
            self._H_ref is None
            or self.last_box is None
            or self._prev_des is None
            or frame_des is None
            or len(self._prev_des) < 4
            or len(frame_des) < 4
        ):
            return None, None, 0
        # Сопоставляем дескрипторы предыдущего и текущего кадров.
        matches = self._cc_matcher.match(self._prev_des, frame_des)
        poly = np.int32(self.last_box)   # полигон прошлой рамки
        # Оставляем только пары, чья точка в предыдущем кадре лежит ВНУТРИ рамки
        # (>= 0 — внутри/на границе полигона по pointPolygonTest).
        kept = [
            m for m in matches
            if cv2.pointPolygonTest(poly, self._prev_kp[m.queryIdx].pt, False) >= 0
        ]
        if len(kept) < 6:   # слишком мало точек внутри рамки — пропагация ненадёжна
            return None, None, 0
        # Координаты пар точек: src — в предыдущем кадре, dst — в текущем.
        src = np.float32(
            [self._prev_kp[m.queryIdx].pt for m in kept]
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [frame_kp[m.trainIdx].pt for m in kept]
        ).reshape(-1, 1, 2)
        # Гомография «предыдущий -> текущий» кадр, оценённая RANSAC с порогом 3px.
        h_prev_cur, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if h_prev_cur is None:
            return None, None, 0
        # Композиция гомографий: H(эталон->текущий) = H(пред->тек) @ H(эталон->пред).
        h_cur = h_prev_cur @ self._H_ref
        # Проецируем углы эталона через итоговую гомографию -> рамка.
        projected = cv2.perspectiveTransform(self.template_corners, h_cur)
        # Отбрасываем геометрически невалидную или скачущую рамку.
        if not self._box_valid(projected) or not self._temporal_ok(projected):
            return None, None, 0
        # Возвращаем новую гомографию, рамку и число inlier'ов (mask.sum()).
        return h_cur, projected, int(mask.sum())

    def track(self, frame):
        """Обрабатывает кадр, возвращает копию с нарисованной рамкой (или без неё)."""
        # Ключевые точки ORB работают по яркости -> переводим кадр в серый.
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Сопоставляем кадр с эталоном (детекция + матчинг + ratio-тест).
        frame_keypoints, frame_descriptors, good, n_good = self._match(frame_gray)
        result = frame.copy()   # рисуем на копии, чтобы не портить исходный кадр
        found = False           # найден ли объект в этом кадре
        H = None                # актуальная гомография «эталон -> кадр»

        # --- Прямое сопоставление с эталоном.
        # Хороших пар достаточно, чтобы попытаться построить гомографию.
        if n_good >= self.min_matches:
            # Координаты соответствующих точек: src — на эталоне, dst — на кадре.
            src_pts = np.float32(
                [self.template_keypoints[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [frame_keypoints[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            # Оценка гомографии по сопоставлениям через RANSAC (см. раздел 6).
            # RANSAC игнорирует выбросы (ложные пары) и возвращает маску inlier'ов.
            homography, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh
            )
            if homography is not None:
                self.inliers = int(mask.sum())   # число inlier'ов
                # Проецируем 4 угла эталона через H -> рамка (см. раздел 5).
                projected = cv2.perspectiveTransform(
                    self.template_corners, homography
                )
                # Принимаем рамку только если: достаточно inlier'ов, рамка
                # геометрически корректна и временно согласована с прошлой.
                if self.inliers >= self.min_inliers and self._box_valid(projected) \
                        and self._temporal_ok(projected):
                    self.last_box = projected.reshape(4, 2)   # запоминаем рамку
                    result = self._draw_box(result, projected, (0, 255, 0))  # зелёная
                    found = True
                    H = homography

        if not found:
            # --- Прямое сопоставление не дало устойчивой рамки — пробуем
            # перенести её из предыдущего кадра (временной трекинг).
            H, projected, n_inl = self._propagate(
                frame_keypoints, frame_descriptors
            )
            if H is not None:
                self.inliers = n_inl
                self.last_box = projected.reshape(4, 2)
                result = self._draw_box(result, projected, (0, 255, 0))
                found = True

        # --- Обновляем состояние временного трекинга для следующего кадра:
        # сохраняем дескрипторы и ключевые точки текущего кадра.
        self._prev_des = frame_descriptors
        self._prev_kp = frame_keypoints
        # Если объект найден — обновляем H(эталон -> текущий кадр) для пропагации.
        if found:
            self._H_ref = H

        if not found:
            # --- Объект не найден: сбрасываем состояние и рисуем красную подпись.
            self.inliers = 0
            self.last_box = None
            cv2.putText(
                result,
                self.LABEL + ": not found",
                (15, 30),                  # координаты левого верхнего угла текста
                cv2.FONT_HERSHEY_SIMPLEX,  # шрифт
                0.8,                       # масштаб шрифта
                (0, 0, 255),               # красный цвет (BGR)
                2,                         # толщина
            )
        return result, found

    def _box_valid(self, projected):
        """Геометрическая проверка рамки.

        Рамка должна быть выпуклым четырёхугольником разумного размера и
        пропорций. Это отсекает вырожденные/ложные гомографии: точка,
        вытянутая полоса, гигантский или крошечный четырёхугольник.
        """
        pts = projected.reshape(4, 2)
        # Гомография сохраняет выпуклость; вогнутый четырёхугольник — ложная.
        if not cv2.isContourConvex(np.int32(pts)):
            return False
        area = abs(cv2.contourArea(np.int32(pts)))   # площадь рамки
        # Рамка не должна быть слишком малой (вырожденной) или больше кадра.
        if area < 800 or area > 0.6 * self.frame_area:
            return False
        # Длины сторон рамки.
        sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
        if min(sides) < 15:   # слишком короткая сторона — рамка вырождена
            return False
        # Не допускаем слишком вытянутые (почти вырожденные) четырёхугольники:
        # отношение длин сторон не больше 5.
        return max(sides) / min(sides) <= 5.0

    def _draw_box(self, frame, projected, color):
        """Рисует рамку и подпись вокруг найденного объекта."""
        # cv2.polylines рисует замкнутую ломаную по 4 углам рамки (True = замкнуть).
        cv2.polylines(
            frame, [np.int32(projected)], True, color, 3, cv2.LINE_AA
        )
        # Подпись над верхним левым углом рамки (projected[0] — ЛВ угол).
        x, y = int(projected[0][0][0]), int(projected[0][0][1])
        cv2.putText(
            frame,
            self.LABEL,
            (x, y - 8),                # текст чуть выше рамки
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,               # сглаживание
        )
        return frame


def process_video(
    input_path,
    output_path,
    show=False,
    stats=True,
):
    """Обрабатывает видео: первый кадр — эталон, далее трекинг."""
    # Открываем видеофайл для чтения кадров.
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():   # видео не открылось (нет файла / битый кодек)
        raise IOError(f"Не удалось открыть видео: {input_path}")

    # Считываем метаданные видео: ширину, высоту кадра и частоту кадров (FPS).
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0   # если FPS не определён — берём 25

    # Создаём VideoWriter для сохранения результата (кодек mp4v).
    writer = None
    if output_path:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    # Читаем первый кадр: он считается крупным изображением объекта (эталоном).
    ret, first = cap.read()
    if not ret:   # видео пустое — нет даже первого кадра
        cap.release()
        raise IOError("Видео пустое (нет первого кадра).")

    # Создаём трекер, передавая эталоном первый кадр видео.
    template = first
    tracker = KeypointTracker(template)

    # Списки для статистики по всем кадрам.
    found_flags = []   # найден ли объект в каждом кадре
    inliers_list = []  # число inlier'ов в каждом кадре

    while True:
        # Читаем следующий кадр; ret=False означает конец видео.
        ret, frame = cap.read()
        if not ret:
            break

        # Обрабатываем кадр трекером -> кадр с рамкой и флаг «найден».
        result, found = tracker.track(frame)
        inliers_list.append(tracker.inliers)
        if writer:   # сохраняем размеченный кадр в выходное видео
            writer.write(result)
        if show:     # показываем кадр в окне (режим просмотра)
            cv2.imshow("tracker", result)
            # Выход по клавише 'q'.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        found_flags.append(found)

    # Освобождаем ресурсы: закрываем видео и (если создан) writer.
    cap.release()
    if writer:
        writer.release()
    if show:   # закрываем все окна OpenCV
        cv2.destroyAllWindows()

    # --- Итоговая статистика по обработке видео.
    if stats:
        total = len(found_flags)    # число обработанных кадров сцены
        tracked = sum(found_flags)  # из них найден объект
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