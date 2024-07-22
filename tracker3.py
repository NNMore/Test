import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class HungarianTracker:
    def __init__(self, distance_threshold=50):
        self.center_points = {}
        self.histograms = {}
        self.id_count = 0
        self.distance_threshold = distance_threshold

    def compute_histogram(self, image, rect):
        x, y, w, h = rect
        roi = image[y:y+h, x:x+w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        hsv_channels = list(cv2.split(roi))
        hsv_channels[2] = cv2.equalizeHist(
            hsv_channels[2])  # Эквализация по яркости
        roi = cv2.merge(hsv_channels)

        # Вычисление гистограммы
        hist = cv2.calcHist([roi], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)  # Нормализация гистограммы
        return hist.flatten()

    def update(self, objects_rect, image):
        num_objects = len(objects_rect)
        cost_matrix = np.zeros((num_objects, len(self.center_points)))

        # Заполнение матрицы затрат на основе гистограмм цветов
        for i, rect1 in enumerate(objects_rect):
            hist1 = self.compute_histogram(image, rect1)
            for j, (id, hist2) in enumerate(self.histograms.items()):
                cost = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                cost_matrix[i, j] = cost

        # Поиск соответствий через венгерский алгоритм
        if num_objects > 0 and len(self.center_points) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = [], []

        objects_bbs_ids = []

        # Сопоставление обнаруженных объектов с известными
        matched_ids = set()  # Множество для хранения соответствующих ID
        for i in range(num_objects):
            found = False
            if i < len(col_ind):  # Если у нас есть столбцы (существующие объекты)
                j = col_ind[i]  # Получаем индекс объекта
                if j < len(self.center_points):  # Проверяем, существует ли объект
                    id = list(self.histograms.keys())[j]
                    x, y, w, h = objects_rect[i]
                    cx, cy = (x + w // 2), (y + h // 2)

                    if (id not in matched_ids and
                        abs(cx - self.center_points[id][0]) < self.distance_threshold and
                            abs(cy - self.center_points[id][1]) < self.distance_threshold):
                        self.center_points[id] = (cx, cy)  # Обновляем центр
                        objects_bbs_ids.append([x, y, w, h, id])
                        matched_ids.add(id)  # Добавляем ID в множество
                        found = True

            # Если объект не найден в соответствующих, то мы создаем новый ID
            if not found:
                x, y, w, h = objects_rect[i]
                new_id = self.id_count
                self.center_points[new_id] = ((x + w // 2), (y + h // 2))
                self.histograms[new_id] = self.compute_histogram(
                    image, (x, y, w, h))
                objects_bbs_ids.append([x, y, w, h, new_id])
                self.id_count += 1

        return objects_bbs_ids

    def retain_histograms(self):
        return self.histograms
