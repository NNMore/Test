import cv2
from threading import Thread
from tracker3 import *


def processing(file, file_name):
    # Открываем видеофайл
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print(f"Error: Невозможно открыть видеофайл {file}")
        return

    alpha = 0.999  # Коэффициент для фоновой обработки
    isFirstTime = True  # Переменная для отслеживания первого кадра
    tracker = HungarianTracker()  # Инициализация трекера

    while True:
        # Чтение кадра из видео
        ret, frame = cap.read()
        if not ret:
            print(f"Конец видеопотока {file}")
            break

        # Изменение размера кадра для уменьшения вычислительной нагрузки
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        frame_copy = frame.copy()  # Копируем текущий кадр для фоновой обработки
        frame_copy[350:360, 340:380] = 127

        if isFirstTime:
            bg_img = frame_copy  # Устанавливаем фоновое изображение на первый кадр
            isFirstTime = False
        else:
            # Обновляем фоновое изображение с использованием смешивания
            bg_img = cv2.addWeighted(frame_copy, (1-alpha), bg_img, alpha, 0)

        # Вычисляем изображение переднего плана
        fg_img = cv2.absdiff(frame_copy, bg_img)
        # Преобразуем в оттенки серого
        gray = cv2.cvtColor(fg_img, cv2.COLOR_RGB2GRAY)
        # Применяем пороговую фильтрацию
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 8)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Находим контуры

        detections = []
        if contours:
            # Находим самый большой контур как основной объект
            largest_contour = max(contours, key=cv2.contourArea)
            # Получаем координаты и размеры bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Добавляем информацию о выделенном объекте
            detections.append([x, y, w, h])

        # Обновление трекера с новыми данными о детекциях
        boxes_ids = tracker.update(detections, frame_copy)
        histogram = tracker.retain_histograms()  # Сохраняем гистограммы для трекера

        for box_id in boxes_ids:
            x, y, w, h, id = box_id  # Распаковываем информацию о треках
            # Отображение идентификатора объекта на кадре
            cv2.putText(frame, str(id), (x, y - 15),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            # Рисуем прямоугольник вокруг обнаруженного объекта
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # Показ обработанного кадра
        cv2.imshow(file_name, frame)
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break


def first_cam():
    # Запуск обработки первого видеопотока
    processing(
        'D:/тестовое задание/3.Camera 2017-05-29 16-23-04_137 [3m3s].avi', 'First Camera')


def second_cam():
    # Запуск обработки второго видеопотока
    processing(
        'D:/тестовое задание/4.Camera 2017-05-29 16-23-04_137 [3m3s].avi', 'Second Camera')


# Создание потоков для обработки видео с двух камер
first_thread = Thread(target=first_cam, args=())
second_thread = Thread(target=second_cam, args=())

# Запуск потоков
first_thread.start()
second_thread.start()
