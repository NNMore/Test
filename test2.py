import numpy as np
import cv2
import onnxruntime as ort
from threading import Thread

# Загрузка ONNX модели
onnx_model_path = 'ssd_resnet34_dpe400_check_onnx.onnx'
session = ort.InferenceSession(onnx_model_path)

# Функция для предобработки изображения


def preprocess_image(frame):
    # Изменение размера до 300x300
    frame = cv2.resize(frame, (300, 300))
    # Переставляем оси: HWC -> CHW
    # (height, width, channels) -> (channels, height, width)
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)  # Добавление размерности батча
    return frame.astype(np.float32)


def processing(file, file_name):
    # Открываем видеофайл
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print(f"Error: Невозможно открыть видеофайл {file}")
        return

    while True:
        # Чтение кадра из видео
        ret, frame = cap.read()
        if not ret:
            print(f"Конец видеопотока {file}")
            break

        frame_copy = preprocess_image(frame)
        input_name = session.get_inputs()[0].name

        # Получаем результаты
        output = session.run(None, {input_name: frame_copy})
        print(output[0])
        cv2.imshow(file_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
