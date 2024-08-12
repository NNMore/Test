import onnxruntime as ort
from torchvision.io import read_image
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import torch

# Подгружаем изображение
img = read_image("C:/Users/meeed/Downloads/40.jpg")

# Шаг 1: Подгружаем модель ONNX
onnx_model_path = "model_simplified.onnx"

# Шаг 2: Инициализация сессии ONNX Runtime
ort_session = ort.InferenceSession(onnx_model_path)

# Шаг 3: Предобработка изображения для модели


def preprocess_image(image):
    # Размеры изображения должны соответствовать входным требованиям модели (1, 3, 300, 300)
    image = F.resize(image, [300, 300])  # Изменение размера изображения
    image = F.convert_image_dtype(image, dtype=torch.float32)
    image = image.unsqueeze(0)  # Добавление батча
    return image.numpy()  # Возвращаем NumPy массив


# Предобработка изображения
input_data = preprocess_image(img)

# Шаг 4: Выполнение инференса
ort_inputs = {ort_session.get_inputs()[0].name: input_data}
ort_outs = ort_session.run(None, ort_inputs)

# Шаг 5: Обработка выходных данных модели
boxes = ort_outs[0]  # Извлекаем предсказанные боксы

# Нарисуем боксы на изображении
for box in boxes:
    x_min, y_min, x_max, y_max, score, class_id = box
    if score > 0.1:  # Отфильтровываем по порогу вероятности
        img = draw_bounding_boxes(img, boxes=torch.tensor([[x_min, y_min, x_max, y_max]]),
                                  labels=[str(int(class_id))], colors="red", width=4)

# Преобразуем обратно в PIL и показываем изображение
img_pil = Image.fromarray(img.permute(1, 2, 0).byte().numpy())
img_pil.show()
