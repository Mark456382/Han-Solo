from ultralytics import YOLO
import os
import torch

# Проверка доступности CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Загрузка модели
model = YOLO("best.pt")  # загрузка предварительно обученной модели (рекомендуется для обучения)

if __name__ == '__main__':
    # Использование модели
    model.train(data="data.yaml", epochs=30, device=0, batch=4, workers=4)
    metrics = model.val()  # оценка производительности модели на валидационном наборе данных