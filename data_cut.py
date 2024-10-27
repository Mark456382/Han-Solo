import os
import shutil
import random
import cv2
import numpy as np

def split_dataset(image_dir, mask_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Проверка, что сумма пропорций равна 1
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Сумма пропорций должна быть равна 1.0")

    # Получаем список файлов в папках с изображениями и масками
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    # Проверяем, что количество файлов совпадает
    if len(image_files) != len(mask_files):
        raise ValueError("Количество изображений и масок не совпадает.")

    # Проверяем, что списки не пусты
    if not image_files or not mask_files:
        raise ValueError("В директориях нет файлов с расширением .png.")

    # Перемешиваем списки файлов
    combined = list(zip(image_files, mask_files))
    random.shuffle(combined)
    image_files, mask_files = zip(*combined)

    # Вычисляем количество файлов для каждой выборки
    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count

    # Создаем директории для каждой выборки в новой корневой папке
    train_image_dir = os.path.join(output_dir, 'images', 'train')
    val_image_dir = os.path.join(output_dir, 'images', 'val')
    test_image_dir = os.path.join(output_dir, 'images', 'test')

    train_mask_dir = os.path.join(output_dir, 'masks', 'train')
    val_mask_dir = os.path.join(output_dir, 'masks', 'val')
    test_mask_dir = os.path.join(output_dir, 'masks', 'test')

    train_label_dir = os.path.join(output_dir, 'labels', 'train')
    val_label_dir = os.path.join(output_dir, 'labels', 'val')
    test_label_dir = os.path.join(output_dir, 'labels', 'test')

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)

    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)

    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # Перемещаем файлы в соответствующие директории и создаем аннотации
    for i, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        if i < train_count:
            shutil.copy(os.path.join(image_dir, image_file), os.path.join(train_image_dir, image_file))
            shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(train_mask_dir, mask_file))
            create_annotation(train_mask_dir, mask_file, train_label_dir, image_file)
        elif i < train_count + val_count:
            shutil.copy(os.path.join(image_dir, image_file), os.path.join(val_image_dir, image_file))
            shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(val_mask_dir, mask_file))
            create_annotation(val_mask_dir, mask_file, val_label_dir, image_file)
        else:
            shutil.copy(os.path.join(image_dir, image_file), os.path.join(test_image_dir, image_file))
            shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(test_mask_dir, mask_file))
            create_annotation(test_mask_dir, mask_file, test_label_dir, image_file)

def create_annotation(mask_dir, mask_file, label_dir, image_file):
    mask_path = os.path.join(mask_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Находим контуры объектов на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем аннотацию
    annotation_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
    with open(annotation_file, 'w') as f:
        if len(contours) == 0:
            # Если нет контуров, добавляем только класс 1 (нет контуров) и нулевые координаты
            f.write('1 0 0 0 0 0 0 0 0 0 0 0 0\n')
        else:
            for contour in contours:
                # Вычисляем ограничивающую рамку
                x, y, w, h = cv2.boundingRect(contour)
                x_center = (x + w / 2) / mask.shape[1]
                y_center = (y + h / 2) / mask.shape[0]
                width = w / mask.shape[1]
                height = h / mask.shape[0]

                # Записываем аннотацию в файл
                f.write('0 ')  # Класс 0 (есть контуры)
                f.write(f'{x_center} {y_center} {width} {height} ')

                # Записываем координаты полигона в формате YOLO
                for point in contour:
                    x_norm = point[0][0] / mask.shape[1]
                    y_norm = point[0][1] / mask.shape[0]
                    f.write(f'{x_norm} {y_norm} ')
                f.write('\n')

if __name__ == "__main__":
    # Фиксированные пути к папкам с изображениями и масками
    image_dir = "image_split_1024_png"
    mask_dir = "mask_split_1024_png"

    # Путь к новой корневой папке для сохранения результатов
    output_dir = "split_dataset_output"

    # Пропорции для разбиения на выборки
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    split_dataset(image_dir, mask_dir, output_dir, train_ratio, val_ratio, test_ratio)