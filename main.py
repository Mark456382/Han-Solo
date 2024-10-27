import os
from pathlib import Path
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import argparse

# Функции для преобразования GeoTIFF в PNG
def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band - band_min) / (band_max - band_min))

def brighten(band):
    alpha = 0.13
    beta = 0
    return np.clip(alpha * band + beta, 0, 255)

def convert_geotiff_to_png(source_directory, png_directory):
    for filename in os.listdir(source_directory):
        if filename.endswith('.tif'):
            try:
                img_path = os.path.join(source_directory, filename)
                with rasterio.open(img_path) as src:
                    image = src.read(masked=True)
                    if image.shape[0] > 3:
                        image = image[:3, :, :]
                    image = np.stack([brighten(normalize(band)) for band in image], axis=0)
                    new_filename = os.path.splitext(filename)[0] + '.png'
                    new_path = os.path.join(png_directory, new_filename)
                    plt.imshow(image.transpose(1, 2, 0))
                    plt.axis('off')
                    plt.savefig(new_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    print(f'Преобразован {filename} в {new_filename}')
            except Exception as e:
                print(f'Ошибка при обработке файла {filename}: {e}')

# Функция для предварительной обработки изображения
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Функция для получения сегментации
def get_segmentation(image_path, output_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
    output = torch.softmax(output, dim=1)
    _, predicted = torch.max(output, 1)
    predicted_image = predicted.squeeze().cpu().numpy()
    plt.imshow(predicted_image, cmap='gray')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Сегментация сохранена в {output_path}")

# Основная функция
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сегментация изображений с использованием нейросетевой модели.")
    parser.add_argument("image_path", type=str, help="Путь к изображению для сегментации.")
    parser.add_argument("--output_directory", type=str, default="output_segmentations", help="Путь к директории для сохранения результатов сегментации.")
    args = parser.parse_args()

    # Путь к директории для сохранения результатов сегментации
    output_directory = args.output_directory

    # Создаем директорию для сохранения результатов, если она не существует
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Загрузка модели с разрешенными глобальными переменными
    from ultralytics.nn.tasks import SegmentationModel
    import builtins
    torch.serialization.add_safe_globals([SegmentationModel, builtins.set])
    model_dict = torch.load('flood-seg.pt', weights_only=True)
    model = model_dict['model']  # Предполагаем, что модель сохранена под ключом 'model'
    model.eval()

    # Обрабатываем изображение
    input_image_path = args.image_path
    output_image_path = os.path.join(output_directory, f"segmented_{os.path.basename(input_image_path)}.png")
    get_segmentation(input_image_path, output_image_path)

    print("Обработка изображения завершена.")