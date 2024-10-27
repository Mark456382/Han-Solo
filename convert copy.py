import os
from pathlib import Path
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np

def normalize(band):
    """
    Нормализует данные канала в диапазон [0, 1].
    :param band: numpy массив канала.
    :return: нормализованный numpy массив канала.
    """
    band_min, band_max = (band.min(), band.max())
    return ((band - band_min) / (band_max - band_min))

def brighten(band):
    """
    Увеличивает яркость канала.
    :param band: numpy массив канала.
    :return: numpy массив канала с увеличенной яркостью.
    """
    alpha = 0.13
    beta = 0
    return np.clip(alpha * band + beta, 0, 255)

def convert_geotiff_to_png(source_directory, root_directory):
    """
    Преобразует все файлы GeoTIFF в указанной директории в формат PNG и сохраняет их в корневой директории.
    :param source_directory: путь к директории с исходными файлами GeoTIFF.
    :param root_directory: путь к корневой директории, куда будут сохраняться преобразованные файлы PNG.
    """
    # Проходим по всем файлам в исходной директории
    for filename in os.listdir(source_directory):
        # Проверяем, что файл имеет расширение .tif
        if filename.endswith('.tif'):
            try:
                # Открываем GeoTIFF
                img_path = os.path.join(source_directory, filename)
                with rasterio.open(img_path) as src:
                    # Читаем данные изображения с использованием masked=True
                    image = src.read(masked=True)
                    
                    # Проверяем форму изображения
                    if image.shape[0] > 3:
                        # Выбираем первые три канала (RGB)
                        image = image[:3, :, :]
                    
                    # Применяем нормализацию и увеличение яркости к каждому каналу
                    image = np.stack([brighten(normalize(band)) for band in image], axis=0)
                    
                    # Создаем новое имя файла с расширением .png
                    new_filename = os.path.splitext(filename)[0] + '.png'
                    
                    # Сохраняем изображение в корневой директории
                    new_path = os.path.join(root_directory, new_filename)
                    
                    # Используем matplotlib для сохранения изображения в формате PNG
                    plt.imshow(image.transpose(1, 2, 0))
                    plt.axis('off')  # Убираем оси
                    plt.savefig(new_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    print(f'Преобразован {filename} в {new_filename}')
            except Exception as e:
                print(f'Ошибка при обработке файла {filename}: {e}')

if __name__ == "__main__":
    # Укажите путь к директории с исходными файлами
    source_directory = "image_split_1024"
    
    # Укажите путь к корневой директории, куда будут сохраняться преобразованные файлы
    root_directory = "image_split_1024_png"
    
    # Преобразуем файлы
    convert_geotiff_to_png(source_directory, root_directory)