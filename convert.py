import os
from pathlib import Path
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np


def binarize_mask(image):
    """
    Бинаризует изображение: все ненулевые значения становятся 1, нулевые остаются 0.
    :param image: массив изображения.
    :return: бинаризованное изображение.
    """
    return (image != 0).astype(np.uint8)

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
                    
                    # Бинаризуем маску
                    binary_image = binarize_mask(image)
                    
                    # Создаем новое имя файла с расширением .png
                    new_filename = os.path.splitext(filename)[0] + '.png'
                    
                    # Сохраняем изображение в корневой директории
                    new_path = os.path.join(root_directory, new_filename)
                    
                    # Используем matplotlib для сохранения изображения в формате PNG
                    plt.imshow(binary_image.squeeze(), cmap='gray')
                    plt.axis('off')  # Убираем оси
                    plt.savefig(new_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    print(f'Преобразован {filename} в {new_filename}')
            except Exception as e:
                print(f'Ошибка при обработке файла {filename}: {e}')

if __name__ == "__main__":
    # Укажите путь к директории с исходными файлами
    source_directory = "mask_split_1024"
    
    # Укажите путь к корневой директории, куда будут сохраняться преобразованные файлы
    root_directory = "mask_split_1024_png"
    
    # Создаем корневую директорию, если она не существует
    Path(root_directory).mkdir(parents=True, exist_ok=True)
    
    # Преобразуем файлы
    convert_geotiff_to_png(source_directory, root_directory)