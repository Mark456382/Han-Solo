import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import os

# Путь к исходному изображению
input_path = r"train_dataset_skoltech_train\train\images\1.tif"

# Размер одного маленького изображения (например, 256x256)
tile_size = 512

# Открываем изображение
with rasterio.open(input_path) as src:
    # Получаем ширину и высоту изображения
    width = src.width
    height = src.height
    
    # Инициализируем директорию для сохранения маленьких изображений
    output_dir = 'image_split_512'
    os.makedirs(output_dir, exist_ok=True)

    # Проходим по каждой строке и колонке изображения
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            # Определяем окно для чтения данных
            window = Window(col, row, tile_size, tile_size)
            
            # Читаем данные из окна
            data = src.read(window=window, out_shape=(src.count, tile_size, tile_size),
                            resampling=Resampling.nearest)
            
            # Создаем профиль для нового изображения
            profile = src.profile.copy()
            profile.update({
                'height': tile_size,
                'width': tile_size,
                'transform': rasterio.transform.from_bounds(*rasterio.windows.bounds(window, transform=src.transform), tile_size, tile_size)
            })

            # Формируем имя файла
            file_name = f"{row // tile_size}_{col // tile_size}.tif"
            output_file = os.path.join(output_dir, file_name)

            # Записываем новое изображение
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(data)