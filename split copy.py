import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import os

def split_images_and_masks(image_dir, mask_dir, tile_size=512):
    # Получаем список файлов в папках с изображениями и масками
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])

    # Проверяем, что количество файлов совпадает
    if len(image_files) != len(mask_files):
        raise ValueError("Количество изображений и масок не совпадает.")

    # Инициализируем директории для сохранения маленьких изображений и масок
    image_output_dir = 'image_split_512'
    mask_output_dir = 'mask_split_512'
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Проходим по каждой паре изображений и масок
    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)

        # Открываем изображение и маску
        with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
            # Получаем ширину и высоту изображения
            width = image_src.width
            height = image_src.height

            # Проходим по каждой строке и колонке изображения
            for row in range(0, height, tile_size):
                for col in range(0, width, tile_size):
                    # Определяем окно для чтения данных
                    window = Window(col, row, tile_size, tile_size)

                    # Читаем данные из окна для изображения
                    image_data = image_src.read(window=window, out_shape=(image_src.count, tile_size, tile_size),
                                                resampling=Resampling.nearest)

                    # Читаем данные из окна для маски
                    mask_data = mask_src.read(window=window, out_shape=(mask_src.count, tile_size, tile_size),
                                              resampling=Resampling.nearest)

                    # Создаем профиль для нового изображения
                    image_profile = image_src.profile.copy()
                    image_profile.update({
                        'height': tile_size,
                        'width': tile_size,
                        'transform': rasterio.transform.from_bounds(*rasterio.windows.bounds(window, transform=image_src.transform), tile_size, tile_size)
                    })

                    # Создаем профиль для новой маски
                    mask_profile = mask_src.profile.copy()
                    mask_profile.update({
                        'height': tile_size,
                        'width': tile_size,
                        'transform': rasterio.transform.from_bounds(*rasterio.windows.bounds(window, transform=mask_src.transform), tile_size, tile_size)
                    })

                    # Формируем имена файлов
                    image_file_name = f"{image_file.split('.')[0]}_{row // tile_size}_{col // tile_size}.tif"
                    mask_file_name = f"{mask_file.split('.')[0]}_{row // tile_size}_{col // tile_size}.tif"

                    image_output_file = os.path.join(image_output_dir, image_file_name)
                    mask_output_file = os.path.join(mask_output_dir, mask_file_name)

                    # Записываем новое изображение
                    with rasterio.open(image_output_file, 'w', **image_profile) as image_dst:
                        image_dst.write(image_data)

                    # Записываем новую маску
                    with rasterio.open(mask_output_file, 'w', **mask_profile) as mask_dst:
                        mask_dst.write(mask_data)

if __name__ == "__main__":
    # Фиксированные пути к папкам с изображениями и масками
    image_dir = "train_dataset_skoltech_train/train/images"
    mask_dir = "train_dataset_skoltech_train/train/masks"
    tile_size = 512


    print("Идет разметка. ожидайте")
    split_images_and_masks(image_dir, mask_dir, tile_size)
    print("Готово")
    