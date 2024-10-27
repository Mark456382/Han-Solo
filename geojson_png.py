import geopandas as gpd
import matplotlib.pyplot as plt

# Путь к файлу .geojson
geojson_file = 'train_dataset_skoltech_train/train/osm/9.geojson'

# Путь для сохранения изображения .png
output_png = 'output.png'

# Загрузка данных из .geojson
gdf = gpd.read_file(geojson_file)

# Создание графика
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Отрисовка геометрии
gdf.plot(ax=ax, color='blue')

# Настройка осей
ax.set_aspect('equal')
ax.set_title('GeoJSON to PNG')

# Сохранение изображения в формате .png
plt.savefig(output_png, dpi=300)

# Закрытие графика
plt.close(fig)

print(f"Изображение сохранено в {output_png}")