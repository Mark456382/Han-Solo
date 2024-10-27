import os

def count_annotations(label_dir):
    class_counts = {}
    bounding_boxes = []

    # Проходим по всем файлам в директории с аннотациями
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        print(f"Предупреждение: некорректная строка в файле {label_file}: {line}")
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Увеличиваем счетчик для класса
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                    else:
                        class_counts[class_id] = 1

                    # Добавляем информацию о bounding box
                    bounding_boxes.append((class_id, x_center, y_center, width, height))

    return class_counts, bounding_boxes

def print_statistics(class_counts, bounding_boxes):
    print("Статистика по классам:")
    class_stats = ", ".join([f"Класс {class_id}: {count} аннотаций" for class_id, count in class_counts.items()])
    print(class_stats)

    print("\nИнформация о bounding box:")
    print(f"{'Класс':<6} {'x_center':<10} {'y_center':<10} {'width':<10} {'height':<10}")
    for bb in bounding_boxes:
        class_id, x_center, y_center, width, height = bb
        print(f"{class_id:<6} {x_center:<10.4f} {y_center:<10.4f} {width:<10.4f} {height:<10.4f}")

if __name__ == "__main__":
    # Путь к директории с аннотациями
    label_dir = "split_dataset_output/labels/train"

    # Считаем аннотации
    class_counts, bounding_boxes = count_annotations(label_dir)

    # Выводим статистику
    print_statistics(class_counts, bounding_boxes)