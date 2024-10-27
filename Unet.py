import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

# 1. Подготовка данных

# Пути к папкам с изображениями и масками
image_dir = 'image_split_256'
mask_dir = 'mask_split_256'

# Загрузка изображений и масок
images = []
masks = []

for filename in os.listdir(image_dir):
    if filename.endswith('.tif'):
        # Загрузка изображения
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)
        images.append(np.array(img))
        
        # Загрузка маски
        mask_path = os.path.join(mask_dir, filename)
        mask = Image.open(mask_path)
        masks.append(np.array(mask))

# Преобразование в массивы numpy
images = np.array(images)
masks = np.array(masks)

# Нормализация данных
images = images / 255.0
masks = masks / 255.0

# 2. Создание датасета и даталоадера

class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        # Преобразование в тензоры
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Пример для RGB изображений
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Добавление канала
        
        return image, mask

# Создание датасета
dataset = SegmentationDataset(images, masks)

# Создание даталоадера
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 3. Определение модели

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Определение слоев модели
        # (здесь упрощенная версия U-Net)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upconv = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x1))
        x2 = self.pool(x1)
        x3 = self.upconv(x2)
        x4 = torch.cat([x1, x3], dim=1)
        x5 = F.relu(self.conv3(x4))
        x6 = torch.sigmoid(self.conv4(x5))
        return x6

# Инициализация модели
model = UNet()

# 4. Обучение модели

# Определение функции потерь и оптимизатора
criterion = nn.BCELoss()  # Бинарная кросс-энтропия для бинарной сегментации
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Логирование
log_file = open('training_log.txt', 'w')

# Цикл обучения
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    iou_scores = []
    
    for images, masks in dataloader:
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Вычисление IoU
        outputs_np = outputs.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()
        for i in range(outputs_np.shape[0]):
            pred_mask = (outputs_np[i] > 0.5).astype(int)
            true_mask = masks_np[i].astype(int)
            iou = jaccard_score(true_mask.flatten(), pred_mask.flatten())
            iou_scores.append(iou)
    
    avg_loss = running_loss / len(dataloader)
    avg_iou = np.mean(iou_scores)
    
    log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}, IoU: {avg_iou}\n')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}, IoU: {avg_iou}')

# Закрытие файла логирования
log_file.close()

# 5. Сохранение модели

torch.save(model.state_dict(), 'segmentation_model.pth')

# 6. Использование модели

# Пример использования модели
model.load_state_dict(torch.load('segmentation_model.pth'))
model.eval()

# Пример нового изображения
new_image = torch.tensor(np.array(Image.open('path/to/new_image.tif')), dtype=torch.float32).permute(2, 0, 1) / 255.0

with torch.no_grad():
    output = model(new_image.unsqueeze(0))  # Добавление размерности батча
    predicted_mask = (output > 0.5).float()

# Сохранение предсказанной маски
predicted_mask_np = predicted_mask.squeeze().cpu().numpy()
plt.imsave('predicted_mask.png', predicted_mask_np, cmap='gray')

print("Training complete. Model saved and predicted mask saved as 'predicted_mask.png'.")