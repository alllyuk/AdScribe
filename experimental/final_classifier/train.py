import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ImageDataset(Dataset):
    def __init__(self, df, transform=None, cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cache_dir = cache_dir
        self.label_to_idx = {'Один предмет': 0, 'Разнообразие предметов': 1, 'Непонятно': 2}
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def __len__(self):
        return len(self.df)
    
    def load_image_from_url(self, image_id):
        """Загружает изображение по ID (предполагается, что это URL или путь к изображению)"""
        try:
            # Если это кэшированное изображение
            if self.cache_dir:
                cache_path = os.path.join(self.cache_dir, f"{image_id}.jpg")
                if os.path.exists(cache_path):
                    return Image.open(cache_path).convert('RGB')
            
            # Формируем URL (нужно будет адаптировать под реальную систему)
            # Пример: если image_id - это числовой ID
            if str(image_id).isdigit():
                # Здесь должна быть реальная логика получения URL по ID
                # Пока используем заглушку
                image = self.create_dummy_image()
            else:
                # Если это уже URL
                response = requests.get(str(image_id), timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Кэшируем изображение
            if self.cache_dir:
                image.save(cache_path, 'JPEG')
            
            return image
            
        except Exception as e:
            print(f"Ошибка загрузки изображения {image_id}: {e}")
            return self.create_dummy_image()
    
    def create_dummy_image(self):
        """Создает заглушку изображения для тестирования"""
        # Создаем RGB изображение 224x224 со случайными пикселями
        return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = row['label']
        
        # Загружаем изображение
        image = self.load_image_from_url(image_id)
        
        # Применяем трансформации
        if self.transform:
            image = self.transform(image)
        
        # Преобразуем метку в индекс
        label_idx = self.label_to_idx[label]
        
        return image, torch.tensor(label_idx, dtype=torch.long)

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ImageClassifier, self).__init__()
        
        # Используем предобученную ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Заменяем последний слой
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.backbone(x)
        return x

class Trainer:
    def __init__(self, model, device, checkpoint_dir):
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # История обучения
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Обучение
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Валидация
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader, criterion)
            
            # Обновление планировщика
            scheduler.step()
            
            # Сохранение истории
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Сохранение лучшей модели
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, val_preds, val_targets)
                print(f"Новая лучшая модель сохранена! Accuracy: {val_acc:.2f}%")
    
    def save_checkpoint(self, epoch, accuracy, predictions, targets):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
        
        # Сохраняем также детальную информацию
        np.save(self.checkpoint_dir / 'predictions.npy', predictions)
        np.save(self.checkpoint_dir / 'targets.npy', targets)
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График потерь
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # График точности
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def evaluate_model(predictions, targets, class_names):
    """Подробная оценка модели"""
    print("=== ДЕТАЛЬНАЯ ОЦЕНКА МОДЕЛИ ===")
    print(f"Общая точность: {accuracy_score(targets, predictions):.4f}")
    print("\nОтчет по классификации:")
    print(classification_report(targets, predictions, target_names=class_names))
    
    # Матрица ошибок
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('/workspace/AAA_project/experimental/final_classifier/confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Параметры
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    
    # Пути
    labeled_csv = "/workspace/AAA_project/experimental/final_classifier/labeled_dataset.csv"
    checkpoint_dir = "/workspace/AAA_project/experimental/final_classifier/model_checkpoint"
    cache_dir = "/workspace/AAA_project/experimental/final_classifier/image_cache"
    
    # Создаем директории
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Загружаем размеченный датасет...")
    df = pd.read_csv(labeled_csv)
    
    print(f"Размер датасета: {len(df)}")
    print("Распределение классов:")
    print(df['label'].value_counts())
    
    # Разделение данных
    train_df, temp_df = train_test_split(df, test_size=TEST_SIZE + VAL_SIZE, 
                                        stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=TEST_SIZE/(TEST_SIZE + VAL_SIZE),
                                      stratify=temp_df['label'], random_state=42)
    
    print(f"\nРазделение данных:")
    print(f"Обучение: {len(train_df)}")
    print(f"Валидация: {len(val_df)}")
    print(f"Тест: {len(test_df)}")
    
    # Трансформации
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Создание датасетов
    train_dataset = ImageDataset(train_df, transform=train_transform, cache_dir=cache_dir)
    val_dataset = ImageDataset(val_df, transform=val_transform, cache_dir=cache_dir)
    test_dataset = ImageDataset(test_df, transform=val_transform, cache_dir=cache_dir)
    
    # Создание загрузчиков данных
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуется устройство: {device}")
    
    # Создание модели
    model = ImageClassifier(num_classes=3, pretrained=True)
    model = model.to(device)
    
    # Создание тренера
    trainer = Trainer(model, device, checkpoint_dir)
    
    # Обучение
    print("\nНачинаем обучение...")
    trainer.train(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)
    
    # Построение графиков
    trainer.plot_training_history()
    
    # Загрузка лучшей модели для тестирования
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Тестирование
    print("\nТестирование модели...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_targets = trainer.validate(test_loader, criterion)
    
    print(f"Точность на тестовой выборке: {test_acc:.2f}%")
    
    # Детальная оценка
    class_names = ['Один предмет', 'Разнообразие предметов', 'Непонятно']
    evaluate_model(test_preds, test_targets, class_names)
    
    print(f"\nМодель и результаты сохранены в: {checkpoint_dir}")

if __name__ == "__main__":
    main()