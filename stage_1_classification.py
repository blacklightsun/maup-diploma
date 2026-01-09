import matplotlib
# Вмикаємо "безоконний" режим для збереження файлів на сервері/в контейнері
matplotlib.use('Agg') 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import zipfile

# --- КОНФІГУРАЦІЯ ---
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Шляхи
DATA_ROOT = './Data'
LFW_ZIP_PATH = os.path.join(DATA_ROOT, 'LFW', 'archive.zip') 
EXTRACT_DIR = os.path.join(DATA_ROOT, 'LFW_Extracted')        
LOCAL_IMAGES_DIR = os.path.join(DATA_ROOT, 'Faces')
MODELS_DIR = './models'
RESULTS_DIR = './results_stage1'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Використовується пристрій: {DEVICE}")

# --- 0. РОЗПАКУВАННЯ ZIP ---

def extract_lfw_zip():
    if os.listdir(EXTRACT_DIR):
        print("Папка розпакування не порожня. Пропускаємо розпакування.")
        return

    if not os.path.exists(LFW_ZIP_PATH):
        print(f"Файл архіву не знайдено: {LFW_ZIP_PATH}")
        exit()

    print(f"Розпакування {LFW_ZIP_PATH}...")
    try:
        with zipfile.ZipFile(LFW_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Розпакування завершено.")
    except Exception as e:
        print(f"Помилка розпакування: {e}")
        exit()

extract_lfw_zip()

# --- 1. ПОШУК КОРЕНЯ ДАТАСЕТУ ---

def find_dataset_root(base_dir):
    target = "Gerhard" 
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.startswith(target):
                return root 
    return base_dir

print("Пошук папки з зображеннями...")
REAL_LFW_ROOT = find_dataset_root(EXTRACT_DIR)
print(f"Корінь датасету знайдено: {REAL_LFW_ROOT}")

# --- 2. ПІДГОТОВКА ДАНИХ ---

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=REAL_LFW_ROOT, transform=train_transforms)

possible_targets = [
    ["Gerhard_Schroeder", "Tony_Blair", "Colin_Powell", "George_W_Bush", "Donald_Rumsfeld"],
    ["Gerhard Schroeder", "Tony Blair", "Colin Powell", "George W Bush", "Donald Rumsfeld"]
]

target_classes = []
class_to_idx = full_dataset.class_to_idx

for candidates in possible_targets:
    if candidates[0] in class_to_idx:
        target_classes = candidates
        break

if not target_classes:
    print("Не знайдено цільових класів!")
    exit()

print(f"Використовуються класи: {target_classes}")

target_indices = [class_to_idx[name] for name in target_classes]
model_idx_to_name = {i: name.replace('_', ' ') for i, name in enumerate(target_classes)}

indices = []
for i, label in enumerate(full_dataset.targets):
    if label in target_indices:
        indices.append(i)

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, original_target_indices):
        self.dataset = dataset
        self.indices = indices
        self.map_label = {old: new for new, old in enumerate(original_target_indices)}

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, self.map_label[label]

    def __len__(self):
        return len(self.indices)

train_subset = FilteredDataset(full_dataset, indices, target_indices)
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(f"Датасет готовий: {len(train_subset)} фото.")

# --- 3. АРХІТЕКТУРИ ---

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_resnet_architecture(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- 4. ТРЕНУВАННЯ ---

def get_or_train_model(model, name, filename):
    save_path = os.path.join(MODELS_DIR, filename)
    model = model.to(DEVICE)
    
    if os.path.exists(save_path):
        print(f"\n[{name}] Завантаження збереженої моделі...")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        return model
    
    print(f"\n[{name}] Починаємо тренування на CPU...")
    start_time = time.time()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {100 * correct / total:.1f}%")
    
    duration = time.time() - start_time
    print(f"[{name}] Завершено за {duration:.1f} сек. Збережено в {save_path}")
    torch.save(model.state_dict(), save_path)
    return model

# --- ІНІЦІАЛІЗАЦІЯ ---

num_classes = len(target_classes)

simple_net = SimpleCNN(num_classes)
simple_model = get_or_train_model(simple_net, "SimpleCNN", "simple_cnn_lfw_manual.pth")

resnet_net = get_resnet_architecture(num_classes)
resnet_model = get_or_train_model(resnet_net, "ResNet50", "resnet50_lfw_manual.pth")

# --- 5. ІНФЕРЕНС ТА ЗБЕРЕЖЕННЯ (PyTorch MTCNN) ---

detector = MTCNN(keep_all=False, select_largest=True, device=DEVICE)

def predict_local_image(model, model_name, image_path):
    if not os.path.exists(image_path):
        print(f"Файл не знайдено: {image_path}")
        return None, None, None

    try:
        pil_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Помилка відкриття: {e}")
        return None, None, None
    
    boxes, probs = detector.detect(pil_img)
    
    if boxes is None:
        print(f"[{model_name}] Обличчя не знайдено на {os.path.basename(image_path)}")
        return None, None, None

    box = boxes[0]
    x1, y1, x2, y2 = [int(b) for b in box]
    x1, y1 = max(0, x1), max(0, y1)
    
    face_img = pil_img.crop((x1, y1, x2, y2))
    face_img_resized = face_img.resize((224, 224))
    
    input_tensor = train_transforms(face_img_resized).unsqueeze(0).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_catid = torch.max(probabilities, 1)
        
    predicted_label = model_idx_to_name[top_catid.item()]
    confidence = top_prob.item() * 100
    
    print(f"Модель: {model_name:<10} | Файл: {os.path.basename(image_path):<15} -> {predicted_label} ({confidence:.2f}%)")
    # Повертаємо зображення та результати
    return face_img_resized, predicted_label, confidence

# НОВА ФУНКЦІЯ: Збереження результату у файл
def save_result_image(img, pred, conf, original_filename, model_name):
    if img is None: return
    
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    # Зелений заголовок, якщо впевненість > 70%, інакше червоний
    title_color = "green" if conf > 70 else "red"
    # Додаємо назву моделі в заголовок
    plt.title(f"{model_name}\n{pred}\n{conf:.1f}%", color=title_color, fontsize=10)
    plt.axis('off')
    
    # Формуємо ім'я файлу: result_originalName_ModelName.png
    base_name = os.path.basename(original_filename).split('.')[0]
    save_name = f"result_{base_name}_{model_name.replace(' ', '')}.png"
    
    plt.savefig(RESULTS_DIR+'/'+save_name, bbox_inches='tight')
    print(f"Збережено зображення: {save_name}")
    plt.close() # Очищення пам'яті обов'язкове

print("\n=== ЕТАП 1: Тестування моделей на локальних фото ===")
test_files = [f for f in os.listdir(LOCAL_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
test_files.sort()

if not test_files:
    print(f"У папці {LOCAL_IMAGES_DIR} немає зображень! Додайте фото (schroeder_1.jpg, abbey_1.jpg).")
else:
    for file_name in test_files:
        full_path = os.path.join(LOCAL_IMAGES_DIR, file_name)
        print("-" * 60)
        print(f"Обробка файлу: {file_name}")
        
        # 1. Тест SimpleCNN
        img_s, pred_s, conf_s = predict_local_image(simple_model, "SimpleCNN", full_path)
        save_result_image(img_s, pred_s, conf_s, file_name, "SimpleCNN")
        
        # 2. Тест ResNet50
        img_r, pred_r, conf_r = predict_local_image(resnet_model, "ResNet50", full_path)
        save_result_image(img_r, pred_r, conf_r, file_name, "ResNet50")

print("\nВисновки для Диплому (Етап 1):")
print("Перевірте збережені зображення. Ви побачите, що для Jeff обидві моделі")
print("видають хибний результат з високою впевненістю (Галюцинація).")