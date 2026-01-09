import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# --- КОНФІГУРАЦІЯ ---
BATCH_SIZE = 32
EPOCHS = 10 
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Шляхи
DATA_DIR = './Data/LFW_Cache'       # Кеш датасету
LOCAL_IMAGES_DIR = './Data/Faces'   # Ваші фото для тестів
MODELS_DIR = './models'             # Папка для збереження моделей

# Створюємо папку для моделей, якщо немає
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Використовується пристрій: {DEVICE}")

# --- 1. ПІДГОТОВКА ДАНИХ (LFW) ---

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Перевірка/Завантаження датасету LFW...")
# Завантажуємо датасет
lfw_dataset = datasets.LFWPeople(root=DATA_DIR, split='train', download=True, transform=train_transforms)

# Класи для тренування (включаючи Шрьодера)
target_classes = [
    "Gerhard Schroeder", 
    "Tony Blair",
    "Colin Powell",
    "George W Bush",
    "Donald Rumsfeld"
]

class_to_idx = lfw_dataset.class_to_idx
target_indices = [class_to_idx[name] for name in target_classes if name in class_to_idx]

# Мапа для виводу результатів (0->Gerhard, 1->Tony...)
model_idx_to_name = {i: name for i, name in enumerate(target_classes)}

# Фільтрація датасету
indices = []
for i, (_, label) in enumerate(lfw_dataset):
    if label in target_indices:
        indices.append(i)

# Кастомний датасет для перепризначення міток у 0..N
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

train_subset = FilteredDataset(lfw_dataset, indices, target_indices)
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Датасет готовий: {len(train_subset)} фото для {len(target_classes)} класів.")

# --- 2. ВИЗНАЧЕННЯ АРХІТЕКТУР ---

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
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- 3. ЛОГІКА "ТРЕНУВАННЯ АБО ЗАВАНТАЖЕННЯ" ---

def get_or_train_model(model, name, filename):
    save_path = os.path.join(MODELS_DIR, filename)
    model = model.to(DEVICE)
    
    # ПЕРЕВІРКА: Чи є вже збережена модель?
    if os.path.exists(save_path):
        print(f"\n[{name}] Знайдено збережену модель. Завантаження з {save_path}...")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        print(f"[{name}] Успішно завантажено.")
        return model
    
    # Якщо немає - тренуємо
    print(f"\n[{name}] Збереженої моделі немає. Починаємо тренування...")
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
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {100 * correct / total:.2f}%")
    
    # ЗБЕРЕЖЕННЯ
    print(f"[{name}] Тренування завершено. Збереження у {save_path}...")
    torch.save(model.state_dict(), save_path)
    return model

# --- ЗАПУСК ПРОЦЕСУ ---

num_classes = len(target_classes)

# 1. SimpleCNN
simple_net = SimpleCNN(num_classes)
simple_model = get_or_train_model(simple_net, "SimpleCNN", "simple_cnn_lfw.pth")

# 2. ResNet50
resnet_net = get_resnet_architecture(num_classes)
resnet_model = get_or_train_model(resnet_net, "ResNet50", "resnet50_lfw.pth")

# --- 4. ІНФЕРЕНС ---

detector = MTCNN()

def predict_local_image(model, model_name, image_path):
    if not os.path.exists(image_path):
        print(f"❌ Файл не знайдено: {image_path}")
        return None, None, None

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    results = detector.detect_faces(img_rgb)
    if not results:
        print(f"[{model_name}] Обличчя не знайдено на {image_path}")
        return None, None, None

    best_face = max(results, key=lambda x: x['confidence'])
    x, y, w, h = best_face['box']
    x, y = max(0, x), max(0, y)
    
    face_img = img_rgb[y:y+h, x:x+w]
    pil_img = Image.fromarray(face_img).resize((224, 224))
    
    input_tensor = train_transforms(pil_img).unsqueeze(0).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_catid = torch.max(probabilities, 1)
        
    predicted_label = model_idx_to_name[top_catid.item()]
    confidence = top_prob.item() * 100
    
    print(f"Модель: {model_name:<10} | Файл: {os.path.basename(image_path):<15} -> {predicted_label} ({confidence:.2f}%)")
    return pil_img, predicted_label, confidence

print("\n=== ЕТАП 1: Тестування моделей на локальних фото ===")
test_files = [f for f in os.listdir(LOCAL_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
test_files.sort()

if not test_files:
    print(f"У папці {LOCAL_IMAGES_DIR} немає зображень!")
else:
    for file_name in test_files:
        full_path = os.path.join(LOCAL_IMAGES_DIR, file_name)
        print("-" * 60)
        
        # Тестуємо SimpleCNN
        predict_local_image(simple_model, "SimpleCNN", full_path)
        
        # Тестуємо ResNet50 (з візуалізацією)
        img, pred, conf = predict_local_image(resnet_model, "ResNet50", full_path)
        
        if img:
            plt.figure(figsize=(3, 3))
            plt.imshow(img)
            # Червоний колір, якщо впевненість низька, зелений - якщо висока
            title_color = "green" if conf > 70 else "red" 
            plt.title(f"{pred}\n{conf:.1f}%", color=title_color)
            plt.axis('off')
            plt.show()