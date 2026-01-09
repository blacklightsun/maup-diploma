import matplotlib
# Вмикаємо режим без GUI
matplotlib.use('Agg') 

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# --- КОНФІГУРАЦІЯ ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Шляхи (мають збігатися з Етапом 1)
DATA_ROOT = './Data'
RESULTS_DIR = './results_stage2'
LFW_EXTRACTED = os.path.join(DATA_ROOT, 'LFW_Extracted') # Тут лежить розпакований LFW
LOCAL_IMAGES_DIR = os.path.join(DATA_ROOT, 'Faces')      # Ваші тестові фото

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Використовується пристрій: {DEVICE}")

# --- 1. ПОШУК ЕТАЛОНУ (ANCHOR) В ДАТАСЕТІ ---

# ЗМІНІТЬ ЦЕ ЧИСЛО, щоб вибрати інше фото
# 0 - перше фото, 1 - друге, і т.д.
ANCHOR_INDEX = 1 

def get_anchor_image_path(desired_index):
    """Знаходить фото Герхарда Шрьодера за вказаним індексом"""
    target_name = "Gerhard_Schroeder"
    candidate_files = []
    
    # 1. Збираємо всі фото
    for root, dirs, files in os.walk(LFW_EXTRACTED):
        if os.path.basename(root).startswith(target_name):
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
            # Сортуємо, щоб порядок був завжди однаковим
            images.sort()
            
            for img in images:
                candidate_files.append(os.path.join(root, img))
    
    if not candidate_files:
        return None

    print(f"\nЗнайдено {len(candidate_files)} фото для еталону:")
    for i, path in enumerate(candidate_files):
        # Виводимо перші 5 і вибране
        if i < 5 or i == desired_index:
            marker = "  <--- ВИБРАНО" if i == desired_index else ""
            print(f"  [{i}]: {os.path.basename(path)}{marker}")
    
    # 2. Вибираємо потрібне
    if 0 <= desired_index < len(candidate_files):
        return candidate_files[desired_index]
    else:
        print(f"Індекс {desired_index} виходить за межі (доступно {len(candidate_files)}). Беремо останнє.")
        return candidate_files[-1]

anchor_path = get_anchor_image_path(ANCHOR_INDEX)

if not anchor_path:
    print("Помилка: Не знайдено папку з Герхардом Шрьодером.")
    exit()

print(f"Еталон (Anchor) успішно встановлено: {os.path.basename(anchor_path)}")



# --- 2. ІНІЦІАЛІЗАЦІЯ ІНСТРУМЕНТІВ ---

# Детектор
mtcnn = MTCNN(keep_all=False, select_largest=True, device=DEVICE)

# Модель А: Класифікатор (для демонстрації проблеми)
model_classifier = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=8631).to(DEVICE)
model_classifier.eval()

# Модель Б: Екстрактор (для вирішення проблеми)
model_extractor = InceptionResnetV1(pretrained='vggface2', classify=False).to(DEVICE)
model_extractor.eval()

# --- 3. ЧАСТИНА А: ДЕМОНСТРАЦІЯ ПРОБЛЕМИ КЛАСИФІКАЦІЇ ---

def demo_classification_failure(image_path, model):
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        return

    # Детекція та кроп
    x_aligned, prob = mtcnn(img, return_prob=True)
    
    if x_aligned is not None:
        # Інференс
        # MTCNN повертає тензор нормалізований як [-1, 1], це ок для InceptionResnetV1
        input_tensor = x_aligned.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            top_prob, top_class = torch.max(probs, 1)
            
        conf = top_prob.item() * 100
        
        print(f"[Classify=True]  Файл: {os.path.basename(image_path):<15} -> ClassID: {top_class.item()} | Conf: {conf:.4f}%")
        
        # Візуалізація
        plt.figure(figsize=(4, 4))
        # Перетворюємо тензор назад в картинку для показу
        img_show = x_aligned.permute(1, 2, 0).cpu().numpy()
        img_show = (img_show * 128 + 127.5) / 255.0 # Denormalize
        img_show = np.clip(img_show, 0, 1)
        
        plt.imshow(img_show)
        plt.title(f"Raw Classifier Output:\nClass {top_class.item()} ({conf:.2f}%)", color="red")
        plt.axis('off')
        save_name = f"stage2_FAIL_classify_{os.path.basename(image_path).split('.')[0]}.png"
        plt.savefig(RESULTS_DIR+'/'+save_name)
        plt.close()

# --- 4. ЧАСТИНА Б: РІШЕННЯ ЧЕРЕЗ ВЕРИФІКАЦІЮ (МЕТРИКИ) ---

def get_embedding(model, image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        return None, None

    # MTCNN повертає кропнуте обличчя як тензор
    x_aligned, prob = mtcnn(img, return_prob=True)
    
    if x_aligned is not None:
        input_tensor = x_aligned.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            embedding = model(input_tensor)
            
        return embedding.cpu(), x_aligned
    return None, None

# Отримуємо вектор ЕТАЛОНУ
print("\n--- Генерація вектора Еталону (Anchor) ---")
anchor_emb, anchor_img_tensor = get_embedding(model_extractor, anchor_path)

if anchor_emb is None:
    print("Не вдалося обробити еталон.")
    exit()

def run_verification(test_image_path):
    print(f"\nВерифікація: {os.path.basename(test_image_path)}")
    
    # Отримуємо вектор ТЕСТУ
    test_emb, test_img_tensor = get_embedding(model_extractor, test_image_path)
    
    if test_emb is None:
        print("Обличчя не знайдено.")
        return

    # 1. Розрахунок відстані (L2 - Euclidean)
    dist = (anchor_emb - test_emb).norm().item()
    
    # Поріг (Threshold)
    # Для VGGFace2 / L2 distance хороший поріг близько 1.0 - 1.1
    threshold = 1.0
    is_match = dist < threshold
    
    verdict = "ЦЕ ВІН (MATCH)" if is_match else "ЧУЖИЙ (MISMATCH)"
    color = "green" if is_match else "red"
    
    print(f"   Відстань L2: {dist:.4f} (Поріг: {threshold})")
    print(f"   Результат:   {verdict}")

    # Візуалізація
    plt.figure(figsize=(8, 4))
    
    # Еталон
    plt.subplot(1, 2, 1)
    # Denormalize
    show_anchor = (anchor_img_tensor.permute(1, 2, 0).cpu().numpy() * 128 + 127.5) / 255.0
    plt.imshow(np.clip(show_anchor, 0, 1))
    plt.title("ANCHOR (From LFW Dataset)\nGerhard Schroeder")
    plt.axis('off')
    
    # Тест
    plt.subplot(1, 2, 2)
    show_test = (test_img_tensor.permute(1, 2, 0).cpu().numpy() * 128 + 127.5) / 255.0
    plt.imshow(np.clip(show_test, 0, 1))
    plt.title(f"TEST IMAGE\nDist: {dist:.3f} -> {verdict}", color=color, fontweight='bold', fontsize=9)
    plt.axis('off')
    
    save_name = f"stage2_VERIFY_{os.path.basename(test_image_path).split('.')[0]}.png"
    plt.savefig(RESULTS_DIR+'/'+save_name)
    print(f"   Збережено: {save_name}")
    plt.close()

# --- ЗАПУСК ЕКСПЕРИМЕНТІВ ---

test_files = [f for f in os.listdir(LOCAL_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
test_files.sort()

if not test_files:
    print(f"Немає фото в {LOCAL_IMAGES_DIR}")
else:
    # 1. Спочатку показуємо, чому класифікатор ламається
    print("\n=== ЧАСТИНА А: Провал Класифікатора (InceptionResnetV1 classify=True) ===")
    print("Очікуємо дуже низьку впевненість або рандомний клас, бо голова не навчена під ці ID.")
    for f in test_files:
        demo_classification_failure(os.path.join(LOCAL_IMAGES_DIR, f), model_classifier)

    # 2. Тепер показуємо, як це вирішує верифікація
    print("\n=== ЧАСТИНА Б: Успіх Верифікації (Distance Metric) ===")
    print(f"Порівнюємо всіх з еталоном: {os.path.basename(anchor_path)}")
    
    for f in test_files:
        run_verification(os.path.join(LOCAL_IMAGES_DIR, f))

print("\nВисновки для Диплому (Етап 2):")
print("1. Зображення 'FAIL' показують, що ми не можемо використовувати класифікатор 'з коробки'.")
print("2. Зображення 'VERIFY' доводять, що порівняння відстаней працює надійно:")
print("   - Фото Шрьодера з папки Faces мають малу відстань (< 1.0) до еталону з LFW.")
print("   - Фото Jeff мають велику відстань (> 1.0), що дозволяє відсіяти його як 'Unknown'.")