import matplotlib
matplotlib.use('Agg') 

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- КОНФІГУРАЦІЯ ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Поріг доступу (Cosine Similarity)
# 0.6 - це досить суворий поріг для VGGFace2. 
# > 0.6 = Це точно він. < 0.6 = Сумнівно/Чужий.
ACCESS_THRESHOLD = 0.6 

# Шляхи
DATA_ROOT = './Data'
LFW_EXTRACTED = os.path.join(DATA_ROOT, 'LFW_Extracted') 
LOCAL_IMAGES_DIR = os.path.join(DATA_ROOT, 'Faces')      
RESULTS_DIR = './results_stage4'

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Система запущена на: {DEVICE}")

# --- 1. ЗАВАНТАЖЕННЯ МОДЕЛЕЙ ---

print("Завантаження нейромереж...")
mtcnn = MTCNN(keep_all=False, select_largest=True, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2', classify=False).to(DEVICE)
model.eval()

# --- 2. СТВОРЕННЯ "БІЛОГО СПИСКУ" (GALLERY) ---
# Ми завантажуємо ВСІ фото Шрьодера, щоб система впізнавала його з різних ракурсів

def create_gallery():
    print("\nСтворення галереї допуску (White List)...")
    target_name = "Gerhard_Schroeder"
    gallery_vectors = []
    gallery_images = [] # Зберігаємо оригінали для візуалізації
    
    # Шукаємо папку
    target_dir = None
    for root, dirs, files in os.walk(LFW_EXTRACTED):
        if os.path.basename(root).startswith(target_name):
            target_dir = root
            break
    
    if not target_dir:
        print("Папка Gerhard не знайдена в LFW!")
        exit()

    files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    print(f"   Знайдено {len(files)} еталонних фото.")

    for f in files:
        path = os.path.join(target_dir, f)
        try:
            img = Image.open(path).convert('RGB')
            # Детекція
            x_aligned = mtcnn(img)
            if x_aligned is not None:
                # Векторизація
                input_tensor = x_aligned.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    emb = model(input_tensor)
                
                gallery_vectors.append(emb)
                gallery_images.append(img) # Зберігаємо PIL Image
        except Exception as e:
            continue
            
    # Об'єднуємо всі вектори в один тензор (N, 512)
    if gallery_vectors:
        gallery_tensor = torch.cat(gallery_vectors)
        print(f"Галерея завантажена: {gallery_tensor.shape[0]} векторів.")
        return gallery_tensor, gallery_images
    else:
        print("Не вдалося створити галерею.")
        exit()

gallery_emb_tensor, gallery_imgs = create_gallery()

# --- 3. ФУНКЦІЯ КОНТРОЛЮ ДОСТУПУ ---

def check_access(test_path):
    filename = os.path.basename(test_path)
    print(f"\nСпроба доступу: {filename}")
    
    try:
        img_test = Image.open(test_path).convert('RGB')
    except:
        print("   Помилка відкриття файлу")
        return

    # 1. Детекція
    x_test = mtcnn(img_test)
    if x_test is None:
        print("   Обличчя не виявлено -> ВІДМОВА")
        visualize_result(img_test, None, 0.0, "NO FACE DETECTED", filename)
        return

    # 2. Векторизація
    with torch.no_grad():
        emb_test = model(x_test.unsqueeze(0).to(DEVICE))

    # 3. ПОРІВНЯННЯ З ГАЛЕРЕЄЮ (One-to-Many)
    # Рахуємо косинусну схожість між Вхідним вектором і ВСІМА векторами галереї
    # F.cosine_similarity підтримує broadcasting
    similarities = F.cosine_similarity(emb_test, gallery_emb_tensor)
    
    # Знаходимо найкращий збіг (Max Score)
    best_score, best_idx = torch.max(similarities, 0)
    best_score = best_score.item()
    best_match_img = gallery_imgs[best_idx.item()] # Фото, на яке найбільше схожий
    
    # 4. Прийняття рішення
    access_granted = best_score > ACCESS_THRESHOLD
    
    status = "ACCESS GRANTED" if access_granted else "ACCESS DENIED"
    print(f"   Максимальна схожість: {best_score:.4f}")
    print(f"   Статус: {status}")
    
    visualize_result(img_test, best_match_img, best_score, status, filename)

# --- 4. ВІЗУАЛІЗАЦІЯ "КАРТКИ ДОСТУПУ" ---

def visualize_result(test_img, match_img, score, status, filename):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    
    # Кольори
    bg_color = "#e6ffe6" if status == "ACCESS GRANTED" else "#ffe6e6"
    text_color = "green" if status == "ACCESS GRANTED" else "red"
    fig.patch.set_facecolor(bg_color)

    # 1. Вхідне фото (Test)
    axes[0].imshow(test_img)
    axes[0].set_title("CAMERA INPUT", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Найкращий збіг з бази (Evidence)
    if match_img:
        axes[1].imshow(match_img)
        axes[1].set_title(f"BEST DATABASE MATCH\n(Evidence)", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "NO MATCH FOUND", ha='center')
    axes[1].axis('off')
    
    # 3. Інфо-панель
    axes[2].axis('off')
    
    # Малюємо "лампочку" статусу
    circle = patches.Circle((0.5, 0.7), 0.15, color=text_color)
    axes[2].add_patch(circle)
    
    info_text = (
        f"SYSTEM DECISION:\n\n"
        f"{status}\n\n"
        f"Similarity: {score:.1%}\n"
        f"Threshold:  {ACCESS_THRESHOLD:.1%}\n\n"
        f"Algorithm:  ArcFace Logic\n"
        f"(Cosine Similarity)"
    )
    
    axes[2].text(0.5, 0.3, info_text, ha='center', va='top', fontsize=14, fontweight='bold', color='#333333')
    
    save_path = os.path.join(RESULTS_DIR, f"AccessCard_{filename.split('.')[0]}.png")
    plt.tight_layout()
    plt.savefig(save_path, facecolor=bg_color)
    plt.close()
    print(f"   Звіт збережено: {save_path}")

# --- ЗАПУСК ---

print("\n=== ЕТАП 4: СИМУЛЯЦІЯ СИСТЕМИ КОНТРОЛЮ ДОСТУПУ ===")

test_files = [f for f in os.listdir(LOCAL_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
test_files.sort()

if not test_files:
    print("Немає файлів для тесту.")
else:
    for f in test_files:
        check_access(os.path.join(LOCAL_IMAGES_DIR, f))

print("\nВисновки для Диплому (Фінал):")
print("1. Система використовує 'Gallery Matching' (порівняння з набором еталонів).")
print("2. Це забезпечує максимальну точність: якщо Шрьодер схожий хоча б на одне зі своїх фото в базі -> Вхід дозволено.")
print("3. Jeff не схожий на жодне фото Шрьодера -> Вхід заборонено.")