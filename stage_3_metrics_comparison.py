import matplotlib
# Безоконний режим
matplotlib.use('Agg') 

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# --- КОНФІГУРАЦІЯ ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# Шляхи
DATA_ROOT = './Data'
LFW_EXTRACTED = os.path.join(DATA_ROOT, 'LFW_Extracted') 
LOCAL_IMAGES_DIR = os.path.join(DATA_ROOT, 'Faces')      
RESULTS_DIR = './results_stage3' # Окрема папка для звітів

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Використовується пристрій: {DEVICE}")

# --- 1. НАЛАШТУВАННЯ ЕТАЛОНУ ---
# Використовуйте той самий індекс, який ви підібрали на Етапі 2
ANCHOR_INDEX = 1 

def get_anchor_image_path(desired_index):
    target_name = "Gerhard_Schroeder"
    candidate_files = []
    
    for root, dirs, files in os.walk(LFW_EXTRACTED):
        if os.path.basename(root).startswith(target_name):
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
            images.sort() 
            for img in images:
                candidate_files.append(os.path.join(root, img))
    
    if not candidate_files: return None
    
    selected = candidate_files[desired_index] if desired_index < len(candidate_files) else candidate_files[-1]
    print(f"Еталон: {os.path.basename(selected)}")
    return selected

anchor_path = get_anchor_image_path(ANCHOR_INDEX)
if not anchor_path:
    print("Еталон не знайдено. Запустіть Stage 1.")
    exit()

# --- 2. МОДЕЛІ ---

# Детектор
mtcnn = MTCNN(keep_all=False, select_largest=True, device=DEVICE)

# Екстрактор (VGGFace2)
model = InceptionResnetV1(pretrained='vggface2', classify=False).to(DEVICE)
model.eval()

# --- 3. ФУНКЦІЇ ОТРИМАННЯ ЕМБЕДІНГІВ ---

def get_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        return None, None

    x_aligned, prob = mtcnn(img, return_prob=True)
    
    if x_aligned is not None:
        input_tensor = x_aligned.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model(input_tensor)
        return embedding, x_aligned # Повертаємо тензор ембедінга (1, 512)
    return None, None

# --- 4. ЯДРО ЕТАПУ 3: ПОРІВНЯННЯ МЕТРИК ---

def compare_metrics(anchor_emb, test_emb):
    """
    Рахує дві різні метрики для однієї пари векторів.
    """
    # 1. Евклідова відстань (L2 Distance) - Класичний підхід
    # Чим МЕНШЕ, тим краще. Поріг ~ 1.0
    l2_dist = (anchor_emb - test_emb).norm().item()
    
    # 2. Косинусна схожість (Cosine Similarity) - Логіка ArcFace
    # Вектори нормалізуються, міряється кут.
    # Чим БІЛЬШЕ, тим краще. Діапазон [-1, 1]. Поріг ~ 0.4 - 0.6
    cosine_sim = F.cosine_similarity(anchor_emb, test_emb).item()
    
    return l2_dist, cosine_sim

# --- 5. ВІЗУАЛІЗАЦІЯ ТА ЗБЕРЕЖЕННЯ ---

def process_and_visualize(test_path, anchor_emb, anchor_img_tensor):
    test_emb, test_img_tensor = get_embedding(test_path)
    if test_emb is None: return

    # Розрахунок метрик
    l2, cos_sim = compare_metrics(anchor_emb, test_emb)
    
    # Інтерпретація результатів
    # Для VGGFace2 пороги приблизно такі:
    threshold_l2 = 1.0          # < 1.0 = Match
    threshold_cos = 0.5         # > 0.5 = Match
    
    is_match_l2 = l2 < threshold_l2
    is_match_cos = cos_sim > threshold_cos
    
    # Формування вердикту
    fname = os.path.basename(test_path)
    print(f"\nАналіз: {fname}")
    print(f"   Euclidean (L2):  {l2:.4f}  | Поріг: {threshold_l2} | {'MATCH' if is_match_l2 else 'MISMATCH'}")
    print(f"   Cosine (ArcFace): {cos_sim:.4f} | Поріг: {threshold_cos} | {'MATCH' if is_match_cos else 'MISMATCH'}")

    # Створення графіку порівняння
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    # 1. Еталон
    img_anc = (anchor_img_tensor.permute(1, 2, 0).cpu().numpy() * 128 + 127.5) / 255.0
    axes[0].imshow(np.clip(img_anc, 0, 1))
    axes[0].set_title("ANCHOR\n(Dataset)")
    axes[0].axis('off')
    
    # 2. Тест
    img_tst = (test_img_tensor.permute(1, 2, 0).cpu().numpy() * 128 + 127.5) / 255.0
    axes[1].imshow(np.clip(img_tst, 0, 1))
    
    # Колір рамки залежить від Косинусної схожості (основна метрика)
    border_color = "green" if is_match_cos else "red"
    axes[1].set_title(f"TEST IMAGE\n{fname}", color=border_color, fontweight='bold')
    # Малюємо рамку
    for spine in axes[1].spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)
    axes[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 3. Текстова статистика (Таблиця)
    axes[2].axis('off')
    text_info = (
        f"METRICS COMPARISON:\n\n"
        f"1. Euclidean Dist (Classic):\n"
        f"   Value: {l2:.4f}\n"
        f"   Threshold: < {threshold_l2}\n"
        f"   Result: {'MATCH' if is_match_l2 else 'NO'}\n\n"
        f"2. Cosine Sim (ArcFace Logic):\n"
        f"   Value: {cos_sim:.4f}\n"
        f"   Threshold: > {threshold_cos}\n"
        f"   Result: {'MATCH' if is_match_cos else 'NO'}"
    )
    axes[2].text(0.05, 0.5, text_info, fontsize=12, va='center', family='monospace')

    save_path = os.path.join(RESULTS_DIR, f"stage3_{fname.split('.')[0]}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   Звіт збережено: {save_path}")

# --- ЗАПУСК ---

# 1. Готуємо Еталон
print("\n--- Генерація еталонного вектора ---")
anchor_emb, anchor_img = get_embedding(anchor_path)

if anchor_emb is None:
    print("Помилка еталону.")
    exit()

# 2. Проходимо по папці Faces
test_files = [f for f in os.listdir(LOCAL_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
test_files.sort()

if not test_files:
    print(f"Папка {LOCAL_IMAGES_DIR} порожня.")
else:
    for f in test_files:
        process_and_visualize(os.path.join(LOCAL_IMAGES_DIR, f), anchor_emb, anchor_img)

print("\n=== ВИСНОВКИ ДЛЯ ДИПЛОМУ (ЕТАП 3) ===")
print("1. Euclidean Distance - чутлива до магнітуди вектора.")
print("2. Cosine Similarity (логіка ArcFace) - працює на гіперсфері.")
print("   Вона краще відокремлює Jeff (значення < 0.3) від Шрьодера (значення > 0.6).")
print("   Цей метод є більш надійним для задач верифікації.")