import cv2
import torch
import os
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import shutil

# --- КОНФІГУРАЦІЯ ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIR = './Data/Images'
RESULTS_ROOT_DIR = './results_stage0'

# Розмір, до якого будемо приводити вирізані обличчя (стандарт для Facenet)
FACE_SIZE = (160, 160) 

# Очищення результатів попереднього запуску
if os.path.exists(RESULTS_ROOT_DIR):
    shutil.rmtree(RESULTS_ROOT_DIR)
os.makedirs(RESULTS_ROOT_DIR)

print(f"Використовується пристрій для MTCNN: {DEVICE}")

# --- ІНІЦІАЛІЗАЦІЯ МОДЕЛЕЙ ---

# 1. Viola-Jones (OpenCV)
# Завантажуємо стандартний XML класифікатор для фронтальних облич
vj_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
vj_detector = cv2.CascadeClassifier(vj_cascade_path)

if vj_detector.empty():
    print("Помилка: Не вдалося завантажити XML для Viola-Jones.")
    exit()

# 2. MTCNN (PyTorch)
# keep_all=True обов'язково для групових фото!
mtcnn = MTCNN(keep_all=True, device=DEVICE)

# --- ДОПОМІЖНІ ФУНКЦІЇ ---

def save_detection_results(method_name, img_name, original_img_pil, boxes):
    """
    Зберігає результати:
    1. Загальне фото з рамками.
    2. Окремі кропнуті обличчя.
    """
    # Шлях: results_stage0 / Method_Name / Image_Name /
    base_path = os.path.join(RESULTS_ROOT_DIR, method_name, img_name.split('.')[0])
    faces_path = os.path.join(base_path, 'faces')
    
    os.makedirs(faces_path, exist_ok=True)
    
    # Конвертуємо в OpenCV формат для малювання рамок
    img_cv2 = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR)
    
    count = 0
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # Координати можуть бути float, приводимо до int
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Захист від виходу за межі (важливо для країв фото)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_img_pil.width, x2)
            y2 = min(original_img_pil.height, y2)
            
            # Якщо рамка некоректна (ширина або висота 0)
            if x2 <= x1 or y2 <= y1:
                continue

            # 1. Малюємо рамку (Зелена, товщина 2)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 2. Вирізаємо обличчя
            face_crop = original_img_pil.crop((x1, y1, x2, y2))
            # Ресайз до стандарту (наприклад, 160x160 для нейромереж)
            face_crop = face_crop.resize(FACE_SIZE, Image.Resampling.BILINEAR)
            
            # Зберігаємо обличчя
            face_filename = f"face_{count}.jpg"
            face_crop.save(os.path.join(faces_path, face_filename))
            count += 1
            
    # Зберігаємо велике фото з рамками
    cv2.imwrite(os.path.join(base_path, "detected_overview.jpg"), img_cv2)
    
    return count

# --- ГОЛОВНИЙ ЦИКЛ ОБРОБКИ ---

image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print(f"Папка {INPUT_DIR} порожня! Додайте туди фото з групами людей.")
    exit()

print(f"Знайдено зображень: {len(image_files)}\n")

for img_file in image_files:
    print(f"Обробка: {img_file}")
    img_path = os.path.join(INPUT_DIR, img_file)
    
    try:
        # Завантажуємо зображення
        # Для MTCNN потрібен PIL (RGB)
        pil_img = Image.open(img_path).convert('RGB')
        # Для Viola-Jones потрібен GrayScale (OpenCV)
        cv_img_gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        
    except Exception as e:
        print(f"   Помилка читання файлу: {e}")
        continue

    # ============================
    # АЛГОРИТМ 1: VIOLA-JONES
    # ============================
    
    # Варіант 1.1: minNeighbors = 3 (Чутливий, багато помилок)
    # detectMultiScale повертає (x, y, w, h), нам треба (x1, y1, x2, y2)
    rects_vj_3 = vj_detector.detectMultiScale(cv_img_gray, scaleFactor=1.1, minNeighbors=3)
    boxes_vj_3 = []
    for (x, y, w, h) in rects_vj_3:
        boxes_vj_3.append([x, y, x+w, y+h])
        
    count_vj_3 = save_detection_results("VJ_neighbors_3", img_file, pil_img, boxes_vj_3)
    
    # Варіант 1.2: minNeighbors = 20 (Дуже суворий)
    rects_vj_20 = vj_detector.detectMultiScale(cv_img_gray, scaleFactor=1.1, minNeighbors=20)
    boxes_vj_20 = []
    for (x, y, w, h) in rects_vj_20:
        boxes_vj_20.append([x, y, x+w, y+h])
        
    count_vj_20 = save_detection_results("VJ_neighbors_20", img_file, pil_img, boxes_vj_20)

    # ============================
    # АЛГОРИТМ 2: MTCNN
    # ============================
    
    # MTCNN повертає boxes і probabilities (впевненість)
    # Ми робимо один виклик, а потім фільтруємо за порогом
    mtcnn_boxes, mtcnn_probs = mtcnn.detect(pil_img)
    
    boxes_conf_90 = []
    boxes_conf_99 = []
    
    if mtcnn_boxes is not None:
        for box, prob in zip(mtcnn_boxes, mtcnn_probs):
            # Варіант 2.1: Confidence > 0.90
            if prob > 0.90:
                boxes_conf_90.append(box)
            
            # Варіант 2.2: Confidence > 0.99
            if prob > 0.99:
                boxes_conf_99.append(box)

    count_mtcnn_90 = save_detection_results("MTCNN_conf_0.90", img_file, pil_img, boxes_conf_90)
    count_mtcnn_99 = save_detection_results("MTCNN_conf_0.99", img_file, pil_img, boxes_conf_99)

    # ============================
    # ЗВІТ ПО ЗОБРАЖЕННЮ
    # ============================
    print(f"   --- Результати ---")
    print(f"   Viola-Jones (n=3):  {count_vj_3} облич")
    print(f"   Viola-Jones (n=20): {count_vj_20} облич")
    print(f"   MTCNN (conf>0.90):  {count_mtcnn_90} облич")
    print(f"   MTCNN (conf>0.99):  {count_mtcnn_99} облич")
    print("-" * 30)

print("\nЕтап 0 завершено. Результат у папці 'results_stage0'.")