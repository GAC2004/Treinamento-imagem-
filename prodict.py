# predict.py
import os
import cv2
import numpy as np
from skimage.feature import hog
from joblib import load

def extract_hog_features(image_path, size=(128,128)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERRO: não consegui ler {image_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    fd, _ = hog(gray,
                block_norm='L2-Hys',
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True)
    return fd

# caminho absoluto para o modelo
model_path = r"C:\Users\Gabriela-Note\OneDrive\Documentos\nicoleCodigos\dataset\modelo_imagens_svm.pkl"
model = load(model_path)

# exemplos de teste — use paths absolutos ou relativos corretos
test_images = [
    r"C:\Users\Gabriela-Note\OneDrive\Documentos\nicoleCodigos\dataset\exemplo_correta.jpg",
    r"C:\Users\Gabriela-Note\OneDrive\Documentos\nicoleCodigos\dataset\exemplo_incorreta.jpg"
]

for img_path in test_images:
    print("Lendo:", img_path)
    features = extract_hog_features(img_path)
    if features is None:
        continue
    pred = model.predict([features])[0]
    classe = "Banana" if pred == 0 else "Maçã"
    print(f"{os.path.basename(img_path)} → {classe}")
