import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

def extract_hog_features(image_path, size=(128,128)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)  # força tamanho fixo
    fd, _ = hog(gray,
                block_norm='L2-Hys',
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True)
    return fd

def load_images_from_folder(folder, label, data, labels):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        features = extract_hog_features(path)
        if features is None:
            print(f"Ignorando: {filename}")
            continue
        data.append(features)
        labels.append(label)

train_dir = r"C:\Users\Gabriela-Note\OneDrive\Documentos\nicoleCodigos\dataset"
banana_dir = os.path.join(train_dir, "banana")
maca_dir   = os.path.join(train_dir, "maca")

data, labels = [], []
load_images_from_folder(banana_dir, 0, data, labels)
load_images_from_folder(maca_dir,   1, data, labels)

# converte usando stack (mesmo comprimento garantido)
X = np.stack(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = svm.SVC(kernel='linear', C=1, max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

dump(model, os.path.join(train_dir, 'modelo_imagens_svm.pkl'))
