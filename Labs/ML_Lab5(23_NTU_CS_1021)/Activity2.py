import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset folder ka path 
data_dir = 'brain_tumor_dataset' 
IMG_SIZE = 64

def load_and_preprocess_images(base_dir):
    X = []
    y = []
    categories = ['no', 'yes']  
    for category in categories:
        path = os.path.join(base_dir, category)
        class_num = categories.index(category)
        if not os.path.exists(path):
            print(f"Warning: Folder not found -> {path}")
            continue
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                # Corrected comma here instead of dot
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is not None:
                    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    X.append(resized_array.flatten())
                    y.append(class_num)
            except Exception as e:
                pass
    return np.array(X), np.array(y)

print("Loading and preprocessing images...")
X, y = load_and_press = load_and_preprocess_images(data_dir)


if len(X) == 0:
    print("Dataset folder nahi mila, is liye dummy data generate kiya ja raha hai taake code run ho sakay...")
    X = np.random.randint(0, 255, size=(200, 64*64))
    y = np.random.randint(0, 2, size=(200,))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training SVM Model...")
svm_model = SVC(kernel='rbf', C=10, gamma='scale')
svm_model.fit(X_train_scaled, y_train)


y_pred = svm_model.predict(X_test_scaled)
print("\n--- RESULTS ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))