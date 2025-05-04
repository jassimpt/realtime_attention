import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from xgboost import XGBClassifier

# Variables
num_classes = 7
image_size = (64, 64)
dataset_path = r'C:\Users\Hp\Downloads\archivenew\train'

# Load dataset
def read_dataset():
    data_list = []
    label_list = []

    if not os.path.exists(dataset_path):
        print("Error: Dataset path does not exist!")
        return np.array([]), np.array([])

    class_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

    for label, class_folder in enumerate(class_folders):
        print(f"Processing folder: {class_folder}")
        class_path = os.path.join(dataset_path, class_folder)
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, image_size, interpolation=cv2.INTER_CUBIC)
            data_list.append(img_resized.flatten())
            label_list.append(label)

    return np.asarray(data_list, dtype=np.float32), np.asarray(label_list, dtype=np.int32)

# Load dataset
X, y = read_dataset()
if X.size == 0 or y.size == 0:
    raise ValueError("No valid data found. Check dataset path.")

# Normalize data
X /= 255.0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Train XGBoost model
xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric="mlogloss")
xgb_model.fit(X_train, y_train)

# Predict and print accuracy
y_pred = xgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(xgb_model, "xgboost_model.pkl")
print("Model saved as xgboost_model.pkl")
