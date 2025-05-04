import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # To save the model

# ------------------------------
# Variables
image_size = (64, 64)
dataset_path = r'C:\Users\Hp\Downloads\archivenew\train'

# Function to extract HOG features
def extract_hog_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, visualize=True)
    return features

# Function to read dataset
def read_dataset():
    data_list = []
    label_list = []
    class_folders = os.listdir(dataset_path)

    for label, class_folder in enumerate(class_folders):
        print(f"Processing {class_folder}...")
        class_path = os.path.join(dataset_path, class_folder)
        for root, _, files in os.walk(class_path):
            for file in files:
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Unable to read {file_path}")
                    continue
                img_resized = cv2.resize(img, image_size)
                hog_features = extract_hog_features(img_resized)
                data_list.append(hog_features)
                label_list.append(label)

    return np.asarray(data_list, dtype=np.float32), np.asarray(label_list, dtype=np.int32)

# Load dataset
X, y = read_dataset()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(svm_model, "svm_model.pkl")

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))
