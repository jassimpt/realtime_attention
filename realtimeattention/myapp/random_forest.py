import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib  # For saving the model

# Variables
num_classes = 7
image_size = (64, 64)

# Load dataset
def read_dataset():
    data_list = []
    label_list = []
    dataset_path = r'C:\Users\Hp\Downloads\archivenew\train'
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
                img_resized = cv2.resize(img, image_size, interpolation=cv2.INTER_CUBIC)
                data_list.append(img_resized.flatten())  # Flattening for RandomForest
                label_list.append(label)

    return np.asarray(data_list, dtype=np.float32), np.asarray(label_list, dtype=np.int32)

# Load dataset
X, y = read_dataset()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the data
X_train /= 255.0
X_test /= 255.0

# Train Random Forest model with improved hyperparameters
rf_model = RandomForestClassifier(
    n_estimators=300,  # Increased number of trees
    max_depth=20,  # Increased depth
    min_samples_split=3,  # Reduced min samples split for better learning
    min_samples_leaf=2,  # Smaller leaf size
    max_features='sqrt',  # Better feature selection
    bootstrap=True,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "random_forest_model.pkl")

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)