import os
import shutil
import cv2

# === 1. SET PATHS ===
dataset_root = r"C:\Users\Hp\Downloads\balanced_og_dataset\og_dataset"
output_folder = r"C:\Users\Hp\Downloads\balanced_og_dataset\og_dataset\realtime_dataset2"

# Paths for new organized dataset
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')

# Create output folders
for folder in [train_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

# === 2. Copy and resize train/test images ===
for mode in ['train', 'test']:
    mode_path = os.path.join(dataset_root, mode)

    for emotion_class in os.listdir(mode_path):
        emotion_folder = os.path.join(mode_path, emotion_class)
        if not os.path.isdir(emotion_folder):
            continue

        # Create emotion folders inside train/test
        output_emotion_folder = os.path.join(output_folder, mode, emotion_class)
        os.makedirs(output_emotion_folder, exist_ok=True)

        for img_file in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_file)
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Read and resize to 32x32
                    img = cv2.imread(img_path)
                    img_resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

                    # Save resized image
                    output_img_path = os.path.join(output_emotion_folder, img_file)
                    cv2.imwrite(output_img_path, img_resized)

                except Exception as e:
                    print(f"⚠️ Error processing {img_path}: {e}")

print("\n✅ Dataset organized successfully into 'train/' and 'test/' folders!")

# === 3. CREATE labels.txt ===
labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
labels_file = os.path.join(output_folder, 'labels.txt')

with open(labels_file, 'w') as f:
    for label in labels:
        f.write(label + '\n')

print(f"✅ Labels file created at: {labels_file}")
