import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D,
                                     GlobalAveragePooling2D, Dense, Multiply, Activation)
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import os, cv2, pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.preprocessing.image import array_to_img

# Swish activation function
def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6

# Squeeze-and-Excitation (SE) block
def se_block(input_tensor, reduction=4):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // reduction, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    return Multiply()([input_tensor, se])

# Depthwise separable convolution block with SE and Hard-Swish
def mobilenetv3_block(inputs, filters, kernel_size, strides, use_se, activation, block_id):
    x = DepthwiseConv2D(kernel_size, strides=strides, padding='same', name=f'dwconv_{block_id}')(inputs)
    x = BatchNormalization(name=f'dw_bn_{block_id}')(x)
    x = Activation(activation, name=f'dw_activation_{block_id}')(x)
    if use_se:
        x = se_block(x)
    x = Conv2D(filters, (1, 1), padding='same', name=f'pwconv_{block_id}')(x)
    x = BatchNormalization(name=f'pw_bn_{block_id}')(x)
    return Activation(activation, name=f'pw_activation_{block_id}')(x)


# ✅ Data Augmentation & Normalization
data_augmentation = tf.keras.Sequential([

     RandomFlip("horizontal_and_vertical"),
     #RandomRotation(0.2),
     RandomZoom(0.2),
     RandomContrast(0.1),
     RandomTranslation(0.1, 0.1),
     Rescaling(1.0 / 255.0)

])

# Build MobileNetV3 model
def build_mobilenetv3(input_shape=(32, 32, 1), num_classes=7):
    input_layer = Input(shape=input_shape)
    x = data_augmentation(input_layer)
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv1')(input_layer)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(hard_swish, name='conv1_hswish')(x)
    x = mobilenetv3_block(x, 16, (3, 3), strides=1, use_se=False, activation='relu', block_id=1)
    x = mobilenetv3_block(x, 24, (3, 3), strides=2, use_se=False, activation='relu', block_id=2)
    x = mobilenetv3_block(x, 40, (3, 3), strides=2, use_se=True, activation=hard_swish, block_id=3)
    x = mobilenetv3_block(x, 80, (3, 3), strides=2, use_se=False, activation=hard_swish, block_id=4)
    x = mobilenetv3_block(x, 112, (3, 3), strides=1, use_se=True, activation=hard_swish, block_id=5)
    x = mobilenetv3_block(x, 160, (3, 3), strides=2, use_se=True, activation=hard_swish, block_id=6)
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)


# Load dataset
def read_dataset(path):
    data_list, label_list = [], []
    i = -1
    for pa in os.listdir(path):
        i += 1
        for root, _, files in os.walk(os.path.join(path, pa)):
            for f in files:
                try:
                    file_path = os.path.join(root, f)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
                    res = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
                    data_list.append(res)
                    label_list.append(i)
                except:
                    pass
    return np.asarray(data_list, dtype=np.float32), np.asarray(label_list)






# Load and preprocess dataset
dataset_path = r"C:\Users\Hp\Downloads\FER 2013\train"
x_dataset, y_dataset = read_dataset(dataset_path)
x_dataset = x_dataset.reshape(-1, 32, 32, 1)  # Reshape for grayscale input
import matplotlib.pyplot as plt

# Visualize original vs augmented image
original = x_dataset[0].reshape(32, 32)
augmented_tensor = data_augmentation(np.expand_dims(x_dataset[0], axis=0))[0].numpy()
augmented = augmented_tensor.reshape(32, 32)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(original, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(augmented, cmap='gray')
plt.title("Augmented")
plt.axis('off')

plt.suptitle("Data Augmentation Check (Flip/Rotate/etc)")
plt.tight_layout()
plt.show()


import pathlib
from tensorflow.keras.preprocessing.image import array_to_img

# Directory to save augmented images
save_augmented_path = r"C:\Users\Hp\Downloads\FER 2013\augmented-saved"
os.makedirs(save_augmented_path, exist_ok=True)

# Sample a few images from your dataset to augment and save
num_to_generate = 10  # Choose how many you want to save
counter = 0

for i in range(len(x_dataset)):
    if counter >= num_to_generate:
        break
    img = x_dataset[i]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    for augmented_img in data_augmentation(img):
        augmented_img = array_to_img(augmented_img)
        filename = f"aug_{i}_{counter}.png"
        augmented_img.save(os.path.join(save_augmented_path, filename))
        counter += 1
        break  # Save only one augmented version per image



# Print dataset info
print(f"Number of original samples: {len(x_dataset)}")
augmented_samples = len(x_dataset) * 5
print(f"Number of augmented samples: {augmented_samples}")
print(f"Total training samples after augmentation: {len(x_dataset) + augmented_samples}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.1, random_state=0)
X_train = x_dataset
y_train = y_dataset

y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)

# Normalize data
x_train, x_test = X_train / 255.0, X_test / 255.0

# Build & compile model
model = build_mobilenetv3()
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Save model and history
model.save("mobilenetv3_grayscale.keras")
with open('mbv3_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Predict & evaluate
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(7), yticklabels=range(7))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_true, y_pred_classes))

# Plot accuracy
history_dict = history.history
accuracy_key = 'accuracy' if 'accuracy' in history_dict else 'acc'
val_accuracy_key = 'val_accuracy' if 'val_accuracy' in history_dict else 'val_acc'

plt.plot(history_dict[accuracy_key], label='Training Accuracy')
plt.plot(history_dict[val_accuracy_key], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
