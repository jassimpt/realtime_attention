import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Add,GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



import os, cv2, keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# manipulate with numpy,load with panda
import numpy as np
# import pandas as pd

# data visualization
import cv2
# import seaborn as sns



import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint




# Data Import
def read_dataset(path):
    data_list = []
    label_list = []
    i=-1
    my_list = os.listdir(r"C:\Users\Hp\Downloads\dataset (1)\archivenew\train")
    for pa in my_list:
        i=i+1
        print(pa,"==================")
        for root, dirs, files in os.walk(r"C:\Users\Hp\Downloads\dataset (1)\archivenew\train\\" + pa):

         for f in files:
             try:
                file_path = os.path.join(r"C:\Users\Hp\Downloads\dataset (1)\archivenew\train\\"+pa, f)
                img = cv2.imread(file_path)
                res = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
                data_list.append(res)
                # label = dirPath.split('/')[-1]
                label = i
                label_list.append(label)
             except:
                 pass
            # label_list.remove("./training")
    return (np.asarray(data_list, dtype=np.float32), np.asarray(label_list))

def read_dataset1(path):
    data_list = []
    label_list = []

    file_path = os.path.join(path)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    data_list.append(res)
    # label = dirPath.split('/')[-1]

            # label_list.remove("./training")
    return (np.asarray(data_list, dtype=np.float32))

# Import train_test_split
from sklearn.model_selection import train_test_split
# load dataset
x_dataset, y_dataset = read_dataset(r"C:\Users\Hp\Downloads\dataset (1)\archivenew\train")
X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)
num_classes=7
y_train1=[]
for i in y_train:
    emotion = keras.utils.to_categorical(i, num_classes)
    print(i,emotion)
    y_train1.append(emotion)

y_train=y_train1
x_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255  # normalize inputs between [0, 1]
x_test /= 255
print("x_train.shape",x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Load CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0


# Load CIFAR-10 dataset
# (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0

# Convert labels to one-hot encoding
train_labels = y_train
test_labels = to_categorical(y_test, num_classes=7)


# Depthwise separable convolution block
def _depthwise_separable_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

    return x


# Build MobileNetV2 model
def build_mobilenetv2(input_shape=(32, 32, 3), num_classes=7):
    input_layer = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='conv1')(input_layer)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(6., name='conv1_relu')(x)

    x = _depthwise_separable_conv_block(x, 64, depth_multiplier=1, strides=(1, 1), block_id=1)
    x = _depthwise_separable_conv_block(x, 128, depth_multiplier=1, strides=(2, 2), block_id=2)
    x = _depthwise_separable_conv_block(x, 128, depth_multiplier=1, strides=(1, 1), block_id=3)
    x = _depthwise_separable_conv_block(x, 256, depth_multiplier=1, strides=(2, 2), block_id=4)
    x = _depthwise_separable_conv_block(x, 256, depth_multiplier=1, strides=(1, 1), block_id=5)
    x = _depthwise_separable_conv_block(x, 512, depth_multiplier=1, strides=(2, 2), block_id=6)

    for _ in range(5):
        x = _depthwise_separable_conv_block(x, 512, depth_multiplier=1, strides=(1, 1), block_id=7 + _)

    x = _depthwise_separable_conv_block(x, 1024, depth_multiplier=1, strides=(2, 2), block_id=12)
    x = _depthwise_separable_conv_block(x, 1024, depth_multiplier=1, strides=(1, 1), block_id=13)

    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


# Create model
model = build_mobilenetv2()

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
# checkpoint = ModelCheckpoint('mobilenetv2_cifar10.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)



import os
import pickle

if not os.path.exists("mobilenet145.h5"):
    # Train the model
    # history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[checkpoint_callback])
    # train/validation result

    # Train model
    history = model.fit(x_train, train_labels, epochs=70, batch_size=128, validation_data=(x_test, test_labels))
    print(history.history.keys())
    print(history.history)
    model.save("mobilenet.h5")
    with open('mb_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)


    from matplotlib import pyplot as plt

    print(history.history.keys())
    history = history.history
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
else:
    import pickle
    import matplotlib.pyplot as plt

    # Read the history file
    with open('mb_history.pkl', 'rb') as file:
        history = pickle.load(file)
    print(history)
    # Plot training and validation loss
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_acc'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()