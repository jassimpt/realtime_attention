import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load the trained model (with custom hard_swish activation)
def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6


# Load model with custom activation
model = load_model("mobilenetv3_grayscale.keras", custom_objects={"hard_swish": hard_swish}, compile=False)

# Define class labels (update with actual class names)
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        print(f"Error: Unable to load image: {image_path}")
        return None

    detected_face = img  # transform to gray scale
    # detected_face = cv2.resize(detected_face, (32, 32)) #resize to 48x48
    detected_face = cv2.resize(detected_face, (32, 32), interpolation=cv2.INTER_CUBIC)
    #                     data_list.append(res)

    img_pixels = np.asarray(detected_face, dtype=np.float32)
    img_pixels = img_pixels.reshape(-1, 32, 32, 1)
    # img_pixels = np.expand_dims(img_pixels, axis = 0)

    img_pixels /= 255 # Add batch dimension
    return img_pixels


# Function to predict an image class
def predict_image(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return None, None  # Return None if the image couldn't be processed

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    print(predicted_class,"predicted_class")
    confidence = np.max(prediction)

    print(f"Predicted Class: {class_labels[predicted_class]} (Confidence: {confidence:.2f})")
    return class_labels[predicted_class], confidence  # Return predicted class and confidence score


# # Run the script only when executed directly
# if __name__ == "__main__":
#     folder_path = r"C:\Users\Hp\Downloads\Emotion\train\disgusted"
#
#     if os.path.exists(folder_path):  # Check if the folder exists
#         file_list = os.listdir(folder_path)  # List all files in the folder
#
#         for filename in file_list:
#             file_path = os.path.join(folder_path, filename)  # Construct full file path
#
#             predicted_class, confidence = predict_image(file_path)  # Get prediction
#
#             if predicted_class:
#                 print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")
#
#                 # Move images to another directory if the predicted class is "happy"
#                 if predicted_class == "fearful":
#                     fearful_folder = r"C:\Users\Hp\Downloads\Emotion\train\fearful"  # Correct folder name
#                     if not os.path.exists(fearful_folder):  # ✅ Fixed Syntax
#                         os.makedirs(fearful_folder)  # Create folder if it doesn't exist
#
#                     output_path = os.path.join(fearful_folder, filename)
#                     img = cv2.imread(file_path)
#                     if img is not None:
#                         cv2.imwrite(output_path, img)
#     else:
#         print("Error: Folder does not exist.")




#Test prediction on an image
image_path =r"C:\Users\Hp\Downloads\FER 2013\train\surprise\Training_86673790.jpg"  # Replace with your image path
predicted_class, confidence = predict_image(image_path)

if predicted_class:
    print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")
else:
   print("Prediction failed.")
