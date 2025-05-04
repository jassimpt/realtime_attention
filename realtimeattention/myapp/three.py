import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load the trained model (with custom hard_swish activation)
def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6


# Load model with custom activation
model = load_model("mobilenetv3.keras", custom_objects={"hard_swish": hard_swish}, compile=False)

# Define class labels (update with actual class names)
class_labels = ["surprised", "sad", "neutral", "happy", "fearful", "disgusted", "angry"]


# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image: {image_path}")
        return None

    img = cv2.resize(img, (32, 32))  # Resize to match model input size
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Function to predict an image class
def predict_image(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return None, None  # Return None if the image couldn't be processed

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    print(f"Predicted Class: {class_labels[predicted_class]} (Confidence: {confidence:.2f})")
    return class_labels[predicted_class], confidence  # Return predicted class and confidence score



# Run the script only when executed directly
#if __name__ == "__main__":
    #folder_path = r"C:\Users\Hp\Downloads\archivenew\train\happy"

    #if os.path.exists(folder_path):  # Check if the folder exists
        #file_list = os.listdir(folder_path)  # List all files in the folder

        #for filename in file_list:
            #file_path = os.path.join(folder_path, filename)  # Construct full file path

            #predicted_class, confidence = predict_image(file_path)  # Get prediction

            #if predicted_class:
                #print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")

                 #Move images to another directory if the predicted class is "happy"
                #if predicted_class == "happy":
                    #happy_folder = r"C:\Users\Hp\Desktop\PROJECT\og_dataset\train\happy"  # Correct folder name
                    #if not os.path.exists(happy_folder):  # ✅ Fixed Syntax
                        #os.makedirs(happy_folder)  # Create folder if it doesn't exist

                    #output_path = os.path.join(happy_folder, filename)
                    #img = cv2.imread(file_path)
                    #if img is not None:
                        #cv2.imwrite(output_path, img)
    #else:
        #print("Error: Folder does not exist.")




# Run the script only when executed directly
#if __name__ == "__main__":
    #folder_path = r"C:\Users\Hp\Downloads\archivenew\train\surprised"

    #if os.path.exists(folder_path):  # Check if the folder exists
        #file_list = os.listdir(folder_path)  # List all files in the folder

        #for filename in file_list:
           # file_path = os.path.join(folder_path, filename)  # Construct full file path

            #predicted_class, confidence = predict_image(file_path)  # Get prediction

            #if predicted_class:
                #print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")

                # Move images to another directory if the predicted class is "happy"
                #if predicted_class == "surprised":
                   # surprised_folder = r"C:\Users\Hp\Desktop\PROJECT\og_dataset\train\surprised"  # Correct folder name
                    #if not os.path.exists(surprised_folder):  # ✅ Fixed Syntax
                        #os.makedirs(surprised_folder)  # Create folder if it doesn't exist

                    #output_path = os.path.join(surprised_folder, filename)
                    #img = cv2.imread(file_path)
                   # if img is not None:
                        #cv2.imwrite(output_path, img)
    #else:
        #print("Error: Folder does not exist.")







# Run the script only when executed directly
#if __name__ == "__main__":
    #folder_path = r"C:\Users\Hp\Downloads\archivenew\train\neutral"

    #if os.path.exists(folder_path):  # Check if the folder exists
        #file_list = os.listdir(folder_path)  # List all files in the folder

        #for filename in file_list:
            #file_path = os.path.join(folder_path, filename)  # Construct full file path

            #predicted_class, confidence = predict_image(file_path)  # Get prediction

            #if predicted_class:
                #print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")

                # Move images to another directory if the predicted class is "happy"
                #if predicted_class == "angry":
                    #angry_folder = r"C:\Users\Hp\Desktop\PROJECT\og_dataset\train\angry"  # Correct folder name
                    #if not os.path.exists(angry_folder):  # ✅ Fixed Syntax
                        #os.makedirs(angry_folder)  # Create folder if it doesn't exist

                    #output_path = os.path.join(angry_folder, filename)
                    #img = cv2.imread(file_path)
                    #if img is not None:
                        #cv2.imwrite(output_path, img)
    #else:
        #print("Error: Folder does not exist.")






# Run the script only when executed directly
#if __name__ == "__main__":
    #folder_path = r"C:\Users\Hp\Downloads\archivenew\train\neutral"

    #if os.path.exists(folder_path):  # Check if the folder exists
        #file_list = os.listdir(folder_path)  # List all files in the folder

        #for filename in file_list:
            #file_path = os.path.join(folder_path, filename)  # Construct full file path

            #predicted_class, confidence = predict_image(file_path)  # Get prediction

            #if predicted_class:
                #print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")

                # Move images to another directory if the predicted class is "happy"
                #if predicted_class == "angry":
                    #angry_folder = r"C:\Users\Hp\Desktop\PROJECT\og_dataset\train\angry"  # Correct folder name
                    #if not os.path.exists(angry_folder):  # ✅ Fixed Syntax
                        #os.makedirs(angry_folder)  # Create folder if it doesn't exist

                    #output_path = os.path.join(angry_folder, filename)
                    #img = cv2.imread(file_path)
                    #if img is not None:
                        #cv2.imwrite(output_path, img)
    #else:
        #print("Error: Folder does not exist.")







# Test prediction on an image
image_path = r"C:\Users\Hp\Downloads\dataset (1)\archivenew\train\sad\im791.png"  # Replace with your image path
predicted_class, confidence = predict_image(image_path)

if predicted_class:
    print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")
else:
    print("Prediction failed.")



