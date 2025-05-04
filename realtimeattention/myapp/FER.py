import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras import layers
from keras.preprocessing import image

# Load MobileNetV3 model for emotion recognition
emotion_model = load_model(
    r"C:\Users\Hp\Downloads\realtimeattention_today20\realtimeattention_today\realtimeattention\myapp\mobilenetv3_grayscale.keras",
    custom_objects={"hard_swish": layers.Activation("hard_swish")}
)

# Emotion labels

emotions = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\Hp\Downloads\realtimeattention_today20\realtimeattention_today\realtimeattention\myapp\haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)
# def read_dataset(path):
#     data_list, label_list = [], []
#     i = -1
#     for pa in os.listdir(path):
#         i += 1
#         for root, _, files in os.walk(os.path.join(path, pa)):
#             for f in files:
#                 try:
#                     file_path = os.path.join(root, f)
#                     img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
#                     res = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
#                     data_list.append(res)
#                     label_list.append(i)
#                 except:
#                     pass
#     return np.asarray(data_list, dtype=np.float32), np.asarray(label_list)


# Emotion labels

emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad','Surprise','Neutral')

def camclick():
    i=0
    while(True):
        ret, img = cap.read()

        # img = cv2.imread('../11.jpg')
        # cv2.imwrite(str(i)+".jpg",img)
        i=i+1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #print(faces) #locations of detected faces
        emotion=None

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image

            detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
            # detected_face = cv2.resize(detected_face, (32, 32)) #resize to 48x48
            detected_face = cv2.resize(detected_face, (32, 32), interpolation=cv2.INTER_CUBIC)
            #                     data_list.append(res)

            img_pixels = np.asarray(detected_face, dtype=np.float32)
            img_pixels= img_pixels.reshape(-1, 32, 32, 1)
            # img_pixels = np.expand_dims(img_pixels, axis = 0)

            img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            predictions = emotion_model.predict(img_pixels) #store probabilities of 7 expressions
            print("+++++++++++==============")
            print(predictions[0])

            #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            max_index = np.argmax(predictions[0])

            emotion = emotions[max_index]
            cv2.putText(img,emotion,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
            print (emotion)

            # if cv2.waitKey(1):
        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

        # kill open cv things
    cap.release()
    cv2.destroyAllWindows()
            # 	pass
        # return emotion
            #write emotion text above rectangle

camclick()