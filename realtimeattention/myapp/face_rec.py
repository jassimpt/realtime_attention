import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import model_from_json
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def head_pose_estimation():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    flag = 0
    count = 0
    text = ""
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while cap.isOpened():
        success, image = cap.read()

        cv2.imwrite("ro.jpg", image)
        img = cv2.imread("ro.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (xx, yy, w, h) in faces:
            cv2.rectangle(img, (xx, yy), (xx + w, yy + h), (255, 0, 0), 2)

            face_img = img[int(yy-10):int(yy + h+10), int(xx-10):int(xx + w+10)]

            face_img = cv2.cvtColor(cv2.flip(face_img, 1), cv2.COLOR_BGR2RGB)
            face_img.flags.writeable = False

            # Get the result
            results = face_mesh.process(face_img)

            face_img.flags.writeable = True
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = face_img.shape
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])

                    # The Distance Matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360
                    
                    # See where the user's head tilting
                    if y < -10:
                        flag = 1
                        count = count + 1
                        text = "Looking Left"
                    elif y > 10:
                        flag = 1
                        count = count + 1
                        text = "Looking Right"
                    elif x < -10:
                        flag = 1
                        count = count + 1
                        text = "Looking Down"
                    else:
                        flag = 0
                        count = 0
                        text = "Forward"

                    cv2.putText(img, text, (xx, yy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                text = "No Face"
                cv2.putText(face_img, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        if flag == 1:
            if count == 50:
                import time
                dt = time.strftime("%Y%m%d_%H%M%S")
                print("+_+_+_+_+_+)+)+)+)+_+_+_+_+_)()_")
                count = 0
                flag = 0
                
        cv2.imshow('Head Pose Estimation', img)
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Example usage
# if __name__ == "__main__":
    # head_pose_estimation()




def emotion_detection():
    # Load model structure from JSON
    model = model_from_json(open(r"C:\Users\Amal Dev\Desktop\realtimeattention_now\realtimeattention\myapp\facial_expression_model_structure.json", "r").read())
    # Load weights
    model.load_weights(r"C:\Users\Amal Dev\Desktop\realtimeattention_now\realtimeattention\myapp\facial_expression_model_weights.h5")

    face_cascade = cv2.CascadeClassifier(r'C:\Users\Amal Dev\Desktop\FER_pos\FER\src\model/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

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
            detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)

            img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            predictions = model.predict(img_pixels) #store probabilities of 7 expressions

            #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            max_index = np.argmax(predictions[0])

            emotion = emotions[max_index]
            cv2.putText(img,emotion,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
            print(f"-----------Detected Emotion:  {emotion}-----------")

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

# Example usage
# if __name__ == "__main__":
#     emotion_detection()
#     head_pose_estimation()

MODEL_JSON = r"C:\Users\Amal Dev\Desktop\realtimeattention_now\realtimeattention\myapp\facial_expression_model_structure.json"
MODEL_WEIGHTS = r"C:\Users\Amal Dev\Desktop\realtimeattention_now\realtimeattention\myapp\facial_expression_model_weights.h5"
FACE_CASCADE = r'C:\Users\Amal Dev\Desktop\FER_pos\FER\src\model/haarcascade_frontalface_default.xml'


def combined_detection():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Load emotion detection model
    model = model_from_json(open(MODEL_JSON, "r").read())
    model.load_weights(MODEL_WEIGHTS)
    
    # Initialize face cascade for both detection methods
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
    
    # Variables for head pose estimation
    flag = 0
    count = 0
    pose_text = ""
    
    # Variables for emotion detection
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    emotion_text = ""
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        # Save the frame for head pose estimation processing
        cv2.imwrite("ro.jpg", frame)
        img = cv2.imread("ro.jpg")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # ---- EMOTION DETECTION SECTION ----
            detected_face = img[int(y):int(y+h), int(x):int(x+w)]
            detected_face_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
            detected_face_resized = cv2.resize(detected_face_gray, (48, 48))
            
            img_pixels = image.img_to_array(detected_face_resized)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotion_text = emotions[max_index]
            
            # Display emotion text
            cv2.putText(img, f"Emotion: {emotion_text}", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"-----------Detected Emotion: {emotion_text}-----------")
            
            # ---- HEAD POSE ESTIMATION SECTION ----
            # Extract face region with margin for head pose
            face_img = img[max(0, int(y-10)):min(img.shape[0], int(y+h+10)), 
                          max(0, int(x-10)):min(img.shape[1], int(x+w+10))]
            
            if face_img.size == 0:  # Skip if face region is empty
                continue
                
            # Process for face mesh
            face_img_rgb = cv2.cvtColor(cv2.flip(face_img, 1), cv2.COLOR_BGR2RGB)
            face_img_rgb.flags.writeable = False
            
            # Get the face mesh result
            results = face_mesh.process(face_img_rgb)
            
            face_img_rgb.flags.writeable = True
            face_img_bgr = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2BGR)
            
            img_h, img_w, img_c = face_img_bgr.shape
            face_3d = []
            face_2d = []
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)
                                
                            x_point, y_point = int(lm.x * img_w), int(lm.y * img_h)
                            
                            # Get coordinates
                            face_2d.append([x_point, y_point])
                            face_3d.append([x_point, y_point, lm.z])
                            
                    # Convert to NumPy arrays
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)
                    
                    # Camera matrix
                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                          [0, focal_length, img_w / 2],
                                          [0, 0, 1]])
                    
                    # Distance matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    
                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    
                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    
                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    
                    # Get rotation degrees
                    x_angle = angles[0] * 360
                    y_angle = angles[1] * 360
                    
                    # Determine head pose
                    if y_angle < -10:
                        flag = 1
                        count = count + 1
                        pose_text = "Looking Left"
                    elif y_angle > 10:
                        flag = 1
                        count = count + 1
                        pose_text = "Looking Right"
                    elif x_angle < -10:
                        flag = 1
                        count = count + 1
                        pose_text = "Looking Down"
                    else:
                        flag = 0
                        count = 0
                        pose_text = "Forward"
                        
                    # Display head pose text
                    cv2.putText(img, f"Head: {pose_text}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Handle attention tracking counter
        if flag == 1:
            if count == 50:
                import time
                dt = time.strftime("%Y%m%d_%H%M%S")
                print("----------------------Attention Alert at {dt}: {pose_text}----------------------")
                count = 0
                flag = 0
        
        # Display the result
        cv2.imshow('Emotion and Head Pose Detection', img)
        
        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27 or cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    combined_detection()