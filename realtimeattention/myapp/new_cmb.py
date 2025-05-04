import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

# Load emotion model from JSON structure and weights
try:
    # Load model structure from JSON file
    json_file_path = r"C:\Users\Amal Dev\Desktop\realtimeattention_now\realtimeattention\myapp\facial_expression_model_structure.json"
    weights_file_path = r"C:\Users\Amal Dev\Desktop\realtimeattention_now\realtimeattention\myapp\facial_expression_model_weights.h5"
    
    with open(json_file_path, "r") as json_file:
        model_json = json_file.read()
    
    emotion_model = model_from_json(model_json)
    emotion_model.load_weights(weights_file_path)
    emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Emotion model loaded successfully")
    
    # Test if the model works
    test_input = np.random.random((1, 48, 48, 1))  # Note: this model expects 48x48 input
    test_pred = emotion_model.predict(test_input, verbose=0)
    print("Model validation successful. Output shape:", test_pred.shape)
    
except Exception as e:
    print(f"Error loading emotion model: {str(e)}")
    import traceback
    traceback.print_exc()
    exit()

# Define emotion labels - using lowercase to match your second implementation
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Alternative face detection option - you can switch between MediaPipe and Haar Cascade
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\Amal Dev\Desktop\FER_pos\FER\src\model/haarcascade_frontalface_default.xml'
)

# Initialize Mediapipe face detection and face mesh for head pose estimation
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables for cooldown logic
last_emotion = ""
last_pose_text = "Looking Forward"
last_emotion_time = 0
last_pose_time = 0
emotion_cooldown = 1.0
pose_cooldown = 1.0

# Function to preprocess face for emotion recognition
def preprocess_face_for_emotion(face_img):
    """
    Preprocess detected face for emotion detection model input
    """
    try:
        # Convert to grayscale if not already
        if len(face_img.shape) > 2 and face_img.shape[2] > 1:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_img
            
        # Resize to model's expected input size (48x48)
        resized_face = cv2.resize(face_gray, (48, 48))
        
        # Convert to format expected by the model
        img_pixels = image.img_to_array(resized_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        
        # Normalize pixel values to [0,1]
        img_pixels /= 255.0
        
        return img_pixels
    except Exception as e:
        print(f"Error in face preprocessing: {str(e)}")
        return None

def detect_emotion(face_img):
    """
    Detect emotion from face image
    Returns emotion label and confidence score
    """
    try:
        processed_face = preprocess_face_for_emotion(face_img)
        if processed_face is None:
            return None, 0
            
        predictions = emotion_model.predict(processed_face, verbose=0)
        max_index = np.argmax(predictions[0])
        max_score = np.max(predictions[0])
        
        return emotions[max_index], max_score
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return None, 0

# Start webcam
cap = cv2.VideoCapture(0)

# Set detection method: 'mediapipe' or 'haar'
detection_method = 'mediapipe'  # Change to 'haar' if mediapipe doesn't work well

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame")
        break

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_h, img_w = img.shape[:2]
    
    # Face detection
    faces_detected = False
    x1, y1, x2, y2 = 0, 0, 0, 0
    
    if detection_method == 'mediapipe':
        # Detect faces using Mediapipe
        face_results = face_detection.process(rgb)
        
        if face_results.detections:
            faces_detected = True
            # Pick the largest face
            detections = sorted(face_results.detections, 
                               key=lambda d: d.location_data.relative_bounding_box.width * 
                                            d.location_data.relative_bounding_box.height, 
                               reverse=True)
            detection = detections[0]
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * img_w)
            y = int(bboxC.ymin * img_h)
            w = int(bboxC.width * img_w)
            h = int(bboxC.height * img_h)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)
    else:
        # Detect faces using Haar cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) > 0:
            faces_detected = True
            # Use the first face detected (or sort by size if needed)
            x, y, w, h = faces[0]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)

    if faces_detected:
        # Draw face bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Face ROI for emotion detection
        face_roi = img[y1:y2, x1:x2]
        
        if face_roi.size > 0:
            # Detect emotion
            emotion, confidence = detect_emotion(face_roi)
            
            if emotion:
                # Update emotion with cooldown
                current_time = time.time()
                if emotion != last_emotion and (current_time - last_emotion_time) > emotion_cooldown:
                    last_emotion = emotion
                    last_emotion_time = current_time
                    print(f"Detected Emotion: {last_emotion} ({confidence*100:.2f}%)")
                
                # Show emotion label on frame
                emotion_text = f"{last_emotion.capitalize()}: {confidence*100:.2f}%"
                cv2.putText(img, emotion_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Process face mesh for head pose estimation if using mediapipe
        face_landmarks = face_mesh.process(rgb)
        if face_landmarks and face_landmarks.multi_face_landmarks:
            for landmarks in face_landmarks.multi_face_landmarks:
                face_2d, face_3d = [], []
                for idx, lm in enumerate(landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:  # Key facial landmarks
                        px, py = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([px, py])
                        face_3d.append([px, py, lm.z])

                if len(face_2d) == 6 and len(face_3d) == 6:  # Ensure we have all required landmarks
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = img_w
                    cam_matrix = np.array([
                        [focal_length, 0, img_w / 2],
                        [0, focal_length, img_h / 2],
                        [0, 0, 1]
                    ])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    try:
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        if success:
                            rmat, _ = cv2.Rodrigues(rot_vec)
                            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                            x_angle = angles[0] * 360
                            y_angle = angles[1] * 360

                            pose_temp = "Looking Forward"
                            if y_angle < -15:
                                pose_temp = "Looking Left"
                            elif y_angle > 15:
                                pose_temp = "Looking Right"
                            elif x_angle < -15:
                                pose_temp = "Looking Down"
                            elif x_angle > 15:
                                pose_temp = "Looking Up"

                            # Update pose label with cooldown
                            if pose_temp != last_pose_text and (time.time() - last_pose_time) > pose_cooldown:
                                last_pose_text = pose_temp
                                last_pose_time = time.time()
                                print(f"Head Pose: {last_pose_text}")
                    except Exception as e:
                        print(f"Error during pose estimation: {str(e)}")
                break  # Only process first face

        # Display pose
        cv2.putText(img, last_pose_text, (x1, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Attention logic - convert emotion to lowercase for comparison
        emotion_lower = last_emotion.lower() if last_emotion else ""
        if last_pose_text == "Looking Forward" and emotion_lower in ['neutral', 'happy', 'surprise']:
            attention_status = "Attentive"
        else:
            attention_status = "Non-Attentive"

        cv2.putText(img, f"Status: {attention_status}", (x1, y2 + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(img, last_pose_text, (x1, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Attention logic - convert emotion to lowercase for comparison
        emotion_lower = last_emotion.lower() if last_emotion else ""
        if last_pose_text == "Looking Forward" and emotion_lower in ['neutral', 'happy', 'surprise']:
            attention_status = "Attentive"
        else:
            attention_status = "Non-Attentive"

        cv2.putText(img, f"Status: {attention_status}", (x1, y2 + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show frame
    cv2.imshow('Emotion & Head Pose Detection', img)
    
    # Check for key press
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        # Toggle detection method
        detection_method = 'haar' if detection_method == 'mediapipe' else 'mediapipe'
        print(f"Switched to {detection_method} detection")

cap.release()
cv2.destroyAllWindows()