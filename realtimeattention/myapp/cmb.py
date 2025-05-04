import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
from keras import layers

# Load MobileNetV3 grayscale emotion model
emotion_model = load_model(
    r"C:\Users\Hp\Downloads\realtimeattention_today20\realtimeattention_today\realtimeattention\myapp\mobilenetv3.h5",
    custom_objects={"hard_swish": layers.Activation("hard_swish")}
)

emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')

# Initialize Mediapipe face detection and face mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Variables for cooldown logic
last_emotion = ""
last_pose_text = ""
last_emotion_time = 0
last_pose_time = 0
emotion_cooldown = 1.0
pose_cooldown = 1.0

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_h, img_w = img.shape[:2]

    # Detect faces using Mediapipe
    face_results = face_detection.process(rgb)
    face_landmarks = face_mesh.process(rgb)

    if face_results.detections:
        # Pick the largest face
        detections = sorted(face_results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height, reverse=True)
        detection = detections[0]
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * img_w)
        y = int(bboxC.ymin * img_h)
        w = int(bboxC.width * img_w)
        h = int(bboxC.height * img_h)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

        # Face ROI for emotion detection
        face_roi = gray[y1:y2, x1:x2]
        if face_roi.size > 0:
            face_roi = cv2.resize(face_roi, (32, 32))
            face_roi = cv2.equalizeHist(face_roi)  # Enhance contrast
            face_roi = face_roi.astype("float32") / 255.0
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = np.expand_dims(face_roi, axis=0)

            # Predict emotion
            emotion_pred = emotion_model.predict(face_roi, verbose=0)
            pred_label = emotions[np.argmax(emotion_pred[0])]
            pred_score = np.max(emotion_pred[0])

            # Update emotion with cooldown
            current_time = time.time()
            if pred_label != last_emotion and (current_time - last_emotion_time) > emotion_cooldown:
                last_emotion = pred_label
                last_emotion_time = current_time
                print(f"Detected Emotion: {last_emotion}")

            # Show emotion label
            cv2.putText(img, f"{last_emotion}: {pred_score*100:.2f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw face bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Head Pose Estimation
        if face_landmarks.multi_face_landmarks:
            for landmarks in face_landmarks.multi_face_landmarks:
                face_2d, face_3d = [], []
                for idx, lm in enumerate(landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:  # Key facial landmarks
                        px, py = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([px, py])
                        face_3d.append([px, py, lm.z])

                if face_2d and face_3d:
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = img_w
                    cam_matrix = np.array([
                        [focal_length, 0, img_w / 2],
                        [0, focal_length, img_h / 2],
                        [0, 0, 1]
                    ])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
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

                    # Update pose label with cooldown
                    if pose_temp != last_pose_text and (time.time() - last_pose_time) > pose_cooldown:
                        last_pose_text = pose_temp
                        last_pose_time = time.time()
                        print(f"Head Pose: {last_pose_text}")

                    break  # Only process first set of landmarks

        # Display pose
        cv2.putText(img, last_pose_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Attention logic
        if last_pose_text == "Looking Forward" and last_emotion in ['Neutral', 'Happy', 'Surprise']:
            attention_status = "Attentive"
        else:
            attention_status = "Non-Attentive"

        cv2.putText(img, f"Status: {attention_status}", (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show video
    cv2.imshow('Emotion & Head Pose Detection', img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
