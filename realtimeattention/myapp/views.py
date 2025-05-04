from django.contrib import messages
from django.contrib.auth import authenticate
from django.http import HttpResponse,JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.models import User

# Create your views here.
from myapp.models import student_table, emotion_table
from myapp.forms import StudentRegisterForm

from django.http import StreamingHttpResponse
import cv2
import numpy as np
import mediapipe as mp
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import math
import time

from tensorflow.keras.preprocessing import image

import cv2
import time
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import mediapipe as mp
from django.core.files.base import ContentFile
from myapp.models import student_table, emotion_table,Attendence
from django.utils.timezone import now
from datetime import datetime, timedelta

from datetime import timedelta, date
from django.utils.timezone import localtime
from .models import Attendence
from collections import defaultdict
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.contrib import messages



def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            user = User.objects.get(username=username)
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, "Login successful!")
                return redirect('admin_home') 
            else:
                messages.error(request, "Invalid credentials or not an admin.")
        except User.DoesNotExist:
            messages.error(request, "User does not exist.")

    return render(request, 'admin_login.html')


def admin_logout(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect('admin_login')



def admin_home(request):
    user = request.user 
    return render(request, 'home.html', {"user": user})


def list_student(request):
    students = student_table.objects.all()
    return render(request,'student_list.html',{'students':students,'user':request.user})

def list_attendence(request):
    logs = Attendence.objects.all().order_by('-timestamp')
    daily_logs = defaultdict(list)
    for log in logs:
        day = localtime(log.timestamp).date()
        daily_logs[day].append(log)

    summary = []
    for day, logs in daily_logs.items():
        inattentive_count = len(logs)  
        status = "Inattentive" if inattentive_count > 0 else "Absent"
        image = logs[0].image.url if logs[0].image else None

        summary.append({
            "date": day.strftime("%B %d, %Y"),
            "status": status,
            "inattentive_count": inattentive_count,
            "image": image
        })
    return render(request,'attendence.html',{'summary':summary,'user':request.user})



def student_login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        try:
            student = student_table.objects.get(email=email, password=password)
            request.session['student_id'] = student.id 
            request.session['student_name'] = student.name
            messages.success(request, "Login successful!")
            return redirect('student_home')
        except student_table.DoesNotExist:
            messages.error(request, "Invalid email or password.")
    return render(request, 'sign_in.html')


def student_logout(request):
    request.session.flush()
    messages.success(request, "You have been logged out successfully.")
    return redirect('student_login')


def student_register(request):
    if request.method == 'POST':
        form = StudentRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Registration successful! Please log in.")
            return redirect('student_login')
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = StudentRegisterForm()
    return render(request, 'sign_up.html', {'form': form})


def student_home(request):
    student_id = request.session.get('student_id')
    if not student_id:
        messages.error(request, "Please log in to continue.")
        return redirect('student_login')
    try:
        student = student_table.objects.get(id=student_id)
    except student_table.DoesNotExist:
        messages.error(request, "Student not found.")
        return redirect('student_login')
    today = date.today()
    logs = Attendence.objects.filter(
            student_id=student_id,
            is_attentive=False 
        ).order_by('-timestamp')

    daily_logs = defaultdict(list)
    for log in logs:
        day = localtime(log.timestamp).date()
        daily_logs[day].append(log)

    summary = []
    for day, logs in daily_logs.items():
        inattentive_count = len(logs)  
        status = "Inattentive" if inattentive_count > 0 else "Absent"
        image = logs[0].image.url if logs[0].image else None

        summary.append({
            "date": day.strftime("%B %d, %Y"),
            "status": status,
            "inattentive_count": inattentive_count,
            "image": image
        })

    return render(request, 'student_home.html', {'student': student,'summary': summary})


try:
    json_file_path = r"C:\Users\Hp\Downloads\Telegram Desktop\realtimeattention_final\realtimeattention\myapp\facial_expression_model_structure.json"
    weights_file_path = r"C:\Users\Hp\Downloads\Telegram Desktop\realtimeattention_final\realtimeattention\myapp\facial_expression_model_weights.h5"
    
    with open(json_file_path, "r") as json_file:
        model_json = json_file.read()
    
    emotion_model = model_from_json(model_json)
    emotion_model.load_weights(weights_file_path)
    emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Emotion model loaded successfully")
    
    # Model test
    test_input = np.random.random((1, 48, 48, 1))
    test_pred = emotion_model.predict(test_input, verbose=0)
    print("Model validation successful. Output shape:", test_pred.shape)
except Exception as e:
    print(f"Error loading emotion model: {str(e)}")
    import traceback
    traceback.print_exc()
    exit()

# Emotion labels
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Haar cascade
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\Hp\Downloads\Telegram Desktop\realtimeattention_final\realtimeattention\myapp\haarcascade_frontalface_default.xml'
)

# Mediapipe initialization
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Globals
last_emotion = ""
last_pose_text = "Looking Forward"
last_emotion_time = 0
last_pose_time = 0
emotion_cooldown = 1.0
pose_cooldown = 1.0
streaming_active = False
last_save_time = None

def preprocess_face_for_emotion(face_img):
    try:
        if len(face_img.shape) > 2 and face_img.shape[2] > 1:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_img
        resized_face = cv2.resize(face_gray, (48, 48))
        img_pixels = image.img_to_array(resized_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0
        return img_pixels
    except Exception as e:
        print(f"Error in face preprocessing: {str(e)}")
        return None


def detect_emotion(face_img):
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


def gen_frames(request):
    global last_emotion, last_emotion_time, last_pose_text, last_pose_time,last_save_time
    cap = cv2.VideoCapture(0)
    detection_method = 'mediapipe'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame")
            break

        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_h, img_w = img.shape[:2]
        faces_detected = False
        x1, y1, x2, y2 = 0, 0, 0, 0

        if detection_method == 'mediapipe':
            face_results = face_detection.process(rgb)
            if face_results.detections:
                faces_detected = True
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
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(img_w, x + w), min(img_h, y + h)
        else:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if len(faces) > 0:
                faces_detected = True
                x, y, w, h = faces[0]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(img_w, x + w), min(img_h, y + h)

        if faces_detected:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            face_roi = img[y1:y2, x1:x2]

            if face_roi.size > 0:
                emotion, confidence = detect_emotion(face_roi)
                if emotion:
                    current_time = time.time()
                    if emotion != last_emotion and (current_time - last_emotion_time) > emotion_cooldown:
                        last_emotion = emotion
                        last_emotion_time = current_time
                        print(f"Detected Emotion: {last_emotion} ({confidence * 100:.2f}%)")

                    cv2.putText(img, f"{last_emotion.capitalize()}: {confidence * 100:.2f}%",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            face_landmarks = face_mesh.process(rgb)
            if face_landmarks and face_landmarks.multi_face_landmarks:
                for landmarks in face_landmarks.multi_face_landmarks:
                    face_2d, face_3d = [], []
                    for idx, lm in enumerate(landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            px, py = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([px, py])
                            face_3d.append([px, py, lm.z])
                    if len(face_2d) == 6 and len(face_3d) == 6:
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
                            success, rot_vec, trans_vec = cv2.solvePnP(
                                face_3d, face_2d, cam_matrix, dist_matrix)
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

                                if pose_temp != last_pose_text and (time.time() - last_pose_time) > pose_cooldown:
                                    last_pose_text = pose_temp
                                    last_pose_time = time.time()
                                    print(f"Head Pose: {last_pose_text}")
                        except Exception as e:
                            print(f"Error during pose estimation: {str(e)}")
                    break

            # Attention logic
            emotion_lower = last_emotion.lower() if last_emotion else ""
            if last_pose_text == "Looking Forward" and emotion_lower in ['neutral', 'happy', 'surprise']:
                attention_status = "Attentive"
            else:
                attention_status = "Non-Attentive"
                if attention_status == "Non-Attentive":
                    student_id = request.session.get('student_id')
                    now_time = datetime.now()
                    if student_id and last_save_time is None or (now_time - last_save_time).total_seconds() > 10:
                        _, img_buffer = cv2.imencode('.jpg', img)
                        try:
                            student = student_table.objects.get(id=student_id)
                            image_content = ContentFile(img_buffer.tobytes(), name=f"non_attentive_{now_time.strftime('%Y%m%d_%H%M%S')}.jpg")
                            Attendence.objects.create(student=student, is_attentive=False, image=image_content)
                            last_save_time = now_time
                            print(f"Saved non-attentive frame for student {student_id}")
                        except student_table.DoesNotExist:
                            print(f"Student with ID {student_id} not found")
                        print(f"Stored non-attentive record for student {student_id}")
                    else:
                        pass
            cv2.putText(img, last_pose_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(img, f"Status: {attention_status}", (x1, y2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            detection_method = 'haar' if detection_method == 'mediapipe' else 'mediapipe'
            print(f"Switched to {detection_method} detection")

    cap.release()
    cv2.destroyAllWindows()


def video_feed(request):
    global streaming_active
    streaming_active = True
    return StreamingHttpResponse(gen_frames(request),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def stop_video_feed(request):
    global streaming_active
    streaming_active = False
    return JsonResponse({"status": "streaming stopped"})
