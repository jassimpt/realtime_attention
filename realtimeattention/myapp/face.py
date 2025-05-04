import cv2
import mediapipe as mp

import numpy as np
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
flag=0
count=0
text=""
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    success, image = cap.read()

    cv2.imwrite("ro.jpg", image)
    img = cv2.imread("ro.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (xx,yy,w,h) in faces:

        cv2.rectangle(img, (xx, yy), (xx + w, yy + h), (255, 0, 0), 2)

        image = img[int(yy-10):int(yy + h+10), int(xx-10):int(xx + w+10)]

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)


        image.flags.writeable = False

        # Get the result
        results = face_mesh.process(image)


        image.flags.writeable = True


        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
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

                # Convert it to the NumPy array
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
                # print(y)
                # See where the user's head tilting
                if y < -10:
                    flag=1
                    count=count+1
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
                    flag=0
                    count=0
                    text = "Forward"

                cv2.putText(img, text, (xx, yy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        else:
            text="No Face"
            cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if flag==1:
        if count==50:
            import time
            dt = time.strftime("%Y%m%d_%H%M%S")

            print("+_+_+_+_+_+)+)+)+)+_+_+_+_+_)()_")
            count=0
            flag=0
    cv2.imshow('Head Pose Estimation', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()


