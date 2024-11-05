import cv2
import mediapipe as mp
import pyautogui
import threading
from keras.models import load_model
import numpy as np
import time

# Khai báo thuộc tính chữ
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 0, 0)
thickness = 1

def move_cursor(x, y):
    pyautogui.moveTo(x, y - 150, duration=0.0001)

def auto_click():
    pyautogui.click()
    time.sleep(5)

# Khởi tạo Mediapipe Hand
loaded_model = load_model('Data/my_model.h5')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo video từ camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    data = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            point8 = hand_landmarks.landmark[8]
            x, y = int(point8.x * frame.shape[1]), int(point8.y * frame.shape[0])

            # Tạo luồng di chuyển con trỏ chuột
            cursor_thread = threading.Thread(target=move_cursor, args=(x, y))
            cursor_thread.start()

            # Xác định toạ độ
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * frame.shape[1]), int(max(x_coords) * frame.shape[1])
            y_min, y_max = int(min(y_coords) * frame.shape[0]), int(max(y_coords) * frame.shape[0])
            text_x = x_min
            text_y = y_min - 10

            # Vẽ hình
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(image, (x, y), radius=10, color=(0, 0, 255), thickness=-1)

            #Dự đoán cử chỉ
            for landmark in hand_landmarks.landmark:
                data.append(landmark.x)
                data.append(landmark.y)
            data_array = np.array(data)
            data_array = data_array.reshape(1, -1)
            y_pred = loaded_model.predict(data_array)


            if y_pred[0][0] > 0.8:
                cv2.putText(image, "Click", (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
                click_thread = threading.Thread(target=auto_click)
                click_thread.start()
            elif y_pred[0][0] < 0.3:
                cv2.putText(image, "Non click", (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
            else:
                pass


    # Hiển thị kết quả
    cv2.imshow("Hand Tracking", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
