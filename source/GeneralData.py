import cv2
import mediapipe as mp
import csv


root_path_data = "../Data/click.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    data = []
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lưu dữ liệu vào file csv
            for landmark in hand_landmarks.landmark:
                data.append(landmark.x)
                data.append(landmark.y)

    cv2.imshow("Hand Tracking", image)

    # Nhấn r thì lưu dữ liệu vào file csv
    if cv2.waitKey(1) & 0xFF == ord('r'):
        with open(root_path_data, mode='a', newline='') as file:  # Mở file ở chế độ append
            writer = csv.writer(file)
            writer.writerow(data)

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
