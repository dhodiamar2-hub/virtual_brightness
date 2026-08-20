import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

# --- Camera & MediaPipe Setup ---
cam_w, cam_h = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

bright_bar = 400
bright_perc = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip horizontally for natural mirror behavior
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # Extract Thumb tip (4) and Index tip (8)
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]

            # Convert normalized coordinates to camera pixels
            tx, ty = int(thumb_tip.x * cam_w), int(thumb_tip.y * cam_h)
            ix, iy = int(index_tip.x * cam_w), int(index_tip.y * cam_h)
            cx, cy = (tx + ix) // 2, (ty + iy) // 2

            # Render landmark tracking visuals
            cv2.circle(frame, (tx, ty), 10, (0, 215, 255), cv2.FILLED)
            cv2.circle(frame, (ix, iy), 10, (0, 215, 255), cv2.FILLED)
            cv2.line(frame, (tx, ty), (ix, iy), (0, 215, 255), 3)
            cv2.circle(frame, (cx, cy), 8, (0, 215, 255), cv2.FILLED)

            # Measure Euclidean distance between finger tips
            distance = np.hypot(ix - tx, iy - ty)

            # Map distance (20px - 200px) to brightness (0% - 100%) and UI height
            bright_perc = np.interp(distance, [20, 200], [0, 100])
            bright_bar = np.interp(distance, [20, 200], [400, 150])

            # Apply brightness level to primary monitor
            sbc.set_brightness(int(bright_perc))

            # Visual feedback on pinch closed
            if distance < 20:
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    # --- Draw On-Screen GUI Gauge ---
    # Brightness Bar Outer Border
    cv2.rectangle(frame, (50, 150), (85, 400), (255, 255, 255), 3)
    # Dynamic Fill Level
    cv2.rectangle(frame, (50, int(bright_bar)), (85, 400), (0, 215, 255), cv2.FILLED)
    # Brightness Percentage Text
    cv2.putText(
        frame, 
        f'{int(bright_perc)} %', 
        (40, 450), 
        cv2.FONT_HERSHEY_COMPLEX, 
        1, 
        (255, 255, 255), 
        2
    )

    cv2.imshow("Virtual Brightness Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()