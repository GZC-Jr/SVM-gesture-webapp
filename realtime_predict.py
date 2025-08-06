import cv2
import mediapipe as mp
import numpy as np
import joblib

# 加载模型
model = joblib.load('svm_gesture_model.pkl')

# 初始化MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制手部关键点
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 提取特征 (与训练时完全一致)
            base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
            features = []
            for landmark in hand_landmarks.landmark:
                features.append(landmark.x - base_x)
                features.append(landmark.y - base_y)
            
            # 预测
            prediction = model.predict([features])
            probability = model.predict_proba([features])
            gesture_id = int(prediction[0])
            confidence = np.max(probability)
            
            # 在画面上显示结果
            text = f"Gesture: {gesture_id} (Conf: {confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Real-time Gesture Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()