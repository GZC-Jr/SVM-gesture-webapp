import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# 初始化MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

DATA_DIR = 'data'
CSV_FILE = 'gesture_features.csv'

# 准备写入CSV文件
with open(CSV_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    # 写入表头，42个特征 (21个点的x,y) + 1个标签
    header = [f'p{i}_{axis}' for i in range(21) for axis in ['x', 'y']] + ['label']
    writer.writerow(header)

    # 遍历每个手势文件夹
    for label in os.listdir(DATA_DIR):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        print(f"Processing gesture: {label}")
        # 遍历文件夹中的每张图片
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 关键点归一化
                    base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
                    features = []
                    for landmark in hand_landmarks.landmark:
                        features.append(landmark.x - base_x)
                        features.append(landmark.y - base_y)
                    
                    # 写入一行数据：42个特征 + 1个标签
                    row = features + [label]
                    writer.writerow(row)

hands.close()
print(f"Dataset created successfully: {CSV_FILE}")