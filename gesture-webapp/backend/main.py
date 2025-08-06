# backend/main.py

import base64
import cv2
import mediapipe as mp
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import uvicorn

# 1. 初始化 FastAPI 应用
app = FastAPI()

# 2. 加载模型和工具 (在服务启动时加载一次，避免重复加载)
try:
    model = joblib.load('backend/svm_gesture_model_tuned.pkl')
    scaler = joblib.load('backend/scaler.pkl')
    print("模型和scaler加载成功！")
except FileNotFoundError:
    print("错误：找不到模型或scaler文件。请确保它们在 'backend' 文件夹中。")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# 3. 定义手势识别的核心函数
def get_gesture_prediction(image_bytes):
    """接收图像字节流，返回预测结果"""
    try:
        # 将字节流解码为图像
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        # 图像处理和模型预测 (这是你脚本的核心逻辑)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
                features = []
                for landmark in hand_landmarks.landmark:
                    features.append(landmark.x - base_x)
                    features.append(landmark.y - base_y)
                
                features_np = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_np)
                
                prediction = model.predict(features_scaled)
                probability = model.predict_proba(features_scaled)
                
                return {
                    "gesture": int(prediction[0]),
                    "confidence": float(np.max(probability))
                }
        return None
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None

# 4. 创建 WebSocket 端点
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket连接成功！")
    try:
        while True:
            # 接收来自前端的图像数据
            data = await websocket.receive_bytes()
            
            # 调用识别函数
            prediction_result = get_gesture_prediction(data)
            
            # 如果有结果，就发回给前端
            if prediction_result:
                await websocket.send_json(prediction_result)

    except WebSocketDisconnect:
        print("WebSocket连接断开。")
    except Exception as e:
        print(f"WebSocket出错: {e}")
    finally:
        await websocket.close()

# 5. (可选) 添加一个根路径，用于测试服务器是否在运行
@app.get("/")
def read_root():
    return {"Hello": "Gesture Recognition World"}

# 这部分代码允许我们直接运行这个Python文件来启动服务器
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)