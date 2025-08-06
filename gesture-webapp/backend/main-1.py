import cv2
import mediapipe as mp
import numpy as np
import pickle
import asyncio
import joblib
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# --- 1. 初始化 ---

# 初始化 FastAPI 应用
app = FastAPI()

# 配置 CORS (跨源资源共享)
# 这允许我们的前端页面(从文件系统或不同端口打开)访问后端API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，为了简单起见
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)

# 加载训练好的SVM模型和Scaler
try:
    model = joblib.load('backend/svm_gesture_model_tuned.pkl')
    scaler = joblib.load('backend/scaler.pkl')
    print("模型和scaler加载成功！")
except FileNotFoundError:
    print("错误：找不到模型或scaler文件。请确保'svm_gesture_model_tuned.pkl'和'scaler.pkl'在backend目录下。")
    exit()


# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=3,         # 不只识别一只手
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 全局变量，用于在 video_feed 和 websocket 之间共享最新的手势
latest_gesture = "Unknown"

# --- 2. 手势识别核心逻辑 ---

def process_frame(frame):
    """处理单帧图像，进行手势识别，并返回处理后的图像和手势结果"""
    global latest_gesture

    # 翻转图像，使其看起来像镜子
    frame = cv2.flip(frame, 1)
    # 将 BGR 图像转换为 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 使用 MediaPipe 处理图像
    results = hands.process(frame_rgb)
    
    current_gesture = "Unknown" # 默认值

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 在图像上绘制手部关键点和连接线
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )
            
            # --- 数据预处理，与训练时保持一致 ---
            # 1. 提取关键点坐标
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            # 2. 相对坐标计算
            base_x, base_y, base_z = landmarks[0], landmarks[1], landmarks[2] # 以手腕为基准点
            relative_landmarks = []
            for i in range(0, len(landmarks), 3):
                relative_landmarks.append(landmarks[i] - base_x)
                relative_landmarks.append(landmarks[i+1] - base_y)
                relative_landmarks.append(landmarks[i+2] - base_z)
            
            # 3. 准备模型输入
            data = np.array(relative_landmarks).reshape(1, -1)
            # 4. 使用加载的scaler进行缩放
            scaled_data = scaler.transform(data)
            # 5. 进行预测
            prediction = model.predict(scaled_data)
            
            current_gesture = str(prediction[0])

            # 在图像上显示预测结果
            cv2.putText(frame, f"Gesture: {current_gesture}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
    # 更新全局手势变量
    latest_gesture = current_gesture
    return frame


# --- 3. FastAPI 端点实现 ---

async def video_generator():
    """视频流生成器"""
    cap = cv2.VideoCapture(0) # 0 代表默认摄像头
    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 处理每一帧
        processed_frame = process_frame(frame)
        
        # 将帧编码为 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        
        # 将 buffer 转换为字节流
        frame_bytes = buffer.tobytes()
        
        # 以 multipart/x-mixed-replace 格式 yield 帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 稍微等待，避免CPU占用过高
        await asyncio.sleep(0.01)

    cap.release()

@app.get("/video_feed")
async def video_feed():
    """提供视频流的HTTP端点"""
    return StreamingResponse(video_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """提供手势数据的WebSocket端点"""
    await websocket.accept()
    print("WebSocket客户端已连接")
    try:
        while True:
            # 每 0.1 秒向客户端发送一次最新的手-势数据
            await websocket.send_json({"gesture": latest_gesture})
            await asyncio.sleep(0.1) 
    except WebSocketDisconnect:
        print("WebSocket客户端已断开")
    except Exception as e:
        print(f"WebSocket 错误: {e}")
    finally:
        # 确保在断开连接时打印信息
        print("WebSocket 连接已关闭")


# --- 4. 运行应用 ---
if __name__ == "__main__":
    # 使用 uvicorn 启动应用
    # host="0.0.0.0" 让局域网内其他设备也可以访问
    # reload=True 会在代码改变时自动重启服务器，方便开发
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)