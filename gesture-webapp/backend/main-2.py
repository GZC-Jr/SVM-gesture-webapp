import cv2
import numpy as np
import mediapipe as mp
import joblib
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles # CHANGED: To serve static files
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn # CHANGED: To run the server programmatically

# ==============================================================================
# 1. 初始化 FastAPI 应用
# ==============================================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 2. 加载模型和初始化工具 (路径已修改)
# ==============================================================================
try:
    # CHANGED: Paths are now relative to the project root (gesture-webapp/)
    model_path = 'backend/svm_gesture_model_tuned.pkl'
    scaler_path = 'backend/scaler.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"模型 '{model_path}' 和 scaler '{scaler_path}' 加载成功。")
except FileNotFoundError as e:
    print(f"错误: 找不到模型文件。 {e}")
    print("请确认你是在 gesture-webapp/ 目录下运行此脚本。")
    exit()

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ==============================================================================
# 3. WebSocket 连接管理器 (无变化)
# ==============================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"新客户端连接: {len(self.active_connections)} 个连接")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"客户端断开: {len(self.active_connections)} 个连接")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# ==============================================================================
# 4. 核心功能：视频处理与手势识别 (无变化)
# ==============================================================================
async def process_video_and_broadcast():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        return
        
    print("摄像头已启动，开始处理视频流...")
    
    # 确保 app.state 属性存在
    if not hasattr(app.state, 'latest_frame'):
        app.state.latest_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            await asyncio.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        gesture_id = "Unknown"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

                base_x, base_y, _ = landmarks[0], landmarks[1], landmarks[2]
                normalized_landmarks = []
                for i in range(0, len(landmarks), 3):
                    normalized_landmarks.append(landmarks[i] - base_x)
                    normalized_landmarks.append(landmarks[i+1] - base_y)
                
                feature_vector = np.array(normalized_landmarks).flatten()
                
                if len(feature_vector) == 42:
                    scaled_features = scaler.transform([feature_vector])
                    prediction = model.predict(scaled_features)
                    gesture_id = str(prediction[0])
                
                cv2.putText(frame, f'Gesture: {gesture_id}', (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

        message = json.dumps({"gesture": gesture_id})
        await manager.broadcast(message)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        app.state.latest_frame = frame_bytes

        await asyncio.sleep(0.01)

# ==============================================================================
# 5. 定义 FastAPI 路由 (路径已修改)
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_video_and_broadcast())

# 视频流路由 (无变化)
async def frame_generator():
    while True:
        if hasattr(app.state, 'latest_frame') and app.state.latest_frame:
            frame = app.state.latest_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        await asyncio.sleep(0.03)

@app.get("/video_feed")
async def video_feed():
    # 这部分解决了 404 问题，因为它现在是一个已知的路由
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

# WebSocket路由 (无变化)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 主页路由 (路径已修改)
@app.get("/")
async def read_index():
    # CHANGED: Path is relative to the project root
    return FileResponse('frontend/index.html')

# CHANGED: 添加一个静态文件目录挂载
# 这样如果你的HTML中引用了CSS或JS文件，它们也能被正确找到
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


# ==============================================================================
# 6. 添加直接运行的能力
# ==============================================================================
# CHANGED: New section to allow running with `python backend/main.py`
if __name__ == "__main__":
    print("正在以直接运行模式启动服务器...")
    # 使用 uvicorn.run() 来启动 app
    # host="0.0.0.0" 允许局域网内其他设备访问
    uvicorn.run(app, host="127.0.0.1", port=8000)