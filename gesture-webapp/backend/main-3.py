import cv2
import numpy as np
import mediapipe as mp
import joblib
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles 
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
# CHANGED: 导入 webockets 异常，用于更精确地捕捉连接关闭错误
from websockets.exceptions import ConnectionClosed

# ==============================================================================
# 1. 初始化 FastAPI 应用 (无变化)
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
# 2. 加载模型和初始化工具 (无变化)
# ==============================================================================
try:
    model_path = 'backend/svm_gesture_model_tuned.pkl'
    scaler_path = 'backend/scaler.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"模型 '{model_path}' 和 scaler '{scaler_path}' 加载成功。")
except FileNotFoundError as e:
    print(f"错误: 找不到模型文件。 {e}")
    print("请确认你是在 gesture-webapp/ 目录下运行此脚本。")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ==============================================================================
# 3. WebSocket 连接管理器 (关键修改)
# ==============================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"新客户端连接: {len(self.active_connections)} 个连接")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"客户端断开: {len(self.active_connections)} 个连接")

    # ==================== CRITICAL CHANGE HERE ====================
    #           修改 broadcast 方法，使其能够处理并移除已断开的连接
    # ==============================================================
    async def broadcast(self, message: str):
        # 创建一个列表来存储待移除的无效连接
        disconnected_clients: list[WebSocket] = []
        
        # 遍历所有活动连接
        for connection in self.active_connections:
            try:
                # 尝试发送消息
                await connection.send_text(message)
            except (WebSocketDisconnect, ConnectionClosed, RuntimeError) as e:
                # 如果发送失败（说明连接已断开），则将此连接添加到待移除列表
                # WebSocketDisconnect: FastAPI优雅断开
                # ConnectionClosed: websockets库的断开异常
                # RuntimeError: 当连接状态不正确时，FastAPI/Starlette可能抛出
                print(f"检测到无效连接，准备移除: {e}")
                disconnected_clients.append(connection)
        
        # 循环结束后，统一移除所有已标记的无效连接
        if disconnected_clients:
            print(f"正在清理 {len(disconnected_clients)} 个无效连接...")
            for connection in disconnected_clients:
                self.disconnect(connection) # 使用disconnect方法来移除

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
    
    if not hasattr(app.state, 'latest_frame'):
        app.state.latest_frame = None

    while True:
        try: # 添加一个顶层异常捕获，确保此循环不会意外崩溃
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
                        landmarks.extend([lm.x, lm.y, lm.z])

                    base_x, base_y = landmarks[0], landmarks[1]
                    normalized_landmarks = []
                    for i in range(0, len(landmarks), 3):
                        normalized_landmarks.extend([landmarks[i] - base_x, landmarks[i+1] - base_y])
                    
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
        except Exception as e:
            print(f"在主处理循环中发生错误: {e}")
            await asyncio.sleep(1) # 发生错误时稍作等待，防止CPU占用过高

# ==============================================================================
# 5. 定义 FastAPI 路由 (无变化，但建议修改主页路由指向新HTML)
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_video_and_broadcast())

async def frame_generator():
    while True:
        if hasattr(app.state, 'latest_frame') and app.state.latest_frame:
            frame = app.state.latest_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        await asyncio.sleep(0.03)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         while True:
#             # 保持连接开放，等待客户端断开
#             await websocket.receive_text()
#     except WebSocketDisconnect:
#         # 这个是“优雅”断开的处理
#         manager.disconnect(websocket)


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         # 我们不再等待一个永远不会到来的消息。
#         # 相反，我们进入一个循环，只是为了让这个连接的上下文保持活动状态。
#         # 后台的 broadcast 任务会通过这个 websocket 对象发送数据。
#         # 当客户端关闭浏览器或断开连接时，这个循环会因为异常而退出。
#         while True:
#             # 使用一个小的休眠来防止这个循环空占CPU，同时保持协程存活。
#             await asyncio.sleep(1) 
#     except (WebSocketDisconnect, ConnectionClosed):
#         # 捕获 FastAPI 的优雅断开和 websockets 库的底层断开异常
#         manager.disconnect(websocket)

# 这是修改后的正确代码
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # 我们不再等待客户端发送消息，因为客户端只接收数据。
        # 这个循环的唯一目的就是保持连接存活，直到客户端断开，
        # 届时将抛出 WebSocketDisconnect 异常。
        while True:
            await asyncio.sleep(1)  # 被动地等待，不消耗CPU
    except (WebSocketDisconnect, ConnectionClosed):
        # 捕获 FastAPI 和 websockets 库的断开异常
        manager.disconnect(websocket)

@app.get("/")
async def read_index():
    # 建议使用我们之前创建的、功能更完整的 index-3.html
    return FileResponse('frontend/index-3.html') 

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# ==============================================================================
# 6. 添加直接运行的能力 (无变化)
# ==============================================================================
if __name__ == "__main__":
    print("正在以直接运行模式启动服务器...")
    uvicorn.run(app, host="127.0.0.1", port=8000)