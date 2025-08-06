import cv2
import os

# 要采集的手势标签 (0-9)
label = "9"  # 每次只采集一种手势，手动修改这个数字

# 创建保存目录
SAVE_PATH = os.path.join('data', label)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# 采集数量
count = 0
cap = cv2.VideoCapture(0) # 打开摄像头

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 翻转图像，使其像镜子一样
    frame = cv2.flip(frame, 1)

    # 显示提示信息
    cv2.putText(frame, f"Collecting data for gesture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Count: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collecting Data", frame)

    # 按's'键保存图片
    key = cv2.waitKey(1)
    if key == ord('s'):
        img_name = os.path.join(SAVE_PATH, f"{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        count += 1
    # 按'q'键退出
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()