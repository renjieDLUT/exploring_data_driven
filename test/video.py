import cv2

# 打开摄像头，0代表第一个摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open video capture")

    exit()

ret, frame = cap.read()

if not ret:
    print("Cannot read frame")

    exit()

while True:
    # 读取摄像头的帧，ret为布尔类型，如果读取帧正确则返回True，img为图像的三维矩阵
    ret, img = cap.read()

    # 显示图像
    cv2.imshow('Camera', img)

    # 如果按下q键，则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 释放摄像头并销毁所有窗口
cap.release()
cv2.destroyAllWindows()