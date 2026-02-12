import cv2
import mediapipe as mp
import numpy as np
import pydirectinput
import pyautogui
import time
import math

# 灵敏度
# 如果觉得太快，改小这个数；觉得太慢，改大。
SENSITIVITY_X = 12
SENSITIVITY_Y = 0

# 非线性加速 (值 > 1.0 时启用)
# 1.0 = 线性映射
# 1.5 = 小幅度转头时慢，大幅度转头时呈指数级加速
POWER_CURVE = 1.2

# 死区设置 (度)
# 在这个角度范围内，鼠标不动，减小微颤
DEAD_ZONE = 0.001

# --- 滤波器设置 ---
# 调小 MIN_CUTOFF 可以降低抖动程度
MIN_CUTOFF = 0.5
BETA = 0.05

# 显示实时摄像头画面（可能带来一定延迟）
SHOW_CAMERA = True

pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False
pyautogui.PAUSE = 0

# One Euro Filter
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.first_time = True
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = 0
        self.dx_prev = 0
        self.t_prev = 0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x, t):
        if self.first_time:
            self.x_prev = x
            self.dx_prev = 0
            self.t_prev = t
            self.first_time = False
            return x
        
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev

        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

screen_w, screen_h = pyautogui.size()
center_x, center_y = screen_w // 2, screen_h // 2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 滤波器实例
oef_x = OneEuroFilter(min_cutoff=MIN_CUTOFF, beta=BETA)
oef_y = OneEuroFilter(min_cutoff=MIN_CUTOFF, beta=BETA)

# 校准变量
calib_nose_rel_x = 0
calib_nose_rel_y = 0
is_calibrated = False

print("Head Tracker 已启动")
print("遮挡摄像头即可自由控制鼠标")

# 启动缓冲
start_time = time.time()

try:
    while cap.isOpened():
        current_time = time.time()
        success, image = cap.read()
        if not success: continue

        img_h, img_w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark
                
                # === 新的核心算法：2D 向量投影 ===
                # 获取鼻尖 (索引 1)
                nose_x = lm[1].x
                nose_y = lm[1].y
                
                # 获取双耳/面部边缘作为基准 (左耳: 234, 右耳: 454)
                # 使用双耳连线的中点作为“面部中心”，比用眼睛更稳，因为转头时眼睛会变形
                face_center_x = (lm[234].x + lm[454].x) / 2
                face_center_y = (lm[234].y + lm[454].y) / 2

                # 计算鼻尖相对于面部中心的“偏移向量”
                # 当你向左转头，鼻尖会向左移动，而面部中心相对保持（或反向），产生差值
                curr_rel_x = nose_x - face_center_x
                curr_rel_y = nose_y - face_center_y

                # 自动校准 (取前几帧作为零点)
                if not is_calibrated or (current_time - start_time < 1.0):
                    calib_nose_rel_x = curr_rel_x
                    calib_nose_rel_y = curr_rel_y
                    is_calibrated = True
                    if current_time - start_time > 0.5:
                        pass # 可以在这里加个提示

                # 计算相对于校准点的 Delta
                delta_x = curr_rel_x - calib_nose_rel_x
                delta_y = curr_rel_y - calib_nose_rel_y

                # 死区
                if abs(delta_x) < DEAD_ZONE: delta_x = 0
                if abs(delta_y) < DEAD_ZONE: delta_y = 0
                
                # 幂函数映射
                final_move_x = np.sign(delta_x) * (abs(delta_x) ** POWER_CURVE) * SENSITIVITY_X * screen_w
                final_move_y = np.sign(delta_y) * (abs(delta_y) ** POWER_CURVE) * SENSITIVITY_Y * screen_h

                # 计算目标绝对坐标
                target_x = center_x - final_move_x # 如果方向反了就把减号改加号
                target_y = center_y + final_move_y

                # 滤波
                filtered_x = oef_x.filter(target_x, current_time)
                filtered_y = oef_y.filter(target_y, current_time)

                # 限位
                screen_x = int(np.clip(filtered_x, 0, screen_w - 1))
                screen_y = int(np.clip(filtered_y, 0, screen_h - 1))

                pydirectinput.moveTo(screen_x, screen_y)
        
        if SHOW_CAMERA:
            cv2.imshow('Head Tracker', cv2.flip(image, 1))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("停止")
cap.release()