import cv2
import numpy as np
import mediapipe as mp
import time
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import threading

# 多線程
# 數學原理
# 打包

 # mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mp_drawing = mp.solutions.drawing_utils

class Tk_window():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('shooting detect')
        self.window.geometry('400x200')
        self.window.resizable(True, False)
        
        self.file_path = "none"
        self.cap = None
        self.running = False
        self.first_plt = True
        
        title_label = tk.Label(self.window, text="shooting detect", font=("Helvetica", 30))
        start_button = tk.Button(self.window, text="start",command=Opevcv.show, width=10)
        stop_button = tk.Button(self.window, text="stop",command=Opevcv.stop_video, width=10)
        self.file_path = "none"
        upload_button = tk.Button(self.window, text="upload file", command=self.open_file_dialog, width=10)
        self.upload_file_name_label = tk.Label(self.window, text=f"file name:{self.file_path}", width=15)
        # place
        title_label.grid(column=0, row=0, columnspan=3)
        upload_button.grid(column=0, row=2)
        self.upload_file_name_label.grid(column=1, row=2)
        start_button.grid(column=0, row=3, pady=20)
        stop_button.grid(column=1, row=3)
    def open_file_dialog(self):
        self.file_path = filedialog.askopenfilename()
        self.upload_file_name_label.config(
            text=os.path.basename(f"file name:{self.file_path}"))
        self.cap = cv2.VideoCapture(self.file_path)
        first_plt = True
        if not first_plt:
            self.new_plt()
        first_plt = False
    def run(self):
        window = tk.Tk()
        window.mainloop()

class Opevcv(Tk_window):
    def __init__(self):
        self.pTime = 0
    # show
    def show(self):
        running = True
        while running:
            self.ret, self.frame = self.cap.read()  # 讀取回傳ret(bool)看是否有畫面 和 當前偵數的畫面frame
            if self.ret:
                imgH = self.frame.shape[0]
                imgW = self.frame.shape[1]
                # frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
                frame_rgb = cv2.cvtColor(
                    self.frame, cv2.COLOR_BGR2RGB)  # 將幀轉換為 RGB 格式
                results = pose.process(frame_rgb)  # 進行姿勢關鍵點檢測
                # 繪製關鍵點
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        self.frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        self.xPos = int(lm.x*imgW)
                        self.yPos = int(lm.y*imgH)
                        self.lms = results.pose_landmarks
                    Angle.R_angle(Relbow_list, 12, 14, 16)
                    Angle.R_angle(Rshoulder_list, 14, 12, 24)
                    Angle.R_angle(Rbody_list, 12, 24, 26)
                    Angle.R_angle(Rknee_list, 24, 26, 28)
                    Opevcv.fps_show()
                    Plt.drawPlt()
                cv2.imshow("Output", self.frame)
            else:
                break
            cv2.waitKey(10)
        self.cap.release()
        cv2.destroyAllWindows()

    def stop_video(self):
        self.running = False  # 停止視頻播放循環
    
    def fps_show(self):
        global pTime
        cTime = time.time()
        fps = 1/(cTime-pTime)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        global total_video_seconds
        total_video_seconds = total_frames * (1 / fps)
        pTime = cTime
        cv2.putText(self.frame, f"fps:{int(fps)}", (30, 50),
        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

class Plt(Opevcv):
    # record_list
    def reset():
        global Relbow_list, Rshoulder_list, Rbody_list, Rknee_list
        Relbow_list = []
        Rshoulder_list = []
        Rbody_list = []
        Rknee_list = []
    reset()
    # RL
    RL = "R"

    # Figure_angle
    def Figure_angle(x1, y1, x2, y2, x3, y3):
        # 計算 AB 和 BC 的x y長度
        a = (x2 - x1, y2 - y1)
        b = (x3 - x2, y3 - y2)

        # 計算 AB 和 BC 的向量大小(畢氏定理計算出向量的長度)
        magnitude_AB = math.sqrt(a[0]**2 + a[1]**2)  # AB線段
        magnitude_BC = math.sqrt(b[0]**2 + b[1]**2)  # BC線段

        # 計算 AB 和 BC 的內積 (公式 : a ⋅ b = |a| |b| cosθ = x1x2 + y1y2 )
        dot_product = a[0] * b[0] + a[1] * b[1]

        # 計算夾角的cosθ (cosθ = a ⋅ b / |a| |b|)
        cosine_angle = dot_product / (magnitude_AB * magnitude_BC)

        #把cosθ轉為兩邊夾角的弧度
        angle_radians = math.acos(cosine_angle)

        # 將弧度轉換為度數(角度)
        angle_deg = math.degrees(angle_radians)

        # 確定較小的夾角
        if angle_deg > 180:
            angle_deg = 360 - angle_deg

        # 伸直180 收起0
        angle_deg = int(180 - angle_deg)

        return angle_deg


    # draw
    plt.style.use('bmh')
    plt.xlabel('time(frame)')
    plt.ylabel('angle')
    plt.plot(Relbow_list, 'b', label='elbow')
    plt.plot(Rshoulder_list, 'g', label='shoulder')
    plt.plot(Rbody_list, 'r', label='body')
    plt.plot(Rknee_list, 'y', label='knee')
    plt.legend(loc='lower left')

    def drawPlt():
        plt.plot(Relbow_list, 'b')
        plt.plot(Rshoulder_list, 'g')
        plt.plot(Rbody_list, 'r')
        plt.plot(Rknee_list, 'y')
        plt.pause(0.01)


    def new_plt():
        Plt.reset()
        plt.figure()
        plt.style.use('bmh')
        plt.xlabel('time(frame)')
        plt.ylabel('angle')
        plt.plot(Relbow_list, 'b', label='elbow')
        plt.plot(Rshoulder_list, 'g', label='shoulder')
        plt.plot(Rbody_list, 'r', label='body')
        plt.plot(Rknee_list, 'y', label='knee')
        plt.legend(loc='lower left')
        plt.show()

class Angle():
    def R_angle(self,list_name, a, b, c):
        x1, y1 = int(self.lms.landmark[a].x * self.imgW), int(self.lms.landmark[a].y * self.imgH)
        x2, y2 = int(self.lms.landmark[b].x * self.imgW), int(self.lms.landmark[b].y * self.imgH)
        x3, y3 = int(self.lms.landmark[c].x * self.imgW), int(self.lms.landmark[c].y * self.imgH)
        angle_deg = Plt.Figure_angle(x1, y1, x2, y2, x3, y3)
        # 顯示
        xPos = int(self.lms.landmark[b].x*self.imgW)
        yPos = int(self.lms.landmark[b].y*self.imgH)
        cv2.putText(self.frame, str(f"{angle_deg}"), (xPos, yPos+20),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # 0.4大小
        list_name.append(angle_deg)

    # def Lelbow():
    #     x1, y1 = int(self.lms.landmark[11].x * imgW), int(lms.landmark[11].y * imgH)
    #     x2, y2 = int(lms.landmark[13].x * imgW), int(lms.landmark[13].y * imgH)
    #     x3, y3 = int(lms.landmark[15].x * imgW), int(lms.landmark[15].y * imgH)
    #     angle_deg = Figure_angle(x1, y1, x2, y2, x3, y3)
    #     # 顯示
    #     if RL == "L":
    #         xPos = int(lms.landmark[13].x*imgW)
    #         yPos = int(lms.landmark[13].y*imgH)
    #         cv2.putText(frame, str(f"L:{angle_deg}"), (xPos, yPos+20),
    #                     cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # 0.4大小
    #         print("L: ", angle_deg)
    
if __name__ == "__main__":
    app = Tk_window()
    app.run()