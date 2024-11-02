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
from data_collect import record as rc
from AIq import outputmotion

#pyinstaller --onefile --windowed --add-data "C:\\Users\\yoyok\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\mediapipe\\modules\\pose_landmark\\pose_landmark_cpu.binarypb;mediapipe/modules/pose_landmark" main.py

#多線程 嘗試方法一 把global改self 失敗
#數學原理

class Main():
    def __init__(self):
        #mediapipe
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        #reset
        self.Relbow_list = []
        self.Rshoulder_list = []
        self.Rbody_list = []
        self.Rknee_list = []
        #fps
        self.pTime=0
        #RL
        self.RL="R"
        #draw
        self.first_plt=True
        plt.style.use('bmh')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.plot(self.Relbow_list,'b',label='elbow')
        plt.plot(self.Rshoulder_list,'g',label='shoulder')
        plt.plot(self.Rbody_list,'r',label='body')
        plt.plot(self.Rknee_list,'y',label='knee')
        plt.legend(loc='lower left')
        #Tk
        self.window = tk.Tk()
        self.window.title('shooting detect')
        self.window.geometry('400x250')
        self.window.resizable(True, False)
        title_label=tk.Label(self.window,text="shooting detect",font=("Helvetica", 30))
        start_button = tk.Button(self.window,text="start",command=self.show,width=10)
        stop_button = tk.Button(self.window,text="stop",command=self.stop_video,width=10)
        self.file_path = "none"
        self.upload_button = tk.Button(self.window,text="upload file",command=self.open_file_dialog,width=10)
        self.upload_file_name_label = tk.Label(self.window,text=f"file name:{self.file_path}", width=15)
        title_label.grid(column=0, row=0, columnspan=3)
        self.upload_button.grid(column=0, row=2)
        self.upload_file_name_label.grid(column=1, row=2)
        start_button.grid(column=0, row=3,pady=20)
        stop_button.grid(column=1, row=3)
        
        one_motion_record=tk.Button(self.window,text="one",command=lambda: rc(self,"one"),width=10)
        one_motion_record.grid(column=0, row=4)
        two_motion_record=tk.Button(self.window,text="two",command=lambda: rc(self,"two"),width=10)
        two_motion_record.grid(column=1, row=4)
        record_motion_record=tk.Button(self.window,text="record",command=lambda: rc(self,"record"),width=10)
        record_motion_record.grid(column=0, row=5)
        distinguish_motion_record=tk.Button(self.window,text="AI",command=outputmotion,width=10)
        distinguish_motion_record.grid(column=1, row=5)

    #record_list
    def reset(self):
        self.Relbow_list = []
        self.Rshoulder_list = []
        self.Rbody_list = []
        self.Rknee_list = []
    
    #fps
    def fps_show(self):
        cTime=time.time()
        fps = 1/(cTime-self.pTime)
        self.pTime=cTime
        cv2.putText(self.frame,f"fps:{int(fps)}",(30,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)

    #Figure_angle
    def figure_angle(self,x1,y1,x2,y2,x3,y3):  
        # 計算 AB 和 BC 的x y長度
        a = (x2 - x1, y2 - y1)
        b = (x3 - x2, y3 - y2)
        # 計算 AB 和 BC 的向量大小(畢氏定理計算出向量的長度)
        magnitude_AB = math.sqrt(a[0]**2 + a[1]**2) #AB線段
        magnitude_BC = math.sqrt(b[0]**2 + b[1]**2) #BC線段  
        # 計算 AB 和 BC 的內積 (公式 : a ⋅ b = |a| |b| cosθ = x1x2 + y1y2 )
        dot_product = a[0] * b[0] + a[1] * b[1]
        # 計算夾角的cosθ (cosθ = a ⋅ b / |a| |b|)
        cosine_angle = dot_product / (magnitude_AB * magnitude_BC)    
        # 把cosθ轉為兩邊夾角的弧度
        angle_radians = math.acos(cosine_angle)
        # 將弧度轉換為度數(角度)
        angle_deg = math.degrees(angle_radians)
        # 確定較小的夾角
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
        #伸直180 收起0
        angle_deg = int(180 - angle_deg)
        return angle_deg

    def drawPlt(self, time_elapsed):
        # 記錄時間點並與角度數據一同繪製
        if not hasattr(self, 'time_points'):
            self.time_points = []  # 初始化時間點列表
    
        # 添加當前時間點
        self.time_points.append(time_elapsed)
    
        # 清空舊的圖表，重新繪製
        if self.RL == "R":
            plt.plot(self.time_points, self.Relbow_list, 'b')
            plt.plot(self.time_points, self.Rshoulder_list, 'g')
            plt.plot(self.time_points, self.Rbody_list, 'r')
            plt.plot(self.time_points, self.Rknee_list, 'y')
        plt.pause(0.01)

    def new_plt(self):
        self.reset()
        plt.figure()
        plt.style.use('bmh')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.legend(loc='lower left')
        plt.plot(self.Relbow_list,'b',label='elbow')
        plt.plot(self.Rshoulder_list,'g',label='shoulder')
        plt.plot(self.Rbody_list,'r',label='body')
        plt.plot(self.Rknee_list,'y',label='knee')
        self.time_points=[]
        plt.show()

    def R_angle(self,list_name,a,b,c):
        x1, y1 = int(self.lms.landmark[a].x * self.imgW), int(self.lms.landmark[a].y * self.imgH)
        x2, y2 = int(self.lms.landmark[b].x * self.imgW), int(self.lms.landmark[b].y * self.imgH)
        x3, y3 = int(self.lms.landmark[c].x * self.imgW), int(self.lms.landmark[c].y * self.imgH)
        angle_deg=self.figure_angle(x1,y1,x2,y2,x3,y3)
        # 顯示    
        xPos = int(self.lms.landmark[b].x*self.imgW)
        yPos = int(self.lms.landmark[b].y*self.imgH)
        cv2.putText(self.frame,str(f"{angle_deg}"),(xPos,yPos+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)# 0.4大小
        list_name.append(angle_deg)
    
    def show(self):
        self.running = True
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # 獲取影片幀率
        time_elapsed = 0  # 用來記錄影片的秒數

        while self.running:
            ret, self.frame = self.cap.read()
            if ret:
                self.imgH = self.frame.shape[0]
                self.imgW = self.frame.shape[1]
                frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                # 繪製關鍵點
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(self.frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        self.xPos = int(lm.x * self.imgW)
                        self.yPos = int(lm.y * self.imgH)
                        self.lms = results.pose_landmarks

                    # 計算角度
                    self.R_angle(self.Relbow_list, 12, 14, 16)
                    self.R_angle(self.Rshoulder_list, 14, 12, 24)
                    self.R_angle(self.Rbody_list, 12, 24, 26)
                    self.R_angle(self.Rknee_list, 24, 26, 28)

                    # 繪製 FPS 和更新繪圖
                    self.fps_show()
                    self.drawPlt(time_elapsed)  # 傳入影片的秒數來畫圖

                cv2.imshow("Output", self.frame)

                # 更新時間（秒數）
                time_elapsed += 1 / fps  # 增加時間（每一幀時間）
            else:
                break

            cv2.waitKey(10)
        self.cap.release()
        cv2.destroyAllWindows()

        
    def stop_video(self):
        self.running = False  # 停止視頻播放循環

    def open_file_dialog(self):
        file_path =  filedialog.askopenfilename()
        self.upload_file_name_label.config(text=os.path.basename(f"file name:{file_path}"))
        self.cap = cv2.VideoCapture(file_path)
        if not self.first_plt:
            self.new_plt()
        self.first_plt = False
                
    
    
    def run(self):
        self.window.mainloop()
        
if __name__ == "__main__":
    app = Main()
    app.run()