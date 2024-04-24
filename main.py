import cv2 
import numpy as np
import mediapipe as mp
import time 
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

#mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mp_drawing = mp.solutions.drawing_utils

#record_list
Relbow_list = []
Rshoulder_list = []
Rbody_list = []
Rknee_list = []


#RL
RL = "R"

#fps
pTime = 0
def fps_show():
    global pTime
    cTime=time.time()
    fps = 1/(cTime-pTime)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    global total_video_seconds
    total_video_seconds = total_frames * (1 / fps)
    pTime=cTime
    cv2.putText(frame,f"fps:{int(fps)}",(30,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)


#Figure_angle
def Figure_angle(x1,y1,x2,y2,x3,y3):
    
    # 計算向量 AB 和 BC
    AB = (x2 - x1, y2 - y1)
    BC = (x3 - x2, y3 - y2)
    
    # 計算 AB 和 BC 的內積
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]
    
    # 計算 AB 和 BC 的大小
    magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
    magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)
    
    # 計算夾角的餘弦值
    cosine_angle = dot_product / (magnitude_AB * magnitude_BC)
    
    # 計算弧度值的夾角
    angle_radians = math.acos(cosine_angle)
    
    # 將弧度轉換為度數
    angle_deg = math.degrees(angle_radians)
    
    # 確定較小的夾角
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    
    #伸直180 收起0
    angle_deg = int(180 - angle_deg)

    return angle_deg
    '''向量的概念：在這個情況下，我們將兩個點之間的連線視為一個向量。例如，點 A(x1, y1) 和點 B(x2, y2) 之間的向量可以表示為 AB = (x2 - x1, y2 - y1)。這個向量的方向和大小可以描述兩點之間的位移和距離。

內積：內積是向量的一種運算，可以用來計算兩個向量之間的相似度。在這個情況下，我們使用內積來計算兩個向量 AB 和 BC 的相似度，它們的內積為 AB[0] * BC[0] + AB[1] * BC[1]。

夾角的餘弦值：當我們知道兩個向量的內積和它們的大小時，我們可以使用夾角的餘弦值來計算夾角。夾角的餘弦值等於兩個向量的內積除以它們的大小的乘積，即 cosine_angle = dot_product / (magnitude_AB * magnitude_BC)。

反餘弦函數：當我們知道夾角的餘弦值時，我們可以使用反餘弦函數（也稱為 arccos 函數）來計算夾角的弧度。反餘弦函數會返回夾角的弧度值，這個值介於 0 到 π 之間。

弧度轉換成度數：最後，我們將夾角的弧度值轉換成度數，這樣我們就可以得到以度為單位的夾角值。'''

#draw
class Draw():
    plt.style.use('bmh')
    plt.xlabel('time(frame)')
    plt.ylabel('angle')
    plt.plot(Relbow_list,'b',label='elbow')
    plt.plot(Rshoulder_list,'g',label='shoulder')
    plt.plot(Rbody_list,'r',label='body')
    plt.plot(Rknee_list,'y',label='knee')
    plt.legend(loc='lower left')
    def drawPlt():
        if RL == "R":
            plt.plot(Relbow_list,'b')
            plt.plot(Rshoulder_list,'g')
            plt.plot(Rbody_list,'r')
            plt.plot(Rknee_list,'y')
            plt.pause(0.01)


class Angle():
    def R_angle(list_name,a,b,c):
        x1, y1 = int(lms.landmark[a].x * imgW), int(lms.landmark[a].y * imgH)
        x2, y2 = int(lms.landmark[b].x * imgW), int(lms.landmark[b].y * imgH)
        x3, y3 = int(lms.landmark[c].x * imgW), int(lms.landmark[c].y * imgH)
        angle_deg=Figure_angle(x1,y1,x2,y2,x3,y3)
        # 顯示    
        xPos = int(lms.landmark[b].x*imgW)
        yPos = int(lms.landmark[b].y*imgH)
        cv2.putText(frame,str(f"{angle_deg}"),(xPos,yPos+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)# 0.4大小
        list_name.append(angle_deg)
        #print("R: ",angle_deg)
    def Lelbow():
        x1, y1 = int(lms.landmark[11].x * imgW), int(lms.landmark[11].y * imgH)
        x2, y2 = int(lms.landmark[13].x * imgW), int(lms.landmark[13].y * imgH)
        x3, y3 = int(lms.landmark[15].x * imgW), int(lms.landmark[15].y * imgH)
        angle_deg=Figure_angle(x1,y1,x2,y2,x3,y3)
        # 顯示    
        if RL == "L":
            xPos = int(lms.landmark[13].x*imgW)
            yPos = int(lms.landmark[13].y*imgH)
            cv2.putText(frame,str(f"L:{angle_deg}"),(xPos,yPos+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)# 0.4大小
            print("L: ",angle_deg)
    

#show
def show():
    global running,ret,frame,xPos,yPos,imgH,imgW,lms
    running = True
    while running:
        ret,frame=cap.read() #讀取回傳ret(bool)看是否有畫面 和 當前偵數的畫面frame
        if ret:
            imgH = frame.shape[0]
            imgW = frame.shape[1]
            #frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 將幀轉換為 RGB 格式
            results = pose.process(frame_rgb)  # 進行姿勢關鍵點檢測
        
            # 繪製關鍵點
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for i,lm in enumerate(results.pose_landmarks.landmark):
                    xPos = int(lm.x*imgW)
                    yPos = int(lm.y*imgH)
                    lms=results.pose_landmarks

                    #print(i, xPos, yPos)
                    #cv2.putText(frame,str(i),(xPos-25,yPos+5),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,0,255),2)#0.4大小

                    # if i in point_body:
                    #     cv2.circle(frame,(xPos,yPos),10,(0,255,0),cv2.FILLED)
        
                # Angle.Lelbow()
                # Angle.Relbow()
                # Angle.Rshoulder()
                # Angle.Rbody()
                # Angle.Rknee()
                Angle.R_angle(Relbow_list,12,14,16)
                Angle.R_angle(Rshoulder_list,14,12,24)
                Angle.R_angle(Rbody_list,12,24,26)
                Angle.R_angle(Rknee_list,24,26,28)
                fps_show()
                Draw.drawPlt()
                
            cv2.imshow("Output",frame)


        else:
            break
        #wait_time = int(1000 / fps)
        if cv2.waitKey(1)==ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    

class analyze_data():
    def detect_start_motion(detected_list):
        if detected_list:
            return
        

def stop_video():
    global running
    running = False  # 停止視頻播放循環

def new_plt():
    global Relbow_list,Rshoulder_list,Rbody_list,Rknee_list
    Relbow_list=[]
    Rshoulder_list = []
    Rbody_list = []
    Rknee_list = []
    plt.figure()
    plt.clf()
    plt.show()
    

#tk
class tk_window():
    global window
    window = tk.Tk()
    window.title('shooting detect')
    window.geometry('400x400')
    window.resizable(True, False)
    title_label=tk.Label(window,text="shooting detect",font=("Helvetica", 30))
    start_button = tk.Button(window,text="start",command=show,width=10)
    stop_button = tk.Button(window,text="stop",command=stop_video,width=10)
    new_plt_button = tk.Button(window,text="new plt",command=new_plt,width=10)
    file_path = "none"
    def open_file_dialog():
        file_path =  filedialog.askopenfilename()
        upload_file_name_label.config(text=os.path.basename(f"file name:{file_path}"))
        global cap
        cap = cv2.VideoCapture(file_path)
    upload_button = tk.Button(window,text="upload file",command=open_file_dialog,width=10)
    global upload_file_name_label
    upload_file_name_label = tk.Label(window,text=f"file name:{file_path}", width=15)

    #place

    title_label.grid(column=0, row=0, columnspan=3)
    upload_button.grid(column=0, row=2)
    upload_file_name_label.grid(column=1, row=2)
    start_button.grid(column=0, row=3,pady=20)
    stop_button.grid(column=1, row=3)
    new_plt_button.grid(column=2, row=3)
    window.mainloop()