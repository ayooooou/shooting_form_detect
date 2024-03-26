import cv2 
import numpy as np
import mediapipe as mp
import time 
import math
import matplotlib.pyplot as plt

#read
cap = cv2.VideoCapture("opencv\shoting_detect\\5.mp4")
#cap = cv2.VideoCapture(0)

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
    
    def Relbow():
        x1, y1 = int(lms.landmark[12].x * imgW), int(lms.landmark[12].y * imgH)
        x2, y2 = int(lms.landmark[14].x * imgW), int(lms.landmark[14].y * imgH)
        x3, y3 = int(lms.landmark[16].x * imgW), int(lms.landmark[16].y * imgH)
        angle_deg=Figure_angle(x1,y1,x2,y2,x3,y3)

        # 顯示    
        if RL == "R":
            xPos = int(lms.landmark[14].x*imgW)
            yPos = int(lms.landmark[14].y*imgH)
            cv2.putText(frame,str(f"Relbow:{angle_deg}"),(xPos,yPos+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)# 0.4大小
            Relbow_list.append(angle_deg)
            #print("R: ",angle_deg)
        
    def Rshoulder():
        x1, y1 = int(lms.landmark[14].x * imgW), int(lms.landmark[14].y * imgH)
        x2, y2 = int(lms.landmark[12].x * imgW), int(lms.landmark[12].y * imgH)
        x3, y3 = int(lms.landmark[24].x * imgW), int(lms.landmark[24].y * imgH)
        angle_deg=Figure_angle(x1,y1,x2,y2,x3,y3)
        # 顯示    
        if RL == "R":
            xPos = int(lms.landmark[12].x*imgW)
            yPos = int(lms.landmark[12].y*imgH)
            cv2.putText(frame,str(f"shoulder:{angle_deg}"),(xPos,yPos+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)# 0.4大小
            Rshoulder_list.append(angle_deg)
    
    def Rbody():
        x1, y1 = int(lms.landmark[12].x * imgW), int(lms.landmark[12].y * imgH)
        x2, y2 = int(lms.landmark[24].x * imgW), int(lms.landmark[24].y * imgH)
        x3, y3 = int(lms.landmark[26].x * imgW), int(lms.landmark[26].y * imgH)
        angle_deg=Figure_angle(x1,y1,x2,y2,x3,y3)
        # 顯示    
        if RL == "R":
            xPos = int(lms.landmark[24].x*imgW)
            yPos = int(lms.landmark[24].y*imgH)
            cv2.putText(frame,str(f"body:{angle_deg}"),(xPos,yPos+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)# 0.4大小
            Rbody_list.append(angle_deg)

    def Rknee():
        x1, y1 = int(lms.landmark[24].x * imgW), int(lms.landmark[24].y * imgH)
        x2, y2 = int(lms.landmark[26].x * imgW), int(lms.landmark[26].y * imgH)
        x3, y3 = int(lms.landmark[28].x * imgW), int(lms.landmark[28].y * imgH)
        angle_deg=Figure_angle(x1,y1,x2,y2,x3,y3)
        # 顯示    
        if RL == "R":
            xPos = int(lms.landmark[26].x*imgW)
            yPos = int(lms.landmark[26].y*imgH)
            cv2.putText(frame,str(f"knee:{angle_deg}"),(xPos,yPos+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)# 0.4大小
            Rknee_list.append(angle_deg)

#show
while True:
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
        
            Angle.Lelbow()
            Angle.Relbow()
            Angle.Rshoulder()
            Angle.Rbody()
            Angle.Rknee()
            drawPlt()
            fps_show()
        cv2.imshow("Output",frame)


    else:
        break
    #wait_time = int(1000 / fps)
    if cv2.waitKey(1)==ord("q"):
        break


#