import cv2
import mediapipe as mp
import time
import numpy as np
import math
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)

RL = "R"
Relbow_list = []
#record_list
Relbow_list = []
Rshoulder_list = []
Rbody_list = []
Rknee_list = []


#計算角度
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

def Relbow():
        x1, y1 = int(lms.landmark[12].x * imgW), int(lms.landmark[12].y * imgH)
        x2, y2 = int(lms.landmark[14].x * imgW), int(lms.landmark[14].y * imgH)
        x3, y3 = int(lms.landmark[16].x * imgW), int(lms.landmark[16].y * imgH)
        angle_deg=Figure_angle(x1,y1,x2,y2,x3,y3)

        # 顯示    
        if RL == "R":
            xPos = int(lms.landmark[14].x*imgW)
            yPos = int(lms.landmark[14].y*imgH)
            cv2.putText(frame,str(f"R:{angle_deg}"),(xPos,yPos+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)# 0.4大小
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
            cv2.putText(frame,str(f"body:{angle_deg}"),(xPos,yPos+20),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)# 0.4大小
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
        cv2.putText(frame,str(f"knee:{angle_deg}"),(xPos,yPos+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)# 0.4大小
        Rknee_list.append(angle_deg)

#mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mp_drawing = mp.solutions.drawing_utils
pTime=0
cTime=0


while True:
    ret,frame=cap.read()
    if ret:
        imgH = frame.shape[0]
        imgW = frame.shape[1]
        #frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 將幀轉換為 RGB 格式
        results = pose.process(frame_rgb)  # 進行姿勢關鍵點檢測
        lms=results.pose_landmarks
        # 繪製關鍵點
        if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for i,lm in enumerate(results.pose_landmarks.landmark):
                    xPos = int(lm.x*imgW)
                    yPos = int(lm.y*imgH)
                Relbow() 
                Rshoulder()
                Rbody()
                Rknee()
                    

        cTime=time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime
        cv2.putText(frame,f"fps:{int(fps)}",(30,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)

        
        cv2.imshow("img",frame)
    else:
        break
    if cv2.waitKey(1)==ord('q'):
        break