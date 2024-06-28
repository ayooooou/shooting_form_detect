import tkinter as tk
from tkinter import filedialog
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

class Main():
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        self.Relbow_list = []
        self.Rshoulder_list = []
        self.Rbody_list = []
        self.Rknee_list = []

        self.pTime = 0
        self.RL = "R"
        self.first_plt = True

        plt.style.use('bmh')
        plt.xlabel('time(frame)')
        plt.ylabel('angle')
        plt.plot(self.Relbow_list, 'b', label='elbow')
        plt.plot(self.Rshoulder_list, 'g', label='shoulder')
        plt.plot(self.Rbody_list, 'r', label='body')
        plt.plot(self.Rknee_list, 'y', label='knee')
        plt.legend(loc='lower left')

        self.window = tk.Tk()
        self.window.title('shooting detect')
        self.window.geometry('400x200')
        self.window.resizable(True, False)

        title_label = tk.Label(self.window, text="shooting detect", font=("Helvetica", 30))
        title_label.pack(pady=10)

        start_button = tk.Button(self.window, text="start", command=self.try_catch(self.show), width=10)
        start_button.pack(pady=10)
        stop_button = tk.Button(self.window, text="stop", command=self.try_catch(self.stop_video), width=10)
        stop_button.pack(pady=10)
        self.file_path = "none"
        self.upload_button = tk.Button(self.window, text="upload file", command=self.try_catch(self.open_file_dialog), width=10)
        self.upload_button.pack(pady=10)
        self.upload_file_name_label = tk.Label(self.window, text=f"file name:{self.file_path}", width=15)
        self.upload_file_name_label.pack(pady=10)

    def try_catch(self, func):
        def wrapper():
            try:
                func()
            except Exception as e:
                print(f"Error: {e}")
        return wrapper

    def show(self):
        self.running = True
        while self.running:
            ret, self.frame = self.cap.read()
            if ret:
                self.imgH = self.frame.shape[0]
                self.imgW = self.frame.shape[1]
                frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(self.frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        self.xPos = int(lm.x * self.imgW)
                        self.yPos = int(lm.y * self.imgH)
                        self.lms = results.pose_landmarks
                    self.R_angle(self.Relbow_list, 12, 14, 16)
                    self.R_angle(self.Rshoulder_list, 14, 12, 24)
                    self.R_angle(self.Rbody_list, 12, 24, 26)
                    self.R_angle(self.Rknee_list, 24, 26, 28)
                    self.fps_show()
                    self.drawPlt()
                cv2.imshow("Output", self.frame)
            else:
                break
            cv2.waitKey(10)
        self.cap.release()
        cv2.destroyAllWindows()

    def stop_video(self):
        self.running = False

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename()
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
