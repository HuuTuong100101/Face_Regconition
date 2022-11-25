import cv2
import numpy as np
import os
import sqlite3
from PIL import Image

# Huấn luyện hình ảnh nhận diện khuôn mặt với thư viên nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier("C:\\Users\\tn732\\PycharmProjects\\pythonProject4\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
recognizer = cv2.face_LBPHFaceRecognizer.create()

recognizer.read("D:\\HK1_2022_2023\\CT466_NLCN\\PycharmProjects\CT466\\recognizer\\trainingData.yml")

# Lấy thông tin khuôn mặt theo id
def GetProfileById(id):
    connect = sqlite3.connect('data.db')
    qr = 'SELECT * FROM People WHERE ID =' + str(id)
    datas = connect.execute(qr)
    profile = None
    for data in datas:
        profile = data
    connect.close()
    return profile

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x+20, y-20), (x+w-20, y+h+20), (250, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        id,confidence = recognizer.predict(roi_gray)
        if confidence < 40:
            profile = GetProfileById(id)
            if profile is not None:
                cv2.putText(frame, ""+str(profile[1]), (x+30, y+h+50), font, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x + 10, y + h + 30), font, 1, (0, 255, 0), 2)
    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()