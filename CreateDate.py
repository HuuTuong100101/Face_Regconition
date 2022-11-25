import sqlite3
import cv2
import os
import numpy as np
import random
def insertAndUpdate(id, name):
    connect = sqlite3.connect('data.db')
    qr = "SELECT * FROM People WHERE ID = " + str(id)
    data = connect.execute(qr)
    checkId = 0
    for row in data:
        checkId = 1
    if checkId == 0:
        qr = "INSERT INTO People(ID, Name) VALUES (" + str(id) + ",'" + str(name) + "')"
    else:
        qr = "UPDATE People SET Name = '" + str(name) + "'WHERE ID = " + str(id)
    connect.execute(qr)
    connect.commit()
    connect.close()


# Load cam
face_cascade = cv2.CascadeClassifier("C:\\Users\\tn732\\PycharmProjects\\pythonProject4\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# insert database
id = round(random.uniform(0,10) * 1000000)
name = input("Nhập tên: ")
insertAndUpdate(id, name)

index = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x+20, y-20), (x+w-20, y+h+20), (250, 0, 0), 2)
        if not os.path.exists('Image'):
            os.mkdir('Image')
        index += 1
        cv2.imwrite('Image/User.' + str(id)+'.'+str(index)+'.jpg', gray[y:y+h, x:x+w])

    cv2.imshow('FaceApp', frame)
    cv2.waitKey(1)

    if index > 200:
        break

cap.release()
cv2.destroyAllWindows()