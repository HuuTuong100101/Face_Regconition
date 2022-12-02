import sqlite3
import cv2
import os
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
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
dir = r'D:\HK1_2022_2023\CT466_NLCN\PycharmProjects\CT466\Video'

label = []

pathVideo = []
pathFolder = []
# index = 0

for i in os.listdir(dir):
    label.append(i)
    path1 = os.path.join(dir, i)
    pathFolder.append(path1)

    for j in os.listdir(path1):
        path2 = os.path.join(path1, j)
        pathVideo.append(path2)

for i in pathVideo:
    id = round(random.uniform(0, 10) * 1000000)
    Name = i.split('\\')[6]
    insertAndUpdate(id, Name)
    index = 0
    cap = cv2.VideoCapture(i)
    ran = random.randint(0, 201)
    while True:
        print(index)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x+20, y-20), (x+w-20, y+h+20), (250, 0, 0), 2)
            if not os.path.exists('Train'):
                os.mkdir('Train')
            if not os.path.exists('Test'):
                os.mkdir('Test')
            if index != ran:
                cv2.imwrite('Train/User.' + str(id)+'.'+str(index)+'.jpg', gray[y:y+h, x:x+w])
            else:
                cv2.imwrite('Test/User.' + str(id) + '.' + str(index) + '.jpg', frame)

            index += 1
        cv2.imshow('FaceApp', frame)
        cv2.waitKey(1)

        if index > 201:
            break

    cap.release()
    cv2.destroyAllWindows()