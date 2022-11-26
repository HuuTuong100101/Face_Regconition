import cv2
import os

# Load cam
face_cascade = cv2.CascadeClassifier("C:\\Users\\tn732\\PycharmProjects\\pythonProject4\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# insert database
name = input("Nhập tên: ")

index = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (250, 0, 0), 2)
        if not os.path.exists('Image/'+str(name)):
            os.mkdir('Image/'+str(name))
        index += 1
        cv2.imwrite('Image/'+ str(name) +'/'+ str(name) + '.' + str(index)+'.jpg', gray[y:y+h, x:x+w])

    cv2.imshow('FaceApp', frame)
    cv2.waitKey(1)

    if index > 200:
        break

cap.release()
cv2.destroyAllWindows()