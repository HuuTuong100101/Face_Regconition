import cv2
import pickle

# Huấn luyện hình ảnh nhận diện khuôn mặt với thư viên nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier("C:\\Users\\tn732\\PycharmProjects\\pythonProject4\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.read("recognizer/training.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (250, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        id, conf = recognizer.predict(roi_gray)
        if conf < 45:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            cv2.putText(frame, name, (x, y), font, 1, (250, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()