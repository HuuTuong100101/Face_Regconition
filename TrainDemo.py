import os
from PIL import Image
import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier("C:\\Users\\tn732\\PycharmProjects\\pythonProject4\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
recognizer = cv2.face_LBPHFaceRecognizer.create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Image")

# print(os.walk(image_dir))
y_labels = []
x_train = []
id = 0
label_ids = {}

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if(file.endswith("png") or file.endswith("jpg")):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

            if not label in label_ids:
                label_ids[label] = id
                id += 1
            _id = label_ids[label]
            pil_image = Image.open(path).convert("L")
            # size = (550, 550)
            # final_img = pil_image.resize(size, Image.LANCZOS)
            image_array = np.array(pil_image, "uint8")
            cv2.imshow('Training...', image_array)
            cv2.waitKey(10)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(_id)


# print(x_train)
# print(y_labels)
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizer/training.yml")