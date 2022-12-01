import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face_LBPHFaceRecognizer.create()

path = 'Train/'

def getImgWithId(path):
    ImgPaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(ImgPaths)
    faces = []
    IDs =[]
    for ImgPath in ImgPaths:
        faceImg = Image.open(ImgPath).convert("L")
        faceNp = np.array(faceImg)
        # print(faceNp)
        ID = int(ImgPath.split('/')[1].split('.')[1])

        cv2.imshow('Training...', faceNp)
        cv2.waitKey(10)

        faces.append(faceNp)
        IDs.append(ID)
    return faces,IDs

faces, IDs = getImgWithId(path)
recognizer.train(faces, np.array(IDs))

if not os.path.exists('recognizer'):
    os.mkdir('recognizer')
recognizer.save('recognizer/trainingData.yml')

cv2.destroyAllWindows()