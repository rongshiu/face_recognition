import cv2
import numpy as np
import os
from PIL import Image

FaceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
ID=1


def capture():
    SampleNum=0
    while(True):
        ret,img=cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=FaceDetect.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            SampleNum=SampleNum+1
            cv2.imwrite('DataSet/User.'+str(ID)+'.'+str(SampleNum)+'.jpg',gray[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.waitKey(100)
        cv2.imshow("Face",img)
        cv2.waitKey(1)
        if SampleNum>99:
            break
    cam.release()
    cv2.destroyAllWindows()
capture()


recognizer=cv2.face.LBPHFaceRecognizer_create()
path='DataSet'

def GetImageID(path):
    ImagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in ImagePaths:
        FaceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(FaceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
    return IDs, faces

IDs, faces= GetImageID(path)
recognizer.train(faces,np.array(IDs))
recognizer.write('recognizer/trainingData.yml')
cv2.destroyAllWindows()

