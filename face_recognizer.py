
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


FaceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer\\trainingData.yml')
ID=0
font=cv2.FONT_HERSHEY_SIMPLEX


# In[3]:


name1=input('Please Enter Name of person ID=1: ')

while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=FaceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),3)
        ID, conf=rec.predict(gray[y:y+h,x:x+w])
        if ID==1 and conf>=20:
            ID=name1  
        else :
            ID='unknown'
        cv2.putText(img,str(ID),(x,y+h),font,1,(0,255,0),2)
        cv2.putText(img,str(round(conf,2))+'%',(x,y),font,1,(0,255,0),2)
    cv2.imshow("Face",img)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

