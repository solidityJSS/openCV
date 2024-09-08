import cv2 as cv
import numpy as np
import os
haar_cascade=cv.CascadeClassifier('haar_face.xml')
people=['Ben Afflek','Elton John', 'Jerry Seinfield','Madonna','Mindy Kaling']
DIR=r'C:\Users\HP PC\Desktop\CV\Photos\train'
# p=[]
# for i in os.listdir(r'C:\Users\HP PC\Desktop\CV\Photos\train'):
#     p.append(i)

# print(p)
features=[]
labels=[]

def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for(x,y,w,h) in faces_rect:
                faces_roi=gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

face_recogniser=cv.face.LBPHFaceRecognizer.create()


