import cv2 as cv
img=cv.imread('Photos/maxine.jpg')
#  cv.imshow('nigga',img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('gray nigga',gray)

haar_cascade=cv.CascadeClassifier('haar_face.xml')

faces=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
print(faces)

for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(w+x,h+y),255,thickness=2)
cv.imshow('face',img)
cv.waitKey(0)