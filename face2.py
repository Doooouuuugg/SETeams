import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascase = cv2.CascadeClassifier('haarcascade_eye.xml')
upperBody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
#lowerBody_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
#fullBody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    #lower = lowerBody_cascade.detectMultiScale(gray,1.3,5)
    #full = fullBody_cascade.detectMultiScale(gray,1.3,5)
    upper = upperBody_cascade.detectMultiScale(gray,1.3,3)
    for (x,y,w,h) in upper:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]
        #eyes = eye_cascase.detectMultiScale(roi_gray)
        #upper = upperBody_cascade.detectMultiScale(gray,1.3,5)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey), (ex+ew, ey+eh ), (0,255,0),2)

    cv2.imshow('img', img)
    k=cv2.waitKey(30) & 0xFF
    if k==27:
        break



    # cv2.imshow('frame',frame)
    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #     break 
cap.release()
cv2.destroyAllWindows() 