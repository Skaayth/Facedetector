import cv2
import numpy as np
import requests
import io

face_cascade = cv2.CascadeClassifier("/home/skaayth/Documents/Computer Vision/haarcascade_frontalface_default.xml")

def detect_face(img):
    face_img = img.copy()
   #face_rects = face_cascade.detectMultiScale(face_img)
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,0,255),5)
    return face_img

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read(0)
    frame = detect_face(frame)
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('video face detect', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

