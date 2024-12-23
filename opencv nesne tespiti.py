# kütüphaneleri import edelim

import matplotlib.pyplot as plt
import numpy as np
import cv2


# resmi siyah beyaz olarak içe aktaralım
img = cv2.imread("odev2.jpg",0)
cv2.imshow("orjinal resim", img)


#resim üzerinde bulunan kenarları tespit edelim ve görselleştirelim
blurred = cv2.GaussianBlur(src = img, ksize = (5,5), sigmaX = 0)
edges = cv2.Canny(blurred, threshold1 = 84, threshold2 = 150)
cv2.imshow("kenar tespt", edges) 

# yüz tespit için cascade içe aktaralım
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# yüz tespit yapıp sonuçları görselleştirelim

face_reacts = face_detect.detectMultiScale(img, scaleFactor = 1.1, minSize=(30, 30),  minNeighbors = 1,  flags=cv2.CASCADE_SCALE_IMAGE)

for (x,y,w,h) in face_reacts:
    cv2.rectangle(img,(x,y), (x + w, y + h), (255,0,0), 10)
cv2.imshow("tespit", img)

# hog insan tespit algoritmamızı çağıralım ve svm yapalım
hog = cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# resime insan tespit algoritmamızı uygulayalım ve görselleştirelim

(react, weigt) = hog.detectMultiScale(img, padding = (8,8), scale = 1.05)

for (x,y,w,h) in react:
    cv2.rectangle(img, (x,y), (x + w, y + h), (0,255,0), 10)
cv2.imshow("son tespit", img)
