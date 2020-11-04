import cv2
import numpy as np
import face_recognition

imgTrump = face_recognition.load_image_file('knownPeople/trump.jpg')
# convert image to rgb
imgTrump = cv2.cvtColor(imgTrump,cv2.COLOR_BGR2RGB)

imgTrumpTest = face_recognition.load_image_file('knownPeople/trumpTest.jpg')
# convert image to rgb
imgTrumpTest = cv2.cvtColor(imgTrumpTest,cv2.COLOR_BGR2RGB)

# since we are sending a single image, we can get index 0.
faceLocation = face_recognition.face_locations(imgTrump)[0]
encodeTrump = face_recognition.face_encodings(imgTrump)[0]
cv2.rectangle(imgTrump,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]),(255,0,255),2)

faceLocationTest = face_recognition.face_locations(imgTrumpTest)[0]
encodeTrumpTest = face_recognition.face_encodings(imgTrumpTest)[0]
cv2.rectangle(imgTrumpTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]),(255,0,255),2)

# time to compare the faces and finding the distance between them 128 measurements of both faces
results = face_recognition.compare_faces([encodeTrump],encodeTrumpTest)
faceDistance = face_recognition.face_distance([encodeTrump],encodeTrumpTest)
print(results,faceDistance)
# print on the result image and facedistance origin, font, scale, color, thickness

cv2.putText(imgTrumpTest,f'{results} {round(faceDistance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
# x1 y1 x2 y2 location
# print(faceLocation)
cv2.imshow('Donald Trump', imgTrump)
cv2.imshow('Donald Trump Test', imgTrumpTest)
cv2.waitKey(0)