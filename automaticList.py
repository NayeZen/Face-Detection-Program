import cv2
import numpy as np
import face_recognition
import os
#Library for time and date
from datetime import datetime
#Folder being accessed
path = 'listOfPeople'
#array of pictures
images = []
#array of names of images
classNames = []

# Grab the list of names from the folder and print out the whole name
myList = os.listdir(path)
print(myList)
# use these names and import the images one by one
for cl in myList:
    #Grab the current image from the path @ cl
    curImg = cv2.imread(f'{path}/{cl}')
    #Add the image to the images array
    images.append(curImg)
    #Add the name of people into the classNames array and split the text and only get the first element.
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    #List of encoded images
    encodeList = []
    for img in images:
        #convert to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Encode image
        encode = face_recognition.face_encodings(img)[0]
        #Append encoding to the list
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    #Use the attendance file cvs=seperated by commmas
    #Read and write at the same time = r+
    #Save as f
    with open('Attendance.cvs', 'r+') as f:
        myDataList = f.readline()
        #All the name that we find, we want to put it in this list
        nameList = []
        for line in myDataList:
            #Entry has two values
            entry = line.split(',')
            nameList.append(entry[0])
        #If the name of the person is not in the list, print the following.
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#Send array of images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

#Initialize web cam
cap = cv2.VideoCapture(0)

#Get each frame one by one
while True:
    success, img = cap.read()
    #resized the image to 1/4 the size.
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    #Convert to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodingOfCurrentFrame = face_recognition.face_encodings(imgS,faceCurrentFrame)

    #Compare all of the faces currently being captured in the frame with the faces of the list.
    #Using zip since we are using them in the same loop.
    for encodeFace,faceLoc in zip(encodingOfCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDistance)
        #Grab the lowest value to determine the match.
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            #Get the name of the index
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            #Scale the image back to its original size. It was 1/4 so multiplied by 4 brings it back to
            #its original size.
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            #Draw a rectangle around the face
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            #Draw a fill under the rectangle
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            #Add the name of the person on the box.
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)


        cv2.imshow('Webcam',img)
        cv2.waitKey(1)
