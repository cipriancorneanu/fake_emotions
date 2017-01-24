__author__ = 'Ikechukwu Ofodile -- ikechukwu.ofodile@estcube.eu'

import csv
import sys
import os
import numpy as np
import cv2
import Image


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture("D2N2Sur.MP4")
#img = cv2.imread('H2N2C.MP4')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#success,image = cap.read()
success = True
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 100.0, (640,480))
images = "C:\Users\Iyke Ofodile\Downloads\Luna_Robotex-master\Luna_Robotex-master/testnew/fake_SURPRISED/"
if not os.path.exists(images):
    os.makedirs(images)

csvFile  = open('C:\Users\Iyke Ofodile\Downloads\Luna_Robotex-master\Luna_Robotex-master/testnew/fake_SURPRISED/fake_SURPRISED.csv', "wb")
#csvFile = open(sys.argv[1], 'wt')
#csvFile = "C:\Users\Iyke Ofodile\Downloads\Luna_Robotex-master\Luna_Robotex-master/fake_SURPRISED/fake_SURPRISED.csv"
count = 1
left = [0,0]
right = [0,0]
writer = csv.writer(csvFile)

while success:
    #while count > 270:
    if cap.grab():
        #flag, frame = cap.retrieve()
        #success, frame = cap.read()
        success, frame = cap.retrieve()
        print 'Read a new frame: ', success , count
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray
        faces = face_cascade.detectMultiScale(roi, 1.3, 3, minSize=(150, 150),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            print (x,y), (x+w,y+h)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            #eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex, ey, ew, eh) in eyes:
                #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            writer.writerow((count, (x,y),(x+w,y+h)))

        print 'frame done', count
        cv2.imwrite("C:\Users\Iyke Ofodile\Downloads\Luna_Robotex-master\Luna_Robotex-master/testnew/fake_SURPRISED/frame%d.png" % count, frame)  # save frame as JPEG file
        count += 1

        #cv2.imshow('frame', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break
    else:
        break

# Release everything if job is finished
cap.release()
#out.release()
cv2.destroyAllWindows()