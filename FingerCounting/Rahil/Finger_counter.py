import os
import cv2
import time
import Hand_Tracking_Module as htm

wCam , Cam = 640 , 480 

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,Cam)

folder = "FingerImages"
myList = os.listdir(folder)
# print(myList)
ptime = 0
overlayList = []

tipIds = [4,8,12,16,20]

for imPath in myList:
    image = cv2.imread(f'{folder}/{imPath}')
    if image is None:
        print(f"Error loading: {imPath}")
    else:
        overlayList.append(image)

detector = htm.handDetector(detectionCon=0.75)

while True : 
    success , img = cap.read()
    img[0:200,0:200] = overlayList[0]
    img = detector.findHands(img)
    lmlist = detector.findPosition(img,False)
    if len(lmlist) != 0:  
        fingers = []  
        for id in range(0,5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-1][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFinger = fingers.count(1)
        h , w ,c = overlayList[totalFinger-1].shape
        img[0:h , 0:w] = overlayList[totalFinger-1]
        cv2.rectangle(img, (20, 225), (170, 425), (0, 0, 0), cv2.FILLED)  # black background
        cv2.putText(img, str(totalFinger), (45, 375), cv2.FONT_HERSHEY_PLAIN,10, (255, 255, 255), 10)  # white text

    ctime= time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,f'FPS: {int(fps)}',(400,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,200),2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
