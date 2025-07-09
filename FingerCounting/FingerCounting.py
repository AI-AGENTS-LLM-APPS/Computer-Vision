import cv2
import HandTrackingModule as htm
import time
import os

wCam , hCam = 640,480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerCounting\FingerImages"
MyList = os.listdir(folderPath)

print(MyList)

overlayList = []
for imPath in MyList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success , img = cap.read()
    img = detector.findHands(img)
    LmList = detector.findPosition(img,draw = False)
    #print(LmList)

    if len(LmList) != 0:
        fingers = []

        #Thumb
        if LmList[tipIds[0]][1] > LmList[tipIds[0]-1][1]:
                fingers.append(1)
        else:
            fingers.append(0)

        #4 fingers
        for id in range(1,5):
            if LmList[tipIds[id]][2] < LmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1) 
        #print(totalFingers)    

        h, w, c = overlayList[totalFingers - 1].shape
        img[ 0:h, 0:w ] = overlayList[totalFingers - 1]

        cv2.rectangle(img, (20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,400),cv2.FONT_HERSHEY_PLAIN,
                    10,(255,0,0), 25)
        

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
      
    cv2.putText(img, f'FPS: {int(fps)}', (400,70),cv2.FONT_HERSHEY_PLAIN,
               3,(255,0,0), 3)

    
    cv2.imshow("Image",img)
    cv2.waitKey(1)