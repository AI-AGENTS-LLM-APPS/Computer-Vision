import mediapipe
import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

###########################
wCam , hCam = 640,480
###########################

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0

detector = htm.handDetector(detectionCon = 0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
#print(volume.GetVolumeRange())


minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success ,img = cap.read()
    img = detector.findHands(img)
    LmList = detector.findPosition(img,draw = False)
    if len(LmList) != 0:
        #print(LmList[4],LmList[8])

        x1,y1 = LmList[4][1] , LmList[4][2]
        x2,y2 = LmList[8][1] , LmList[8][2]
        cx , cy = (x1+x2)//2 , (y1+y2)//2

        cv2.circle(img,(x1,y1), 15 , (255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2), 15 , (255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy), 15 , (255,0,255),cv2.FILLED)
    
        length = math.hypot(x2-x1,y2-y1)
        

        #Hand range 50 -300
        #Volume range -65 - 0
        vol = np.interp(length,[50,300],[minVol,maxVol])
        volBar = np.interp(length,[50,300],[400,150])
        volPer = np.interp(length,[50,300],[0,100])

        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img,(cx,cy), 15 , (0,255,0),cv2.FILLED)
    
    cv2.rectangle(img, (50,150), (85,400), (255,0,0), 3)
    cv2.rectangle(img, (50,int(volBar)), (85,400), (255,0,0),cv2.FILLED)
    
        
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',(30,50),
                cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    cv2.putText(img, f'{int(volPer)} %',(40,450),
                cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    
    cv2.imshow("Img",img)
    cv2.waitKey(1)

