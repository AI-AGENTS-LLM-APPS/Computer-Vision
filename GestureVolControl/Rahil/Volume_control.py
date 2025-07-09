import cv2
import time
import numpy as np
import mediapipe as mp
import Hand_Tracking_Module as htm
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume 
from comtypes import CLSCTX_ALL

###################
wCam , hCam  = 640 ,480
##################

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
ptime = 0
detector = htm.handDetector()


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# print(f"Audio output: {device.FriendlyName}")
# print(f"- Muted: {bool(volume.GetMute())}")
# print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
volrange = volume.GetVolumeRange()
minvol = volrange[0]
maxvol = volrange[1]
volBar = 400
vol = 0
volPer = 0
while True:
    success , img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img,draw=False)
    if len(lmlist) != 0:
        # print(lmlist[4],lmlist[8])

        x1,y1 = lmlist[4][1],lmlist[4][2];
        x2,y2 = lmlist[8][1],lmlist[8][2];
        cx,cy = (x1 +x2) // 2 ,(y1 + y2) //2

        cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
        cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
        length = math.hypot(x2-x1,y2-y1)
        # print(length)
        #Hand range = 15-230
        # Volume Range -65 - 0
        vol = np.interp(length,[15,230],[minvol,maxvol])
        volPer = np.interp(length,[15,230],[0,100])
        volBar = np.interp(length,[50,300],[230,150])
        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 30 :
            cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)
    cv2.rectangle(img, (50, 150), (85, 400), (255, 255, 0), 2)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%' , (40,450),cv2.FONT_HERSHEY_COMPLEX,1,(255255,255,0),2)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime =ctime

    cv2.putText(img, f'FPS: {int(fps)}' , (20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)