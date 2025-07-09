import cv2
import mediapipe as mp
import time
import pose_detection_module as pd


cap = cv2.VideoCapture('C:/Users/nazar/Desktop/VS Code/Projects/Computer_Vison/Pose Estimation/1.mp4');
ptime = 0 
detector = pd.poseDetector()

while True:
    success,img = cap.read()
    img = detector.findPose(img)
    lmlist = detector.findPosition(img)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img , str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,0,100),3)

    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
