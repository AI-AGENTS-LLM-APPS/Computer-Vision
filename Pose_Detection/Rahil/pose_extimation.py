import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose  #This give whether hand,pose,fase mesh are what ever
pose = mpPose.Pose()     #Gives codinates to work with 
mpDraw = mp.solutions.drawing_utils  

cap = cv2.VideoCapture('C:/Users/nazar/Desktop/VS Code/Projects/Computer_Vison/Pose Estimation/1.mp4');
ptime = 0 


while True:
    success,img = cap.read()
    img = cv2.resize(img, (640, 480))
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)  #Changing cour to rgb
    # print(results.pose_landmarks)
    if results.pose_landmarks: #These pose. landmarksare the results like the x,y,z, axis of the person point 1-32 point x,y,z per frame is given
        mpDraw.draw_landmarks(img , results.pose_landmarks,mpPose.POSE_CONNECTIONS)  #Result.landmark in draw .Puts dots on the specified point by usinf pose_connection it connects all
        for id , lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x * w) ,int(lm.y*h)
            # print(cx,cy)
            if id == 12 :
                cv2.circle(img, (cx,cy),10 , (255,0,0),cv2.FILLED)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img , str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,0,100),3)

    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break