import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode=False, model_complexity=1, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon 
        self.mpPose = mp.solutions.pose  #This give whether hand,pose,fase mesh are what ever
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, model_complexity=self.model_complexity, smooth_landmarks=self.smooth, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  


    def findPose(self , img , draw = True ): #Draw is given for whether to display or not
        img = cv2.resize(img, (640, 480))
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)  #Changing cour to rgb
        if self.results.pose_landmarks: #These pose. landmarksare the results like the x,y,z, axis of the person point 1-32 point x,y,z per frame is given
            if draw :   
                self.mpDraw.draw_landmarks(img , self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)  #Result.landmark in draw .Puts dots on the specified point by usinf pose_connection it connects all
        return img
    
    def findPosition(self, img , draw = True):
        self.lmlist = []
        if self.results.pose_landmarks: 
            for id , lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x * w) ,int(lm.y*h)
                # print(cx,cy)
                self.lmlist.append([id,cx,cy])
                if draw :
                    cv2.circle(img, (cx,cy),10 , (255,0,0),cv2.FILLED)
        return self.lmlist
    def findAngle(self,img,p1,p2,p3,draw =True):

        #Get Landmarks
        x1,y1 = self.lmlist[p1][1:]
        x2,y2 = self.lmlist[p2][1:]
        x3,y3 = self.lmlist[p3][1:]

        #Calculate the Angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
        if angle < 0:
            angle += 360
        # print(angle)
        #Draw
        if draw :
            cv2.line(img,(x1,y1),(x2,y2),(255,255,0),3)
            cv2.line(img,(x3,y3),(x2,y2),(255,255,0),3)
            cv2.circle(img, (x1,y1),10 , (255,0,0),cv2.FILLED)
            cv2.circle(img, (x1,y1),15 , (255,0,0),2)
            cv2.circle(img, (x2,y2),10 , (255,0,0),cv2.FILLED)
            cv2.circle(img, (x2,y2),15 , (255,0,0),2)
            cv2.circle(img, (x3,y3),10 , (255,0,0),cv2.FILLED)
            cv2.circle(img, (x3,y3),15 , (255,0,0),2)
            cv2.putText(img,str(int(angle)),(x2-20,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        return angle


def main():
    
    cap = cv2.VideoCapture('C:/Users/nazar/Desktop/VS Code/Projects/Computer_Vison/Pose Estimation/1.mp4');
    ptime = 0 
    detector = poseDetector()

    while True:
        success,img = cap.read()
        img = detector.findPose(img)
        lmlist = detector.findPosition(img)
        angle = detector.findAngle(img,12 ,14 ,16)
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img , str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,0,100),3)

        cv2.imshow("Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__": 
    main() 