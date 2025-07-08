import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands#initialising the hands module ig
        #self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)#giving the function of mpHands to handds
        self.hands = self.mpHands.Hands(
        static_image_mode=self.mode,
        max_num_hands=self.maxHands,
        min_detection_confidence=self.detectionCon,
        min_tracking_confidence=self.trackCon
        )
        
        self.mpDraw = mp.solutions.drawing_utils#for drawing something in the video

    def findHands(self, img, draw = True):

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)#on image draw dots and lines
                
        return img
                
    def findPosition(self, img, handNo = 0,draw =True):    
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
        
            for id,lm in enumerate(myHand.landmark):
                # print(id,lm)
                h , w , c = img.shape;
                cx , cy = int(lm.x*w) ,int(lm.y*h)
                #print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),7,(255,0,0),cv2.FILLED) 

        return lmList

def main():

    pTime = 0
    cTime = 0
    
    cap = cv2.VideoCapture(0)   
    detector = handDetector() 
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        LmList = detector.findPosition(img)
        if len(LmList) != 0:
            print(LmList[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()