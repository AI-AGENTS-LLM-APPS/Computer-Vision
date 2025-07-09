import cv2  
import mediapipe as mp 
import time 
 
class handDetector(): 
    def __init__(self, mode=False, maxHands=2, detectionCon=0.75, trackCon=0.5): #Constructor and all the parameter are given as self for further use of that value
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils 

    def findHands(self, img, draw=True):   #To find hands and draw a line diagroam of hands 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img #As it is not a part of main and we call a fn we need to return this line doted imag to main

    def findPosition(self , img , handNo = 0 , draw = True): #Gives out the position of the hands in the terminal for further use case also it crates a colout to designed finger for further use case
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]  #To detect one hand and give its value 
            for id, lm in enumerate(myHand.landmark):   
                h,w,c = img.shape  
                cx,cy = int(lm.x*w) , int (lm.y * h) 
                # print(id,cx,cy)
                lmlist.append([id,cx,cy]) 
                # if id == 8: 
                #     cv2.circle(img,(cx,cy),15,(255,0,0),cv2.FILLED)
                # if id == 4: 
                #     cv2.circle(img,(cx,cy),15,(255,0,0),cv2.FILLED)
                

        return lmlist
    
def main():  #main() that return all then fn in this code
    cap = cv2.VideoCapture(0)
    ptime = 0
    ctime = 0 
    detector = handDetector()  #Calling that class 
    while(True):
        success, img = cap.read()
        img = detector.findHands(img) #calling fn findhands
        lmlist = detector.findPosition(img) #calling fn findPositon,feeding the hand data to a list
        # if len(lmlist) != 0 :  #Only if find hands unless it crashes thus we use this
        #     print(lmlist[4])
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
