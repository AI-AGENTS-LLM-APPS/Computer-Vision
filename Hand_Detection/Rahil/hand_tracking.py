import cv2  
import mediapipe as mp 
import time 

cap = cv2.VideoCapture(0)  #
 
mpHands = mp.solutions.hands   #Starting use the module 
hands = mpHands.Hands()          #For using hands in it DEFAULT PARAMETER ARE THERE
mpDraw = mp.solutions.drawing_utils   # Drawing a line requires a lot of understanding in the API thus they created to understand all the draw compand so we drawing utilies  

ptime = 0
ctime = 0

while(True):
    success,img = cap.read() #Reads frame from web cam
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  #Converting BGR TO RGB because this module only uses rgb  
    results = hands.process(imgRGB)  #Processing the frame and giving the result
    # print(results.multi_hand_landmarks)  #Detecting whether the hand is there 

    if results.multi_hand_landmarks:  #If there are multiple hands and create a problem while creating thus we use one hand configuration              
        for handLms in results.multi_hand_landmarks:  #Using one hand
            for id, lm in enumerate(handLms.landmark):   # Take al; thne index of x,y and z .This is used for creating take all changing frames and take its id and,x,y,z, There are 20 id the landmarks have x,y and z
                # print(id,lm)  
                h,w,c = img.shape  #Width and height of the image  shape
                cx,cy = int(lm.x*w) , int (lm.y * h) #(landmark.x value multiplied by weidth);
                print(id,cx,cy)  #This is for all 21 values to understand the id of hand 
                if id == 8: #Position of API FINGER
                    cv2.circle(img,(cx,cy),15,(255,0,0),cv2.FILLED)  #Change the colour of all the points
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)  #Drawing the  connecting the lines in one hand and connectes all the 21 lines
    
    ctime = time.time()
    fps = 1/(ctime-ptime)  #Current - present time by 1 is fps
    ptime = ctime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3) #Printing fps() on the img
    
    cv2.imshow("Image",img)  #Show the image
    cv2.waitKey(1)