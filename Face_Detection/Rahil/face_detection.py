import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
ptime = 0
mpFaceDetection = mp.solutions.face_detection   #fACE DETECT WITHOUT GOING TO ITS DETAILS
mpDraw = mp.solutions.drawing_utils  #DRAW WITHOUT GOING TO ITS DETAILS
facedetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results =facedetection.process(imgRGB)
    if results.detections:
        for id,detection in enumerate(results.detections):
           
            # print(id,detection)
            h,w,c = img.shape
            # print(detection.score)
            #print(detection.location_data.relative_bounding_box)

             # mpDraw.draw_detection(img,detection) #Doing this manually 
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * w) , int(bboxC.ymin *h) ,\
                    int(bboxC.width * w) , int(bboxC.height *h) 
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img,f' {int(detection.score[0]*100)}% ',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)



    cTime = time.time()
    fps = 1/(cTime -ptime)
    ptime = cTime

    cv2.putText(img,f'fps : {(int(fps))} ',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)